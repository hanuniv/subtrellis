import numpy as np
import sympy
from sympy import symbols, factor
from scipy.ndimage.interpolation import shift as spshift
from collections import namedtuple, Counter
# from recordclass import recordclass
import itertools
import plotly.offline as py
import plotly.graph_objs as go

from pprint import pprint

# import networkx as nx


class SubDecode:
    """
    class for testing which coset a codeword belongs to.
    C is the row space of coset leaders and S spans the coset.
    [C;S] has full row rank.
    """

    def __init__(self, C, S):
        """ save C and S matrices """
        if C.shape[1] != S.shape[1]:
            raise ValueError("C and S should have the same number of columns!")
        self.C = C
        self.S = S
        self.G = np.vstack((C, S))
        self.T = 0  # TODO, precalculate inversion matrix

    def decode(self, c):
        """ Gives the C component of the decoded word, ignores the residual """
        mc = self.C.shape[0]
        x, r, *_ = np.linalg.lstsq(self.G.T, c)
        return x[:mc].round()


class SubDecodeFirstGenerator(SubDecode):
    """
    use the first mc rows of the generator for coset leaders
    """

    def __init__(self, G, mc):
        super().__init__(G[:mc], G[:mc])


class SubDecodeSelectedGenerator(SubDecode):
    """
    use the selected set of rows of the generator for coset leaders
    """

    def __init__(self, G, lstmc):
        lstmc = list(set(lstmc))
        lstmcc = list(set(range(G.shape[0])) - set(lstmc))
        super().__init__(G[lstmc], G[lstmcc])


class SubDecodeCombineGenerator(SubDecode):
    """
    use a combination of rows in the generator matrix for coset leaders
    the decoder will decode to the solution in the subspace basis
    The input combination should be a full rank combination
    """

    def __init__(self, G, CB):
        C = np.array(CB).dot(G) % 2
        _, ind = sympy.Matrix(CB).rref()
        indc = list(set(range(G.shape[0])) - set(ind))  # ind's complement
        SB = np.zeros(shape=(G.shape[0] - C.shape[0], G.shape[0]))
        SB[range(SB.shape[0]), indc] = 1              # S's index
        S = SB.dot(G)
        assert np.linalg.matrix_rank(np.vstack((CB, SB))) == G.shape[0], \
            "Basis coefficients do not yield a full rank generator"
        super().__init__(C, S)


def cor(a, b):
    """
    obtain correlation between the two codewords whose entries are between 0 and 1.
    returns (1-2a).(1-2b)

    >>> cor([1, 1], [1, 1])
    2
    >>> cor([1, 0], [1, 1])
    0
    """
    return np.inner(1 - 2 * np.array(a), 1 - 2 * np.array(b))


TrellisEdge = namedtuple('TrellisEdge', 'begin end weight')


class Trellis:
    def __init__(self, G, S=None):
        """ Does not enforce G to be Trellis oriented"""
        if S is None:
            S = minspan(G)
        self.G = G
        self.S = S
        self.A, self.B, self.alpha, self.beta, self.rhoplus, self.rhominus, self.V, self.E = trellis(
            G, S)
        self.codewords = []
        self.messages = np.array(
            list(itertools.product([0, 1], repeat=G.shape[0])))
        for i in self.messages:
            self.codewords.append(i.dot(G) % 2)
        self.codewords = np.array(self.codewords)


def closest(c, codewords):
    """
    return codewords in trellis T that are closest to c
    """
    d = np.array([sum((c + w) % 2) for w in codewords[:, :len(c)]])
    return min(d), codewords[d == min(d)]


def ncloeset(c, codewords):
    dmin, codes = closest(c, codewords)
    return dmin, codes.shape[0]


def closestat(c, codewords):
    """
    return a list, the first entry counts the number of codewords that has 0 error from c, the second counts 1 error, etc.

    """
    n = codewords.shape[1]
    if len(c) > n:
        raise Exception("Cannot deal with strings longer than n.")
    d = np.array([sum((c + w) % 2) for w in codewords[:, :len(c)]])
    c = Counter(d)
    return [c[i] if i in c else 0 for i in range(n + 1)]


def select_subcode(T, nodes):
    """
    select codewords that passes through a given set of nodes
    nodes = [(level, [markers, ... ]), (level, [markers, ... ])]
    not all level has to be specified. select codewords that passes through all markers
    return a boolearn array that can be used to slice T.codewords and T.messages
    """
    m = T.messages
    mcount = np.zeros(m.shape[0])
    for l, vs in nodes:
        ml = m[:, list(T.B[l])]
        mt = np.zeros(m.shape[0])
        for v in vs:
            av = np.array([int(_) for _ in v])
            mt[np.sum((ml - av) % 2, axis=1) == 0] += 1
        mcount[mt > 0] += 1
    return mcount == len(nodes)


def select_subtrellis(T, nodes):
    """
    return the edge set passed by the node subsets
    """
    m = T.messages[select_subcode(T, nodes)]
    n = T.G.shape[1]
    V = []
    E = []
    if m.size > 0:
        V.append(T.V[0])
    for i in range(1, n + 1):
        # states that are passed through by the selected message
        mi = m[:, list(T.B[i])]
        vi = [''.join(str(b) for b in _) for _ in mi]
        vi = list(set(vi))  # drop duplicates, sort to preserve order
        vi.sort()
        V.append(vi)
    for i in range(n):
        Ei = []
        for e in T.E[i]:
            if e.begin in V[i] and e.end in V[i + 1]:
                Ei.append(e)
        E.append(Ei)
    return V, E


def minspan(G):
    """
    >>> G = np.array([[1, 1, 0, 1, 1, 0, 0],[0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]);
    >>> minspan(G)
    [(0, 4), (1, 3), (4, 6)]
    """
    span = []
    for row in G:
        loc = np.nonzero(row)
        l = np.min(loc)
        h = np.max(loc)
        span.append((l, h))
    return span


def trellis(G, S):
    """
    Get the row covers items from S
    returns A, B, alpha, beta, rhoplus, rhominus, V, E
    """
    n = G.shape[1]
    A = [set()]
    alpha = [0]
    for i in range(n):
        a = []
        for j in range(len(S)):
            if S[j][0] <= i and i <= S[j][1]:
                a.append(j)
        A.append(set(a))
        alpha.append(len(a))
    B = [set()]
    beta = [0]
    for i in range(1, n):
        B.append(A[i].intersection(A[i + 1]))
        beta.append(len(B[i]))
    B.append(set())
    beta.append(0)
    # rhos
    rhoplus = []
    rhominus = [0]
    for i in range(n):
        rhoplus.append(2**(alpha[i + 1] - beta[i]))
        rhominus.append(2**(alpha[i + 1] - beta[i + 1]))
    rhoplus.append(0)
    # V and E, here E[i] = E_{i,i+1} in the paper
    V = []
    for b in beta:
        V.append(bits(b))
    E = []
    for i in range(0, n):
        gt = G[:, i][list(A[i + 1])]
        gt = ''.join([str(_) for _ in gt])
        us = bits(alpha[i + 1])
        Ei = []
        for u in us:
            lu = inner(u, gt)
            initu = sindex(u, relindex(B[i], A[i + 1]))
            finu = sindex(u, relindex(B[i + 1], A[i + 1]))
            Ei.append(TrellisEdge(*[initu, finu, lu]))
        E.append(Ei)
    return A, B, alpha, beta, rhoplus, rhominus, V, E


def plottrellis(T, subE=None, title='Trelis', statelabel=None):
    def edgetrace(V, E, width=2, color='#888'):
        edge_trace0 = go.Scatter(
            x=[],
            y=[],
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines')
        edge_trace1 = go.Scatter(
            x=[],
            y=[],
            line=dict(width=width, color=color, dash='dot'),
            hoverinfo='none',
            mode='lines')
        for i, Ei in enumerate(E):
            for e in Ei:
                x0 = i
                x1 = i + 1
                y0 = V[i].index(e.begin)
                y1 = V[i + 1].index(e.end)
                if e.weight == 0:
                    edge_trace0['x'] += tuple([x0, x1, None])
                    edge_trace0['y'] += tuple([y0, y1, None])
                else:
                    edge_trace1['x'] += tuple([x0, x1, None])
                    edge_trace1['y'] += tuple([y0, y1, None])
        return edge_trace0, edge_trace1
    V = T.V
    E = T.E
    n = T.G.shape[1]
    a_trace = go.Scatter(
        x=[-0.5] + list(range(1, n)),
        y=[-0.2] * (n + 1),
        text=['A'] + [str(_) for _ in T.A[1:n]],
        mode='text',
        hoverinfo='none'
    )
    b_trace = go.Scatter(
        x=[-0.5] + list(range(1, n)),
        y=[-0.5] * n,
        text=['B'] + [str(_) for _ in T.B[1:n]],
        mode='text',
        hoverinfo='none'
    )
    rhoplus_trace = go.Scatter(
        x=[-0.5] + list(range(n)),
        y=[-0.8] * (n + 1),
        text=['rho+'] + [str(_) for _ in T.rhoplus[0:n]],
        mode='text',
        hoverinfo='none'
    )
    rhominus_trace = go.Scatter(
        x=[-0.5] + list(range(n)),
        y=[-1.0] * (n + 1),
        text=['rho-'] + [str(_) for _ in T.rhominus[0:n]],
        mode='text',
        hoverinfo='none'
    )
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='top center',  # 'bottom center',
        hoverinfo='none'
    )
    for i, vi in enumerate(V):
        for j, v in enumerate(vi):
            x, y = i, j
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            try:
                node_trace['text'] += tuple(
                    [np.array2string(statelabel[(i, v)])])
            except:
                node_trace['text'] += tuple([v])
    edge_trace0, edge_trace1 = edgetrace(V, E)
    if subE is not None:
        bedge_trace0, bedge_trace1 = edgetrace(
            V, subE, width=3.5, color='rgba(0, 0, 152, .8)')
        data = [edge_trace0, edge_trace1, bedge_trace0, bedge_trace1, node_trace,
                a_trace, b_trace, rhoplus_trace, rhominus_trace]
    else:
        data = [edge_trace0, edge_trace1, node_trace,
                a_trace, b_trace, rhoplus_trace, rhominus_trace]
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        title=title,
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[ dict(
                        #     text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    py.init_notebook_mode(connected=True)
    py.iplot(fig, filename='trellis')


def bits(b):
    """
    return all binary string of length b

    >>> bits(0)
    ['']
    >>> bits(1)
    ['0', '1']
    """
    if b == 0:
        return ['']
    else:
        return [format(_, '0' + str(b) + 'b') for _ in range(2**b)]


def inner(u, gt):
    """
    >>> inner('11', '00')
    0
    >>> inner('01', '10')
    0
    >>> inner('01', '11')
    1
    """
    return sum(ui == '1' and gi == '1' for ui, gi in zip(u, gt)) % 2


def sindex(u, ind):
    """
    helper function that indexes a bit string with a set

    >>> sindex('010001', {0, 1, 4, 5})
    '0101'
    """
    return ''.join(u[i] for i in ind)


def relindex(B, A):
    """
    relative index of B in A, A and B are sorted.
    >>> relindex({1, 2}, {0, 1, 2})
    [1, 2]
    >>> relindex({1}, {0, 2, 1})
    [1]
    """
    A = list(A)
    B = list(B)
    A.sort()
    B.sort()
    return [A.index(b) for b in B]


def gensort(G):
    S = minspan(G)
    S = np.fliplr(np.array(S))
    S[:, 0] *= -1   # sort the R reversely
    ind = np.lexsort(S.T)
    S[:, 0] *= -1
    S = np.fliplr(S)
    return S[ind], G[ind]


def minspangen(G):
    """
    obtain a minimum span generator matrix
    """
    # # Wrong implementation, neighboring add is not enough
    # while True:
    #     S, G = gensort(G)
    #     print(G)
    #     Gt = G.copy()
    #     for i in range(G.shape[0]-1):
    #         if 0 in S[i]-S[i+1]:
    #             Gt[i+1] += G[i]
    #             Gt[i+1] = Gt[i+1] % 2
    #     if np.all(G == Gt):
    #         break
    #     else:
    #         G = Gt
    while True:
        # We always have when j<i, L(j) <= L(i), when L(i)==L(j), R(j) >= R(i)
        S, G = gensort(G)
        Gt = G.copy()
        for j in range(G.shape[0] - 1):
            for i in range(j + 1, G.shape[0]):
                # Seems S[i,1] == S[j, 1] is enough,
                if (S[i, 0] == S[j, 0] and S[i, 1] <= S[j, 1]) or (S[i, 0] >= S[j, 0] and S[i, 1] == S[j, 1]):
                    Gt[j] += G[i]
                    Gt[j] = Gt[j] % 2
        if np.all(G == Gt):
            break
        else:
            G = Gt
    if not isminspan(G):
        raise Exception('Not Minimum Span!')
    return G


def isminspan(G):
    S = minspan(G)
    return np.unique(np.array(S)[:, 0]).shape[0] == G.shape[0] and np.unique(np.array(S)[:, 1]).shape[0] == G.shape[0]


def maxsub_strategy(n0, n1):
    """picking 0 yields more closest subword, return True means sending 0"""
    return n0.dsub[1] >= n1.dsub[1]


def maxratio_strategy(n0, n1):
    # return n0.dsub[0] >= n1.dsub[0] and n0.dsub[1] / n0.dcode[1] >= n1.dsub[1] / n1.dcode[1]
    # return n0.dsub[0] == n1.dsub[0] and n0.dsub[1] / n0.dcode[1] >= n1.dsub[1] / n1.dcode[1]
    # return n0.dsub[0] == n1.dsub[0] and n0.dsub[1] / n0.dcode[1] > n1.dsub[1] / n1.dcode[1]
    return n0.dsub[1] / n0.dcode[1] >= n1.dsub[1] / n1.dcode[1]
    # return n0.dsub[1] / n0.dcode[1] > n1.dsub[1] / n1.dcode[1]


def no_strategy(n0, n1):
    return True


def winlose(a, b):
    """ Combine wining / losing / boundary states in [Y N _]"""
    if a == 'Y' and 'b' == 'Y':
        return 'Y'
    elif a in ['_', 'Y'] or 'b' in ['_', 'Y']:
        return '_'
    else:
        return 'N'


def simulate_subcode(sub, T, strategy=maxsub_strategy):
    """
    Simulate the forward evolution of the code, given trellis and strategy.

    The strategy is assumed to based solely on the comparison of the number of path in the subcode
    and the number of path in the base code.

    returns: piles, a list of length n, each item is a list (pile) of Codeprob items
    """

    # Codeprob class: c = feedback word, dsub = number of path in the subcode, dcode = paths in base code.
    Codeprob = namedtuple('Codeprob', 'c prob dsub dcode winning')
    n = T.G.shape[1]
    p = symbols('p')  # crossover probability
    pile = []
    pile.append(Codeprob(c=[0], dsub=ncloeset([0], sub), dcode=ncloeset(
        [0], T.codewords), prob=p, winning=None))
    pile.append(Codeprob(c=[1], dsub=ncloeset([1], sub), dcode=ncloeset(
        [1], T.codewords), prob=1 - p, winning=None))
    piles = [pile]
    # Going through paths, calculate forward probability
    for i in range(n - 1):
        newpile = []
        for w, prob, *_ in pile:
            n0 = Codeprob(*[w + [0], prob, ncloeset(w + [0], sub),
                            ncloeset(w + [0], T.codewords), None])
            n1 = Codeprob(*[w + [1], prob, ncloeset(w + [1], sub),
                            ncloeset(w + [1], T.codewords), None])
            if strategy(n0, n1):
                q = 1 - p
            else:
                q = p
            # n0.prob *= q
            # n1.prob *= 1 - q
            # newpile.extend([n0, n1])
            newpile.extend([n0._replace(prob=prob * q),
                            n1._replace(prob=prob * (1 - q))])
        pile = newpile
        piles.append(pile)
    # Going back, calculate winning and losing states
    pile = piles[n - 1]
    for i in range(len(pile)):
        if (pile[i].dsub[0] < pile[i].dcode[0]) or (pile[i].dsub[0] == pile[i].dcode[0] and pile[i].dsub[1] > pile[i].dcode[1] / 2):
            pile[i] = pile[i]._replace(winning='Y')
        elif (pile[i].dsub[0] == pile[i].dcode[0] and pile[i].dsub[1] == pile[i].dcode[1] / 2):
            pile[i] = pile[i]._replace(winning='_')
        else:
            pile[i] = pile[i]._replace(winning='N')
    backpropagatewl(piles)
    return piles


def steptuple(codetuple, T, level, bit, p):
    """ return a list of possible states for the next level given bits received"""
    c, prob, *_ = codetuple
    return [codetuple._replace(prob=prob * (1 - p), c=c + [bit]),
            codetuple._replace(prob=prob * p, c=c + [1 - bit])]


class LookaheadStrategy:
    """
    a lookahead strategy class based on the trellis T and number of lookahead
    """

    def __init__(self, T, nlook):
        self.T = T
        self.nlook = nlook  # number of look ahaed

    def hitme(self, piles, level, p=0.9):
        """ return the next bit to send, given current piles and level"""
        lookpiles = [piles[-1]]
        for _ in range(self.nlook):
            step(lookpiles, T, level)  # TODO,add next level probability
        # Skip: assign winlose to the last pile
        # backpropagatewl(lookpile)
        # find the one with the best wining probability, when there is a tie, include them all
        bestprob = 0
        bestc = []
        for endings in lookpiles[-1]:
            if endings.prob == bestprob:
                bestc.append(endings.c)
            # TODO, find substitute function
            elif substitute(endings.prob, p) > substitute(bestprob, p):
                bestprob = endings.prob
                bestc = [endings.c]
        return bestc[0][0]  # decide the first choice of the first best policy


def send0always(ns):
    return True


def send0subs(ns):
    """
    Decide whether to send 0 based on the lookahead results in ns
    ns is a list of distance statistics

    Decide to send 0 if it is the majority choice of the wining state
    """
    ds = [_.c[0] for _ in ns if _.winning == 'Y']
    c = Counter(ds)
    # pprint(c)
    return c[0] >= c[1]


def simulate_lookahead(subs, T, nlook, ne=2, send0=send0subs):
    """
    A reworked version of simulation that applies to look ahead policies

    subs: a list of subtrellises, the goal is to transmit the first subtrellis.

    returns: piles, a list of length n, each item is a list (pile) of Codeprob items
    """
    # Codeprob class: c = feedback word, dsub = number of path in the subcode, dcode = paths in base code.
    Codeprob = namedtuple('Codeprob', 'c prob dsub winning')
    n = T.G.shape[1]
    p = symbols('p')  # crossover probability
    # The first pile
    pile = [Codeprob(c=[], dsub=[], prob=1, winning=None)]
    piles = [pile]
    # Going through paths, calculate forward probability
    for i in range(n):
        newpile = []
        w = {}
        for c, prob, *_ in pile:
            ns = []
            for _ in itertools.product([0, 1], repeat=min(nlook, n - i)):
                # w = viterbilist(T, c+list(_), ne, start=i, init=w)
                # pprint(w)
                _dsub = [closestat(c + list(_), sub) for sub in subs]
                nnode = Codeprob(c=c + list(_), prob=prob,
                                 dsub=_dsub, winning=iswiningstat(_dsub))
                ns.append(nnode)
            pprint(ns)
            if send0(ns):
                q = 1 - p
            else:
                q = p
            _dsub0 = [closestat(c + [0], sub) for sub in subs]
            _dsub1 = [closestat(c + [1], sub) for sub in subs]
            newpile.extend([Codeprob(*[c + [0], prob * q, _dsub0, iswiningstat(_dsub0)]),
                            Codeprob(*[c + [1], prob * (1 - q), _dsub1, iswiningstat(_dsub1)])])
        pile = newpile
        piles.append(pile)
    # Going back, calculate winning and losing states
    pile = piles[n - 1]
    for i in range(len(pile)):
        pile[i] = pile[i]._replace(winning=iswiningstat(pile[i].dsub))
    backpropagatewl(piles)
    return piles


def iswiningstat(dsub):
    """
    tell if the statistics of distance dsub allows the first subtrellis count to outweigh others

    returns:
    'Y' if among the closest the first subtrellis has the most
    """
    a = np.array(dsub)
    i = np.sort(np.where(np.sum(a, axis=0) > 0)[0])[0]
    if np.all(a[1:, i] < a[0, i]):
        return 'Y'
    elif np.all(a[1:, i] <= a[0, i]):
        return '_'
    else:
        return 'N'


def viterbi(T, c):
    """
    run viterbi algorithm on a trellis with codeword c, c does not have to be complete.

    returns: a pair of dictionary (d, w)
    d marks distance  to every node in T.V[:len(c)] that has the fewest difference from c
    w records the corresponding best paths, if there is a tie, preserve all of them.
    """
    n = T.G.shape[1]
    if len(c) > n:
        raise Exception("Senseword length exceeds codeword.")
    d = {}  # dictionary of minimum diviation, indexed by (level, state)
    # dictionary of path, indexed by (level, state), the path is a list of bits
    w = {}
    for e in T.E[0]:
        d[(1, e.end)] = (e.weight - c[0]) % 2
        w[(1, e.end)] = [[e.weight]]
    for i in range(1, len(c)):
        for e in T.E[i]:
            if (i + 1, e.end) not in d or d[(i, e.begin)] + (e.weight - c[i]) % 2 < d[(i + 1, e.end)]:
                d[(i + 1, e.end)] = d[(i, e.begin)] + (e.weight - c[i]) % 2
                w[(i + 1, e.end)] = [p + [e.weight] for p in w[(i, e.begin)]]
            elif d[(i, e.begin)] + (e.weight - c[i]) % 2 == d[(i + 1, e.end)]:
                w[(i + 1, e.end)].extend([p + [e.weight]
                                          for p in w[(i, e.begin)]])
    return d, w


def viterbilist(T, c, ne, start=0, init=None):
    """
    run viterbi algorithm on the trellis T with c, starting from a given level,
    track all the paths with fewer than ne errors

    If the start is 0, ignore the initials.

    returns w, dictionary of path, indexed by (level, state, error).
    Old items in w is overwritten if the level is in [start, start + len(c))
    """
    n = T.G.shape[1]
    if start + len(c) > n:
        raise Exception("Senseword length exceeds codeword.")
    if start > 0:
        if init is None:
            raise Exception("No initials for start > 0.")
        else:
            start = init
    elif start == 0:
        w = {}  # dictionary of path, indexed by (level, state, error)
        w[(0, T.V[0][0], 0)] = [[]]
        for _ in range(1, ne + 1):
            w[(0, T.V[0][0], _)] = []  # will always remain []
    for i in range(start, start + len(c)):
        for v in T.V[i + 1]:
            for _ in range(ne + 1):
                w[(i + 1, v, _)] = []
        for e in T.E[i]:
            if e.weight == c[i]:
                for _ in range(ne + 1):
                    w[(i + 1, e.end, _)].extend([p + [e.weight]
                                                 for p in w[(i, e.begin, _)]])
            else:
                for _ in range(ne):
                    w[(i + 1, e.end, _ + 1)].extend([p + [e.weight]
                                                     for p in w[(i, e.begin, _)]])
    return w


def volumes(T, c, ne):
    """
    run viterbi algorithm on a trellis with codeword c, c does not have to be complete.

    returns: a dictionary ds, index by (level, state); to every node in T.V[:len(c)]
            ds gives a e+1 vector that tallies the number of paths with errors no greater than ne
    """
    m, n = T.G.shape
    if len(c) > n:
        raise Exception("Senseword length exceeds codeword")
    ds = {}
    ds[(0, '')] = np.zeros(ne + 1, dtype=np.int)
    ds[(0, '')][0] = 2 ** m
    for i in range(len(c)):
        for e in T.E[i]:
            if (i + 1, e.end) not in ds:
                ds[(i + 1, e.end)] = np.zeros(ne + 1, dtype=np.int)
            if e.weight == c[i]:
                ds[(i + 1, e.end)] += ds[(i, e.begin)] // T.rhoplus[i]
            else:
                ds[(i + 1, e.end)] += spshift(ds[(i, e.begin)],
                                              1, cval=0) // T.rhoplus[i]
    return ds


def backpropagatewl(piles):
    """
    assign wining labels (in place) to all pile in piles, assuming the last level YN_ label is assigned
    """
    for l in reversed(range(len(piles) - 1)):
        for i in range(len(piles[l])):
            piles[l][i] = piles[l][i]._replace(winning=winlose(
                piles[l + 1][2 * i].winning, piles[l + 1][2 * i + 1].winning))


def tallypile(pl):
    """ adding up probability in the pile pl"""
    totalprob = 0
    for p in pl:
        if p.winning == 'Y':
            totalprob += p.prob
        elif p.winning == '_':
            totalprob += 0  # p.prob / 2
    return totalprob


def tallypile2(pl):
    """ adding up probability in the pile pl with randomization 1/2"""
    totalprob = 0
    for p in pl:
        if p.winning == 'Y':
            totalprob += p.prob
        elif p.winning == '_':
            totalprob += p.prob / 2
    return totalprob


def main():
    s, T = rm13trellis()
    sub = T.codewords[s]
    piles = simulate_subcode(sub, T, maxsub_strategy)
    totalprob = tallypile(piles[-1])
    print(totalprob)
    piles = simulate_subcode(sub, T, maxratio_strategy)
    totalprob = tallypile(piles[-1])
    print(totalprob)
    piles = simulate_subcode(sub, T, no_strategy)
    totalprob = tallypile(piles[-1])
    print(totalprob)


if __name__ == '__main__':
    main()
