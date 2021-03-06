import numpy as np
import sympy
from scipy.ndimage.interpolation import shift as spshift
from collections import namedtuple, Counter, OrderedDict
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

    def get_codewords(self):
        """
        return D, codewords, where codeword[d] contains the codeword for d in D
        """
        mc = self.C.shape[0]  # length of coset messages
        ms = self.S.shape[0]  # length of the span basis
        # index of messages, might have been repeated elsewhere, but here sweep under the rug of subdec and generate as needed
        D = list(itertools.product((0, 1), repeat=mc))
        s = np.array(list(itertools.product((0, 1), repeat=ms)))
        codewords = {}
        for d in D:
            codewords[d] = (np.array(d).dot(self.C) + s.dot(self.S)) % 2  # group codewords by cosets
        return D, codewords

    def compute_codewords(self):
        self.D, self.codewords = self.get_codewords()

    def decode(self, c):
        """ Gives the C component of the decoded word, ignores the residual """
        mc = self.C.shape[0]
        x, r, *_ = np.linalg.lstsq(self.G.T, c, rcond=None)
        return x[:mc].round()


class SubDecodeFirstGenerator(SubDecode):
    """
    use the first mc rows of the generator for coset leaders
    """

    def __init__(self, G, mc):
        super().__init__(G[:mc], G[mc:])


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

    def __init__(self, G, SB):                          # should specify basis for SB!
        S = np.array(SB).dot(G) % 2
        _, ind = sympy.Matrix(SB).rref()                # obtain indices of pivot rows
        indc = list(set(range(G.shape[0])) - set(ind))  # ind's complement
        CB = np.zeros(shape=(G.shape[0] - S.shape[0], G.shape[0]))
        CB[range(CB.shape[0]), indc] = 1              # S's index
        C = CB.dot(G) % 2
        assert np.linalg.matrix_rank(np.vstack((CB, SB))) == G.shape[0], \
            "Basis coefficients do not yield a full rank generator"
        super().__init__(C, S)


def cor(a, b):
    """
    obtain correlation between the two codewords whose entries are between 0 and 1.
    returns (1-2a).(1-2b). If a, b are matrices, returns inner of their rows (last dimension must match)

    >>> cor([1, 1], [1, 1])
    2
    >>> cor([1, 0], [1, 1])
    0
    >>> cor([1,1],[[0,1],[1,0]])
    array([0, 0])
    >>> cor([[1,1], [1,0]],[[0,1],[1,0]])
    array([[ 0,  0],
           [-2,  2]])
    >>> cor([[1,1], [1,0]],[[0,1],[0,0]])
    array([[ 0, -2],
           [-2,  0]])
    """
    return np.inner(1 - 2 * np.array(a), 1 - 2 * np.array(b))


TrellisEdge = namedtuple('TrellisEdge', 'begin end weight')


class Trellis:
    """
    A Trellis object, contains
        G: the generateor matrix that is in minspan form
        A, B, alpha, beta, rhoplus, rhominus: used in generating labels of the trellis
        V: list of vertices, vertices are strings
        E: list of TrellisEdge objects with (begin, end, weight) fields
    """

    def __init__(self, G, S=None):
        """ Does not enforce G to be Trellis oriented"""
        if S is None:
            S = minspan(G)
        self.G = G
        self.n = G.shape[1]
        self.m = G.shape[0]
        self.S = S
        self.A, self.B, self.alpha, self.beta, self.rhoplus, self.rhominus, self.V, self.E = trellis(
            G, S)
        self.codewords = []
        self.messages = np.array(
            list(itertools.product([0, 1], repeat=G.shape[0])))
        for i in self.messages:
            self.codewords.append(i.dot(G) % 2)
        self.codewords = np.array(self.codewords)

    def setsub(self, SB):
        """ set subdec objects and related indexing D, edge node pass"""
        self.SB = SB
        self.subdec = SubDecodeCombineGenerator(self.G, SB)
        self.subdec.compute_codewords()   # save codewords dictionary in subdec
        self.D = self.subdec.D
        self.edgepass, self.nodepass = edgenodepass(self, self.subdec.codewords, self.subdec.D)
        self.state_space = None  # compute as needed
        self.P = None            # compute as needed
        self.p = None
        self.j = None
        self.a = None

    def compute_mdp(self):
        """
        compute state_space transitions and P as needed, to be called after setsub
        returns state_space and P
        """
        if self.state_space is None:
            state_space, P = get_state_space(self)
            self.state_space = state_space
            self.P = P
            return state_space, P
        else:
            return self.state_space, self.P     # already computed

    def dpsolve(self, p, jinit):
        """compute optimal policies, to be called after setsub"""
        if self.p == p:
            return self.j, self.a
        if self.state_space is None:
            self.compute_mdp()
        self.p = p
        self.j, self.a = dpsolve_cor(self.state_space, self.P, p, jinit=jinit)
        return self.j, self.a

    def cal_psuccess(self, a=None, p=None):
        """
        returns a real number psuccess, if a, p is None, use the a, p from self.dpsolve
        """
        if a is None:
            a = self.a
            if a is None:
                raise AttributeError("Please specify optimal policies in the argument or call dpsolve() first")
        if p is None:
            p = self.p
            if p is None:
                raise AttributeError(
                    "Please specify crossover probability in the argument or call dpsolve() first")
        return cal_psuccess(self.state_space, self.P, a, p)


class StateTrellis():
    """
    View the state transition of correlation as a trellis

    Has attributes n, V, E, T (base trellis)

    The hovertest function returns a info[(level, state)] for plotting
    """

    def __init__(self, T):
        self.n = len(T.state_space) - 1
        self.V = T.state_space
        P = T.P
        E = []
        for i in range(self.n):
            Ei = []
            for s in self.V[i]:
                Ei.append(TrellisEdge(s, P[i, 0][s], 0))
                Ei.append(TrellisEdge(s, P[i, 1][s], 1))
            E.append(Ei)
        self.E = E
        self.T = T

    def hovertext(self):
        """
        return information at a particular node, which is the vertices:correlation to every coset
        """
        info = {}
        for i, Statei in enumerate(self.V):
            for state in Statei:
                info[i, state] = ""
                co = dict(zip([(i, v, d) for v in self.T.V[i] for d in self.T.D], state))
                for v in self.T.V[i]:
                    info[i, state] += v + ':' + str([co[i, v, d] for d in self.T.D]) + '\n'
                if self.T.j is not None:
                    info[i, state] += str(self.T.j[i, state])
                if self.T.a is not None and (i, state) in self.T.a:
                    info[i, state] += ' _ ' + str(self.T.a[i, state])
        return info


def closest(c, codewords):
    """
    return codewords in trellis T (whose prefixes) are closest to c
    """
    d = np.array([sum((c + w) % 2) for w in codewords[:, :len(c)]])
    return min(d), codewords[d == min(d)]


def ncloeset(c, codewords):
    """
    return number of closest codewords
    """
    dmin, codes = closest(c, codewords)
    return dmin, codes.shape[0]


def closestat(c, codewords):
    """
    return a list, the first entry counts the number of codewords that has 0 error from c,
    the second counts 1 error, etc.
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
    return the edge and node set passed by the node subsets.
    First call select_subcode and then, with the all the passed messages,
    obtain the nodes from the trellis B matrix.
    """
    m = T.messages[select_subcode(T, nodes)]
    n = T.n
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
        gt = ''.join([str(int(_)) for _ in gt])
        us = bits(alpha[i + 1])
        Ei = []
        for u in us:
            lu = inner(u, gt)
            initu = sindex(u, relindex(B[i], A[i + 1]))
            finu = sindex(u, relindex(B[i + 1], A[i + 1]))
            Ei.append(TrellisEdge(*[initu, finu, lu]))
        E.append(Ei)
    return A, B, alpha, beta, rhoplus, rhominus, V, E


def plottrellis(T, subE=None, title='Trelis', statelabel=None, edgelabel=None, maxlevel=None, hovertext=None, plotabr=False):
    """
    if statelabel == None, do not add state label
    if statelabel == {} or some keys are missing, use default label

    requires T to have (T.n, T.V, T.E)
    """
    def edgetrace(V, E, width=2, color='#888', edgelabel=None):
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
        text_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            hoverinfo='none',
            mode='text',
            textposition="top center")
        for i, Ei in enumerate(E):
            if i <= maxlevel - 1:
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
                    if edgelabel and (i, e) in edgelabel:
                        text_trace['x'] += tuple([(x0 + x1) / 2])
                        text_trace['y'] += tuple([(y0 + y1) / 2])
                        text_trace['text'] += tuple([edgelabel[(i, e)]])
        return edge_trace0, edge_trace1, text_trace
    V = T.V
    E = T.E
    n = T.n
    if maxlevel is None:
        maxlevel = T.n
    data = []
    if plotabr:
        if hasattr(T, 'A'):
            a_trace = go.Scatter(
                x=[-0.5] + list(range(1, n)),
                y=[-0.2] * (n + 1),
                text=['A'] + [str(_) for _ in T.A[1:n]],
                mode='text',
                hoverinfo='none'
            )
            data.append(a_trace)
        if hasattr(T, 'B'):
            b_trace = go.Scatter(
                x=[-0.5] + list(range(1, n)),
                y=[-0.5] * n,
                text=['B'] + [str(_) for _ in T.B[1:n]],
                mode='text',
                hoverinfo='none'
            )
            data.append(b_trace)
        if hasattr(T, 'rhoplus') and hasattr(T, 'rhominus'):
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
            data.extend([rhoplus_trace, rhominus_trace])
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='top center',  # 'bottom center',
        hoverinfo="none",
        hovertext=[]
    )
    if hovertext:
        node_trace.hoverinfo = "text"
    for i, vi in enumerate(V):
        if i <= maxlevel:
            for j, v in enumerate(vi):
                x, y = i, j
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                if statelabel is None:
                    pass
                elif statelabel and (i, v) in statelabel:
                    node_trace['text'] += tuple([statelabel[(i, v)]])
                else:
                    node_trace['text'] += tuple([v])
                if hovertext:
                    node_trace['hovertext'] += (hovertext[i, v],)
    edge_trace0, edge_trace1, text_trace = edgetrace(V, E, edgelabel=edgelabel)
    data.extend([edge_trace0, edge_trace1, node_trace, text_trace])
    if subE is not None:
        bedge_trace0, bedge_trace1, subedge_text_trace = edgetrace(
            V, subE, width=3.5, color='rgba(0, 0, 152, .8)', edgelabel=edgelabel)
        data.extend([bedge_trace0, bedge_trace1, subedge_text_trace])
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
    >>> inner('11', '11')
    0
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
    """ get indices that sort the rows of G first so L is increasing while R is decreasing """
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


def edgenodepattern(T, codewords, D):
    """
    mark for every edge and node the codewords that passes through them
    returns:
        edgepass, edgepass[i, e, d][index of word in codewords[d]] = True/ False
        nodepass, nodepass[i, v, d][index of word in codewords[d]] = True/ False
    """
    n = T.n
    E = T.E
    V = T.V
    edgepass = {}
    nodepass = {}
    for i in range(n):
        for e in E[i]:
            for d in D:
                edgepass[i, e, d] = np.array([False] * len(codewords[d]))
    for i in range(n + 1):
        for v in V[i]:
            for d in D:
                nodepass[i, v, d] = np.array([False] * len(codewords[d]))
    for d in D:
        for iword, word in enumerate(codewords[d]):
            v = T.V[0][0]
            for i in range(n):
                nodepass[i, v, d][iword] = True
                for e in E[i]:
                    if e.begin == v and e.weight == word[i]:
                        edgepass[i, e, d][iword] = True
                        v = e.end
                        break
            nodepass[n, v, d][iword] = True
    return edgepass, nodepass


def edgenodepass(T, codewords, D):
    """
    mark for every edge and node existence of codewords that passes through them
    returns:
        edgepass, edgepass[i, e, d] = True if there is some paths in T that passes through it
        nodepass, nodepass[i, v, d] = True if there is some paths in T that passes through it
    """
    n = T.n
    E = T.E
    V = T.V
    edgepass = {}
    nodepass = {}
    for i in range(n):
        for e in E[i]:
            for d in D:
                edgepass[i, e, d] = False
    for i in range(n + 1):
        for v in V[i]:
            for d in D:
                nodepass[i, v, d] = False
    for d in D:
        for iword, word in enumerate(codewords[d]):
            v = T.V[0][0]
            for i in range(n):
                nodepass[i, v, d] = True
                for e in E[i]:
                    if e.begin == v and e.weight == word[i]:
                        edgepass[i, e, d] = True
                        v = e.end
                        break
            nodepass[n, v, d] = True
    return edgepass, nodepass


def viterbicorpattern(T, c, ne=3, start=0, init=None):
    """
    obtain state transitions with the codewords
    T has a subdec object with precomputed codewords
    """
    n = T.n
    ep = T.edgepass
    codewords = T.subdec.codewords
    D = T.D
    # First obtain edge / node -> codewords look-up table
    if start + len(c) > n:
        raise Exception("Senseword length exceeds codeword.")
    if start > 0:
        if init is None:
            raise Exception("No initials for start > 0.")
        else:
            if ne != n:
                co, closest = init
            else:
                co, closest = init, T.nodepass
            co = co.copy()
            closest = closest.copy()  # very important!
    elif start == 0:
        v = T.V[0][0]
        # dictionary of minimum correlation, indexed by (level, node, d)
        co = {(0, v, d): 0 for d in D}
        # record which codewords are the closest
        closest = {(0, v, d): np.full(len(codewords[d]), True) for d in D}
    for i in range(start, start + len(c)):
        for e in T.E[i]:
            for d in D:
                co[(i + 1, e.end, d)] = (i + 1) - 2 * ne - 1                   # set cor to cutoff - 1
                closest[(i + 1, e.end, d)] = np.full(len(codewords[d]), False)
    for i in range(start, start + len(c)):
        for e in T.E[i]:
            for d in D:
                activethrough = np.logical_and(ep[i, e, d], closest[i, e.begin, d])
                if np.any(activethrough):
                    if (i + 1, e.end, d) not in co or \
                            co[i, e.begin, d] + cor(e.weight, c[i - start]) > co[i + 1, e.end, d]:
                        co[i + 1, e.end, d] = co[i, e.begin, d] + cor(e.weight, c[i - start])
                        closest[i + 1, e.end, d] = activethrough
                    elif co[(i, e.begin, d)] + cor(e.weight, c[i - start]) == co[i + 1, e.end, d]:
                        closest[i + 1, e.end, d] = np.logical_or(closest[i + 1, e.end, d], activethrough)
                # else No active paths lead to e.end, no updates
    return co, closest


def viterbicorpass(T, c, ne=3, start=0, init=None):
    """
    Another implementation with only pass

    Input:
        T       the trellis
        c       the received codeword
        D       indexes set of D
        ne      number of error tolerated
        start and init: allows restart from a previous state, if ne=n, init contains only co

    Returns
        co, closest : correlation and closest path from running the viterbi.
            co[i, v, d] best correlation to vertex v in coset d, default i - 2*ne -1.
            closest[i, v, d] whether there exists path <=ne error in coset d that lead to vertex v.
            if ne = n, closest will be the same as nodepass.
    """
    n = T.n
    ep = T.edgepass
    D = T.D
    # First obtain edge / node -> codewords look-up table
    if start + len(c) > n:
        raise Exception("Senseword length exceeds codeword.")
    if start > 0:
        if init is None:
            raise Exception("No initials for start > 0.")
        else:
            if ne < n:
                co, closest = init
            else:
                co, closest = init, T.nodepass
            co = co.copy()
            closest = closest.copy()  # very important!
    elif start == 0:
        # dictionary of minimum correlation, indexed by (level, node, d)
        v = T.V[0][0]
        co = {(0, v, d): 0 for d in D}
        # record existence of paths are the closest
        closest = {(0, v, d): True for d in D}
    for i in range(start, start + len(c)):
        for e in T.E[i]:
            for d in D:
                if (i + 1, e.end, d) not in co:  # initialize, note that new edges might visit a node more than once!
                    co[(i + 1, e.end, d)] = (i + 1) - 2 * ne - 1        # set cor to cutoff - 1
                    closest[(i + 1, e.end, d)] = False
                activethrough = ep[i, e, d] and closest[i, e.begin, d]
                if activethrough:  # exists active paths lead to e.end
                    if co[i, e.begin, d] + cor(e.weight, c[i - start]) > co[i + 1, e.end, d]:
                        co[i + 1, e.end, d] = co[i, e.begin, d] + cor(e.weight, c[i - start])
                        closest[i + 1, e.end, d] = True
                    elif co[(i, e.begin, d)] + cor(e.weight, c[i - start]) == co[i + 1, e.end, d]:
                        closest[i + 1, e.end, d] = True
                    # else correlation is too small, keep initialization.
                # else no active path, keep initialization.
    return co, closest


# alias
viterbicor = viterbicorpass


def get_state_space_enumerate(T, ne=None):
    n = T.n
    V = T.V
    subdec = T.subdec
    if ne is None:
        ne = n
    # corcutoff = n - 2 * ne  # do not consider codewords too far away, too big for code prefixes!
    # corspace = tuple(range(corcutoff, 2 * n + 1, 2))  # space of possible correlations
    mc = subdec.C.shape[0]  # length of coset messages
    ms = subdec.S.shape[0]  # length of the span basis
    D = list(itertools.product((0, 1), repeat=mc))  # index of messages
    s = np.array(list(itertools.product((0, 1), repeat=ms)))
    codewords = {}
    for d in D:
        codewords[d] = (np.array(d).dot(subdec.C) + s.dot(subdec.S)) % 2  # group codewords by cosets
    # Ds = [''.join(_) for _ in D] # strings if needed

    # Figure out all the state transitions by enumerating the received strings
    state_space = [list() for i in range(n + 1)]
    # Transition matrix, P[i,Action][state]=Next state
    P = {(i, a): dict() for i in range(n) for a in range(2)}
    # # Inverse transition matrix, Q[i,Action][Next state] = list of Previous State
    # Q = {(i, a): dict() for i in range(n) for a in range(2)}
    # assert P[0,0] is not P[0,1] and  P[0,0]==P[0,1], 'Dictionary P init overload!'
    # assert state_space[0] is not state_space[1], 'state_space init overload '
    for r in itertools.product((0, 1), repeat=n):
        co, *_ = viterbicor(T, r, ne=ne)
        # initial state is always the same
        pstate = tuple(co[0, v, d] for v in V[0] for d in D)
        if pstate not in state_space[0]:
            state_space[0].append(pstate)
        # assert len(state_space[0]) == 1, "Initial State not Unique: {}".format(state_space[0])
        # look at the transition
        for i, ri in enumerate(r):
            nstate = tuple(co[i + 1, v, d] for v in V[i + 1] for d in D)
            if nstate not in state_space[i + 1]:
                state_space[i + 1].append(nstate)
            if pstate in P[i, ri]:
                assert nstate == P[i, ri][pstate], "Transition is not consistent!"
            else:
                P[i, ri][pstate] = nstate
                # if nstate not in Q[i, ri]:
                #     Q[i, ri][nstate] = [pstate]
                # else:
                #     if pstate not in Q[i, ri][nstate]:
                #         Q[i, ri][nstate].append(pstate)
            pstate = nstate
        # # assert that P is the actual
        # for i in range(n):
        #     for a in range(2):
        #         for pstate, nstate in P[i, a].items():
        #             assert pstate in Q[i,a][nstate], "P, Q not inverse related!"
    return state_space, P  # , Q


def get_state_space_progressive(T, ne=None):
    """
        return the state space progressively run viterbi for one step rather than enumerating c.

    Input
        T      trellis, has a subdec object with C and S matrices
        ne     number of error tolerated, if None, ne = T.n

    Returns
        state_space: list of length n+1, each is the list of states
        P          : P[i,Action][state]=Next state, here action is means what is received.
    """
    n = T.n
    V = T.V
    if ne is None:
        ne = n
    D = T.D
    # Figure out all the state transitions by enumerating the received strings
    state_space = [list() for i in range(n + 1)]  # tuple used in indexing
    costate_space = [list() for i in range(n + 1)]  # dictionary for calculation.
    # Transition matrix, P[i,Action][state]=Next state
    P = {(i, a): dict() for i in range(n) for a in range(2)}
    # Time 0 initial state
    co, *_ = viterbicor(T, [], ne=ne)
    pstate = tuple(co[0, v, d] for v in V[0] for d in D)  # previous state
    state_space[0].append(pstate)
    costate_space[0].append(co)
    for i in range(n):
        for ri in range(2):
            for pstate, co in zip(state_space[i], costate_space[i]):
                nco, *_ = viterbicor(T, [ri], ne=ne, start=i, init=co)
                nstate = tuple(nco[i + 1, v, d] for v in V[i + 1] for d in D)
                P[i, ri][pstate] = nstate
                if nstate not in state_space[i + 1]:
                    state_space[i + 1].append(nstate)
                    costate_space[i + 1].append({(i + 1, v, d): nco[i + 1, v, d] for v in V[i + 1] for d in D})
    return state_space, P


# Alias
get_state_space = get_state_space_progressive


def cal_psuccess(state_space, P, a, p):
    """
    calculate the probability of successful transmition with given policy a.
    p is the crossover probability

    returns a real number psuccess.
    """
    n = len(state_space) - 1
    ej = {}    # ej for expected j.
    for i in range(n + 1):
        for s in state_space[i]:
            ej[i, s] = 0          # expected probability of reaching state s in level i.
    ej[0, state_space[0][0]] = 1.  # better have a real number
    for i in range(n):
        for s in state_space[i]:
            if a[i, s] == 'indifferent':
                ej[i + 1, P[i, 0][s]] += ej[i, s] / 2
                ej[i + 1, P[i, 1][s]] += ej[i, s] / 2
            else:
                ej[i + 1, P[i, a[i, s]][s]] += (1 - p) * ej[i, s]
                ej[i + 1, P[i, 1 - a[i, s]][s]] += p * ej[i, s]
    psuccess = 0
    for s in state_space[n]:
        if len(s) == 1 or s[0] > max(s[1:]):
            psuccess += ej[n, s]
    return psuccess             # sum(ej[n, s] for s in state_space[n] if s[0] > max(s[1:]))


def jinit_psucc(state_space):
    """
    initialize with probability of success, if there is only one coset, psucc=1
    """
    n = len(state_space) - 1
    j = {}
    for s in state_space[n]:
        j[n, s] = 1 if len(s) == 1 or s[0] > max(s[1:]) else 0   # most states are failed decoding
    return j


def jinit_cordiff(state_space):
    """
    initialize with correlation difference, if there is only one coset, init with absolute correlation
    """
    n = len(state_space) - 1
    j = {}
    for s in state_space[n]:
        if len(s) == 1:
            j[n, s] = s[0]
        else:
            j[n, s] = s[0] - max(s[1:])
    return j


def jinit_abscor(state_space):
    n = len(state_space) - 1
    j = {}
    for s in state_space[n]:
        j[n, s] = s[0]                           # absolute correlatioon
    return j


def dpsolve_cor(state_space, P, p, jinit=jinit_psucc):
    """
    simulate with correlation as states

    subdec: The subcode classifier, we will attempt to transmit the message [0]_1^mc, i.e. in D[0]

    returns: j and a, optimal cost and policies
    """
    n = len(state_space) - 1
    # p = symbols('p')  # crossover probability
    j = jinit(state_space)
    a = {}
    for i in reversed(range(n)):
        for s in state_space[i]:
            if j[i + 1, P[i, 0][s]] > j[i + 1, P[i, 1][s]]:
                a[i, s] = 0
                j[i, s] = (1 - p) * j[i + 1, P[i, 0][s]] + p * j[i + 1, P[i, 1][s]]
            elif j[i + 1, P[i, 0][s]] == j[i + 1, P[i, 1][s]]:
                a[i, s] = "indifferent"
                j[i, s] = 1 / 2 * (j[i + 1, P[i, 0][s]] + j[i + 1, P[i, 1][s]])
            else:
                a[i, s] = 1
                j[i, s] = (1 - p) * j[i + 1, P[i, 1][s]] + p * j[i + 1, P[i, 0][s]]
    return j, a  # , codewords


def viterbi(T, c):
    """
    run viterbi algorithm on a trellis with codeword c, c does not have to be complete.

    returns: a pair of dictionary (d, w)
    d marks distance  to every node in T.V[:len(c)] that has the fewest difference from c
    w records the corresponding best paths, if there is a tie, preserve all of them.
    """
    n = T.n
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
    run viterbi algorithm on the trellis T with c, over [start, start+len(c)]
    track all the paths with fewer than ne errors.

    If the start is 0, ignore the initials.

    returns w, dictionary of path, indexed by (level, state, error).
    Old items in w is overwritten if the level is in [start, start + len(c))
    """
    n = T.n
    if start + len(c) > n:
        raise Exception("Senseword length exceeds codeword.")
    if start > 0:
        if init is None:
            raise Exception("No initials for start > 0.")
        else:
            w = init
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
            if e.weight == c[i - start]:
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


def main():
    pass


if __name__ == '__main__':
    main()
