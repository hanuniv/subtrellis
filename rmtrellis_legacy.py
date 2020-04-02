"""
Legacy code for old formulations of rmtrellis
"""

import numpy as np
from rmtrellis import closestat, ncloeset
from sympy import symbols, factor
from collections import namedtuple, Counter
from pprint import pprint
import itertools

# class lookaheadstrategy:
#     """
#     a lookahead strategy class based on the trellis T and number of lookahead
#     This is a template stub, to be developed in the future.
#     """

#     def __init__(self, T, nlook):
#         self.T = T
#         self.nlook = nlook  # number of look ahaed

#     def hitme(self, piles, level, p=0.9):
#         """ return the next bit to send, given current piles and level"""
#         lookpiles = [piles[-1]]
#         for _ in range(self.nlook):
#             step(lookpiles, T, level)  # STUB: add next level probability
#         # Skip: assign winlose to the last pile
#         # backpropagatewl(lookpile)
#         # find the one with the best wining probability, when there is a tie, include them all
#         bestprob = 0
#         bestc = []
#         for endings in lookpiles[-1]:
#             if endings.prob == bestprob:
#                 bestc.append(endings.c)
#             # stub: find substitute function
#             elif substitute(endings.prob, p) > substitute(bestprob, p):
#                 bestprob = endings.prob
#                 bestc = [endings.c]
#         return bestc[0][0]  # decide the first choice of the first best policy


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


def steptuple(codetuple, T, level, bit, p):
    """ return a list of possible states for the next level given bits received"""
    c, prob, *_ = codetuple
    return [codetuple._replace(prob=prob * (1 - p), c=c + [bit]),
            codetuple._replace(prob=prob * p, c=c + [1 - bit])]


def winlose(a, b):
    """ Combine wining / losing / boundary states in [Y N _]"""
    if a == 'Y' and 'b' == 'Y':
        return 'Y'
    elif a in ['_', 'Y'] or 'b' in ['_', 'Y']:
        return '_'
    else:
        return 'N'


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


def simulate_subcode(sub, T, strategy=maxsub_strategy):
    """
    Simulate the forward evolution of the code, given trellis and strategy.

    The strategy is assumed to based solely on the comparison of the number of path in the subcode
    and the number of path in the base code.

    returns: piles, a list of length n, each item is a list (pile) of Codeprob items
    """

    # Codeprob class: c = feedback word,
    #                 dsub = number of path in the subcode, dcode = paths in base code.
    Codeprob = namedtuple('Codeprob', 'c prob dsub dcode winning')
    n = T.n
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
        if (pile[i].dsub[0] < pile[i].dcode[0]) or \
           (pile[i].dsub[0] == pile[i].dcode[0] and pile[i].dsub[1] > pile[i].dcode[1] / 2):
            pile[i] = pile[i]._replace(winning='Y')
        elif (pile[i].dsub[0] == pile[i].dcode[0] and pile[i].dsub[1] == pile[i].dcode[1] / 2):
            pile[i] = pile[i]._replace(winning='_')
        else:
            pile[i] = pile[i]._replace(winning='N')
    backpropagatewl(piles)
    return piles


def simulate_lookahead(subs, T, nlook, ne=2, send0=send0subs):
    """
    A reworked version of simulation that applies to look ahead policies

    subs: a list of subtrellises, the goal is to transmit the first subtrellis.

    returns: piles, a list of length n, each item is a list (pile) of Codeprob items
    """
    # Codeprob class: c = feedback word,
    #              dsub = number of path in the subcode, dcode = paths in base code.
    Codeprob = namedtuple('Codeprob', 'c prob dsub winning')
    n = T.n
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

class LookaheadTest(unittest.TestCase):
    @unittest.skip("Skipping LookaheadTest")
    def test_low_rate_codes(self):
        tps = []
        for nlook in range(1, 5):
            nodes, subs, T = rm13trelliscode1()
            piles = simulate_lookahead(subs, T, nlook=nlook, ne=2)
            totalprob = tallypile(piles[-1])
            tps.append(totalprob)
        piles = simulate_lookahead(subs, T, nlook=1, ne=2, send0=send0always)
        totalprob = tallypile(piles[-1])
        tps.append(totalprob)
        # print(latex(tps))
        # print(latex(_) for _ in tps)
        # print(tps)
        tps = []
        for nlook in range(1, 5):
            nodes, subs, T = rm13trelliscode2()
            piles = simulate_lookahead(subs, T, nlook=nlook, ne=2)
            totalprob = tallypile(piles[-1])
            tps.append(totalprob)
        piles = simulate_lookahead(subs, T, nlook=1, ne=2, send0=send0always)
        totalprob = tallypile(piles[-1])
        tps.append(totalprob)
        # print(latex(tps))
        # print(latex(_) for _ in tps)
        # print(tps)

def rm13simulate():
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
