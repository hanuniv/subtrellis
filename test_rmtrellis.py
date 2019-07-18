import unittest
import numpy as np
from rmtrellis import *
from sympy import latex, plot

class Test(unittest.TestCase):

    def test_covers(self):
        G = np.array([[1, 1, 0, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1]])
        k, n = G.shape
        S = [(0, 4), (0, 4), (4, 6)]
        A, B, alpha, beta, rhoplus, rhominus, V, E = trellis(G, S)
        self.assertEqual(A, [set(), {0, 1}, {0, 1}, {0, 1}, {
                         0, 1}, {0, 1, 2}, {2, }, {2, }])
        self.assertEqual(B, [set(), {0, 1}, {0, 1}, {
                         0, 1}, {0, 1}, {2, }, {2, }, set()])
        self.assertEqual(alpha, [0, 2, 2, 2, 2, 3, 1, 1])
        self.assertEqual(beta, [0, 2, 2, 2, 2, 1, 1, 0])
        self.assertEqual(rhoplus, [4, 1, 1, 1, 2, 1, 1, 0])
        self.assertEqual(rhominus, [0, 1, 1, 1, 1, 4, 1, 2])
        self.assertEqual(V, [[''],
                             ['00', '01', '10', '11'],
                             ['00', '01', '10', '11'],
                             ['00', '01', '10', '11'],
                             ['00', '01', '10', '11'],
                             ['0', '1'], ['0', '1'], ['']])
        self.assertEqual(E, [
            [('', '00', 0), ('', '01', 0), ('', '10', 1), ('', '11', 1)],
            [('00', '00', 0), ('01', '01', 1), ('10', '10', 1), ('11', '11', 0)],
            [('00', '00', 0), ('01', '01', 1), ('10', '10', 0), ('11', '11', 1)],
            [('00', '00', 0), ('01', '01', 1), ('10', '10', 1), ('11', '11', 0)],
            [('00', '0', 0), ('00', '1', 1), ('01', '0', 0), ('01', '1', 1),
             ('10', '0', 1), ('10', '1', 0), ('11', '0', 1), ('11', '1', 0)],
            [('0', '0', 0), ('1', '1', 1)],
            [('0', '', 0), ('1', '', 1)]])
        self.assertEqual([len(v) for v in V], [2**b for b in beta])
        self.assertEqual([len(e) for e in E], [2**a for a in alpha[1:]])

    def test_mingen(self):
        G = np.array([[1, 1, 0, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1]])
        S, Gt = gensort(G)
        # TOOD self.assertEqual(1, 1)
        G = np.array([[1, 1, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 1, 0, 1]])
        G = minspangen(G)
        self.assertEqual(isminspan(G), True)
        # print(S)
        # print(np.array(S))

    def test_closest(self):
        G = np.array([[1, 1, 0, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1]])
        T = Trellis(G)
        np.testing.assert_array_equal(closest(np.array([0, 0, 0]), T.codewords)[
                                      1], np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]))
        np.testing.assert_array_equal(closest(np.array([0, 0, 0, 0, 0, 0, 0]), T.codewords)[
                                      1], np.array([[0, 0, 0, 0, 0, 0, 0]]))
    def test_closestat(self): 
        self.assertEqual(closestat([0],np.array([[0,0],[0,1], [1,0],[1,1]])), [2,2,0])
        self.assertEqual(closestat([0,0],np.array([[0,0],[0,1], [1,0],[1,1]])), [1,2,1])

    def test_iswiningstat(self):
        self.assertEqual(iswiningstat([[0, 1, 0], [0, 0, 1]]),'Y')
        self.assertEqual(iswiningstat([[0, 1, 0], [0, 1, 1]]),'_')
        self.assertEqual(iswiningstat([[0, 1, 0], [0, 2, 1]]),'N')
        self.assertEqual(iswiningstat([[0, 1, 0], [1, 0, 1]]),'N')

    def test_subcode(self):
        """ test on RM(3, 1)"""
        G = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1],
                      [0, 1, 0, 1, 0, 1, 0, 1]])
        G = minspangen(G)
        T = Trellis(G)
        s = select_subcode(T, [(3, ['111', '110'])])
        self.assertEqual(sum(s), 4)
        s = select_subcode(T, [(1, ['0']), (2, ['10'])])
        self.assertEqual(sum(s), 0)
        s = select_subcode(T, [(1, ['0']), (2, ['10', '00'])])
        # print(T.codewords[s])
        self.assertEqual(sum(s), 4)
        V, E = select_subtrellis(T, [(1, ['0']), (2, ['10', '00'])])
        self.assertEqual(V, [[''], ['0'], ['00'], ['000', '001'], ['00', '01'], [
                         '000', '001', '010', '011'], ['00', '01'], ['0', '1'], ['']])
        self.assertEqual(E, [[('', '0', 0)], [('0', '00', 0)],
                             [('00', '000', 0), ('00', '001', 1)],
                             [('000', '00', 0), ('001', '01', 1)],
                             [('00', '000', 0), ('00', '001', 1),
                              ('01', '010', 1), ('01', '011', 0)],
                             [('000', '00', 0), ('001', '01', 1),
                              ('010', '00', 1), ('011', '01', 0)],
                             [('00', '0', 0), ('01', '1', 1)], [('0', '', 0), ('1', '', 1)]])
        V, E = select_subtrellis(
            T, [(2, ['00', '11']), (5, ['111', '011', '100', '000'])])
        self.assertEqual(V, [[''], ['0', '1'], ['00', '11'], ['000', '001', '110', '111'],
                             ['00', '01', '10', '11'], ['000', '011',
                                                        '100', '111'], ['00', '01', '10', '11'],
                             ['0', '1'], ['']])
        self.assertEqual(E, [[('', '0', 0), ('', '1', 1)], [('0', '00', 0), ('1', '11', 0)],
                             [('00', '000', 0), ('00', '001', 1),
                              ('11', '110', 0), ('11', '111', 1)],
                             [('000', '00', 0), ('001', '01', 1),
                              ('110', '10', 1), ('111', '11', 0)],
                             [('00', '000', 0), ('01', '011', 0),
                              ('10', '100', 0), ('11', '111', 0)],
                             [('000', '00', 0), ('011', '01', 0),
                              ('100', '10', 1), ('111', '11', 1)],
                             [('00', '0', 0), ('01', '1', 1),
                              ('10', '0', 1), ('11', '1', 0)],
                             [('0', '', 0), ('1', '', 1)]])

    def test_viterbi(self):
        G = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1],
                      [0, 1, 0, 1, 0, 1, 0, 1]])
        G = minspangen(G)
        T = Trellis(G)
        d, w = viterbi(T, [0, 0, 0])
        self.assertEqual(d, {(1, '0'): 0, (1, '1'): 1,
                             (2, '00'): 0, (2, '01'): 1, (2, '10'): 2, (2, '11'): 1,
                             (3, '000'): 0, (3, '001'): 1, (3, '010'): 2, (3, '011'): 1,
                             (3, '100'): 3, (3, '101'): 2, (3, '110'): 1, (3, '111'): 2})

        self.assertEqual(w, {(1, '0'): [[0]], (1, '1'): [[1]],
                             (2, '00'): [[0, 0]], (2, '01'): [[0, 1]], (2, '10'): [[1, 1]], (2, '11'): [[1, 0]],
                             (3, '000'): [[0, 0, 0]], (3, '001'): [[0, 0, 1]], (3, '010'): [[0, 1, 1]], (3, '011'): [[0, 1, 0]],
                             (3, '100'): [[1, 1, 1]], (3, '101'): [[1, 1, 0]], (3, '110'): [[1, 0, 0]], (3, '111'): [[1, 0, 1]]})

        d, w = viterbi(T, [0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(d[(7, '1')], 2)
        self.assertEqual(w[(7, '1')], [[0, 0, 1, 1, 0, 0, 1],
                                       [0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 1, 0]])

        w = viterbilist(T, [0, 0, 0], 2)
        # print(w)
        self.assertEqual(w, {(0, '', 0): [[]], (0, '', 1): [], (0, '', 2): [],
                             (1, '0', 0): [[0]], (1, '0', 1): [], (1, '0', 2): [],
                             (1, '1', 0): [], (1, '1', 1): [[1]], (1, '1', 2): [],
                             (2, '00', 0): [[0, 0]], (2, '00', 1): [], (2, '00', 2): [],
                             (2, '01', 0): [], (2, '01', 1): [[0, 1]], (2, '01', 2): [],
                             (2, '10', 0): [], (2, '10', 1): [], (2, '10', 2): [[1, 1]],
                             (2, '11', 0): [], (2, '11', 1): [[1, 0]], (2, '11', 2): [],
                             (3, '000', 0): [[0, 0, 0]], (3, '000', 1): [], (3, '000', 2): [],
                             (3, '001', 0): [], (3, '001', 1): [[0, 0, 1]], (3, '001', 2): [],
                             (3, '010', 0): [], (3, '010', 1): [], (3, '010', 2): [[0, 1, 1]],
                             (3, '011', 0): [], (3, '011', 1): [[0, 1, 0]], (3, '011', 2): [],
                             (3, '100', 0): [], (3, '100', 1): [], (3, '100', 2): [],
                             (3, '101', 0): [], (3, '101', 1): [], (3, '101', 2): [[1, 1, 0]],
                             (3, '110', 0): [], (3, '110', 1): [[1, 0, 0]], (3, '110', 2): [],
                             (3, '111', 0): [], (3, '111', 1): [], (3, '111', 2): [[1, 0, 1]]})

    def test_volumes(self):
        G = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1],
                      [0, 1, 0, 1, 0, 1, 0, 1]])
        G = minspangen(G)
        T = Trellis(G)
        ds = volumes(T, [0, 0, 0, 0, 0], 3)
        # print(ds) # TODO, is it right?


class LookaheadTest(unittest.TestCase):
    def test_low_rate_codes(self):
        tps = []
        for nlook in range(1, 5):
            nodes, subs, T = rm13trelliscode1()
            piles=simulate_lookahead(subs, T, nlook=nlook, ne=2)
            totalprob=tallypile(piles[-1])
            tps.append(totalprob)
        piles=simulate_lookahead(subs, T, nlook=1, ne=2, send0=send0always)
        totalprob=tallypile(piles[-1])
        tps.append(totalprob)
        print(latex(tps))
        print(latex(_) for _ in tps)
        print(tps)
        tps = []
        for nlook in range(1, 5):
            nodes, subs, T = rm13trelliscode2()
            piles=simulate_lookahead(subs, T, nlook=nlook, ne=2)
            totalprob=tallypile(piles[-1])
            tps.append(totalprob)
        piles=simulate_lookahead(subs, T, nlook=1, ne=2, send0=send0always)
        totalprob=tallypile(piles[-1])
        tps.append(totalprob)
        print(latex(tps))
        print(latex(_) for _ in tps)
        print(tps)

def mceliece():
    G = np.array([[1, 1, 0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1]])

    S = [(0, 4), (0, 4), (4, 6)]
    plottrellis(Trellis(G, S))
    plottrellis(Trellis(G))
    G = np.array([[1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 1, 1]])
    plottrellis(Trellis(G))


def minmceliece():
    ''' testing the dual example with TOGM operation'''
    G = np.array([[1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 0, 1]])
    print(G)
    G = minspangen(G)
    print(G)
    plottrellis(Trellis(G))


def rm13trellis():
    ''' for Reed-Muller Code '''
    G = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1]])
    G = minspangen(G)
    n = G.shape[1]
    # print(G)
    T = Trellis(G)
    plottrellis(T, title='')
    nodes = [(2, ['00', '11']), (5, ['111', '011', '100', '000'])]
    s = select_subcode(T, nodes)
    subV, subE = select_subtrellis(T, nodes)
    plottrellis(T, title='', subE=subE)
    sub = T.codewords[s]
    # print(T.codewords[s])
    return s, T
    # return simulate_subcode(sub, T, maxsub_strategy)

def rm13trelliscode1(plot=False):
    G = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1]])
    G = minspangen(G)
    n = G.shape[1]
    T = Trellis(G)
    # obtain all combinations of subnodes in nodes 
    node = {}
    node[0] = [(2, ['00', '11'])]
    node[1] = [(i, list(set(T.V[i]) - set(subV))) for i, subV in node[0]]
    nodes = []
    for choice in itertools.product([0, 1], repeat=len(node[0])):
        nodes.append([node[i][k] for (k, i) in enumerate(choice)])
    # for all subnodes get the subcode, which is the indexing list for the codewords
    subs = []
    for subnode in nodes:
        s = select_subcode(T, subnode)
        subV, subE = select_subtrellis(T, subnode)
        if plot: plottrellis(T, title='', subE=subE)
        sub = T.codewords[s]
        subs.append(sub)
    return nodes, subs, T  # cut pattern, codewords for subtrellis, and the trellis

def rm13trelliscode2(plot=False):
    G = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1]])
    G = minspangen(G)
    n = G.shape[1]
    T = Trellis(G)
    # obtain all combinations of subnodes in nodes 
    node = {}
    node[0] = [(2, ['00', '11']), (5, ['111', '011', '100', '000'])]
    node[1] = [(i, list(set(T.V[i]) - set(subV))) for i, subV in node[0]]
    nodes = []
    for choice in itertools.product([0, 1], repeat=len(node[0])):
        nodes.append([node[i][k] for (k, i) in enumerate(choice)])
    # for all subnodes get the subcode, which is the indexing list for the codewords
    subs = []
    for subnode in nodes:
        s = select_subcode(T, subnode)
        subV, subE = select_subtrellis(T, subnode)
        if plot: plottrellis(T, title='', subE=subE)
        sub = T.codewords[s]
        subs.append(sub)
    return nodes, subs, T  # cut pattern, codewords for subtrellis, and the trellis


if __name__ == '__main__':
    unittest.main()
