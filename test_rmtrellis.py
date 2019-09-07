import unittest
import numpy as np
from rmtrellis import *
from sympy import latex
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.style.use('fivethirtyeight')
import pickle
import sys
import json

# constants for testing
_G_rm13 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1]])
G_rm13 = minspangen(_G_rm13)
T_rm13 = Trellis(G_rm13)

_G_rm14 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
G_rm14 = minspangen(_G_rm14)
T_rm14 = Trellis(G_rm14)

_G_rm15 = np.vstack((np.ones(2**5), np.hstack((np.diag([0] + [1] * 4).dot(_G_rm14), _G_rm14))))
G_rm15 = minspangen(_G_rm15)
T_rm15 = Trellis(G_rm15)


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
        self.assertEqual(closestat([0], np.array([[0, 0], [0, 1], [1, 0], [1, 1]])), [2, 2, 0])
        self.assertEqual(closestat([0, 0], np.array([[0, 0], [0, 1], [1, 0], [1, 1]])), [1, 2, 1])

    def test_iswiningstat(self):
        self.assertEqual(iswiningstat([[0, 1, 0], [0, 0, 1]]), 'Y')
        self.assertEqual(iswiningstat([[0, 1, 0], [0, 1, 1]]), '_')
        self.assertEqual(iswiningstat([[0, 1, 0], [0, 2, 1]]), 'N')
        self.assertEqual(iswiningstat([[0, 1, 0], [1, 0, 1]]), 'N')

    def test_subcode(self):
        """ test on RM(3, 1)"""
        T = T_rm13
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
        T = T_rm13
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
        T = T_rm13
        ds = volumes(T, [0, 0, 0, 0, 0], 3)
        # print(ds) # TODO, is it right?


class SubDecodeTest(unittest.TestCase):

    @staticmethod
    def rm13_data():
        G = G_rm13
        T = T_rm13
        CB = [[1, 0, 1, 0], [1, 0, 0, 1]]
        subdec = SubDecodeCombineGenerator(G, CB)
        n = G.shape[1]
        mc = subdec.C.shape[0]  # length of coset messages
        ms = subdec.S.shape[0]  # length of the span basis
        D = list(itertools.product((0, 1), repeat=mc))  # index of messages
        s = np.array(list(itertools.product((0, 1), repeat=ms)))
        codewords = {}
        for d in D:
            codewords[d] = (np.array(d).dot(subdec.C) + s.dot(subdec.S)) % 2  # group codewords by cosets
        return G, T, subdec, D, codewords

    def test_decode(self):
        G = G_rm13
        mc = 2
        ms = G.shape[0] - mc
        subdec = SubDecode(G[:mc], G[mc:])
        w = np.random.randint(0, 2, G.shape[0])
        # print(w, subdec.decode(w.dot(G)))
        np.testing.assert_array_equal(w[:mc], subdec.decode(w.dot(G)))

    def test_generator_decode(self):
        G = G_rm13
        mc = 2
        subdec = SubDecodeFirstGenerator(G, mc)
        w = np.random.randint(0, 2, G.shape[0])
        np.testing.assert_array_equal(w[:mc], subdec.decode(w.dot(G)))
        subdec = SubDecodeSelectedGenerator(G, range(2))  # work with iterator as well
        indc = [1, 2]
        subdec = SubDecodeSelectedGenerator(G, indc)
        w = np.random.randint(0, 2, G.shape[0])
        np.testing.assert_array_equal(w[indc], subdec.decode(w.dot(G)))

    def test_generator_combination(self):
        G = G_rm13
        CB = [[1, 0, 1, 0], [1, 0, 0, 1]]
        subdec = SubDecodeCombineGenerator(G, CB)
        self.assertEqual(np.linalg.matrix_rank(subdec.G), G.shape[0])

    def test_edgenodepattern(self):
        """ See if D[0]'s trajectory is correct """
        G, T, subdec, D, codewords = self.rm13_data()
        ep, vp = edgenodepattern(T, codewords, D)
        V = T.V
        E = T.E
        n = T.n
        d = D[0]
        np.testing.assert_array_equal(codewords[d], np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 1, 1, 1, 1],
                                                              [0, 1, 1, 0, 0, 1, 1, 0],
                                                              [0, 1, 1, 0, 1, 0, 0, 1]])
                                      )
        vp0 = {(i, v): list(vp[i, v, d]) for i in range(n + 1) for v in V[i]}
        self.assertEqual(vp0, {(0, ''): [True, True, True, True],
                               (1, '0'): [True, True, True, True],
                               (1, '1'): [False, False, False, False],
                               (2, '00'): [True, True, False, False],
                               (2, '01'): [False, False, True, True],
                               (2, '10'): [False, False, False, False],
                               (2, '11'): [False, False, False, False],
                               (3, '000'): [True, True, False, False],
                               (3, '001'): [False, False, False, False],
                               (3, '010'): [False, False, True, True],
                               (3, '011'): [False, False, False, False],
                               (3, '100'): [False, False, False, False],
                               (3, '101'): [False, False, False, False],
                               (3, '110'): [False, False, False, False],
                               (3, '111'): [False, False, False, False],
                               (4, '00'): [True, True, False, False],
                               (4, '01'): [False, False, False, False],
                               (4, '10'): [False, False, True, True],
                               (4, '11'): [False, False, False, False],
                               (5, '000'): [True, False, False, False],
                               (5, '001'): [False, True, False, False],
                               (5, '010'): [False, False, False, False],
                               (5, '011'): [False, False, False, False],
                               (5, '100'): [False, False, True, False],
                               (5, '101'): [False, False, False, True],
                               (5, '110'): [False, False, False, False],
                               (5, '111'): [False, False, False, False],
                               (6, '00'): [True, False, False, False],
                               (6, '01'): [False, True, False, False],
                               (6, '10'): [False, False, True, False],
                               (6, '11'): [False, False, False, True],
                               (7, '0'): [True, False, True, False],
                               (7, '1'): [False, True, False, True],
                               (8, ''): [True, True, True, True]})
        ep0 = {(i, e): list(ep[i, e, d]) for i in range(n) for e in E[i]}
        self.assertEqual(ep0, {(0, TrellisEdge(begin='', end='0', weight=0)): [True, True, True, True],
                               (0, TrellisEdge(begin='', end='1', weight=1)): [False, False, False, False],
                               (1, TrellisEdge(begin='0', end='00', weight=0)): [True, True, False, False],
                               (1, TrellisEdge(begin='0', end='01', weight=1)): [False, False, True, True],
                               (1, TrellisEdge(begin='1', end='10', weight=1)): [False, False, False, False],
                               (1, TrellisEdge(begin='1', end='11', weight=0)): [False, False, False, False],
                               (2, TrellisEdge(begin='00', end='000', weight=0)): [True, True, False, False],
                               (2, TrellisEdge(begin='00', end='001', weight=1)): [False, False, False, False],
                               (2, TrellisEdge(begin='01', end='010', weight=1)): [False, False, True, True],
                               (2, TrellisEdge(begin='01', end='011', weight=0)): [False, False, False, False],
                               (2, TrellisEdge(begin='10', end='100', weight=1)): [False, False, False, False],
                               (2, TrellisEdge(begin='10', end='101', weight=0)): [False, False, False, False],
                               (2, TrellisEdge(begin='11', end='110', weight=0)): [False, False, False, False],
                               (2, TrellisEdge(begin='11', end='111', weight=1)): [False, False, False, False],
                               (3, TrellisEdge(begin='000', end='00', weight=0)): [True, True, False, False],
                               (3, TrellisEdge(begin='001', end='01', weight=1)): [False, False, False, False],
                               (3, TrellisEdge(begin='010', end='10', weight=0)): [False, False, True, True],
                               (3, TrellisEdge(begin='011', end='11', weight=1)): [False, False, False, False],
                               (3, TrellisEdge(begin='100', end='00', weight=1)): [False, False, False, False],
                               (3, TrellisEdge(begin='101', end='01', weight=0)): [False, False, False, False],
                               (3, TrellisEdge(begin='110', end='10', weight=1)): [False, False, False, False],
                               (3, TrellisEdge(begin='111', end='11', weight=0)): [False, False, False, False],
                               (4, TrellisEdge(begin='00', end='000', weight=0)): [True, False, False, False],
                               (4, TrellisEdge(begin='00', end='001', weight=1)): [False, True, False, False],
                               (4, TrellisEdge(begin='01', end='010', weight=1)): [False, False, False, False],
                               (4, TrellisEdge(begin='01', end='011', weight=0)): [False, False, False, False],
                               (4, TrellisEdge(begin='10', end='100', weight=0)): [False, False, True, False],
                               (4, TrellisEdge(begin='10', end='101', weight=1)): [False, False, False, True],
                               (4, TrellisEdge(begin='11', end='110', weight=1)): [False, False, False, False],
                               (4, TrellisEdge(begin='11', end='111', weight=0)): [False, False, False, False],
                               (5, TrellisEdge(begin='000', end='00', weight=0)): [True, False, False, False],
                               (5, TrellisEdge(begin='001', end='01', weight=1)): [False, True, False, False],
                               (5, TrellisEdge(begin='010', end='00', weight=1)): [False, False, False, False],
                               (5, TrellisEdge(begin='011', end='01', weight=0)): [False, False, False, False],
                               (5, TrellisEdge(begin='100', end='10', weight=1)): [False, False, True, False],
                               (5, TrellisEdge(begin='101', end='11', weight=0)): [False, False, False, True],
                               (5, TrellisEdge(begin='110', end='10', weight=0)): [False, False, False, False],
                               (5, TrellisEdge(begin='111', end='11', weight=1)): [False, False, False, False],
                               (6, TrellisEdge(begin='00', end='0', weight=0)): [True, False, False, False],
                               (6, TrellisEdge(begin='01', end='1', weight=1)): [False, True, False, False],
                               (6, TrellisEdge(begin='10', end='0', weight=1)): [False, False, True, False],
                               (6, TrellisEdge(begin='11', end='1', weight=0)): [False, False, False, True],
                               (7, TrellisEdge(begin='0', end='', weight=0)): [True, False, True, False],
                               (7, TrellisEdge(begin='1', end='', weight=1)): [False, True, False, True]})

    def test_viterbicorpasspattern(self):
        """make sure pass and pattern are the same"""
        G, T, subdec, D, codewords = self.rm13_data()
        r = np.random.randint(0, 2, T.n)
        co, closest = viterbicorpattern(T, r, codewords, D, ne=3)
        co1, closest1 = viterbicorpass(T, r, codewords, D, ne=3)
        self.assertEqual(co, co1)
        self.assertEqual(closest1, dict((k, np.any(v)) for k, v in closest.items()))

    def test_viterbicor(self, plot=False):
        """
        Test three steps of Viterbi, see notebook for output
        Want to make sure nodepass, edgepass are correctly made (seems correct)
        Want to make sure viberbi works right, (finally seems correct!)
        Want to clarify the relationship between closest and nodepass: when ne=T.n they are the same, otherwise it is the cutoff version
        """
        G, T, subdec, D, codewords = self.rm13_data()
        ne = 3
        r = np.full(T.n, 0)
        # r = np.random.randint(0, 2, T.n)
        co, closest = viterbicorpass(T, r, codewords, D, ne=ne)
        edgepass, nodepass = edgenodepass(T, codewords, D)
        if ne == T.n:
            self.assertEqual(closest, nodepass)  # not the same as nodepass in general!
        if plot:
            d0 = 1
            subE = [[e for e in Ei if edgepass[i, e, D[d0]]] for i, Ei in enumerate(T.E)]
            # edgelabel = {(i,e): 'true' if t else '' for (i, e, d),t in edgepass.items() if d == D[0]}
            # plottrellis(T, statelabel={(i,v):str(t) for (i,v,d),t in co.items() if d == D[0]}, subE=subE, edgelabel=edgelabel)
            # when d = D[1], nodepass != closest because of ne cutoff
            plottrellis(T, statelabel={(i, v): str(t) for (i, v, d), t in co.items() if d == D[d0]}, subE=subE)
            plottrellis(T, statelabel={(i, v): str(t)
                                       for (i, v, d), t in closest.items() if d == D[d0]}, subE=subE)
            plottrellis(T, statelabel={(i, v): str(t)
                                       for (i, v, d), t in nodepass.items() if d == D[d0]}, subE=subE)
        return co, closest, nodepass

    def test_get_state_space(self, G=G_rm13, T=T_rm13, CB=[[1, 0, 1, 0], [1, 0, 0, 1]], ne=None):
        subdec = SubDecodeCombineGenerator(G, CB)
        state_space, P = get_state_space(subdec, T, ne=ne)
        return (subdec, state_space, P)

    def test_simulate_cor(self, plot=False, ps=np.arange(0, 0.55, 0.1),
                          CB=[[1, 0, 1, 0], [1, 0, 0, 1]], T=T_rm13, G=G_rm13,
                          jinits=[jinit_psucc, jinit_cordiff, jinit_abscor], suffix='rm13'):
        """
        for a given construction, compare policies in jinits
        TODO: Should we save the optimal policy as well for examination?
        """
        psuccs = {}
        js = {}
        subdec, state_space, P = self.test_get_state_space(G=G, T=T, CB=CB)
        # print("State space size: \t", [len(s) for s in state_space])
        # print("State_space Obtained!")
        for jinit in jinits:
            psucc = []
            for p in ps:
                j, a = simulate_cor(state_space, P, T, p=p, jinit=jinit)
                psucc.append(cal_psuccess(state_space, P, a, p=p))
            psuccs[jinit] = psucc
            js[jinit] = p
        plot_psucc(ps, np.array(list(psuccs.values())), suffix=suffix, plot=plot)
        return ps, psuccs, state_space

    def test_simulate_cor_cosets(self, plot=False, ps=np.arange(0, 0.55, 0.1),
                                 T=T_rm13, G=G_rm13, dim=2, suffix='rm13'):
        """
        Test the performance for all possible coset divisions
        """
        results = {}
        # results['G'] = G.tolist()
        # results['suffix'] = suffix
        for i, CB in enumerate(echelons(dim, T.m)):
            ps, psuccs, state_space = \
            self.test_simulate_cor(CB=CB, T=T, G=G, ps=ps, plot=plot, suffix=suffix + '_{0}dim_{1}'.format(dim, i))
            results[i] = [CB.tolist(), [len(s) for s in state_space], list(psuccs.values())]
        save_obj(results, 'data/'+suffix+'_{}dim_results'.format(dim))
        return results

    def slice_results(self, results):
        dpps = np.array([psucc[0] for i, (_, _, psucc) in results.items()])
        CBs = np.array([CB for _, (CB, _, _) in results.items()])
        state_space_all = np.array([ss for _, (_, ss, _) in results.items()])
        return {'psucc': dpps, 'CBs': CBs, 'states': state_space_all}

    def analyze_results(self, results, ps = np.arange(0, 0.55, 0.1), dim=2, suffix='rm13'):
        # TODO: dim should be part of results
        sresults = self.slice_results(results)
        dpps, CBs, state_space_all = sresults['psucc'], sresults['CBs'], sresults['states']
        # analyze psucc order
        print("Ordering of psucces along different ps")
        for i in range(dpps.shape[1]):
            dporder = np.lexsort((dpps[:,i],))
            print(dporder)
        # print(dpps[dporder])
        # for analyzation of state_space size
        print("Size of state spaces")
        print(state_space_all)
        # plot a combo graph of all probability of errors
        pspan = slice(0,len(ps)) # adjust span as needed
        plt.figure()
        plt.plot(ps[pspan], dpps[dporder, pspan].T)
        plt.xlabel('Crossover Probability')
        plt.ylabel('Probability of Successful Transmission')
        plt.savefig('output/mdpcor_'+suffix+'_{}dim_'.format(dim)+'all'+'.png')
        plt.clf()
        # group together curves by the last ps
        group = groupps(dpps, -1, 1e-3)
        # [[CBs[i].dot(G) % 2 for i in g] for g in group]; # Look at its CBs if necessary
        print(group)

    def getresults(self, suffix='rm13', dim=1):
        return load_obj('data/'+suffix+'_{}dim_results'.format(dim))

    def getstategroup(self, suffix='rm13', dim=1):
        results = self.getresults(suffix=suffix, dim=dim)
        sresults = self.slice_results(results)
        _, _, state_space_all = sresults['psucc'], sresults['CBs'], sresults['states']
        return np.unique(state_space_all, axis=0, return_inverse=True)

    def getpsuccgroup(self, suffix='rm13', dim=1, axis=-1):
        # TODO: Have some states left in the test class.
        G = getattr(sys.modules[__name__], "G_" + suffix)
        results = self.getresults(suffix=suffix, dim=dim)
        sresults = self.slice_results(results)
        dpps, CBs, _ = sresults['psucc'], sresults['CBs'], sresults['states']
        # group = groupps(dpps, axis, 1e-3)
        return np.unique(dpps.round(decimals=10), axis = 0, return_inverse=True)

    def groupCBstates(self, group=None, suffix='rm13', dim=1):
        """
        obtain basis vector from filenames
        """
        G = getattr(sys.modules[__name__], "G_" + suffix)
        results = load_obj('data/'+suffix+'_{}dim_results'.format(dim))
        sresults = self.slice_results(results)
        dpps, CBs, state_space_all = sresults['psucc'], sresults['CBs'], sresults['states']
        if group is None:
            group = groupps(dpps, -1, 1e-3)
        CBgroup = [[CBs[i].dot(G) % 2 for i in g] for g in group]
        state_group = [[state_space_all[i] for i in g] for g in group]
        return CBgroup, state_group

    def groupanalysis(self, suffix='rm13', dim=1, group=[[0, 1, 2, 3, 4, 5, 6, 7], [14], [8, 9, 10, 11] ,[12, 13]]):
        with open(suffix + '_' + str(dim) + 'dim.tex', 'w') as f:
            f.write(r'''\begin{tabular}{cccc}
            \toprule
            Coset Generator & $|S_i|, i=0, 1, \ldots, 8$ & Plots \\''')
            for axis in [-1]:
                cbgroup, stategroup = self.groupCBstates(group=group, suffix=suffix, dim=dim)
                for ig, cbg, stateg in zip(group, cbgroup, stategroup):
                    l = len(ig)
                    f.write(r'\midrule' + '\n')
                    f.write(r'{\begin{minipage}{0.3\textwidth}\begin{align*}'+'\n')
                    for i, cb, state in zip(ig, cbg, stateg):
                        f.write(matrix2tex(cb.astype(np.int)) + r'\\' + '\n')
                    f.write(r'\end{align*}' + r'\end{minipage}}' + '\n')
                    f.write(r' & {\begin{minipage}{0.3\textwidth}$' + matrix2tex(state) + r'$\end{minipage}}' + '\n')
                    f.write(r' & ' + r'\begin{minipage}{\textwidth/3}\includegraphics[width=\textwidth]{' + \
                          'subtrellis/output/mdpcor_{0}_{1}dim_{2}.png'.format(suffix, dim, i)+ r'}\end{minipage} \\' + '\n')
            f.write(r'\bottomrule\end{tabular}' + '\n')

    def groupanalysis_longtable(self, caption='', label='tab:', suffix='rm13', dim=1, group=[[0, 1, 2, 3, 4, 5, 6, 7], [14], [8, 9, 10, 11] ,[12, 13]]):
        with open(suffix + '_' + str(dim) + 'dim.tex', 'w') as f:
            f.write(r'''
            {\setlength\tabcolsep{0pt}%
            \begin{center}
            \begin{longtable}{p{0.33\textwidth}p{0.33\textwidth}p{0.33\textwidth}}
            \caption{\bfseries ''' + caption + r'\label{' + label + r'''}} \\
            \toprule
            Coset Generator & $|S_i|, i=0, 1, \ldots, n$ & Plots \\
            \endfirsthead %Line(s) to appear as head of the table on the first page

            \multicolumn{3}{l}{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
            \toprule
            Coset Generator & $|S_i|, i=0, 1, \ldots, n$ & Plots \\*\midrule
            \endhead %Line(s) to appear at top of every page (except first)

            \multicolumn{3}{r}{{Continued on next page}}\\
            \endfoot%: Last line(s) to appear at the bottom of every page (except last)

            \bottomrule
            \endlastfoot''')
            for axis in [-1]:
                cbgroup, stategroup = self.groupCBstates(group=group, suffix=suffix, dim=dim)
                for ig, cbg, stateg in zip(group, cbgroup, stategroup):
                    l = len(ig)
                    f.write(r'\midrule' + '\n')
                    f.write(r'{\begin{minipage}{0.2\textwidth}\begin{align*}'+'\n')
                    for i, cb, state in zip(ig, cbg, stateg):
                        f.write(matrix2tex(cb.astype(np.int)) + r'\\' + '\n')
                    f.write(r'\end{align*}' + r'\end{minipage}}' + '\n')
                    f.write(r' & {\begin{minipage}{0.2\textwidth}$' + matrix2tex(state) + r'$\end{minipage}}' + '\n')
                    f.write(r' & ' + r'\begin{minipage}{\textwidth/3}\includegraphics[width=\textwidth]{' + \
                          'subtrellis/output/mdpcor_{0}_{1}dim_{2}.png'.format(suffix, dim, i)+ r'}\end{minipage} \\' + '\n')
            f.write(r'' + '\n' + r'\end{longtable}' + '\n' + r'\end{center} }')

    def test_cal_psuccess(self):
        ps = np.arange(0, 0.55, 0.1)
        CB = [[1, 0, 1, 0], [1, 0, 0, 1]]
        perror = []
        psucc = []
        T = T_rm13
        subdec, state_space, P = self.test_get_state_space(T=T_rm13, G=G_rm13, CB=CB)
        for p in ps:
            j, a = simulate_cor(state_space, P, T, p=p)
            perror.append(j[0, state_space[0][0]])
            psucc.append(cal_psuccess(state_space, P, a, p=p))
        np.testing.assert_allclose(np.array(perror), np.array(psucc))


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


def matrix2tex(a):
    if a.ndim == 2:
        return r"\setlength\arraycolsep{2pt}\begin{bmatrix}" + "\n".join([" & ".join(map(str,line))+"\\\\ " for line in a]) + r'\end{bmatrix}'
    elif a.ndim == 1:
        return r"\setlength\arraycolsep{2pt}\begin{bmatrix}" + " & ".join(str(e) for e in a) + r'\\ \end{bmatrix}'
    else:
        raise ValueError("Cannot deal with ndim > 2")

def plotresults(results, ps=np.arange(0, 0.55, 0.1),  suffix='rm13'):
    """
    Replot all the figures with results
    """
    for i, (_, _, psucc) in results.items():
        plot_psucc(ps, np.array(psucc), suffix=suffix+'_'+str(i))

def save_obj(obj, name):
    # with open(name + '.pkl', 'w') as f:
        # pickle.dump(obj, f, )
    with open(name + '.json', 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)


def load_obj(name):
    # with open(name + '.pkl', 'rb') as f:
        # return pickle.load(f)
    with open(name + '.json', 'r') as f:
        return json.load(f)


def groupps(dpps, pi=-1, tol=1e-2):
    dporder = np.lexsort((dpps[:, pi],))
    group = [[dporder[0]]]
    for i in range(len(dporder) - 1):
        if dpps[dporder[i + 1], pi] - dpps[dporder[i], pi] > tol:
            group.append([dporder[i + 1]])
        else:
            group[-1].append(dporder[i + 1])
    return group


def plot_psucc(ps, psuccs, suffix='cb1', plot=False):
    " Plot the probability of errors"
    if plot: plt.figure()
    plt.plot(ps, psuccs.T)
    plt.legend(['Prob Succ', 'Cor Diff', 'Cor'])
    plt.xlabel('Crossover Probability')
    plt.ylabel('Probability of Successful Transmission')
    plt.savefig('output/mdpcor_' + suffix + '.png')
    if not plot: plt.clf()

def echelons(nrow, ncol):
    """generate all possible echelon forms"""
    rows, cols = list(range(nrow)), list(range(ncol))
    for pivots in itertools.combinations(cols, nrow):
        # obtain places of free entries
        freecols = list(set(cols) - set(pivots))
        freepos = [(i, j) for i in rows for j in freecols if j > pivots[i]]
        for freevalue in itertools.product((0, 1), repeat=len(freepos)):
            CB = np.zeros((nrow, ncol))
            for fp, fv in zip(freepos, freevalue):
                CB[fp] = fv
            for fp in zip(rows, pivots):
                CB[fp] = 1
            # print(CB)
            yield CB


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
    # print(G)
    G = minspangen(G)
    # print(G)
    plottrellis(Trellis(G))


def rm13trellis():
    ''' for Reed-Muller Code '''
    G = G_rm13
    T = T_rm13
    n = T.n
    plottrellis(T, title='')
    nodes = [(2, ['00', '11']), (5, ['111', '011', '100', '000'])]
    s = select_subcode(T, nodes)
    subV, subE = select_subtrellis(T, nodes)
    plottrellis(T, title='', subE=subE)
    sub = T.codewords[s]
    # print(T.codewords[s])
    return s, T
    # return simulate_subcode(sub, T, maxsub_strategy)


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


def rm13trelliscode1(plot=False):
    G = G_rm13
    T = T_rm13
    n = T.n
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
        if plot:
            plottrellis(T, title='', subE=subE)
        sub = T.codewords[s]
        subs.append(sub)
    return nodes, subs, T  # cut pattern, codewords for subtrellis, and the trellis


def rm13trelliscode2(plot=False):
    G = G_rm13
    T = T_rm13
    n = T.n
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
        if plot:
            plottrellis(T, title='', subE=subE)
        sub = T.codewords[s]
        subs.append(sub)
    return nodes, subs, T  # cut pattern, codewords for subtrellis, and the trellis


if __name__ == '__main__':
    unittest.main()
