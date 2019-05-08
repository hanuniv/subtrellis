import unittest
import numpy as np
from rmtrellis import *


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
        V, E = select_subtrellis(T, [(2, ['00', '11']), (5, ['111', '011', '100', '000'])])
        self.assertEqual(V, [[''], ['0', '1'], ['00', '11'], ['000', '001', '110', '111'], 
            ['00', '01', '10', '11'], ['000', '011', '100', '111'], ['00', '01', '10', '11'], 
            ['0', '1'], ['']])
        self.assertEqual(E, [[('', '0', 0), ('', '1', 1)], [('0', '00', 0), ('1', '11', 0)], 
            [('00', '000', 0), ('00', '001', 1), ('11', '110', 0), ('11', '111', 1)], 
            [('000', '00', 0), ('001', '01', 1), ('110', '10', 1), ('111', '11', 0)], 
            [('00', '000', 0), ('01', '011', 0), ('10', '100', 0), ('11', '111', 0)], 
            [('000', '00', 0), ('011', '01', 0), ('100', '10', 1), ('111', '11', 1)], 
            [('00', '0', 0), ('01', '1', 1), ('10', '0', 1), ('11', '1', 0)], 
            [('0', '', 0), ('1', '', 1)]])


if __name__ == '__main__':
    unittest.main()
