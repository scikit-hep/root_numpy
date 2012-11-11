import unittest
from root_numpy import *
from os.path import dirname, join
from ROOT import TChain
import numpy as np


class TestRootNumpy(unittest.TestCase):

    def setUp(self):
        self.datadir = dirname(__file__)

    def ld(self, data):
        if isinstance(data, list):
            return [join(self.datadir, x) for x in data]
        else:
            return join(self.datadir, data)

    def check_single(self, single, n=100, id=1):
        self.assertEqual(
            single.dtype,
            [('n_int', '<i4'), ('f_float', '<f4'), ('d_double', '<f8')])
        self.assertEqual(len(single), n)
        for i in range(len(single)):
            id = (i / 100) + 1
            self.assertEqual(single[i][0], i % 100 + id)
            self.assertAlmostEqual(single[i][1], i % 100 * 2.0 + id)
            self.assertAlmostEqual(single[i][2], i % 100 * 3.0 + id)

    def test_lt(self):
        trees = lt(self.ld('vary1.root'))
        self.assertEqual(trees, ['tree'])

    def test_lb(self):
        branches = lb(self.ld('single1.root'))
        self.assertEqual(branches, ['n_int', 'f_float', 'd_double'])

    def test_single(self):
        f = self.ld('single1.root')
        a = root2array(f)
        self.check_single(a)

    def test_singlechain(self):
        f = self.ld(['single1.root', 'single2.root'])
        a = root2array(f)
        self.check_single(a, 200)

    def test_fixed(self):
        f = self.ld(['fixed1.root', 'fixed2.root'])
        a = root2array(f)
        self.assertEqual(
            a.dtype,
            [('n_int', '<i4', (5,)),
             ('f_float', '<f4', (7,)),
             ('d_double', '<f8', (10,))])
        #TODO: Write a proper check method
        self.assertEqual(a[0][0][0], 1)
        self.assertEqual(a[0][0][1], 2)
        self.assertAlmostEqual(a[-1][2][-1], 1514.5)

    def test_vary(self):
        f = self.ld(['vary1.root', 'vary2.root'])
        a = root2rec(f)
        self.assertEqual(
            a.dtype,
            [('len_n', '<i4'), ('len_f', '<i4'), ('len_d', '<i4'),
             ('n_int', 'O'), ('f_float', 'O'), ('d_double', 'O')])
        #check length
        for i in range(len(a)):
            self.assertEqual(a.len_n[i], len(a.n_int[i]))
            self.assertEqual(a.len_f[i], len(a.f_float[i]))
            self.assertEqual(a.len_d[i], len(a.d_double[i]))
        #couple element check
        self.assertEqual(a.len_n[0], 0)
        self.assertEqual(a.len_f[0], 1)
        self.assertEqual(a.len_d[0], 2)
        self.assertEqual(a.n_int[-1][-1], 417)
        self.assertEqual(a.f_float[-1][0], 380.5)
        self.assertEqual(a.f_float[-1][-1], 456.5)
        self.assertEqual(a.d_double[-1][0], 380.25)
        self.assertEqual(a.d_double[-1][-1], 497.25)

    def test_tree2array(self):
        chain = TChain('tree')
        chain.Add(self.ld('single1.root'))
        self.check_single(tree2array(chain))

    def test_tree2rec(self):
        chain = TChain('tree')
        chain.Add(self.ld('single1.root'))
        #just make sure it doesn't crash

    def test_specific_branch(self):
        a = root2rec(self.ld('single1.root'), branches=['f_float'])
        self.assertEqual(a.dtype, [('f_float', '<f4')])

    def test_vector(self):
        a = root2rec(self.ld('hvector.root'))
        self.assertEqual(
            a.dtype,
            [('v_i', 'O'),
             ('v_f', 'O'),
             ('v_d', 'O'),
             ('v_l', 'O'),
             ('v_c', 'O')])

        self.assertEqual(a.v_i[1].dtype, np.int32)
        self.assertEqual(a.v_f[1].dtype, np.float32)
        self.assertEqual(a.v_d[1].dtype, np.float64)
        self.assertEqual(a.v_l[1].dtype, np.int64)
        self.assertEqual(a.v_c[1].dtype, np.int8)

        #check couple value
        self.assertEqual(a.v_i[1][0], 1)
        self.assertEqual(a.v_i[2][1], 3)
        self.assertEqual(a.v_i[-1][0], 99)
        self.assertEqual(a.v_i[-1][-1], 107)

        self.assertEqual(a.v_f[1][0], 2.0)
        self.assertEqual(a.v_f[2][1], 5.0)
        self.assertEqual(a.v_f[-1][0], 198.0)
        self.assertEqual(a.v_f[-1][-1], 206.0)

    def test_offset_N(self):
        a = root2rec(self.ld('single1.root'), N=10)
        self.assertEqual(len(a), 10)
        self.assertEqual(a.n_int[-1], 10)

        a = root2rec(self.ld('single1.root'), N=10, offset=1)
        self.assertEqual(len(a), 10)
        self.assertEqual(a.n_int[-1], 11)

        a = root2rec(self.ld('single1.root'), N=10, offset=95)
        self.assertEqual(len(a), 5)
        self.assertEqual(a.n_int[-1], 100)


if __name__ == '__main__':
    import nose
    nose.runmodule()
