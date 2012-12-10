from os.path import dirname, join
from root_numpy import *
from ROOT import TChain, TFile, TTree
import numpy as np
import unittest
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal


class TestRootNumpy(unittest.TestCase):

    def setUp(self):
        self.datadir = dirname(__file__)


    def ld(self, data):
        if isinstance(data, list):
            return [join(self.datadir, x) for x in data]
        else:
            return join(self.datadir, data)


    def check_single(self, single, n=100, id=1):
        assert_equal(
            single.dtype,
            [('n_int', '<i4'), ('f_float', '<f4'), ('d_double', '<f8')])
        assert_equal(len(single), n)
        for i in range(len(single)):
            id = (i / 100) + 1
            assert_equal(single[i][0], i % 100 + id)
            assert_almost_equal(single[i][1], i % 100 * 2.0 + id)
            assert_almost_equal(single[i][2], i % 100 * 3.0 + id)


    def test_lt(self):
        trees = lt(self.ld('vary1.root'))
        assert_equal(trees, ['tree'])


    def test_lb(self):
        branches = lb(self.ld('single1.root'))
        assert_equal(branches, ['n_int', 'f_float', 'd_double'])


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
        assert_equal(
            a.dtype,
            [('n_int', '<i4', (5,)),
             ('f_float', '<f4', (7,)),
             ('d_double', '<f8', (10,))])
        #TODO: Write a proper check method
        assert_equal(a[0][0][0], 1)
        assert_equal(a[0][0][1], 2)
        assert_almost_equal(a[-1][2][-1], 1514.5)


    def test_vary(self):
        f = self.ld(['vary1.root', 'vary2.root'])
        a = root2rec(f)
        assert_equal(
            a.dtype,
            [('len_n', '<i4'), ('len_f', '<i4'), ('len_d', '<i4'),
             ('n_int', 'O'), ('f_float', 'O'), ('d_double', 'O')])
        #check length
        for i in range(len(a)):
            assert_equal(a.len_n[i], len(a.n_int[i]))
            assert_equal(a.len_f[i], len(a.f_float[i]))
            assert_equal(a.len_d[i], len(a.d_double[i]))
        #couple element check
        assert_equal(a.len_n[0], 0)
        assert_equal(a.len_f[0], 1)
        assert_equal(a.len_d[0], 2)
        assert_equal(a.n_int[-1][-1], 417)
        assert_equal(a.f_float[-1][0], 380.5)
        assert_equal(a.f_float[-1][-1], 456.5)
        assert_equal(a.d_double[-1][0], 380.25)
        assert_equal(a.d_double[-1][-1], 497.25)


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
        assert_equal(a.dtype, [('f_float', '<f4')])


    def test_vector(self):
        a = root2rec(self.ld('hvector.root'))
        assert_equal(
            a.dtype,
            [('v_i', 'O'),
             ('v_f', 'O'),
             ('v_d', 'O'),
             ('v_l', 'O'),
             ('v_c', 'O')])

        assert_equal(a.v_i[1].dtype, np.int32)
        assert_equal(a.v_f[1].dtype, np.float32)
        assert_equal(a.v_d[1].dtype, np.float64)
        assert_equal(a.v_l[1].dtype, np.int64)
        assert_equal(a.v_c[1].dtype, np.int8)

        #check couple value
        assert_equal(a.v_i[1][0], 1)
        assert_equal(a.v_i[2][1], 3)
        assert_equal(a.v_i[-1][0], 99)
        assert_equal(a.v_i[-1][-1], 107)

        assert_equal(a.v_f[1][0], 2.0)
        assert_equal(a.v_f[2][1], 5.0)
        assert_equal(a.v_f[-1][0], 198.0)
        assert_equal(a.v_f[-1][-1], 206.0)


    def test_offset_N(self):
        a = root2rec(self.ld('single1.root'), N=10)
        assert_equal(len(a), 10)
        assert_equal(a.n_int[-1], 10)

        a = root2rec(self.ld('single1.root'), N=10, offset=1)
        assert_equal(len(a), 10)
        assert_equal(a.n_int[-1], 11)

        a = root2rec(self.ld('single1.root'), N=10, offset=95)
        assert_equal(len(a), 5)
        assert_equal(a.n_int[-1], 100)


    def test_weights(self):
        f = TFile(self.ld('test.root'))
        tree = f.Get('tree')
        tree.SetWeight(5.)
        rec = tree2rec(tree, include_weight=True, weight_name='treeweight')
        assert_array_equal(rec['treeweight'], np.ones(10) * 5)


    def test_PyRoot(self):
        f = TFile(self.ld('single1.root'))
        tree = f.Get('tree')
        tree2array(tree)


if __name__ == '__main__':
    import nose
    nose.runmodule()
