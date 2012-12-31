from os.path import dirname, join
from root_numpy import *
from ROOT import TChain, TFile, TTree, TH1D, TH2D, TH3D
import numpy as np
import unittest
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal
from collections import OrderedDict
from nose.tools import *
from root_numpy.nputil import stretch

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


    def test_lst(self):
        structure = lst(self.ld('single1.root'))
        expected = OrderedDict([
            ('n_int', [('n_int', 'Int_t')]),
            ('f_float', [('f_float', 'Float_t')]),
            ('d_double', [('d_double', 'Double_t')])])
        assert_equal(structure, expected)


    def test_single(self):
        f = self.ld('single1.root')
        a = root2array(f)
        self.check_single(a)


    @raises(IOError)
    def test_single_pattern_not_exist(self):
        f = self.ld(['single1.root','does_not_exists.root'])
        a = root2array(f)


    @raises(IOError)
    def test_single_filename_not_exist(self):
        f = self.ld('does_not_exists.root')
        a = root2array(f)


    @raises(ValueError)
    def test_doubel_tree_name_not_specified(self):
        f = self.ld('doubletree1.root')
        a = root2array(f)


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
        self.check_single(tree2array(chain))


    @raises(TypeError)
    def test_tree2array_wrongtype(self):
        a = list()
        tree2array(a)


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


    def test_offset_entries(self):
        a = root2rec(self.ld('single1.root'), entries=10)
        assert_equal(len(a), 10)
        assert_equal(a.n_int[-1], 10)

        a = root2rec(self.ld('single1.root'), entries=10, offset=1)
        assert_equal(len(a), 10)
        assert_equal(a.n_int[-1], 11)

        a = root2rec(self.ld('single1.root'), entries=10, offset=95)
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


    def test_fill_array(self):
        np.random.seed(0)
        data1D = np.random.randn(1E6)
        w1D = np.empty(1E6)
        w1D.fill(2.)
        data2D = np.random.randn(1E6, 2)
        data3D = np.random.randn(1E4, 3)

        a = TH1D('th1d', 'test', 1000, -5, 5)
        fill_array(a, data1D)
        #one of them lies beyond hist range that's why it's not 1e6
        assert_almost_equal(a.Integral(), 999999.0)

        a_w = TH1D('th1dw', 'test', 1000, -5, 5)
        fill_array(a_w, data1D, w1D)
        assert_almost_equal(a_w.Integral(), 999999.0*2)

        b = TH2D('th2d', 'test', 100, -5, 5, 100, -5, 5)
        fill_array(b, data2D)
        assert_almost_equal(b.Integral(), 999999.0)

        c = TH3D('th3d', 'test', 10, -5, 5, 10, -5, 5, 10, -5, 5)
        fill_array(c, data3D)
        assert_almost_equal(c.Integral(), 10000.0)


    @raises(TypeError)
    def test_fill_array_wrongtype(self):
        h = list()
        a = np.random.randn(100)
        fill_array(h,a)


    def test_stretch(self):
        nrec = 5
        arr = np.empty(nrec, dtype=[('scalar',np.int), ('df1', 'O'),
                                    ('df2', 'O'), ('df3', 'O')])
        for i in range(nrec):
            scalar = i
            df1 = np.array(range(i+1), dtype=np.float)
            df2 = np.array(range(i+1), dtype=np.int)*2
            df3 = np.array(range(i+1), dtype=np.double)*3
            arr[i] = (i, df1, df2, df3)

        stretched =  stretch(arr,['scalar','df1','df2','df3'])

        assert_equal(stretched.dtype,
            [('scalar', np.int), ('df1', np.float), ('df2', np.int), ('df3', np.double)])
        assert_equal(stretched.size, 15)

        assert_almost_equal(stretched.df1[14],4.0)
        assert_almost_equal(stretched.df2[14],8)
        assert_almost_equal(stretched.df3[14],12.0)
        assert_almost_equal(stretched.scalar[14],4)
        assert_almost_equal(stretched.scalar[13],4)
        assert_almost_equal(stretched.scalar[12],4)
        assert_almost_equal(stretched.scalar[11],4)
        assert_almost_equal(stretched.scalar[10],4)
        assert_almost_equal(stretched.scalar[9],3)


if __name__ == '__main__':
    import nose
    nose.runmodule()
