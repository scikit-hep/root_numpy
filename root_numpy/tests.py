import os
from os.path import dirname, join
import tempfile
import warnings
from contextlib import contextmanager

import numpy as np
from numpy.lib import recfunctions
from numpy.testing import assert_array_equal
from numpy.random import RandomState

import ROOT
from ROOT import (
    TChain, TFile, TTree,
    TH1D, TH2D, TH3D,
    TProfile, TProfile2D, TProfile3D,
    TGraph, TGraph2D,
    TF1, TF2, TF3, TFormula,
    TLorentzVector)

import root_numpy as rnp
from root_numpy.testdata import get_filepath, get_file

try:
    from collections import OrderedDict
except ImportError:
    from root_numpy.extern.ordereddict import OrderedDict

from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_almost_equal)


ROOT.gErrorIgnoreLevel = ROOT.kFatal
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=rnp.RootNumpyUnconvertibleWarning)
RNG = RandomState(42)


def load(data):
    if isinstance(data, list):
        return [get_filepath(x) for x in data]
    else:
        return get_filepath(data)


@contextmanager
def temp():
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.root')
    tmp_root = ROOT.TFile.Open(tmp_path, 'recreate')
    yield tmp_root
    tmp_root.Close()
    os.close(tmp_fd)
    os.remove(tmp_path)


def check_single(single, n=100, offset=1):
    assert_equal(
        single.dtype,
        [('n_int', '<i4'), ('f_float', '<f4'), ('d_double', '<f8')])
    assert_equal(len(single), n)
    for i in range(len(single)):
        offset = (i // 100) + 1
        assert_equal(single[i][0], i % 100 + offset)
        assert_almost_equal(single[i][1], i % 100 * 2.0 + offset)
        assert_almost_equal(single[i][2], i % 100 * 3.0 + offset)


def test_list_trees():
    # TTree
    trees = rnp.list_trees(load('vary1.root'))
    assert_equal(trees, ['tree'])
    # TNtuple
    trees = rnp.list_trees(load('ntuple.root'))
    assert_equal(trees, ['ntuple'])


def test_list_branches():
    branches = rnp.list_branches(load('single1.root'))
    assert_equal(branches, ['n_int', 'f_float', 'd_double'])


def test_list_directories():
    directories = rnp.list_directories(load('directories.root'))
    assert_equal(set(directories), set(['Dir1', 'Dir2']))


def test_list_structures():
    structure = rnp.list_structures(load('single1.root'))
    expected = OrderedDict([
        ('n_int', [('n_int', 'int')]),
        ('f_float', [('f_float', 'float')]),
        ('d_double', [('d_double', 'double')])])
    assert_equal(structure, expected)


def test_ntuple():
    f = load('ntuple.root')
    a = rnp.root2array(f)
    assert_equal(len(a), 10)
    assert_equal(len(a.dtype.names), 3)


def test_single():
    f = load('single1.root')
    a = rnp.root2array(f)
    check_single(a)
    # specify tree name
    a = rnp.root2array(f, treename='tree')
    check_single(a)


@raises(IOError)
def test_single_pattern_not_exist():
    f = load(['single1.root', 'does_not_exist.root'])
    a = rnp.root2array(f)


@raises(ValueError)
def test_no_filename():
    rnp.root2array([])


def test_no_trees_in_file():
    with temp() as tmp:
        tmp.Close()
        assert_raises(IOError, rnp.root2array, [tmp.GetName()], treename=None)


@raises(IOError)
def test_single_filename_not_exist():
    f = load('does_not_exist.root')
    a = rnp.root2array(f)


@raises(ValueError)
def test_double_tree_name_not_specified():
    f = load('trees.root')
    a = rnp.root2array(f)


def test_single_chain():
    f = load(['single1.root', 'single2.root'])
    a = rnp.root2array(f)
    check_single(a, 200)


def test_tree_without_branches():
    tree = TTree('test', 'test')
    assert_raises(ValueError, rnp.tree2rec, tree)


def test_empty_branches():
    f = load('single1.root')
    assert_raises(ValueError, rnp.root2array, f, branches=[])


def test_empty_tree():
    from array import array
    tree = TTree('tree', 'tree')
    d = array('d', [0.])
    tree.Branch('double', d, 'double/D')
    rnp.tree2array(tree)


def test_duplicate_branch_name():
    from array import array
    tree = TTree('tree', 'tree')
    d = array('d', [0.])
    tree.Branch('double', d, 'double/D')
    tree.Branch('double', d, 'double/D')
    tree.Fill()

    # check that a warning was emitted
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = rnp.tree2array(tree)
        assert_equal(len(w), 1)
        assert_true(issubclass(w[-1].category, RuntimeWarning))
        assert_true("ignoring duplicate branch named" in str(w[-1].message))
    assert_equal(
        a.dtype,
        [('double', '<f8')])


def test_unsupported_branch_in_branches():
    tree = TTree('test', 'test')
    vect = TLorentzVector()
    double = np.array([0], dtype=float)
    tree.Branch('vector', vect)
    tree.Branch('double', double, 'double/D')
    rnp.tree2array(tree)
    assert_raises(TypeError, rnp.tree2array, tree, branches=['vector'])


def test_no_supported_branches():
    tree = TTree('test', 'test')
    vect = TLorentzVector()
    tree.Branch('vector', vect)
    assert_raises(RuntimeError, rnp.tree2array, tree)


def test_preserve_branch_order():
    a = rnp.root2array(load('test.root'))
    assert_equal(a.dtype.names, ('i', 'x', 'y', 'z'))

    a = rnp.root2array(load('test.root'), branches=['y', 'x', 'z'])
    assert_equal(a.dtype.names, ('y', 'x', 'z'))


def test_fixed_length_arrays():
    f = load(['fixed1.root', 'fixed2.root'])
    a = rnp.root2array(f)
    assert_equal(
        a.dtype,
        [('n_int', '<i4', (5,)),
         ('f_float', '<f4', (7,)),
         ('d_double', '<f8', (10,)),
         ('n2_int', '<i4', (5, 2)),
         ('f2_float', '<f4', (7, 3)),
         ('d2_double', '<f8', (10, 4))])

    # Check values
    assert_equal(a['n_int'][0][0], 1)
    assert_equal(a['n_int'][0][1], 2)
    assert_almost_equal(a['d_double'][-1][-1], 1514.5)
    assert_array_equal(a['n2_int'][0],
                       np.array([[1, 2],
                                 [2, 3],
                                 [3, 4],
                                 [4, 5],
                                 [5, 6]]))


def test_variable_length_arrays():
    f = load(['vary1.root', 'vary2.root'])
    a = rnp.root2rec(f)
    assert_equal(
        a.dtype,
        [('len_n', '<i4'), ('len_f', '<i4'), ('len_d', '<i4'),
         ('n_char', 'O'), ('n_uchar', 'O'),
         ('n_short', 'O'), ('n_ushort', 'O'),
         ('n_int', 'O'), ('n_uint', 'O'),
         ('n_long', 'O'), ('n_ulong', 'O'),
         ('f_float', 'O'), ('d_double', 'O'),
         ('n2_int', 'O'), ('f2_float', 'O'), ('d2_double', 'O')])

    # check lengths
    for i in range(len(a)):
        assert_equal(a.len_n[i], len(a.n_int[i]))
        assert_equal(a.len_f[i], len(a.f_float[i]))
        assert_equal(a.len_d[i], len(a.d_double[i]))

        assert_equal((a.len_n[i], 2), a.n2_int[i].shape)
        assert_equal((a.len_f[i], 3), a.f2_float[i].shape)
        assert_equal((a.len_d[i], 4), a.d2_double[i].shape)

    # check elements
    assert_equal(a.len_n[0], 0)
    assert_equal(a.len_f[0], 1)
    assert_equal(a.len_d[0], 2)
    assert_equal(a.n_int[-1][-1], 417)
    assert_equal(a.f_float[-1][0], 380.5)
    assert_equal(a.f_float[-1][-1], 456.5)
    assert_equal(a.d_double[-1][0], 380.25)
    assert_equal(a.d_double[-1][-1], 497.25)


def test_tree2array():
    chain = TChain('tree')
    chain.Add(load('single1.root'))
    check_single(rnp.tree2array(chain))

    f = get_file('single1.root')
    tree = f.Get('tree')
    check_single(rnp.tree2array(tree))

    assert_raises(ValueError, get_file, 'file_does_not_exist.root')


def test_tree2rec():
    chain = TChain('tree')
    chain.Add(load('single1.root'))
    check_single(rnp.tree2rec(chain))


def test_single_branch():
    f = get_file('single1.root')
    tree = f.Get('tree')
    arr1_1d = rnp.tree2array(tree, branches='n_int')
    arr2_1d = rnp.root2array(load('single1.root'), branches='n_int')
    assert_equal(arr1_1d.dtype, np.dtype('<i4'))
    assert_equal(arr2_1d.dtype, np.dtype('<i4'))


def test_selection():
    chain = TChain('tree')
    chain.Add(load('single1.root'))
    chain.Add(load('single2.root'))
    a = rnp.tree2rec(chain)
    assert_equal((a['d_double'] <= 100).any(), True)
    a = rnp.tree2rec(chain, selection="d_double > 100")
    assert_equal((a['d_double'] <= 100).any(), False)

    # selection with differing variables in branches and expression
    a = rnp.tree2array(chain,
        branches=['d_double'],
        selection="f_float < 100 && n_int%2 == 1")

    # selection with TMath
    a = rnp.tree2rec(chain,
        selection="TMath::Erf(d_double) < 0.5")


def test_expression():
    rec = rnp.root2rec(load('single*.root'))
    rec2 = rnp.root2rec(load('single*.root'), branches=['f_float*2'])
    assert_array_equal(rec['f_float'] * 2, rec2['f_float*2'])


def test_selection_and_expression():
    ref = len(rnp.root2rec(
        load('test.root'), branches=['x', 'y'], selection='z>0'))
    assert_equal(ref,
        len(rnp.root2rec(
            load('test.root'), branches=['x', 'y', 'z'], selection='z>0')))
    assert_equal(ref,
        len(rnp.root2rec(
            load('test.root'), branches=['x', 'x*y'], selection='z>0')))
    assert_equal(ref,
        len(rnp.root2rec(
            load('test.root'), branches=['x', 'x*z'], selection='z>0')))


def test_object_expression():
    rec = rnp.root2rec(load(['object1.root', 'object2.root']),
                       branches=['vect.Pt()'])
    assert_array_equal(
        rec['vect.Pt()'],
        np.concatenate([
            np.arange(10, dtype='d') + 1,
            np.arange(10, dtype='d') + 2]))


@raises(ValueError)
def test_branch_DNE():
    chain = TChain('tree')
    chain.Add(load('single1.root'))
    rnp.tree2array(chain, branches=['my_net_worth'])


@raises(TypeError)
def test_tree2array_wrong_type():
    rnp.tree2array(list())


def test_specific_branch():
    a = rnp.root2rec(load('single1.root'), branches=['f_float'])
    assert_equal(a.dtype, [('f_float', '<f4')])


def test_vector():
    a = rnp.root2rec(load('vector.root'))
    types = [
        ('v_i', 'O'),
        ('v_f', 'O'),
        ('v_F', 'O'),
        ('v_d', 'O'),
        ('v_l', 'O'),
        ('v_c', 'O'),
        ('v_b', 'O'),
        ('vv_i', 'O'),
        ('vv_f', 'O'),
        ('vv_F', 'O'),
        ('vv_d', 'O'),
        ('vv_l', 'O'),
        ('vv_c', 'O'),
        ('vv_b', 'O'),
    ]
    assert_equal(a.dtype, types)

    assert_equal(a.v_i[0].dtype, np.int32)
    assert_equal(a.v_f[0].dtype, np.float32)
    assert_equal(a.v_F[0].dtype, np.float32)
    assert_equal(a.v_d[0].dtype, np.float64)
    assert_equal(a.v_l[0].dtype, np.int64)
    assert_equal(a.v_c[0].dtype, np.int8)
    assert_equal(a.v_b[0].dtype, np.bool)

    # assert that wrapper array is np.object
    assert_equal(a.vv_i[0].dtype, np.object)
    assert_equal(a.vv_f[0].dtype, np.object)
    assert_equal(a.vv_F[0].dtype, np.object)
    assert_equal(a.vv_d[0].dtype, np.object)
    assert_equal(a.vv_l[0].dtype, np.object)
    assert_equal(a.vv_c[0].dtype, np.object)
    assert_equal(a.vv_b[0].dtype, np.object)

    assert_equal(a.vv_i[0][0].dtype, np.int32)
    assert_equal(a.vv_f[0][0].dtype, np.float32)
    assert_equal(a.vv_F[0][0].dtype, np.float32)
    assert_equal(a.vv_d[0][0].dtype, np.float64)
    assert_equal(a.vv_l[0][0].dtype, np.int64)
    assert_equal(a.vv_c[0][0].dtype, np.int8)
    assert_equal(a.vv_b[0][0].dtype, np.bool)

    # check a few values
    assert_equal(a.v_i[0][0], 1)
    assert_equal(a.v_i[1][1], 3)
    assert_equal(a.v_i[-2][0], 9)
    assert_equal(a.v_i[-2][-1], 17)

    assert_equal(a.v_f[0][0], 2.0)
    assert_equal(a.v_f[1][1], 5.0)
    assert_equal(a.v_f[-2][0], 18.0)
    assert_equal(a.v_f[-2][-1], 26.0)

    assert_equal(a.v_F[0][0], 2.0)
    assert_equal(a.v_F[1][1], 5.0)
    assert_equal(a.v_F[-2][0], 18.0)
    assert_equal(a.v_F[-2][-1], 26.0)

    # more strict conditioning for numpy arrays
    def assert_equal_array(arr1, arr2):
        return assert_equal((arr1 == arr2).all(), True,
            "array mismatch: {0} != {1}".format(arr1, arr2))

    assert_equal_array(a.vv_i[0][0], np.array([1], dtype=np.int32) )
    assert_equal_array(a.vv_i[1][1], np.array([2, 3], dtype=np.int32) )
    assert_equal_array(a.vv_i[-2][0], np.array([9], dtype=np.int32) )
    assert_equal_array(a.vv_i[-2][-1],
                       np.array([ 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                dtype=np.int32))

    assert_equal_array(a.vv_f[0][0], np.array([ 2.], dtype=np.float32) )
    assert_equal_array(a.vv_f[1][1], np.array([ 4.,  5.], dtype=np.float32) )
    assert_equal_array(a.vv_f[-2][0], np.array([ 18.], dtype=np.float32) )
    assert_equal_array(a.vv_f[-2][-1],
                       np.array([ 18.,  19.,  20.,  21.,  22.,
                                  23.,  24.,  25.,  26.],
                                dtype=np.float32))

    assert_equal_array(a.vv_F[0][0], np.array([ 2.], dtype=np.float32) )
    assert_equal_array(a.vv_F[1][1], np.array([ 4.,  5.], dtype=np.float32) )
    assert_equal_array(a.vv_F[-2][0], np.array([ 18.], dtype=np.float32) )
    assert_equal_array(a.vv_F[-2][-1],
                       np.array([ 18.,  19.,  20.,  21.,  22.,
                                  23.,  24.,  25.,  26.],
                                dtype=np.float32))


def test_string():
    a = rnp.root2rec(load('string.root'))
    types = [
        ('message', 'O'),
        ('vect', 'O'),
        ('vect2d', 'O'),
    ]
    assert_equal(a.dtype, types)
    assert_equal(a[0][0], 'Hello World!')
    assert_equal(a[0][1][0], 'Hello!')
    assert_equal(a[0][2][0][0], 'Hello!')


def test_slice():
    a = rnp.root2rec(load('single1.root'), stop=10)
    assert_equal(len(a), 10)
    assert_equal(a.n_int[-1], 10)

    a = rnp.root2rec(load('single1.root'), stop=11, start=1)
    assert_equal(len(a), 10)
    assert_equal(a.n_int[-1], 11)

    a = rnp.root2rec(load('single1.root'), stop=105, start=95)
    assert_equal(len(a), 5)
    assert_equal(a.n_int[-1], 100)


def test_weights():
    f = TFile(load('test.root'))
    tree = f.Get('tree')
    tree.SetWeight(5.)
    rec = rnp.tree2rec(tree, include_weight=True, weight_name='treeweight')
    assert_array_equal(rec['treeweight'], np.ones(100) * 5)
    f = load(['single1.root', 'single2.root'])
    a = rnp.root2array(f, include_weight=True)
    assert_array_equal(
        a['weight'],
        np.concatenate((np.ones(100) * 2., np.ones(100) * 3.)))


def test_PyROOT():
    f = TFile(load('single1.root'))
    tree = f.Get('tree')
    rnp.tree2array(tree)


def test_fill_hist():
    n_samples = 1000
    data1D = RNG.randn(n_samples)
    w1D = np.empty(n_samples)
    w1D.fill(2.)
    data2D = RNG.randn(n_samples, 2)
    data3D = RNG.randn(n_samples, 3)

    a = TH1D('th1d', 'test', 100, -5, 5)
    rnp.fill_hist(a, data1D)
    assert_almost_equal(a.Integral(), n_samples)

    a_w = TH1D('th1dw', 'test', 100, -5, 5)
    rnp.fill_hist(a_w, data1D, w1D)
    assert_almost_equal(a_w.Integral(), n_samples * 2)

    b = TH2D('th2d', 'test', 100, -5, 5, 100, -5, 5)
    rnp.fill_hist(b, data2D)
    assert_almost_equal(b.Integral(), n_samples)

    c = TH3D('th3d', 'test', 10, -5, 5, 10, -5, 5, 10, -5, 5)
    rnp.fill_hist(c, data3D)
    assert_almost_equal(c.Integral(), n_samples)

    # array and weights lengths do not match
    assert_raises(ValueError, rnp.fill_hist, c, data3D, np.ones(10))

    # weights is not 1D
    assert_raises(ValueError, rnp.fill_hist, c, data3D,
        np.ones((data3D.shape[0], 1)))

    # array not 2-d when filling 2D/3D histogram
    for h in (b, c):
        assert_raises(ValueError, rnp.fill_hist, h, RNG.randn(10))

    # length of second axis does not match dimensionality of histogram
    for h in (a, b, c):
        assert_raises(ValueError, rnp.fill_hist, h, RNG.randn(10, 4))

    # wrong type
    h = list()
    a = RNG.randn(10)
    assert_raises(TypeError, rnp.fill_hist, h, a)


def test_fill_profile():
    n_samples = 1000
    w1D = np.empty(n_samples)
    w1D.fill(2.)
    data1D = RNG.randn(n_samples, 2)
    data2D = RNG.randn(n_samples, 3)
    data3D = RNG.randn(n_samples, 4)

    a = TProfile('th1d', 'test', 100, -5, 5)
    rnp.fill_profile(a, data1D)
    assert_true(a.Integral() != 0)

    a_w = TProfile('th1dw', 'test', 100, -5, 5)
    rnp.fill_profile(a_w, data1D, w1D)
    assert_true(a_w.Integral() != 0)
    assert_equal(a_w.Integral(), a.Integral())

    b = TProfile2D('th2d', 'test', 100, -5, 5, 100, -5, 5)
    rnp.fill_profile(b, data2D)
    assert_true(b.Integral() != 0)

    c = TProfile3D('th3d', 'test', 10, -5, 5, 10, -5, 5, 10, -5, 5)
    rnp.fill_profile(c, data3D)
    assert_true(c.Integral() != 0)

    # array and weights lengths do not match
    assert_raises(ValueError, rnp.fill_profile, c, data3D, np.ones(10))

    # weights is not 1D
    assert_raises(ValueError, rnp.fill_profile, c, data3D,
                  np.ones((data3D.shape[0], 1)))

    # array is not 2D
    assert_raises(ValueError, rnp.fill_profile, c, np.ones(10))

    # length of second axis is not one more than dimensionality of the profile
    for h in (a, b, c):
        assert_raises(ValueError, rnp.fill_profile, h, RNG.randn(10, 5))

    # wrong type
    assert_raises(TypeError, rnp.fill_profile,
                  TH1D("test", "test", 1, 0, 1), data1D)


def test_fill_graph():
    n_samples = 1000
    data2D = RNG.randn(n_samples, 2)
    data3D = RNG.randn(n_samples, 3)

    graph = TGraph()
    rnp.fill_graph(graph, data2D)

    graph2d = TGraph2D()
    rnp.fill_graph(graph2d, data3D)

    # array not 2-d
    for g in (graph, graph2d):
        assert_raises(ValueError, rnp.fill_graph, g, RNG.randn(10))

    # length of second axis does not match dimensionality of histogram
    for g in (graph, graph2d):
        assert_raises(ValueError, rnp.fill_graph, g, RNG.randn(10, 4))

    # wrong type
    h = list()
    a = RNG.randn(10)
    assert_raises(TypeError, rnp.fill_graph, h, a)


def test_stretch():
    arr = np.empty(5,
        dtype=[
            ('scalar', np.int),
            ('vl1', 'O'),
            ('vl2', 'O'),
            ('vl3', 'O'),
            ('fl1', np.int, (2, 2)),
            ('fl2', np.float, (2, 3)),
            ('fl3', np.double, (3, 2))])

    for i in range(arr.shape[0]):
        vl1 = np.array(range(i + 1), dtype=np.int)
        vl2 = np.array(range(i + 2), dtype=np.float) * 2
        vl3 = np.array(range(2), dtype=np.double) * 3
        fl1 = np.array(range(4), dtype=np.int).reshape((2, 2))
        fl2 = np.array(range(6), dtype=np.float).reshape((2, 3))
        fl3 = np.array(range(6), dtype=np.double).reshape((3, 2))
        arr[i] = (i, vl1, vl2, vl3, fl1, fl2, fl3)

    # no array columns included
    assert_raises(RuntimeError, rnp.stretch, arr, ['scalar',])

    # lengths don't match
    assert_raises(ValueError, rnp.stretch, arr, ['scalar', 'vl1', 'vl2',])
    assert_raises(ValueError, rnp.stretch, arr, ['scalar', 'fl1', 'fl3',])
    assert_raises(ValueError, rnp.stretch, arr)

    # variable-length stretch
    stretched = rnp.stretch(arr, ['scalar', 'vl1',])
    assert_equal(stretched.dtype,
                 [('scalar', np.int),
                  ('vl1', np.int)])
    assert_equal(stretched.shape[0], 15)
    assert_array_equal(
        stretched['scalar'],
        np.repeat(arr['scalar'], np.vectorize(len)(arr['vl1'])))

    # fixed-length stretch
    stretched = rnp.stretch(arr, ['scalar', 'vl3', 'fl1', 'fl2',])
    assert_equal(stretched.dtype,
                 [('scalar', np.int),
                  ('vl3', np.double),
                  ('fl1', np.int, (2,)),
                  ('fl2', np.float, (3,))])
    assert_equal(stretched.shape[0], 10)
    assert_array_equal(
        stretched['scalar'], np.repeat(arr['scalar'], 2))

    # optional argument return_indices
    stretched, idx = rnp.stretch(arr, ['scalar', 'vl1'], return_indices=True)
    assert_equal(stretched.shape[0], idx.shape[0])

    from_arr = list(map(lambda x: x['vl1'][0], arr))
    from_stretched = stretched[idx == 0]['vl1']
    assert_array_equal(from_arr, from_stretched)


def test_blockwise_inner_join():
    test_data = np.array([
        (1.0, np.array([11,12,13]), np.array([1,0,1]), 0, np.array([1,2,3])),
        (2.0, np.array([21,22,23]), np.array([-1,2,-1]), 1, np.array([31,32,33]))],
        dtype=[
            ('sl', np.float),
            ('al', 'O'),
            ('fk', 'O'),
            ('s_fk', np.int),
            ('ar', 'O')])
    # vector join
    a1 = rnp.blockwise_inner_join(
        test_data, ['sl', 'al'], test_data['fk'], ['ar'])

    # specify fk with string
    a1 = rnp.blockwise_inner_join(
        test_data, ['sl', 'al'], 'fk', ['ar'])

    exp1 = np.array([
        (1.0, 11, 2, 1),
        (1.0, 12, 1, 0),
        (1.0, 13, 2, 1),
        (2.0, 22, 33, 2)],
        dtype=[
            ('sl', '<f8'),
            ('al', '<i8'),
            ('ar', '<i8'),
            ('fk', '<i8')])
    assert_array_equal(a1, exp1, verbose=True)

    # vector join with force repeat
    a2 = rnp.blockwise_inner_join(
        test_data, ['sl','al'], test_data['fk'], ['ar'], force_repeat=['al'])
    exp2 = np.array([
        (1.0, np.array([11, 12, 13]), 2, 1),
        (1.0, np.array([11, 12, 13]), 1, 0),
        (1.0, np.array([11, 12, 13]), 2, 1),
        (2.0, np.array([21, 22, 23]), 33, 2)],
        dtype=[
            ('sl', '<f8'),
            ('al', '|O8'),
            ('ar', '<i8'),
            ('fk', '<i8')])
    assert_equal(str(a2), str(exp2)) # numpy testing doesn't like subarray
    assert_equal(a2.dtype, exp2.dtype)

    # scalar join
    a3 = rnp.blockwise_inner_join(
        test_data, ['sl', 'al'], test_data['s_fk'], ['ar'])
    exp3 = np.array([
        (1.0, [11, 12, 13], 1, 0),
        (2.0, [21, 22, 23], 32, 1)],
        dtype=[
            ('sl', '<f8'),
            ('al', '|O8'),
            ('ar', '<i8'),
            ('fk', '<i8')])
    assert_equal(str(a3), str(exp3)) # numpy testing doesn't like subarray
    assert_equal(a3.dtype, exp3.dtype)


def test_struct():
    assert_array_equal(rnp.root2rec(load('struct.root')),
        np.array([(10, 15.5, 20, 781.2)],
            dtype=[
                ('branch1_intleaf', '<i4'),
                ('branch1_floatleaf', '<f4'),
                ('branch2_intleaf', '<i4'),
                ('branch2_floatleaf', '<f4')]))


def test_array2tree():
    a = np.array([
        (12345, 2., 2.1, True),
        (3, 4., 4.2, False),],
        dtype=[
            ('x', np.int32),
            ('y', np.float32),
            ('z', np.float64),
            ('w', np.bool)])

    with temp() as tmp:
        tree = rnp.array2tree(a)
        a_conv = rnp.tree2array(tree)
        assert_array_equal(a, a_conv)
        # extend the tree
        tree2 = rnp.array2tree(a, tree=tree)
        assert_equal(tree2.GetEntries(), len(a) * 2)
        a_conv2 = rnp.tree2array(tree2)
        assert_array_equal(np.hstack([a, a]), a_conv2)

    assert_raises(TypeError, rnp.array2tree, a, tree=object)


def test_array2tree_charstar():
    a = np.array([b'', b'a', b'ab', b'abc', b'xyz', b''],
                 dtype=[('string', 'S3')])

    with temp() as tmp:
        rnp.array2root(a, tmp.GetName(), mode='recreate')
        a_conv = rnp.root2array(tmp.GetName())
        assert_array_equal(a, a_conv)


def test_array2tree_fixed_length_arrays():
    f = load(['fixed1.root', 'fixed2.root'])
    a = rnp.root2array(f)
    with temp() as tmp:
        rnp.array2root(a, tmp.GetName(), mode='recreate')
        a_conv = rnp.root2array(tmp.GetName())
        assert_array_equal(a, a_conv)


def test_array2root():
    a = np.array([
        (12345, 2., 2.1, True),
        (3, 4., 4.2, False),],
        dtype=[
            ('x', np.int32),
            ('y', np.float32),
            ('z', np.float64),
            ('w', np.bool)])
    with temp() as tmp:
        rnp.array2root(a, tmp.GetName(), mode='recreate')
        a_conv = rnp.root2array(tmp.GetName())
        assert_array_equal(a, a_conv)
        # extend the tree
        rnp.array2root(a, tmp.GetName(), mode='update')
        a_conv2 = rnp.root2array(tmp.GetName())
        assert_array_equal(np.hstack([a, a]), a_conv2)


def check_random_sample(obj):
    sample = rnp.random_sample(obj, 100)
    ndim = getattr(obj, 'GetDimension',
                   getattr(obj, 'GetNdim', None))()
    if ndim > 1:
        assert_equal(sample.shape, (100, ndim))
    else:
        assert_equal(sample.shape, (100,))
    a = rnp.random_sample(obj, 10, seed=1)
    b = rnp.random_sample(obj, 10, seed=1)
    c = rnp.random_sample(obj, 10, seed=2)
    assert_array_equal(a, b)
    assert_true((a != c).any())


def test_random_sample():
    funcs = [
        TF1("f1", "TMath::DiLog(x)"),
        TF2("f2", "sin(x)*sin(y)/(x*y)"),
        TF3("f3", "sin(x)*sin(y)*sin(z)/(x*y*z)"),
    ]
    hists = [
        TH1D("h1", "h1", 10, -3, 3),
        TH2D("h2", "h2", 10, -3, 3, 10, -3, 3),
        TH3D("h3", "h3", 10, -3, 3, 10, -3, 3, 10, -3, 3),
    ]
    for i, hist in enumerate(hists):
        hist.FillRandom(funcs[i].GetName())
    for obj in funcs + hists:
        yield check_random_sample, obj


def test_random_sample_bad_input():
    func = TF1("f1", "TMath::DiLog(x)")
    assert_raises(ValueError, rnp.random_sample, func, 0)
    assert_raises(ValueError, rnp.random_sample, func, 10, seed=-1)
    assert_raises(TypeError, rnp.random_sample, object(), 10)


def check_array(cls, copy):
    a = cls(10)
    a[2] = 2
    b = rnp.array(a, copy=copy)
    assert_equal(b[2], 2)
    assert_equal(b.shape[0], 10)


def test_array():
    for copy in (True, False):
        for cls in (getattr(ROOT, 'TArray{0}'.format(atype))
                for atype in 'DFLIS'):
            yield check_array, cls, copy
        a = ROOT.TArrayC(10)
        b = rnp.array(a, copy=copy)
        assert_equal(b.shape[0], 10)
    assert_raises(TypeError, rnp.array, object)


def check_matrix(cls):
    mat = cls(5, 5)
    mat[1][2] = 2
    np_mat = rnp.matrix(mat)
    assert_equal(np_mat[1, 2], 2)


def check_matrix_sym(cls):
    mat = cls(5)
    mat[2][2] = 2
    np_mat = rnp.matrix(mat)
    assert_equal(np_mat[2, 2], 2)


def test_matrix():
    for cls in (getattr(ROOT, 'TMatrix{0}'.format(atype)) for atype in 'DF'):
        yield check_matrix, cls

    for cls in (getattr(ROOT, 'TMatrix{0}Sym'.format(atype)) for atype in 'DF'):
        yield check_matrix_sym, cls

    assert_raises(TypeError, rnp.matrix, object)


def test_rec2array():
    a = np.array([
        (12345, 2., 2.1, True),
        (3, 4., 4.2, False),],
        dtype=[
            ('x', np.int32),
            ('y', np.float32),
            ('z', np.float64),
            ('w', np.bool)])
    arr = rnp.rec2array(a)
    assert_array_equal(arr,
        np.array([
            [12345, 2, 2.1, 1],
            [3, 4, 4.2, 0]]))
    arr = rnp.rec2array(a, fields=['x', 'y'])
    assert_array_equal(arr,
        np.array([
            [12345, 2],
            [3, 4]]))
    # single field
    arr = rnp.rec2array(a, fields=['x'])
    assert_equal(arr.ndim, 1)
    assert_equal(arr.shape, (a.shape[0],))


def test_stack():
    rec = rnp.root2rec(load('test.root'))
    s = rnp.stack([rec, rec])
    assert_equal(s.shape[0], 2 * rec.shape[0])
    assert_equal(s.dtype.names, rec.dtype.names)
    s = rnp.stack([rec, rec], fields=['x', 'y'])
    assert_equal(s.shape[0], 2 * rec.shape[0])
    assert_equal(s.dtype.names, ('x', 'y'))
    # recs don't have identical fields
    rec2 = recfunctions.drop_fields(rec, ['i', 'x'])
    s = rnp.stack([rec, rec2])
    assert_equal(set(s.dtype.names), set(['y', 'z']))


def test_dup_idx():
    a = [1, 2, 3, 4, 3, 2]
    assert_array_equal(rnp.dup_idx(a), [1, 2, 4, 5])


def test_evaluate():
    # create functions and histograms
    f1 = TF1("f1", "x")
    f2 = TF2("f2", "x*y")
    f3 = TF3("f3", "x*y*z")
    h1 = TH1D("h1", "", 10, 0, 1)
    h1.FillRandom("f1")
    h2 = TH2D("h2", "", 10, 0, 1, 10, 0, 1)
    h2.FillRandom("f2")
    h3 = TH3D("h3", "", 10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3.FillRandom("f3")
    # generate random arrays
    arr_1d = RNG.rand(5)
    arr_2d = RNG.rand(5, 2)
    arr_3d = RNG.rand(5, 3)
    arr_4d = RNG.rand(5, 4)
    # evaluate the functions
    assert_array_equal(rnp.evaluate(f1, arr_1d),
                       [f1.Eval(x) for x in arr_1d])
    assert_array_equal(rnp.evaluate(f1.GetTitle(), arr_1d),
                       [f1.Eval(x) for x in arr_1d])
    assert_array_equal(rnp.evaluate(f2, arr_2d),
                       [f2.Eval(*x) for x in arr_2d])
    assert_array_equal(rnp.evaluate(f2.GetTitle(), arr_2d),
                       [f2.Eval(*x) for x in arr_2d])
    assert_array_equal(rnp.evaluate(f3, arr_3d),
                       [f3.Eval(*x) for x in arr_3d])
    assert_array_equal(rnp.evaluate(f3.GetTitle(), arr_3d),
                       [f3.Eval(*x) for x in arr_3d])
    # 4d formula
    f4 = TFormula('test', 'x*y+z*t')
    assert_array_equal(rnp.evaluate(f4, arr_4d),
                       [f4.Eval(*x) for x in arr_4d])
    # evaluate the histograms
    assert_array_equal(rnp.evaluate(h1, arr_1d),
                       [h1.GetBinContent(h1.FindBin(x)) for x in arr_1d])
    assert_array_equal(rnp.evaluate(h2, arr_2d),
                       [h2.GetBinContent(h2.FindBin(*x)) for x in arr_2d])
    assert_array_equal(rnp.evaluate(h3, arr_3d),
                       [h3.GetBinContent(h3.FindBin(*x)) for x in arr_3d])
    # create a graph
    g = TGraph(2)
    g.SetPoint(0, 0, 1)
    g.SetPoint(1, 1, 2)
    assert_array_equal(rnp.evaluate(g, [0, .5, 1]), [1, 1.5, 2])
    from ROOT import TSpline3
    s = TSpline3("spline", g)
    assert_array_equal(rnp.evaluate(s, [0, .5, 1]),
                       [s.Eval(x) for x in [0, .5, 1]])
    # test exceptions
    assert_raises(TypeError, rnp.evaluate, object(), [1, 2, 3])
    assert_raises(ValueError, rnp.evaluate, h1, arr_2d)
    assert_raises(ValueError, rnp.evaluate, h2, arr_3d)
    assert_raises(ValueError, rnp.evaluate, h2, arr_1d)
    assert_raises(ValueError, rnp.evaluate, h3, arr_1d)
    assert_raises(ValueError, rnp.evaluate, h3, arr_2d)
    assert_raises(ValueError, rnp.evaluate, f1, arr_2d)
    assert_raises(ValueError, rnp.evaluate, f2, arr_3d)
    assert_raises(ValueError, rnp.evaluate, f2, arr_1d)
    assert_raises(ValueError, rnp.evaluate, f3, arr_1d)
    assert_raises(ValueError, rnp.evaluate, f3, arr_2d)
    assert_raises(ValueError, rnp.evaluate, g, arr_2d)
    assert_raises(ValueError, rnp.evaluate, s, arr_2d)
    assert_raises(ValueError, rnp.evaluate, "f", arr_1d)
    assert_raises(ValueError, rnp.evaluate, "x*y", arr_1d)
    assert_raises(ValueError, rnp.evaluate, "x", arr_2d)
    assert_raises(ValueError, rnp.evaluate, "x*y", arr_3d)


def make_histogram(hist_type, shape, fill=True):
    # shape=([[z_bins,] y_bins,] x_bins)
    ndim = len(shape)
    hist_cls = getattr(ROOT, 'TH{0}{1}'.format(ndim, hist_type))
    if ndim == 1:
        hist = hist_cls(hist_cls.__name__, '',
                        shape[0], 0, 1)
        func = ROOT.TF1('func', 'x')
    elif ndim == 2:
        hist = hist_cls(hist_cls.__name__, '',
                        shape[1], 0, 1, shape[0], 0, 1)
        func = ROOT.TF2('func', 'x*y')
    elif ndim == 3:
        hist = hist_cls(hist_cls.__name__, '',
                        shape[2], 0, 1, shape[1], 0, 1, shape[0], 0, 1)
        func = ROOT.TF3('func', 'x*y*z')
    else:
        raise ValueError("ndim must be 1, 2, or 3")
    if fill:
        hist.FillRandom('func')
    return hist


def check_hist2array(hist, include_overflow, copy):
    array = rnp.hist2array(hist, include_overflow=include_overflow, copy=copy)
    assert_equal(hist.GetDimension(), array.ndim)
    for iaxis, axis in enumerate('XYZ'[:array.ndim]):
        if include_overflow:
            assert_equal(array.shape[iaxis],
                         getattr(hist, 'GetNbins{0}'.format(axis))() + 2)
        else:
            assert_equal(array.shape[iaxis],
                         getattr(hist, 'GetNbins{0}'.format(axis))())
    # non-zero elements
    assert_true(np.any(array))


def check_hist2array_THn(hist):
    hist_thn = ROOT.THn.CreateHn("", "", hist)
    array = rnp.hist2array(hist)
    array_thn = rnp.hist2array(hist_thn)
    # non-zero elements
    assert_true(np.any(array))
    # arrays should be identical
    assert_array_equal(array, array_thn)


def check_hist2array_THnSparse(hist):
    hist_thnsparse = ROOT.THnSparse.CreateSparse("", "", hist)
    array = rnp.hist2array(hist)
    array_thnsparse = rnp.hist2array(hist_thnsparse)
    # non-zero elements
    assert_true(np.any(array))
    # arrays should be identical
    assert_array_equal(array, array_thnsparse)


def test_hist2array():
    assert_raises(TypeError, rnp.hist2array, object())
    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            yield check_hist2array, hist, False, False
            yield check_hist2array, hist, False, True
            yield check_hist2array, hist, True, False
            yield check_hist2array, hist, True, True
            yield check_hist2array_THn, hist


def test_hist2array_THn():
    assert_raises(TypeError, rnp.hist2array, object())
    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            yield check_hist2array_THn, hist


def test_hist2array_THnSparse():
    assert_raises(TypeError, rnp.hist2array, object())
    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            yield check_hist2array_THnSparse, hist


def check_array2hist(hist):
    shape = np.array([hist.GetNbinsX(), hist.GetNbinsY(), hist.GetNbinsZ()])
    shape = shape[:hist.GetDimension()]
    arr = RNG.randint(0, 10, size=shape)
    rnp.array2hist(arr, hist)
    arr_hist = rnp.hist2array(hist)
    assert_array_equal(arr_hist, arr)

    shape_overflow = shape + 2
    arr_overflow = RNG.randint(0, 10, size=shape_overflow)
    hist_overflow = hist.Clone()
    hist_overflow.Reset()
    rnp.array2hist(arr_overflow, hist_overflow)
    arr_hist_overflow = rnp.hist2array(hist_overflow, include_overflow=True)
    assert_array_equal(arr_hist_overflow, arr_overflow)

    if len(shape) == 1:
        return

    # overflow not specified on all axes
    arr_overflow2 = arr_overflow[1:-1]
    hist_overflow2 = hist.Clone()
    hist_overflow2.Reset()
    rnp.array2hist(arr_overflow2, hist_overflow2)
    arr_hist_overflow2 = rnp.hist2array(hist_overflow2, include_overflow=True)
    assert_array_equal(arr_hist_overflow2[1:-1], arr_overflow2)


def test_array2hist():
    # wrong type
    assert_raises(TypeError, rnp.array2hist,
                  object(), ROOT.TH1D('test', '', 10, 0, 1))
    # wrong type
    assert_raises(TypeError, rnp.array2hist,
                  np.array([1, 2]), object())
    # dimensions don't match
    assert_raises(ValueError, rnp.array2hist,
                  np.arange(4).reshape(2, 2), ROOT.TH1D('test', '', 10, 0, 1))
    # shape not compatible
    assert_raises(ValueError, rnp.array2hist,
                  np.arange(4).reshape(2, 2),
                  ROOT.TH2D('test', '', 4, 0, 1, 3, 0, 1))

    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim, fill=False)
            yield check_array2hist, hist
