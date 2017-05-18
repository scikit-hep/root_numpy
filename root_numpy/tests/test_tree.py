import ROOT
import root_numpy as rnp
from root_numpy.testdata import get_file
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_almost_equal)
import warnings
from . import load, RNG, temp

try:
    from collections import OrderedDict
except ImportError:  # pragma: no cover
    from root_numpy.extern.ordereddict import OrderedDict


def test_testdata():
    assert_raises(ValueError, get_file, 'file_does_not_exist.root')


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
    # Multiple key cycles of the same tree
    with temp() as rfile:
        tree = ROOT.TTree('tree', 'tree')
        rfile.Write()
        assert_equal(len(rnp.list_trees(rfile.GetName())), 1)
        rfile.Write()
        assert_equal(len(rnp.list_trees(rfile.GetName())), 1)
        rdir = rfile.mkdir('dir')
        rdir.cd()
        tree = ROOT.TTree('tree', 'tree')
        rfile.Write()
        assert_equal(set(rnp.list_trees(rfile.GetName())),
                     set(['tree', 'dir/tree']))


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


def test_single():
    f = load('single1.root')
    a = rnp.root2array(f)
    check_single(a)

    # specify tree name
    a = rnp.root2array(f, treename='tree')
    check_single(a)

    # tree2array
    f = get_file('single1.root')
    tree = f.Get('tree')
    check_single(rnp.tree2array(tree))


def test_chain():
    chain = ROOT.TChain('tree')
    chain.Add(load('single1.root'))
    check_single(rnp.tree2array(chain))

    f = load(['single1.root', 'single2.root'])
    a = rnp.root2array(f)
    check_single(a, 200)


def test_ntuple():
    f = load('ntuple.root')
    a = rnp.root2array(f)
    assert_equal(len(a), 10)
    assert_equal(len(a.dtype.names), 3)


@raises(IOError)
def test_root2array_single_pattern_DNE():
    f = load(['single1.root', 'does_not_exist.root'])
    a = rnp.root2array(f)


@raises(ValueError)
def test_root2array_no_filename():
    rnp.root2array([])


def test_root2array_no_trees_in_file():
    with temp() as tmp:
        tmp.Close()
        assert_raises(IOError, rnp.root2array, [tmp.GetName()], treename=None)


@raises(IOError)
def test_root2array_single_filename_DNE():
    f = load('does_not_exist.root')
    a = rnp.root2array(f)


@raises(ValueError)
def test_root2array_multiple_trees_and_name_not_specified():
    f = load('trees.root')
    a = rnp.root2array(f)


def test_empty_branches():
    f = load('single1.root')
    assert_raises(ValueError, rnp.root2array, f, branches=[])


def test_tree_without_branches():
    tree = ROOT.TTree('test', 'test')
    assert_raises(ValueError, rnp.tree2array, tree)


def test_empty_tree():
    from array import array
    tree = ROOT.TTree('tree', 'tree')
    d = array('d', [0.])
    tree.Branch('double', d, 'double/D')
    assert_equal(len(rnp.tree2array(tree)), 0)


def test_duplicate_branch_name():
    from array import array
    tree = ROOT.TTree('tree', 'tree')
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
    tree = ROOT.TTree('test', 'test')
    vect = ROOT.TLorentzVector()
    double = np.array([0], dtype=float)
    tree.Branch('vector', vect)
    tree.Branch('double', double, 'double/D')
    rnp.tree2array(tree)
    assert_raises(TypeError, rnp.tree2array, tree, branches=['vector'])


def test_no_supported_branches():
    tree = ROOT.TTree('test', 'test')
    vect = ROOT.TLorentzVector()
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
    a = rnp.root2array(f).view(np.recarray)

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

    # read only array without "length leaf"
    b = rnp.root2array(f, branches='n_int')
    for i in range(len(b)):
        assert_equal(len(b[i]), a.len_n[i])


def test_single_branch():
    f = get_file('single1.root')
    tree = f.Get('tree')
    arr1_1d = rnp.tree2array(tree, branches='n_int')
    arr2_1d = rnp.root2array(load('single1.root'), branches='n_int')
    assert_equal(arr1_1d.dtype, np.dtype('<i4'))
    assert_equal(arr2_1d.dtype, np.dtype('<i4'))


def test_selection():
    chain = ROOT.TChain('tree')
    chain.Add(load('single1.root'))
    chain.Add(load('single2.root'))
    a = rnp.tree2array(chain)
    assert_equal((a['d_double'] <= 100).any(), True)
    a = rnp.tree2array(chain, selection="d_double > 100")
    assert_equal((a['d_double'] <= 100).any(), False)

    # selection with differing variables in branches and expression
    a = rnp.tree2array(chain,
        branches=['d_double'],
        selection="f_float < 100 && n_int%2 == 1")

    # selection with TMath
    a = rnp.tree2array(chain,
        selection="TMath::Erf(d_double) < 0.5")


def test_expression():
    rec = rnp.root2array(load('single*.root'))
    rec2 = rnp.root2array(load('single*.root'), branches=['f_float*2'])
    assert_array_equal(rec['f_float'] * 2, rec2['f_float*2'])

    a = rnp.root2array(load('single*.root'), branches='Entry$')
    assert_equal(a.dtype, np.int32)
    assert_array_equal(a, np.arange(a.shape[0]))


def test_selection_and_expression():
    ref = len(rnp.root2array(
        load('test.root'), branches=['x', 'y'], selection='z>0'))
    assert_equal(ref,
        len(rnp.root2array(
            load('test.root'), branches=['x', 'y', 'z'], selection='z>0')))
    assert_equal(ref,
        len(rnp.root2array(
            load('test.root'), branches=['x', 'x*y'], selection='z>0')))
    assert_equal(ref,
        len(rnp.root2array(
            load('test.root'), branches=['x', 'x*z'], selection='z>0')))


def test_object_expression():
    rec = rnp.root2array(load(['object1.root', 'object2.root']),
                       branches=['vect.Pt()'])
    assert_array_equal(
        rec['vect.Pt()'],
        np.concatenate([
            np.arange(10, dtype='d') + 1,
            np.arange(10, dtype='d') + 2]))


def test_variable_length_array_expression():
    # variable length array
    a = rnp.root2array(load('vary*.root'), branches='n_int * 2')
    assert_equal(a.ndim, 1)
    assert_equal(a.dtype, 'O')


def test_fixed_length_array_expression():
    # fixed length array
    a = rnp.root2array(load('fixed*.root'), branches='n_int * 2')
    assert_equal(a.ndim, 2)
    assert_equal(a.shape[1], 5)
    assert_true(np.all(rnp.root2array(load('fixed*.root'), branches='Length$(n_int)') == 5))


def test_object_selection():
    a = rnp.root2array(load('vary*.root'), branches='n_int',
                       object_selection={'n_int % 2 == 0': 'n_int'})
    for suba in a:
        assert_true((suba % 2 == 0).all())

    # branch does not exist
    assert_raises(ValueError, rnp.root2array, load('vary*.root'),
                  branches='n_int', object_selection={'n_int % 2 == 0': 'DNE'})

    # duplicate branch in selection list
    assert_raises(ValueError, rnp.root2array, load('vary*.root'),
                  branches='n_int', object_selection={'n_int % 2 == 0': ['n_int', 'n_int']})

    # test object selection on variable-length expression
    a = rnp.root2array(load('object*.root'), branches='lines.GetX1()',
                       object_selection={'lines.GetX1() > 3': 'lines.GetX1()'})

    for suba in a:
        assert_true((suba > 3).all())

    # attempting to apply object selection on fixed-length array
    # currently not implemented since this changes the output type from
    # fixed-length to variable-length
    assert_raises(TypeError, rnp.root2array, load("fixed*.root"),
                  branches='n_int',
                  object_selection={'n_int % 2 == 0': 'n_int'})

    # test with vectors
    a = rnp.root2array(load('vector.root'), branches='v_i',
                       object_selection={'v_i % 2 == 0': 'v_i'})

    for suba in a:
        assert_true((suba % 2 == 0).all())


@raises(ValueError)
def test_branch_DNE():
    chain = ROOT.TChain('tree')
    chain.Add(load('single1.root'))
    rnp.tree2array(chain, branches=['my_net_worth'])


@raises(TypeError)
def test_tree2array_wrong_type():
    rnp.tree2array(list())


def test_specific_branch():
    a = rnp.root2array(load('single1.root'), branches=['f_float'])
    assert_equal(a.dtype, [('f_float', '<f4')])


def test_vector():
    a = rnp.root2array(load('vector.root')).view(np.recarray)
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
    a = rnp.root2array(load('string.root'))
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
    a = rnp.root2array(load('single1.root'), stop=10).view(np.recarray)
    assert_equal(len(a), 10)
    assert_equal(a.n_int[-1], 10)

    a = rnp.root2array(load('single1.root'), stop=11, start=1).view(np.recarray)
    assert_equal(len(a), 10)
    assert_equal(a.n_int[-1], 11)

    a = rnp.root2array(load('single1.root'), stop=105, start=95).view(np.recarray)
    assert_equal(len(a), 5)
    assert_equal(a.n_int[-1], 100)


def test_weights():
    f = ROOT.TFile(load('test.root'))
    tree = f.Get('tree')
    tree.SetWeight(5.)
    rec = rnp.tree2array(tree, include_weight=True, weight_name='treeweight')
    assert_array_equal(rec['treeweight'], np.ones(100) * 5)
    f = load(['single1.root', 'single2.root'])
    a = rnp.root2array(f, include_weight=True)
    assert_array_equal(
        a['weight'],
        np.concatenate((np.ones(100) * 2., np.ones(100) * 3.)))


def test_struct():
    assert_array_equal(rnp.root2array(load('struct.root')),
        np.array([(10, 15.5, 20, 781.2)],
            dtype=[
                ('branch1_intleaf', '<i4'),
                ('branch1_floatleaf', '<f4'),
                ('branch2_intleaf', '<i4'),
                ('branch2_floatleaf', '<f4')]))


def check_truncate_impute(filename):
    filename = load(filename)
    # first convert array and find object columns
    arr = rnp.root2array(filename)
    assert_true(len(arr))
    object_fields = [field for field in arr.dtype.names if arr.dtype[field] == 'O']
    fields_1d = [field for field in object_fields
                 if arr[field][0].dtype != 'O' and len(arr[field][0].shape) == 1]
    fields_md = list(set(object_fields) - set(fields_1d))
    assert_true(fields_1d)
    assert_true(fields_md)
    fields_1d.sort()
    fields_md.sort()

    rfile = ROOT.TFile.Open(filename)
    tree = rfile.Get(rnp.list_trees(filename)[0])

    # test both root2array and tree2array
    for func, arg in [(rnp.root2array, filename), (rnp.tree2array, tree)]:

        arr1 = func(arg, branches=[(f, 0) for f in fields_1d])
        assert_true(len(arr1))
        assert_equal(set(arr1.dtype.names), set(fields_1d))
        # Giving length of 1 will result in the same output
        arr2 = func(arg, branches=[(f, 0, 1) for f in fields_1d])
        assert_array_equal(arr1, arr2)
        # fill_value of 1 instead of 0 should change output array
        arr2 = func(arg, branches=[(f, 1, 1) for f in fields_1d])
        assert_raises(AssertionError, assert_array_equal, arr1, arr2)
        # check dtype shape
        arr3 = func(arg, branches=[(f, 0, 3) for f in fields_1d])
        for field in fields_1d:
            assert_equal(arr3.dtype[field].shape, (3,))

        # length must be at least 1
        assert_raises(ValueError, func, arg, branches=[(fields_1d[0], 0, 0)])
        # tuple is not of length 2 or 3
        assert_raises(ValueError, func, arg, branches=[(fields_1d[0], 1, 1, 1)])
        assert_raises(ValueError, func, arg, branches=(fields_1d[0], 1, 1, 1))
        # can only truncate 1d arrays
        assert_raises(TypeError, func, arg, branches=(fields_md[0], 0))

        # expressions
        arr1 = func(arg, branches='{0}==0'.format(fields_1d[0]))
        assert_equal(arr1.dtype, 'O')
        arr2 = func(arg, branches=('{0}==0'.format(fields_1d[0]), 0))
        assert_equal(arr2.dtype, arr1[0].dtype)


def test_truncate_impute():
    for filename in ['vector.root', 'vary1.root']:
        yield check_truncate_impute, filename


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
