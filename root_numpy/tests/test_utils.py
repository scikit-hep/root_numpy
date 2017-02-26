import numpy as np
from numpy.lib import recfunctions
import root_numpy as rnp
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_equal, assert_raises
from . import load


def test_rec2array():
    # scalar fields
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

    # single scalar field
    arr = rnp.rec2array(a, fields=['x'])
    assert_array_equal(arr, np.array([[12345], [3]], dtype=np.int32))
    # single scalar field simplified
    arr = rnp.rec2array(a, fields='x')
    assert_array_equal(arr, np.array([12345, 3], dtype=np.int32))

    # case where array has single record
    assert_equal(rnp.rec2array(a[:1]).shape, (1, 4))
    assert_equal(rnp.rec2array(a[:1], fields=['x']).shape, (1, 1))
    assert_equal(rnp.rec2array(a[:1], fields='x').shape, (1,))

    # array fields
    a = np.array([
        ([1, 2, 3], [4.5, 6, 9.5],),
        ([4, 5, 6], [3.3, 7.5, 8.4],),],
        dtype=[
            ('x', np.int32, (3,)),
            ('y', np.float32, (3,))])

    arr = rnp.rec2array(a)
    assert_array_almost_equal(arr,
        np.array([[[1, 4.5],
                   [2, 6],
                   [3, 9.5]],
                  [[4, 3.3],
                   [5, 7.5],
                   [6, 8.4]]]))

    # single array field
    arr = rnp.rec2array(a, fields=['y'])
    assert_array_almost_equal(arr,
        np.array([[[4.5], [6], [9.5]],
                  [[3.3], [7.5], [8.4]]]))
    # single array field simplified
    arr = rnp.rec2array(a, fields='y')
    assert_array_almost_equal(arr,
        np.array([[4.5, 6, 9.5],
                  [3.3, 7.5, 8.4]]))

    # case where array has single record
    assert_equal(rnp.rec2array(a[:1], fields=['y']).shape, (1, 3, 1))
    assert_equal(rnp.rec2array(a[:1], fields='y').shape, (1, 3))

    # lengths mismatch
    a = np.array([
        ([1, 2], [4.5, 6, 9.5],),
        ([4, 5], [3.3, 7.5, 8.4],),],
        dtype=[
            ('x', np.int32, (2,)),
            ('y', np.float32, (3,))])
    assert_raises(ValueError, rnp.rec2array, a)

    # mix of scalar and array fields should fail
    a = np.array([
        (1, [4.5, 6, 9.5],),
        (4, [3.3, 7.5, 8.4],),],
        dtype=[
            ('x', np.int32),
            ('y', np.float32, (3,))])
    assert_raises(ValueError, rnp.rec2array, a)


def test_stack():
    rec = rnp.root2array(load('test.root'))
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

    # stretch single field and produce unstructured output
    stretched = rnp.stretch(arr, 'vl1')
    assert_equal(stretched.dtype, np.int)


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
    assert_array_equal(a2, exp2)
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
    assert_array_equal(a3, exp3)
    assert_equal(a3.dtype, exp3.dtype)
