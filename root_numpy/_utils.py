import numpy as np
import operator

from .extern.six import string_types
from ._librootnumpy import _blockwise_inner_join


__all__ = [
    'rec2array',
    'stack',
    'stretch',
    'dup_idx',
    'blockwise_inner_join',
]


VLEN = np.vectorize(len)


def rec2array(rec, fields=None):
    """Convert a record/structured array into an ndarray with a homogeneous data type.

    Parameters
    ----------
    rec : NumPy record/structured array
        A NumPy structured array that will be cast into a homogenous data type.
    fields : list of strings, optional (default=None)
        The fields to include as columns in the output array. If None, then all
        columns will be included. All fields must have the same shape.

    Returns
    -------
    array : NumPy ndarray
        A new NumPy ndarray with homogeneous data types for all columns.

    Notes
    -----
    If the fields are scalars the shape of the output ``array`` will be
    ``(len(rec), num_fields)``. If the fields are arrays of length
    ``num_things`` the shape of the output ``array`` will be ``(len(rec),
    num_things, num_fields)``.

    Examples
    --------

    >>> from root_numpy import rec2array
    >>> import numpy as np
    >>> a = np.array([
    ...         (12345, 2., 2.1, True),
    ...         (3, 4., 4.2, False),],
    ...         dtype=[
    ...             ('x', np.int32),
    ...             ('y', np.float32),
    ...             ('z', np.float64),
    ...             ('w', np.bool)])
    >>> arr = rec2array(a)
    >>> arr
    array([[  1.23450000e+04,   2.00000000e+00,   2.10000000e+00,
              1.00000000e+00],
           [  3.00000000e+00,   4.00000000e+00,   4.20000000e+00,
              0.00000000e+00]])
    >>> arr.dtype
    dtype('float64')
    >>>
    >>> a = np.array([
    ...         ([1, 2, 3], [4.5, 6, 9.5],),
    ...         ([4, 5, 6], [3.3, 7.5, 8.4],),],
    ...         dtype=[
    ...             ('x', np.int32, (3,)),
    ...             ('y', np.float32, (3,))])
    >>> arr = rec2array(a)
    >>> arr
    array([[[ 1.        ,  4.5       ],
            [ 2.        ,  6.        ],
            [ 3.        ,  9.5       ]],
    <BLANKLINE>
           [[ 4.        ,  3.29999995],
            [ 5.        ,  7.5       ],
            [ 6.        ,  8.39999962]]])
    >>> arr.shape
    (2, 3, 2)

    """
    if fields is None:
        fields = rec.dtype.names
    if len(fields) == 1:
        return rec[fields[0]]
    # Creates a copy and casts all data to the same type
    return np.squeeze(np.dstack([rec[field] for field in fields]))


def stack(recs, fields=None):
    """Stack common fields in multiple record arrays (concatenate them).

    Parameters
    ----------
    recs : list
        List of NumPy record arrays
    fields : list of strings, optional (default=None)
        The list of fields to include in the stacked array. If None, then
        include the fields in common to all the record arrays.

    Returns
    -------
    rec : NumPy record array
        The stacked array.

    """
    if fields is None:
        fields = list(set.intersection(
            *[set(rec.dtype.names) for rec in recs]))
        # preserve order of fields wrt first record array
        if set(fields) == set(recs[0].dtype.names):
            fields = list(recs[0].dtype.names)
    return np.hstack([rec[fields] for rec in recs])


def stretch(arr, fields=None, return_indices=False):
    """Stretch an array.

    Stretch an array by ``hstack()``-ing  multiple array fields while
    preserving column names and record array structure. If a scalar field is
    specified, it will be stretched along with array fields.

    Parameters
    ----------
    arr : NumPy structured or record array
        The array to be stretched.
    fields : list of strings, optional (default=None)
        A list of column names to stretch. If None, then stretch all fields.
    return_indices : bool, optional (default=False)
        If True, the array index of each stretched array entry will be
        returned in addition to the stretched array.
        This changes the return type of this function to a tuple consisting
        of a structured array and a numpy int64 array.

    Returns
    -------
    ret : A NumPy structured array
        The stretched array.

    Examples
    --------
    >>> import numpy as np
    >>> from root_numpy import stretch
    >>> arr = np.empty(2, dtype=[('scalar', np.int), ('array', 'O')])
    >>> arr[0] = (0, np.array([1, 2, 3], dtype=np.float))
    >>> arr[1] = (1, np.array([4, 5, 6], dtype=np.float))
    >>> stretch(arr, ['scalar', 'array'])
    array([(0, 1.0), (0, 2.0), (0, 3.0), (1, 4.0), (1, 5.0), (1, 6.0)],
        dtype=[('scalar', '<i8'), ('array', '<f8')])

    """
    dtype = []
    len_array = None

    if fields is None:
        fields = arr.dtype.names

    # Construct dtype and check consistency
    for field in fields:
        dt = arr.dtype[field]
        if dt == 'O' or len(dt.shape):
            if dt == 'O':
                # Variable-length array field
                lengths = VLEN(arr[field])
            else:
                lengths = np.repeat(dt.shape[0], arr.shape[0])
            # Fixed-length array field
            if len_array is None:
                len_array = lengths
            elif not np.array_equal(lengths, len_array):
                raise ValueError(
                    "inconsistent lengths of array columns in input")
            if dt == 'O':
                dtype.append((field, arr[field][0].dtype))
            else:
                dtype.append((field, arr[field].dtype, dt.shape[1:]))
        else:
            # Scalar field
            dtype.append((field, dt))

    if len_array is None:
        raise RuntimeError("no array column in input")

    # Build stretched output
    ret = np.empty(np.sum(len_array), dtype=dtype)
    for field in fields:
        dt = arr.dtype[field]
        if dt == 'O' or len(dt.shape) == 1:
            # Variable-length or 1D fixed-length array field
            ret[field] = np.hstack(arr[field])
        elif len(dt.shape):
            # Multidimensional fixed-length array field
            ret[field] = np.vstack(arr[field])
        else:
            # Scalar field
            ret[field] = np.repeat(arr[field], len_array)

    if return_indices:
        idx = np.concatenate(list(map(np.arange, len_array)))
        return ret, idx

    return ret


def dup_idx(arr):
    """Return the indices of all duplicated array elements.

    Parameters
    ----------
    arr : array-like object
        An array-like object

    Returns
    -------
    idx : NumPy array
        An array containing the indices of the duplicated elements

    Examples
    --------
    >>> from root_numpy import dup_idx
    >>> dup_idx([1, 2, 3, 4, 5])
    array([], dtype=int64)
    >>> dup_idx([1, 2, 3, 4, 5, 5])
    array([4, 5])
    >>> dup_idx([1, 2, 3, 4, 5, 5, 1])
    array([0, 4, 5, 6])

    """
    _, b = np.unique(arr, return_inverse=True)
    return np.nonzero(np.logical_or.reduce(
        b[:, np.newaxis] == np.nonzero(np.bincount(b) > 1),
        axis=1))[0]


def blockwise_inner_join(data, left, foreign_key, right,
                         force_repeat=None,
                         foreign_key_name=None):
    """Perform a blockwise inner join.

    Perform a blockwise inner join from names specified in ``left`` to
    ``right`` via ``foreign_key``: left->foreign_key->right.

    Parameters
    ----------
    data : array
        A structured NumPy array.
    left : array
        Array of left side column names.
    foreign_key : array or string
        NumPy array or string ``foreign_key`` column name. This column can be
        either an integer or an array of ints. If ``foreign_key`` is an array
        of int column, left column will be treated according to left column
        type:

        * Scalar columns or columns in ``force_repeat`` will be repeated
        * Array columns not in ``force_repeat`` will be assumed to the
          same length as ``foreign_key`` and will be stretched by index
    right : array
        Array of right side column names. These are array columns that each
        index ``foreign_key`` points to. These columns are assumed to have the
        same length.
    force_repeat : array, optional (default=None)
        Array of left column names that will be forced to stretch even if it's
        an array (useful when you want to emulate a multiple join).
    foreign_key_name : str, optional (default=None)
        The name of foreign key column in the output array.

    Examples
    --------
    >>> import numpy as np
    >>> from root_numpy import blockwise_inner_join
    >>> test_data = np.array([
    (1.0, np.array([11, 12, 13]), np.array([1, 0, 1]), 0, np.array([1, 2, 3])),
    (2.0, np.array([21, 22, 23]), np.array([-1, 2, -1]), 1, np.array([31, 32, 33]))],
    dtype=[('sl', np.float), ('al', 'O'), ('fk', 'O'), ('s_fk', np.int), ('ar', 'O')])

    >>> blockwise_inner_join(test_data, ['sl', 'al'], test_data['fk'], ['ar'])
    array([(1.0, 11, 2, 1), (1.0, 12, 1, 0), (1.0, 13, 2, 1), (2.0, 22, 33, 2)],
    dtype=[('sl', '<f8'), ('al', '<i8'), ('ar', '<i8'), ('fk', '<i8')])

    >>> blockwise_inner_join(test_data, ['sl', 'al'], test_data['fk'], ['ar'], force_repeat=['al'])
    array([(1.0, [11, 12, 13], 2, 1), (1.0, [11, 12, 13], 1, 0),
    (1.0, [11, 12, 13], 2, 1), (2.0, [21, 22, 23], 33, 2)],
    dtype=[('sl', '<f8'), ('al', '|O8'), ('ar', '<i8'), ('fk', '<i8')])

    """
    if isinstance(foreign_key, string_types):
        foreign_key = data[foreign_key]
    return _blockwise_inner_join(data, left, foreign_key, right,
                                 force_repeat, foreign_key_name)
