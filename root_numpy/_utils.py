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


def _is_object_field(arr, col):
    return arr.dtype[col] == 'O'


def rec2array(rec, fields=None):
    """Convert a record array into a ndarray with a homogeneous data type.

    Parameters
    ----------
    rec : NumPy record/structured array
        A NumPy structured array that will be cast into a homogenous data type.
    fields : list of strings, optional (default=None)
        The fields to include as columns in the output array. If None, then all
        columns will be included.

    Returns
    -------
    array : NumPy ndarray
        A new NumPy ndarray with homogeneous data types for all columns.

    """
    if fields is None:
        fields = rec.dtype.names
    if len(fields) == 1:
        return rec[fields[0]]
    # Creates a copy and recasts data to a consistent datatype
    return np.vstack([rec[field] for field in fields]).T


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


def stretch(arr, fields):
    """Stretch an array.

    Stretch an array by ``hstack()``-ing  multiple array fields while
    preserving column names and record array structure. If a scalar field is
    specified, it will be stretched along with array fields.

    Parameters
    ----------
    arr : NumPy structured or record array
        The array to be stretched.
    fields : list of strings
        A list of column names to stretch.

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
    dt = []
    has_array_field = False
    has_scalar_filed = False
    first_array = None

    # Construct dtype
    for c in fields:
        if _is_object_field(arr, c):
            dt.append((c, arr[c][0].dtype))
            has_array_field = True
            first_array = c if first_array is None else first_array
        else:
            # Assume scalar
            dt.append((c, arr[c].dtype))
            has_scalar_filed = True

    if not has_array_field:
        raise RuntimeError("No array column specified")

    len_array = VLEN(arr[first_array])
    numrec = np.sum(len_array)
    ret = np.empty(numrec, dtype=dt)

    for c in fields:
        if _is_object_field(arr, c):
            # FIXME: this is rather inefficient since the stack
            # is copied over to the return value
            stack = np.hstack(arr[c])
            if len(stack) != numrec:
                raise ValueError(
                    "Array lengths do not match: "
                    "expected %d but found %d in %s" %
                    (numrec, len(stack), c))
            ret[c] = stack
        else:
            # FIXME: this is rather inefficient since the repeat result
            # is copied over to the return value
            ret[c] = np.repeat(arr[c], len_array)

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
