import numpy as np
from _librootnumpy import blockwise_inner_join

__all__ = [
    'stretch',
    'blockwise_inner_join',
]

VLEN = np.vectorize(len)


def _is_array_field(arr, col):
    # For now:
    return arr.dtype[col] == 'O'


def stretch(arr, fields):
    """
    Stretch an array by ``hstack()``-ing  multiple array fields while
    preserving column names and record array structure. If a scalar field
    is specified, it will be stretched along with array fields.

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
    >>>
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
        if _is_array_field(arr, c):
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
        if _is_array_field(arr, c):
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
