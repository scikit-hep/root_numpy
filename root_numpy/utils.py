import numpy as np
import numpy.lib.recfunctions as nprf
from _librootnumpy import blockwise_inner_join

__all__ = [
    'stretch',
    'blockwise_inner_join',
]


def _is_array_field(arr, col):
    # For now:
    return arr.dtype[col] == 'O'


def stretch(arr, col_names, asrecarray=True):
    """
    Stretch an array. ``hstack()`` multiple array fields while preserving
    column names and record array structure. If a scalar field is specified,
    it will be stretched along with array fields.

    Parameters
    ----------
    arr : NumPy structured or record array

    colnames : list of column names to stretch

    asrecarray : bool, optional (default=True)
        If `True`, return a record array, else return a structured array.

    """
    dt = []
    has_array_field = False
    has_scalar_filed = False
    first_array = None

    # Construct dtype
    for c in col_names:
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

    vl = np.vectorize(len)
    len_array = vl(arr[first_array])

    numrec = np.sum(len_array)

    ret = np.empty(numrec, dtype=dt)

    for c in col_names:
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

    if asrecarray:
        ret = ret.view(np.recarray)

    return ret
