import numpy as np
import numpy.lib.recfunctions as nprf
from _libinnerjoin import blockwise_inner_join
__all__ = [
    'stretch',
    'blockwise_inner_join'
]


def _is_array_field(arr, col):
    # For now:
    return arr.dtype[col] == 'O'


def stretch(arr, col_names, asrecarray=True):
    """
    Stretch an array. ``hstack()`` multiple array fields while preserving
    column names and record array structure. If a scalar field is specified,
    it will be stretched along with array field.

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
        raise RuntimeError('No array column specified. '
                           'What are you trying to do?')

    vl = np.vectorize(len)
    len_array = vl(arr[first_array])

    numrec = np.sum(len_array)

    ret = np.empty(numrec, dtype=dt)

    for c in col_names:
        if _is_array_field(arr, c):
            # FIXME: this is kinda stupid since it put the stack
            # some where and copy over to return value
            stack = np.hstack(arr[c])
            if len(stack) != numrec:
                raise RuntimeError(
                    'Array filed length doesn\'t match'
                    'Expect %d found %d in %s' %
                    (numrec, len(stack), c))
            ret[c] = stack
        else:
            # FIXME: this is kinda stupid since it put the repeat result
            # some where and copy over to return value
            ret[c] = np.repeat(arr[c], len_array)

    if asrecarray:
        ret = ret.view(np.recarray)

    return ret
