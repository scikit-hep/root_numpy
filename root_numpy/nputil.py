__all__ = ['stretch']

import numpy as np
import numpy.lib.recfunctions as nprf

def _is_array_field(arr,col):
    return arr.dtype[col]=='O'#for now

def stretch(arr, col_names, asrecarray=True):
    """
    Stretch array. hstack multiple array fields and preserving
    column names and rec array structure. If scalar field is specified,
    it's stretched along with array field.

    **Arguments***

        - **arr** numpy structured array or recarray
        - **colnames** list of column names to stretch
        - **asrecarray** optional boolean. If `True` return recarray,
          `False` returns structured array. Default `True`
    """
    dt = []
    has_array_field = False
    has_scalar_filed = False
    first_array = None

    #construct dtype
    for c in col_names:
        if _is_array_field(arr,c):
            dt.append((c, arr[c][0].dtype))
            has_array_field = True
            first_array = c if first_array is None else first_array
        else:#assume scalar
            dt.append((c, arr[c].dtype))
            has_scalar_filed = True

    if not has_array_field:
        raise RuntimeError('No array column specified.'
                           ' What are you trying to do?')

    vl = np.vectorize(len)
    len_array = vl(arr[first_array])

    numrec = np.sum(len_array)

    ret = np.empty(numrec, dtype=dt)

    for c in col_names:
        if _is_array_field(arr,c):
            #FIXME: this is kinda stupid since it put the stack
            #some where and copy over to return value
            stack = np.hstack(arr[c])
            if len(stack)!= numrec:
                raise RuntimeError('Array filed length doesn\'t match'
                    'Expect %d found %d in %s'%(numrec, len(stack), c))
            ret[c] = stack
        else:
            #FIXME: this is kinda stupid since it put the repeat result
            #some where and copy over to return value
            ret[c] = np.repeat(arr[c],len_array)

    if asrecarray:
        ret = ret.view(np.recarray)

    return ret
