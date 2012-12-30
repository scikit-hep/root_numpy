__all__ = ['stretch']

import numpy as np
import numpy.lib.recfunctions as nprf
def stretch(arr, col_names, asrecarray=True):
    """
    Stretch array. hstack multiple array fileds and preserving
    column names and rec array structure.

    **Arguments***

        - **arr** numpy structured array or recarray
        - **colnames** list of column names to stretch
        - **asrecarray** optional boolean. If `True` return recarray,
          `False` returns structured array. Default `True`
    """
    #this can be implemented in a more memory efficient way but it works for now
    #TODO: support scalar field copying
    hs = np.hstack

    flat_list = [hs(arr[s]) for s in col_names]

    numrec = flat_list[0].size
    for f in flat_list:
        if f.size!=numrec:
            raise RuntimeError('the length of given arrays does not match.'
                ' expect: %d found %d in %r'%(numrec, f.size,f.dtype))

    dt = [(s, f.dtype) for s, f in zip(col_names, flat_list)]

    ret = np.empty(numrec, dtype=dt)

    for s,f in zip(col_names, flat_list):
        ret[s] = f

    if asrecarray:
        ret = ret.view(np.recarray)

    return ret
