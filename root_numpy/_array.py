import numpy as np
from . import _librootnumpy


__all__ = [
    'array',
]


def array(arr, copy=True):
    """Convert a ROOT TArray into a NumPy array.

    Parameters
    ----------
    arr : ROOT TArray
        A ROOT TArrayD, TArrayF, TArrayL, TArrayI or TArrayS
    copy : bool, optional (default=True)
        If True (the default) then copy the underlying array, otherwise the
        NumPy array will view (and not own) the same memory as the ROOT array.

    Returns
    -------
    arr : NumPy array
        A NumPy array

    Examples
    --------
    >>> from root_numpy import array
    >>> from ROOT import TArrayD
    >>> a = TArrayD(5)
    >>> a[3] = 3.141
    >>> array(a)
    array([ 0.   ,  0.   ,  0.   ,  3.141,  0.   ])

    """
    import ROOT
    if isinstance(arr, ROOT.TArrayD):
        arr = _librootnumpy.array_d(ROOT.AsCObject(arr))
    elif isinstance(arr, ROOT.TArrayF):
        arr = _librootnumpy.array_f(ROOT.AsCObject(arr))
    elif isinstance(arr, ROOT.TArrayL):
        arr = _librootnumpy.array_l(ROOT.AsCObject(arr))
    elif isinstance(arr, ROOT.TArrayI):
        arr = _librootnumpy.array_i(ROOT.AsCObject(arr))
    elif isinstance(arr, ROOT.TArrayS):
        arr = _librootnumpy.array_s(ROOT.AsCObject(arr))
    elif isinstance(arr, ROOT.TArrayC):
        arr = _librootnumpy.array_c(ROOT.AsCObject(arr))
    else:
        raise TypeError(
            "unable to convert object of type {0} "
            "into a numpy array".format(type(arr)))
    if copy:
        return np.copy(arr)
    return arr
