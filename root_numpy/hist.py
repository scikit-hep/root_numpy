"""Efficient functions for working with ROOT histograms and numpy arrays""" 
import numpy as np
import ROOT


def _get_hist_dtype(hist):
    """Determine the hist_dtype for a ROOT hist"""
    for dim in [1, 2, 3]:
        for hist_dtype in 'C S I F D'.split():
            if eval('isinstance(hist, ROOT.TH{0}{1})'
                    ''.format(dim, hist_dtype)):
                return hist_dtype

def _get_hist_ndim(hist):
    """Determine the number of dimensions for a ROOT hist"""
    # Note that the order is important here, because
    # ROOT.TH3 is a ROOT.TH1
    if isinstance(hist, ROOT.TH3):
        return 3
    elif isinstance(hist, ROOT.TH2):
        return 2
    elif isinstance(hist, ROOT.TH1):
        return 1
    else:
        raise TypeError("Can't process type: {0}".format(type(hist)))
    

def _get_numpy_dtype(hist):
    """Determine the corresponding numpy dtype for a given ROOT hist"""
    hist_dtype = _get_hist_dtype(hist)
    d = dict(C='u1', S='i2', I='i4', F='f4', D='f8')
    return np.dtype(d[hist_dtype])



def hist2array(hist, include_overflow=False):
    """Convert a ROOT histogram into a numpy array

    Parameters
    -----------
    hist : `~ROOT.TH` histogram (1-, 2- or 3-dimensional)

    include_overflow : bool, optional
        If True, the over- and underflow bins will be included in the
        output numpy array

    Returns
    -------
    array : `~numpy.ndarray`
        Numpy array containing the histogram bin values
    """

    if isinstance(hist, ROOT.TH3):
        shape = (hist.GetNbinsZ(), hist.GetNbinsY(), hist.GetNbinsX())
    elif isinstance(hist, ROOT.TH2):
        shape = (hist.GetNbinsY(), hist.GetNbinsX())
    elif isinstance(hist, ROOT.TH1):
        shape = (hist.GetNbinsX(),)
    else:
        raise TypeError("Can't process type: %s" % type(hist))

    # ROOT.TH array always includes the over- and underflow bins
    shape = [_ + 2 for _ in shape]
    dtype = _get_numpy_dtype(hist)
    array = np.ndarray(shape=shape, dtype=dtype, buffer=hist.GetArray())

    if not include_overflow:
        # remove overflow and underflow bins
        if array.ndim == 1:
            array = array[1:-1]
        elif array.ndim == 2:
            array = array[1:-1, 1:-1]
        elif array.ndim == 3:
            array = array[1:-1, 1:-1, 1:-1]
            
    return array


def array2hist(array, has_overflow=False):
    """Convert a numpy array into a ROOT histogram

    Parameters
    -----------
    array : `~numpy.ndarray`
        Numpy array (1-, 2- or 3-dimensional)

    include_overflow : bool, optional
        If True, the input numpy array is assumed to contain
        over- and underflow bins.

    Returns
    -------
    hist : `~ROOT.TH` histogram
    """

    raise NotImplementedError
    """
    array = np.asarray(array)
    import ctypes
    NULL_DOUBLE_P = ctypes.POINTER(ctypes.c_double)()
    memmove()
    """


def fill_hist_from_array(hist, array, weights=None):
    """Fill a ROOT histogram from a numpy array

    Parameters
    -----------
    hist : `~ROOT.TH` histogram (1-, 2- or 3-dimensional)

    array : `~numpy.ndarray`
        2-dimensional numpy array
        (number of columns must match the hist dimensionality)

    weights : `~numpy.ndarray`
        Numpy 1-dimensional array (same number of rows as `array`)

    Notes
    -----
    
    If you have (x, y) separately, ypu can use array = np.vstack([x, y])
    when calling this function.
    """
    ndim = _get_hist_ndim(hist)

    array = np.asarray(array, dtype='float64')

    # check that array number of columns matches the hist dimensionality
    if ndim == 3:
        if array.ndim != 2:
            raise ValueError('Hist is 3D. Array must be 2D, but is {0}D.'.format(array.ndim))
        if array.shape[1] != 3:
            raise ValueError('Hist is 3D. Array must have 3 columns, but has {0}'.format(array.shape[1]))
    elif ndim == 2:
        if array.ndim != 2:
            raise ValueError('Hist is 2D. Array must be 2D, but is {0}D.'.format(array.ndim))
        if array.shape[1] != 2:
            raise ValueError('Hist is 2D. Array must have 2 columns, but has {0}'.format(array.shape[1]))
    elif ndim == 1:
        if array.ndim != 1:
            raise ValueError('Hist is 1D. Array must be 1D, but is {0}D.'.format(array.ndim))
    else:
        raise TypeError("Can't process type: {0}".format(type(hist)))

    ntimes = array.shape[0]

    # Check that weights are OK
    if weights is not None:
        weights = np.asarray(weights, dtype='float64')
        if weights.shape !=  ntimes:
            raise ValueError("array shape %s and weights shape %s do not match." %
                             (ntimes, weights.shape))
    else:
        weights = np.ones(ntimes, dtype='float64')

    # Fill the hist; hist.FillN takes x, y, z
    if ndim == 1:
        hist.FillN(ntimes, array, weights)
    elif ndim == 2:
        x, y = array[:,0].copy(), array[:,1].copy()
        # C++ method signature:
        # void TH2::FillN(Int_t, const Double_t*, const Double_t*, Int_t)
        # void TH2::FillN(Int_t ntimes, const Double_t* x, const Double_t* y, const Double_t* w, Int_t stride = 1)
        hist.FillN(ntimes, x, y, weights)
    elif ndim == 3:
        x, y, z = array[:,0].copy(), array[:,1].copy(), array[:,2].copy()
        # import IPython; IPython.embed(); 1/0
        raise NotImplementedError
        # This doesn't work because there is no TH3::FillN
        # C++ method signature:
        # void TH1::FillN(Int_t ntimes, const Double_t* x, const Double_t* w, Int_t stride = 1)
        # void TH1::FillN(Int_t, const Double_t*, const Double_t*, const Double_t*, Int_t)
        hist.FillN(ntimes, x, y, z, weights)
    else:
        raise ValueError("Can't handle histograms of dimension {0}.".format(ntimes))
