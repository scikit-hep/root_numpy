import numpy as np
from . import _librootnumpy


__all__ = [
    'fill_hist',
    'fill_profile',
    'hist2array',
    'array2hist',
]

DTYPE_ROOT2NUMPY = dict(C='i1', S='i2', I='i4', L='i8', F='f4', D='f8')
ARRAY_NUMPY2ROOT = dict(
    [(ndim, dict([
        (hist_type,
            getattr(_librootnumpy, 'h{0}{1}_array'.format(
                ndim, hist_type.lower())))
        for hist_type in 'DFISC']))
        for ndim in (1, 2, 3)])


def fill_hist(hist, array, weights=None, return_indices=False):
    """Fill a ROOT histogram with a NumPy array.

    Parameters
    ----------
    hist : ROOT TH1, TH2, or TH3
        The ROOT histogram to fill.
    array : numpy array of shape [n_samples, n_dimensions]
        The values to fill the histogram with. The number of columns must match
        the dimensionality of the histogram. Supply a flat numpy array when
        filling a 1D histogram.
    weights : numpy array
        A flat numpy array of weights for each sample in ``array``.
    return_indices : bool, optional (default=False)
        If True then return an array of the bin indices filled for each element
        in ``array``.

    Returns
    -------
    indices : numpy array or None
        If ``return_indices`` is True, then return an array of the bin indices
        filled for each element in ``array`` otherwise return None.

    """
    import ROOT
    array = np.asarray(array, dtype=np.double)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.double)
        if weights.shape[0] != array.shape[0]:
            raise ValueError("array and weights must have the same length")
        if weights.ndim != 1:
            raise ValueError("weight must be 1-dimensional")
    if isinstance(hist, ROOT.TH3):
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != 3:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the histogram")
        return _librootnumpy.fill_h3(
            ROOT.AsCObject(hist), array, weights, return_indices)
    elif isinstance(hist, ROOT.TH2):
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != 2:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the histogram")
        return _librootnumpy.fill_h2(
            ROOT.AsCObject(hist), array, weights, return_indices)
    elif isinstance(hist, ROOT.TH1):
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.fill_h1(
            ROOT.AsCObject(hist), array, weights, return_indices)
    raise TypeError(
        "hist must be an instance of ROOT.TH1, ROOT.TH2, or ROOT.TH3")


def fill_profile(profile, array, weights=None, return_indices=False):
    """Fill a ROOT profile with a NumPy array.

    Parameters
    ----------
    profile : ROOT TProfile, TProfile2D, or TProfile3D
        The ROOT profile to fill.
    array : numpy array of shape [n_samples, n_dimensions]
        The values to fill the histogram with. There must be one more column
        than the dimensionality of the profile.
    weights : numpy array
        A flat numpy array of weights for each sample in ``array``.
    return_indices : bool, optional (default=False)
        If True then return an array of the bin indices filled for each element
        in ``array``.

    Returns
    -------
    indices : numpy array or None
        If ``return_indices`` is True, then return an array of the bin indices
        filled for each element in ``array`` otherwise return None.

    """
    import ROOT
    array = np.asarray(array, dtype=np.double)
    if array.ndim != 2:
        raise ValueError("array must be 2-dimensional")
    if array.shape[1] != profile.GetDimension() + 1:
        raise ValueError(
            "there must be one more column than the "
            "dimensionality of the profile")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.double)
        if weights.shape[0] != array.shape[0]:
            raise ValueError("array and weights must have the same length")
        if weights.ndim != 1:
            raise ValueError("weight must be 1-dimensional")
    if isinstance(profile, ROOT.TProfile3D):
        return _librootnumpy.fill_p3(
            ROOT.AsCObject(profile), array, weights, return_indices)
    elif isinstance(profile, ROOT.TProfile2D):
        return _librootnumpy.fill_p2(
            ROOT.AsCObject(profile), array, weights, return_indices)
    elif isinstance(profile, ROOT.TProfile):
        return _librootnumpy.fill_p1(
            ROOT.AsCObject(profile), array, weights, return_indices)
    raise TypeError(
        "profile must be an instance of "
        "ROOT.TProfile, ROOT.TProfile2D, or ROOT.TProfile3D")


def hist2array(hist, include_overflow=False, copy=True):
    """Convert a ROOT histogram into a NumPy array

    Parameters
    ----------
    hist : ROOT TH1, TH2, TH3, THn, or THnSparse
        The ROOT histogram to convert into an array
    include_overflow : bool, optional (default=False)
        If True, the over- and underflow bins will be included in the
        output numpy array. These bins are excluded by default.
    copy : bool, optional (default=True)
        If True (the default) then copy the underlying array, otherwise the
        NumPy array will view (and not own) the same memory as the ROOT
        histogram's array.

    Returns
    -------
    array : numpy array
        A NumPy array containing the histogram bin values

    Raises
    ------
    TypeError
        If hist is not a ROOT histogram.

    See Also
    --------
    array2hist

    """
    import ROOT
    # Determine dimensionality and shape
    simple_hist = True
    if isinstance(hist, ROOT.TH3):
        shape = (hist.GetNbinsZ() + 2,
                 hist.GetNbinsY() + 2,
                 hist.GetNbinsX() + 2)
    elif isinstance(hist, ROOT.TH2):
        shape = (hist.GetNbinsY() + 2, hist.GetNbinsX() + 2)
    elif isinstance(hist, ROOT.TH1):
        shape = (hist.GetNbinsX() + 2,)
    elif isinstance(hist, ROOT.THnBase):
        shape = tuple([hist.GetAxis(i).GetNbins() + 2
                       for i in range(hist.GetNdimensions())])
        simple_hist = False
    else:
        raise TypeError(
            "hist must be an instance of ROOT.TH1, "
            "ROOT.TH2, ROOT.TH3, or ROOT.THnBase")

    # Determine the corresponding numpy dtype
    if simple_hist:
        for hist_type in 'DFISC':
            if isinstance(hist, getattr(ROOT, 'TArray{0}'.format(hist_type))):
                break
        else:
            raise AssertionError(
                "hist is somehow an instance of TH[1|2|3] "
                "but not TArray[D|F|I|S|C]")
    else:  # THn, THnSparse
        if isinstance(hist, ROOT.THnSparse):
            cls_string = 'THnSparse{0}'
        else:
            cls_string = 'THn{0}'
        for hist_type in 'CSILFD':
            if isinstance(hist, getattr(ROOT, cls_string.format(hist_type))):
                break
        else:
            raise AssertionError(
                "unsupported THn or THnSparse bin type")

    if simple_hist:
        # Constuct a NumPy array viewing the underlying histogram array
        if hist_type == 'C':
            array_func = getattr(_librootnumpy,
                                 'array_h{0}c'.format(len(shape)))
            array = array_func(ROOT.AsCObject(hist))
            array.shape = shape
        else:
            dtype = np.dtype(DTYPE_ROOT2NUMPY[hist_type])
            array = np.ndarray(shape=shape, dtype=dtype,
                               buffer=hist.GetArray())
    else:  # THn THnSparse
        dtype = np.dtype(DTYPE_ROOT2NUMPY[hist_type])
        if isinstance(hist, ROOT.THnSparse):
            array = _librootnumpy.thnsparse2array(ROOT.AsCObject(hist),
                                                  shape, dtype)
        else:
            array = _librootnumpy.thn2array(ROOT.AsCObject(hist),
                                            shape, dtype)

    if not include_overflow:
        # Remove overflow and underflow bins
        array = array[tuple([slice(1, -1) for idim in range(array.ndim)])]

    if simple_hist:
        # Preserve x, y, z -> axis 0, 1, 2 order
        array = np.transpose(array)
        if copy:
            return np.copy(array)
    return array


def array2hist(array, hist):
    """Convert a NumPy array into a ROOT histogram

    Parameters
    ----------
    array : numpy array
        A 1, 2, or 3-d numpy array that will set the bin contents of the
        ROOT histogram.
    hist : ROOT TH1, TH2, or TH3
        A ROOT histogram.

    Returns
    -------
    hist : ROOT TH1, TH2, or TH3
        The ROOT histogram with bin contents set from the array.

    Raises
    ------
    TypeError
        If hist is not a ROOT histogram.
    ValueError
        If the array and histogram are not compatible in terms of
        dimensionality or number of bins along any axis.

    Notes
    -----
    The NumPy array is copied into the histogram's internal array. If the input
    NumPy array is not of the same data type as the histogram bin contents
    (i.e. TH1D vs TH1F, etc.) and/or the input array does not contain overflow
    bins along any of the axes, an additional copy is made into a temporary
    array with all values converted into the matching data type and with
    overflow bins included. Avoid this second copy by ensuring that the NumPy
    array data type matches the histogram data type and that overflow bins are
    included.

    See Also
    --------
    hist2array

    Examples
    --------

    >>> from root_numpy import array2hist, hist2array
    >>> import numpy as np
    >>> from rootpy.plotting import Hist2D
    >>> hist = Hist2D(5, 0, 1, 3, 0, 1, type='F')
    >>> array = np.random.randint(0, 10, size=(7, 5))
    >>> array
    array([[6, 7, 8, 3, 4],
           [8, 9, 7, 6, 2],
           [2, 3, 4, 5, 2],
           [7, 6, 5, 7, 3],
           [2, 0, 5, 6, 8],
           [0, 0, 6, 5, 2],
           [2, 2, 1, 5, 4]])
    >>> _ = array2hist(array, hist)
    >>> # dtype matches histogram type (D, F, I, S, C)
    >>> hist2array(hist)
    array([[ 9.,  7.,  6.],
           [ 3.,  4.,  5.],
           [ 6.,  5.,  7.],
           [ 0.,  5.,  6.],
           [ 0.,  6.,  5.]], dtype=float32)
    >>> # overflow is excluded by default
    >>> hist2array(hist, include_overflow=True)
    array([[ 6.,  7.,  8.,  3.,  4.],
           [ 8.,  9.,  7.,  6.,  2.],
           [ 2.,  3.,  4.,  5.,  2.],
           [ 7.,  6.,  5.,  7.,  3.],
           [ 2.,  0.,  5.,  6.,  8.],
           [ 0.,  0.,  6.,  5.,  2.],
           [ 2.,  2.,  1.,  5.,  4.]], dtype=float32)
    >>> array2 = hist2array(hist, include_overflow=True, copy=False)
    >>> hist[2, 2] = -10
    >>> # array2 views the same memory as hist because copy=False
    >>> array2
    array([[  6.,   7.,   8.,   3.,   4.],
           [  8.,   9.,   7.,   6.,   2.],
           [  2.,   3., -10.,   5.,   2.],
           [  7.,   6.,   5.,   7.,   3.],
           [  2.,   0.,   5.,   6.,   8.],
           [  0.,   0.,   6.,   5.,   2.],
           [  2.,   2.,   1.,   5.,   4.]], dtype=float32)
    >>> # x, y, z axes correspond to axes 0, 1, 2 in numpy
    >>> hist[2, 3] = -10
    >>> array2
    array([[  6.,   7.,   8.,   3.,   4.],
           [  8.,   9.,   7.,   6.,   2.],
           [  2.,   3., -10., -10.,   2.],
           [  7.,   6.,   5.,   7.,   3.],
           [  2.,   0.,   5.,   6.,   8.],
           [  0.,   0.,   6.,   5.,   2.],
           [  2.,   2.,   1.,   5.,   4.]], dtype=float32)

    """
    import ROOT
    if isinstance(hist, ROOT.TH3):
        shape = (hist.GetNbinsX() + 2,
                 hist.GetNbinsY() + 2,
                 hist.GetNbinsZ() + 2)
    elif isinstance(hist, ROOT.TH2):
        shape = (hist.GetNbinsX() + 2, hist.GetNbinsY() + 2)
    elif isinstance(hist, ROOT.TH1):
        shape = (hist.GetNbinsX() + 2,)
    else:
        raise TypeError(
            "hist must be an instance of ROOT.TH1, ROOT.TH2, or ROOT.TH3")

    # Determine the corresponding numpy dtype
    for hist_type in 'DFISC':
        if isinstance(hist, getattr(ROOT, 'TArray{0}'.format(hist_type))):
            break
    else:
        raise AssertionError(
            "hist is somehow an instance of TH[1|2|3] "
            "but not TArray[D|F|I|S|C]")

    # Constuct a NumPy array viewing the underlying histogram array
    dtype = np.dtype(DTYPE_ROOT2NUMPY[hist_type])
    # No copy is made if the dtype is the same as input
    _array = np.ascontiguousarray(array, dtype=dtype)
    if _array.ndim != len(shape):
        raise ValueError(
            "array and histogram do not have "
            "the same number of dimensions")
    if _array.shape != shape:
        # Check for overflow along each axis
        slices = []
        for axis, bins in enumerate(shape):
            if _array.shape[axis] == bins - 2:
                slices.append(slice(1, -1))
            elif _array.shape[axis] == bins:
                slices.append(slice(None))
            else:
                raise ValueError(
                    "array and histogram are not compatible along "
                    "the {0}-axis".format("xyz"[axis]))
        array_overflow = np.zeros(shape, dtype=dtype)
        array_overflow[tuple(slices)] = _array
        _array = array_overflow
    ARRAY_NUMPY2ROOT[len(shape)][hist_type](
        ROOT.AsCObject(hist), np.ravel(np.transpose(_array)))
    # Set the number of entries to the number of array elements
    hist.SetEntries(_array.size)
    return hist
