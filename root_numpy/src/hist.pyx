@cython.boundscheck(False)
@cython.wraparound(False)
def fill_h1(hist,
            np.ndarray[np.double_t, ndim=1] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TH1* _hist = <TH1*> PyCObject_AsVoidPtr(hist)
    cdef SIZE_t size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef SIZE_t i
    cdef int bin_idx
    if return_indices:
        idx = np.empty(size, dtype=np.int)
    if weights is not None:
        for i from 0 <= i < size:
            bin_idx = _hist.Fill(array[i], weights[i])
            if return_indices:
                idx[i] = bin_idx
    else:
        for i from 0 <= i < size:
            bin_idx = _hist.Fill(array[i])
            if return_indices:
                idx[i] = bin_idx
    if return_indices:
        return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_h2(hist,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TH2* _hist = <TH2*> PyCObject_AsVoidPtr(hist)
    cdef SIZE_t size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef SIZE_t i
    cdef int bin_idx
    if return_indices:
        idx = np.empty(size, dtype=np.int)
    if weights is not None:
        for i from 0 <= i < size:
            bin_idx = _hist.Fill(array[i, 0], array[i, 1], weights[i])
            if return_indices:
                idx[i] = bin_idx
    else:
        for i from 0 <= i < size:
            bin_idx = _hist.Fill(array[i, 0], array[i, 1])
            if return_indices:
                idx[i] = bin_idx
    if return_indices:
        return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_h3(hist,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TH3* _hist = <TH3*> PyCObject_AsVoidPtr(hist)
    cdef SIZE_t size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef SIZE_t i
    cdef int bin_idx
    if return_indices:
        idx = np.empty(size, dtype=np.int)
    if weights is not None:
        for i from 0 <= i < size:
            bin_idx = _hist.Fill(array[i, 0], array[i, 1], array[i, 2], weights[i])
            if return_indices:
                idx[i] = bin_idx
    else:
        for i from 0 <= i < size:
            bin_idx = _hist.Fill(array[i, 0], array[i, 1], array[i, 2])
            if return_indices:
                idx[i] = bin_idx
    if return_indices:
        return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_p1(profile,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TProfile* _profile = <TProfile*> PyCObject_AsVoidPtr(profile)
    cdef SIZE_t size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef SIZE_t i
    cdef int bin_idx
    if return_indices:
        idx = np.empty(size, dtype=np.int)
    if weights is not None:
        for i from 0 <= i < size:
            bin_idx = _profile.Fill(array[i, 0], array[i, 1], weights[i])
            if return_indices:
                idx[i] = bin_idx
    else:
        for i from 0 <= i < size:
            bin_idx = _profile.Fill(array[i, 0], array[i, 1])
            if return_indices:
                idx[i] = bin_idx
    if return_indices:
        return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_p2(profile,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TProfile2D* _profile = <TProfile2D*> PyCObject_AsVoidPtr(profile)
    cdef SIZE_t size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef SIZE_t i
    cdef int bin_idx
    if return_indices:
        idx = np.empty(size, dtype=np.int)
    if weights is not None:
        for i from 0 <= i < size:
            bin_idx = _profile.Fill(array[i, 0], array[i, 1], array[i, 2], weights[i])
            if return_indices:
                idx[i] = bin_idx
    else:
        for i from 0 <= i < size:
            bin_idx = _profile.Fill(array[i, 0], array[i, 1], array[i, 2])
            if return_indices:
                idx[i] = bin_idx
    if return_indices:
        return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_p3(profile,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TProfile3D* _profile = <TProfile3D*> PyCObject_AsVoidPtr(profile)
    cdef SIZE_t size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef SIZE_t i
    cdef int bin_idx
    if return_indices:
        idx = np.empty(size, dtype=np.int)
    if weights is not None:
        for i from 0 <= i < size:
            bin_idx = _profile.Fill(array[i, 0], array[i, 1], array[i, 2], array[i, 3], weights[i])
            if return_indices:
                idx[i] = bin_idx
    else:
        for i from 0 <= i < size:
            bin_idx = _profile.Fill(array[i, 0], array[i, 1], array[i, 2], array[i, 3])
            if return_indices:
                idx[i] = bin_idx
    if return_indices:
        return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def thn2array(hist, shape, dtype):
    cdef THnBase* _hist = <THnBase*> PyCObject_AsVoidPtr(hist)
    cdef double content
    cdef long long ibin
    cdef long long nbins = _hist.GetNbins()
    cdef np.ndarray array = np.zeros(shape, dtype=dtype)
    cdef np.ndarray array_ravel_view = np.ravel(array)
    for ibin in range(nbins):
        array_ravel_view[ibin] = _hist.GetBinContent(ibin)
    return array


@cython.boundscheck(False)
@cython.wraparound(False)
def thnsparse2array(hist, shape, dtype):
    cdef THnBase* _hist = <THnBase*> PyCObject_AsVoidPtr(hist)
    cdef double content
    cdef long long ibin
    cdef long long nbins = _hist.GetNbins()
    cdef np.ndarray array = np.zeros(shape, dtype=dtype)
    cdef np.ndarray coord = np.empty(array.ndim, dtype=np.int32)
    itemset = array.itemset
    for ibin in range(nbins):
        content = _hist.GetBinContent(ibin, <int*> coord.data)
        itemset(tuple(coord), content)
    return array
