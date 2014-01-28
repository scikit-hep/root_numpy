def fill_h1(hist,
            np.ndarray[np.double_t, ndim=1] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TH1* _hist = <TH1*> PyCObject_AsVoidPtr(hist)
    cdef long size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef long i
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

def fill_h2(hist,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TH2* _hist = <TH2*> PyCObject_AsVoidPtr(hist)
    cdef long size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef long i
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

def fill_h3(hist,
            np.ndarray[np.double_t, ndim=2] array,
            np.ndarray[np.double_t, ndim=1] weights=None,
            bool return_indices=False):
    # perform type checking on python side
    cdef TH3* _hist = <TH3*> PyCObject_AsVoidPtr(hist)
    cdef long size = array.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] idx = np.empty(0, dtype=np.int)
    cdef long i
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
