"""
ROOT TMatrixT -> numpy matrix conversion
"""
@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_d(root_mat):
    cdef TMatrixDBase* _mat = <TMatrixDBase*> PyCObject_AsVoidPtr(root_mat)
    cdef np.ndarray[np.double_t, ndim=2] arr = np.empty((_mat.GetNrows(), _mat.GetNcols()), dtype=np.double)
    cdef int i
    cdef int j
    for i from 0 <= i < _mat.GetNrows():
        for j from 0 <= j < _mat.GetNcols():
            arr[i, j] = _mat.get(i, j)
    return np.matrix(arr)


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_f(root_mat):
    cdef TMatrixFBase* _mat = <TMatrixFBase*> PyCObject_AsVoidPtr(root_mat)
    cdef np.ndarray[np.float32_t, ndim=2] arr = np.empty((_mat.GetNrows(), _mat.GetNcols()), dtype=np.float32)
    cdef int i
    cdef int j
    for i from 0 <= i < _mat.GetNrows():
        for j from 0 <= j < _mat.GetNcols():
            arr[i, j] = _mat.get(i, j)
    return np.matrix(arr)
