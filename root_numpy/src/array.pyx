"""
ROOT TArray -> NumPy array conversion
"""
cdef inline np.ndarray tonumpyarray(void* data, int size, int dtype) with gil:
    cdef np.npy_intp dims = size
    #NOTE: it doesn't take ownership of `data`. You must free `data` yourself
    return np.PyArray_SimpleNewFromData(1, &dims, dtype, data)


def array_d(root_arr):
    cdef TArrayD* _arr = <TArrayD*> PyCObject_AsVoidPtr(root_arr)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_DOUBLE)


def array_f(root_arr):
    cdef TArrayF* _arr = <TArrayF*> PyCObject_AsVoidPtr(root_arr)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_FLOAT32)


def array_l(root_arr):
    cdef TArrayL* _arr = <TArrayL*> PyCObject_AsVoidPtr(root_arr)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_LONG)


def array_i(root_arr):
    cdef TArrayI* _arr = <TArrayI*> PyCObject_AsVoidPtr(root_arr)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_INT)


def array_s(root_arr):
    cdef TArrayS* _arr = <TArrayS*> PyCObject_AsVoidPtr(root_arr)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_SHORT)


def array_c(root_arr):
    cdef TArrayC* _arr = <TArrayC*> PyCObject_AsVoidPtr(root_arr)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_BYTE)


def array_h1c(root_hist):
    cdef TH1C* _hist = <TH1C*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayC* _arr = dynamic_cast["TArrayC*"](_hist)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_BYTE)


def array_h2c(root_hist):
    cdef TH2C* _hist = <TH2C*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayC* _arr = dynamic_cast["TArrayC*"](_hist)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_BYTE)


def array_h3c(root_hist):
    cdef TH3C* _hist = <TH3C*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayC* _arr = dynamic_cast["TArrayC*"](_hist)
    return tonumpyarray(_arr.GetArray(), _arr.GetSize(), np.NPY_BYTE)

"""
NumPy array -> ROOT TArray[DFISC] or TH[123][DFISC] conversion
"""
@cython.boundscheck(False)
@cython.wraparound(False)
def h1d_array(root_hist, np.ndarray[np.double_t, ndim=1] array):
    cdef TH1D* _hist = <TH1D*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayD* _arr = dynamic_cast["TArrayD*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h2d_array(root_hist, np.ndarray[np.double_t, ndim=1] array):
    cdef TH2D* _hist = <TH2D*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayD* _arr = dynamic_cast["TArrayD*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h3d_array(root_hist, np.ndarray[np.double_t, ndim=1] array):
    cdef TH3D* _hist = <TH3D*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayD* _arr = dynamic_cast["TArrayD*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h1f_array(root_hist, np.ndarray[np.float32_t, ndim=1] array):
    cdef TH1F* _hist = <TH1F*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayF* _arr = dynamic_cast["TArrayF*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h2f_array(root_hist, np.ndarray[np.float32_t, ndim=1] array):
    cdef TH2F* _hist = <TH2F*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayF* _arr = dynamic_cast["TArrayF*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h3f_array(root_hist, np.ndarray[np.float32_t, ndim=1] array):
    cdef TH3F* _hist = <TH3F*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayF* _arr = dynamic_cast["TArrayF*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h1i_array(root_hist, np.ndarray[int, ndim=1] array):
    cdef TH1I* _hist = <TH1I*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayI* _arr = dynamic_cast["TArrayI*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h2i_array(root_hist, np.ndarray[int, ndim=1] array):
    cdef TH2I* _hist = <TH2I*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayI* _arr = dynamic_cast["TArrayI*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h3i_array(root_hist, np.ndarray[int, ndim=1] array):
    cdef TH3I* _hist = <TH3I*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayI* _arr = dynamic_cast["TArrayI*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h1s_array(root_hist, np.ndarray[short, ndim=1] array):
    cdef TH1S* _hist = <TH1S*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayS* _arr = dynamic_cast["TArrayS*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h2s_array(root_hist, np.ndarray[short, ndim=1] array):
    cdef TH2S* _hist = <TH2S*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayS* _arr = dynamic_cast["TArrayS*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h3s_array(root_hist, np.ndarray[short, ndim=1] array):
    cdef TH3S* _hist = <TH3S*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayS* _arr = dynamic_cast["TArrayS*"](_hist)
    _arr.Set(array.shape[0], &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h1c_array(root_hist, np.ndarray[np.int8_t, ndim=1] array):
    cdef TH1C* _hist = <TH1C*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayC* _arr = dynamic_cast["TArrayC*"](_hist)
    _arr.Set(array.shape[0], <char*> &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h2c_array(root_hist, np.ndarray[np.int8_t, ndim=1] array):
    cdef TH2C* _hist = <TH2C*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayC* _arr = dynamic_cast["TArrayC*"](_hist)
    _arr.Set(array.shape[0], <char*> &array[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def h3c_array(root_hist, np.ndarray[np.int8_t, ndim=1] array):
    cdef TH3C* _hist = <TH3C*> PyCObject_AsVoidPtr(root_hist)
    cdef TArrayC* _arr = dynamic_cast["TArrayC*"](_hist)
    _arr.Set(array.shape[0], <char*> &array[0])
