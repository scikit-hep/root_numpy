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
