
@cython.boundscheck(False)
@cython.wraparound(False)
def reader_evaluate(reader, name, np.ndarray[np.double_t, ndim=2] events):
    cdef Reader* _reader = <Reader*> PyCObject_AsVoidPtr(reader)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef vector[double] event
    cdef np.ndarray[np.double_t, ndim=1] output = np.empty(size, dtype=np.double)
    for i from 0 <= i < size:
        event.clear()
        for j from 0 <= j < n_features:
            event.push_back(<double> events[i, j])
        output[i] = _reader.EvaluateMVA(event, name)
    return output
