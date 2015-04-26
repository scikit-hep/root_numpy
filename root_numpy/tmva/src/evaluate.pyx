
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_twoclass(method, np.ndarray[np.double_t, ndim=2] events):
    cdef MethodBase* _method = <MethodBase*> PyCObject_AsVoidPtr(method)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef vector[float] features
    cdef Event* event = new Event(features, 0)
    _method.fTmpEvent = event
    cdef np.ndarray[np.double_t, ndim=1] output = np.empty(size, dtype=np.double)
    for i from 0 <= i < size:
        for j from 0 <= j < n_features:
            event.SetVal(j, events[i, j])
        output[i] = _method.GetMvaValue()
    _method.fTmpEvent = NULL
    del event
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_multiclass(method, np.ndarray[np.double_t, ndim=2] events, unsigned int n_classes):
    cdef MethodBase* _method = <MethodBase*> PyCObject_AsVoidPtr(method)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef vector[float] features
    cdef Event* event = new Event(features, 0)
    _method.fTmpEvent = event
    cdef np.ndarray[np.float32_t, ndim=2] output = np.empty((size, n_classes), dtype=np.float32)
    for i from 0 <= i < size:
        for j from 0 <= j < n_features:
            event.SetVal(j, events[i, j])
        memcpy(&output[i, 0], &(_method.GetMulticlassValues()[0]), sizeof(np.float32_t) * n_classes)
    _method.fTmpEvent = NULL
    del event
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_regression(method, np.ndarray[np.double_t, ndim=2] events, unsigned int n_targets):
    cdef MethodBase* _method = <MethodBase*> PyCObject_AsVoidPtr(method)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef vector[float] features
    cdef Event* event = new Event(features, 0)
    _method.fTmpEvent = event
    cdef np.ndarray[np.float32_t, ndim=2] output = np.empty((size, n_targets), dtype=np.float32)
    for i from 0 <= i < size:
        for j from 0 <= j < n_features:
            event.SetVal(j, events[i, j])
        memcpy(&output[i, 0], &(_method.GetRegressionValues()[0]), sizeof(np.float32_t) * n_targets)
    _method.fTmpEvent = NULL
    del event
    return output
