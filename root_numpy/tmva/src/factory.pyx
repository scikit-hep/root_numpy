
@cython.boundscheck(False)
@cython.wraparound(False)
def factory_add_events(factory,
                       np.ndarray[np.double_t, ndim=2] events,
                       np.ndarray[np.int_t, ndim=1] labels,
                       int signal_label,
                       np.ndarray[np.double_t, ndim=1] weights=None,
                       bool test=False):
    cdef Factory* _factory = <Factory*> PyCObject_AsVoidPtr(factory)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef double weight = 1.
    cdef int label
    cdef vector[double] event
    for i from 0 <= i < size:
        event.clear()
        label = labels[i]
        if weights is not None:
            weight = weights[i]
        for j from 0 <= j < n_features:
            event.push_back(<double> events[i, j])
        if test:
            if label == signal_label:
                _factory.AddSignalTestEvent(event, weight)
            else:
                _factory.AddBackgroundTestEvent(event, weight)
        else:
            if label == signal_label:
                _factory.AddSignalTrainingEvent(event, weight)
            else:
                _factory.AddBackgroundTrainingEvent(event, weight)
