
@cython.boundscheck(False)
@cython.wraparound(False)
def factory_add_events_twoclass(
        factory,
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
    cdef vector[double]* event = new vector[double](n_features)
    cdef ETreeType treetype = kTraining
    if test:
        treetype = kTesting
    for i from 0 <= i < size:
        label = labels[i]
        if weights is not None:
            weight = weights[i]
        for j from 0 <= j < n_features:
            event[0][j] = events[i, j]
        if label == signal_label:
            _factory.AddEvent("Signal", treetype, event[0], weight)
        else:
            _factory.AddEvent("Background", treetype, event[0], weight)
    del event


@cython.boundscheck(False)
@cython.wraparound(False)
def factory_add_events_multiclass(
        factory,
        np.ndarray[np.double_t, ndim=2] events,
        np.ndarray[np.int_t, ndim=1] labels,
        np.ndarray[np.double_t, ndim=1] weights=None,
        bool test=False):
    cdef Factory* _factory = <Factory*> PyCObject_AsVoidPtr(factory)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef double weight = 1.
    cdef int label
    cdef vector[double]* event = new vector[double](n_features)
    cdef ETreeType treetype = kTraining
    if test:
        treetype = kTesting
    for i from 0 <= i < size:
        label = labels[i]
        if weights is not None:
            weight = weights[i]
        for j from 0 <= j < n_features:
            event[0][j] = events[i, j]
        _factory.AddEvent("Class_{0:d}".format(label), treetype, event[0], weight)
    del event


@cython.boundscheck(False)
@cython.wraparound(False)
def factory_add_events_regression(
        factory,
        np.ndarray[np.double_t, ndim=2] events,
        np.ndarray[np.double_t, ndim=2] targets,
        np.ndarray[np.double_t, ndim=1] weights=None,
        bool test=False):
    cdef Factory* _factory = <Factory*> PyCObject_AsVoidPtr(factory)
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long n_targets = targets.shape[1]
    cdef long i, j
    cdef double weight = 1.
    cdef vector[double]* event = new vector[double](n_features + n_targets)
    cdef ETreeType treetype = kTraining
    if test:
        treetype = kTesting
    for i from 0 <= i < size:
        if weights is not None:
            weight = weights[i]
        for j from 0 <= j < n_features:
            event[0][j] = events[i, j]
        for j from 0 <= j < n_targets:
            event[0][n_features + j] = targets[i, j]
        _factory.AddEvent("Regression", treetype, event[0], weight)
    del event
