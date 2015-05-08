@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_reader(reader, name, np.ndarray[np.double_t, ndim=2] events, double aux):
    cdef Reader* _reader = <Reader*> PyCObject_AsVoidPtr(reader)
    cdef IMethod* imeth = _reader.FindMVA(name)
    if imeth == NULL:
        raise ValueError(
            "method '{0}' is not booked in this reader".format(name))
    cdef MethodBase* method = dynamic_cast["MethodBase*"](imeth)
    return evaluate_method_dispatch(method, events, aux)


@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_method(method, np.ndarray[np.double_t, ndim=2] events, double aux):
    cdef MethodBase* _method = <MethodBase*> PyCObject_AsVoidPtr(method)
    return evaluate_method_dispatch(_method, events, aux)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef evaluate_method_dispatch(MethodBase* _method, np.ndarray[np.double_t, ndim=2] events, double aux):
    cdef long n_features = events.shape[1]
    cdef unsigned int n_classes, n_targets
    if n_features != _method.GetNVariables():
        raise ValueError(
            "this method was trained with events containing "
            "{0} variables, but these events contain {1} variables".format(
                _method.GetNVariables(), n_features))
    cdef EAnalysisType analysistype
    analysistype = _method.GetAnalysisType()
    if analysistype == kClassification:
        return evaluate_twoclass(_method, events, aux)
    elif analysistype == kMulticlass:
        n_classes = _method.DataInfo().GetNClasses()
        if n_classes < 2:
            raise AssertionError("there must be at least two classes")
        return evaluate_multiclass(_method, events, n_classes)
    elif analysistype == kRegression:
        n_targets = _method.DataInfo().GetNTargets()
        if n_targets < 1:
            raise AssertionError("there must be at least one regression target")
        output = evaluate_regression(_method, events, n_targets)
        if n_targets == 1:
            return np.ravel(output)
        return output
    raise AssertionError("the analysis type of this method is not supported")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef evaluate_twoclass(MethodBase* _method, np.ndarray[np.double_t, ndim=2] events, double aux):
    cdef MethodCuts* mc
    cdef long size = events.shape[0]
    cdef long n_features = events.shape[1]
    cdef long i, j
    cdef vector[float] features
    cdef Event* event = new Event(features, 0)
    _method.fTmpEvent = event
    cdef np.ndarray[np.double_t, ndim=1] output = np.empty(size, dtype=np.double)
    if _method.GetMethodType() == kCuts:
        mc = dynamic_cast["MethodCuts*"](_method)
        if mc != NULL:
            mc.SetTestSignalEfficiency(aux)
    for i from 0 <= i < size:
        for j from 0 <= j < n_features:
            event.SetVal(j, events[i, j])
        output[i] = _method.GetMvaValue()
    _method.fTmpEvent = NULL
    del event
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cdef evaluate_multiclass(MethodBase* _method, np.ndarray[np.double_t, ndim=2] events, unsigned int n_classes):
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
cdef evaluate_regression(MethodBase* _method, np.ndarray[np.double_t, ndim=2] events, unsigned int n_targets):
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
