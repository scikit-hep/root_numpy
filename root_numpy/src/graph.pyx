@cython.boundscheck(False)
@cython.wraparound(False)
def fill_g1(graph,
            np.ndarray[np.double_t, ndim=2] array):
    # perform type checking on python side
    cdef TGraph* _graph = <TGraph*> PyCObject_AsVoidPtr(graph)
    cdef int size = array.shape[0]
    cdef int i
    _graph.Set(size)
    for i from 0 <= i < size:
        _graph.SetPoint(i, array[i, 0], array[i, 1])


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_g2(graph,
            np.ndarray[np.double_t, ndim=2] array):
    # perform type checking on python side
    cdef TGraph2D* _graph = <TGraph2D*> PyCObject_AsVoidPtr(graph)
    cdef int size = array.shape[0]
    cdef int i
    _graph.Set(size)
    for i from 0 <= i < size:
        _graph.SetPoint(i, array[i, 0], array[i, 1], array[i, 2])
