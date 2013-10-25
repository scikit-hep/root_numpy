from cpython cimport PyObject

cdef extern from "hist.h":
    cdef PyObject* fill_hist(PyObject* hist_, PyObject* array_, PyObject* weights_) 
