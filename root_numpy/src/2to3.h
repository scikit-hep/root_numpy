#ifndef __2TO3_H_
#define __2TO3_H_
#include <Python.h>

#if PY_MAJOR_VERSION >= 3
    #define PyCObject_FromVoidPtr(C, ignored_destroy) PyCapsule_New(C, NULL, NULL)
    #define PyCObject_AsVoidPtr(P) PyCapsule_GetPointer(P, NULL)
    #define PyCObject_Check(P) PyCapsule_CheckExact(P)
    //extern "C" void destroy_32(PyObject *P)
    //{
    //    destroy(PyCapsule_GetPointer(P, NULL));
    //}
#endif
#endif
