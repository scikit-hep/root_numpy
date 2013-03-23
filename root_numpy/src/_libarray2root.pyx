# cython: experimental_cpp_class_def=True
import numpy as np
import numpy as pynp # for forcing the use of python numpy
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.string cimport string, const_char
from libc.string cimport memcpy
include "all.pxi"
from warnings import warn
from libc.stdlib cimport malloc, free
np.import_array()

cdef cppclass NP2CConverter:
    TBranch* make_branch(TTree* tree):
        return NULL
    
    void read(void* source):
        pass

cdef cppclass ScalarNP2CConverter(NP2CConverter):
    int nbytes
    string roottype
    string name
    void* value
    # don't use copy constructor of this one since it will screw up
    # tree binding and/or ownership of value
    __init__(string name, string roottype, int nbytes):
        this.nbytes = nbytes
        this.roottype = roottype
        this.name = name
        this.value = malloc(nbytes)
        
    __del__(self): # does this do what I want?
        free(this.value)

    TBranch* make_branch(TTree* tree):
        print this.name, this.roottype
        cdef string leaflist = this.name+'/'+this.roottype
        print leaflist
        tree.Branch(this.name.c_str(), this.value, leaflist.c_str())

    void read(void* source):
        memcpy(this.value, source, this.nbytes)

cdef NP2CConverter* find_np2c_converter(name, dtype, peekvalue=None):
    scalarlist = {
        #np.int8 from cython means something else
        np.dtype(np.int8): (1, 'B'),
        np.dtype(np.int16): (2, 'S'),
        np.dtype(np.int32): (4, 'I'),
        np.dtype(np.int64): (8, 'L'),
        np.dtype(np.uint8): (1, 'b'),
        np.dtype(np.uint16): (2, 's'),
        np.dtype(np.uint32): (4, 'i'),
        np.dtype(np.uint64): (8, 'l'),
        
        np.dtype(np.float): (8, 'D'),
        np.dtype(np.float32): (4, 'F'),
        np.dtype(np.float64): (8, 'D')
    }
    #TODO:
    #np.float16: #this needs special treatment root doesn't have 16 bit float?
    #np.bool #this is need special case
    #np.object #this too should detect basic numpy array
    #How to detect fixed length array?
    print dtype
    if dtype in scalarlist:
        nbytes, roottype = scalarlist[dtype]
        return new ScalarNP2CConverter(name, roottype, nbytes)
    elif dtype==np.dtype(np.object):
        #lets peek
        if type(peekvalue) == type(np.array([])):
            ndim = peekvalue.ndim
            dtype = peekvalue.dtype
            #TODO finish this
    else:
        warn('Converter for %r not implemented yet. Skip.'%dtype)
    return NULL

cdef TTree* array2ttree(np.ndarray arr, treename='tree') except *:
    cdef vector[NP2CConverter*] cvarray # hmm how do I catch all python exception
                                        # and clean up before throwing ?
    cdef vector[int] posarray
    cdef vector[int] roffsetarray
    cdef int icol
    cdef auto_ptr[NP2CConverter] tmp
    cdef TTree* ret = NULL
    cdef int icv = 0
    cdef int arr_len = 0
    cdef int pos_len = 0
    cdef void* source = NULL
    cdef void* thisrow = NULL
    cdef NP2CConverter* tmpcv
    
    try: 
        names = arr.dtype.names
        fields = arr.dtype.fields
        
        
        # figure out the structure
        for icol, name in enumerate(names):
            # roffset is an offset of particular field in each record
            dtype, roffset = fields[name] 
            cvt = find_np2c_converter(name, dtype, arr[0][name])
            if cvt is not NULL:
                roffsetarray.push_back(roffset)
                cvarray.push_back(cvt)
                posarray.push_back(icol)

        ret = new TTree(treename, treename)
        
        # make branches
       
        for icv in range(cvarray.size()):
            cvarray[icv].make_branch(ret)

        # fill in data
        arr_len = len(arr)
        pos_len = posarray.size()

        for idata in range(arr_len):
            thisrow = np.PyArray_GETPTR1(arr, idata)
            for ipos in range(pos_len):
                roffset = roffsetarray[ipos]
                source = shift(thisrow, roffset)
                cvarray[ipos].read(source)
            ret.Fill()
    
    except:
        raise
    
    finally:
        # how do I clean up TTree?
        # root has some global funny memory management...
        # need to make sure no double free
        for icv in range(cvarray.size()):
            tmpcv = cvarray[icv]
            del tmpcv
    
    return ret

def test(a):
    #a = np.array([(1234567890123,2.),(3,4.)], dtype=[('x',np.int),('y',np.float32)])
    cdef TTree* tree = array2ttree(a)
    tree.Print()
    tree.Scan()
    del tree
