import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string, const_char
from libc.string cimport memcpy
from cpython cimport array
from cython.operator cimport dereference as deref
try:
    from collections import OrderedDict
except ImportError:
    # fall back to drop in
    from OrderedDict import OrderedDict
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cpython cimport PyObject
from cpython.cobject cimport PyCObject_AsVoidPtr, PyCObject_Check
from glob import glob
import warnings
from root_numpy_warnings import RootNumpyUnconvertibleWarning
include "all.pxi"
np.import_array()


def list_trees(fname):
    fname = glob(fname) # poor man support for globbing
    if len(fname) == 0: raise IOError('File not found: %s' % fname)
    fname = fname[0]

    cdef TFile* f = new TFile(fname)
    if f is NULL: raise IOError('Cannot read: %s' % fname)

    cdef TList* keys = f.GetListOfKeys()
    if keys is NULL: raise IOError('Not a valid root file: %s' % fname)
    ret = []
    cdef int n = keys.GetEntries()
    cdef TObject* obj
    for i in range(n):
        name = keys.At(i).GetName()
        obj = f.Get(name)
        if obj is not NULL:
            clname = str(obj.ClassName())
            if  clname == 'TTree':
                ret.append(name)
    return ret


def list_structures(fname, tree=None):
    if tree is None:#support for automatically find tree
        tree = list_trees(fname)
        if len(tree) != 1:
            raise ValueError('Multiple Tree Found: %s' % str(tree))
        else:
            tree = tree[0]

    cdef TFile* f = new TFile(fname)
    fname = glob(fname)#poor man support for globbing
    if len(fname)==0: raise IOError('File not found: %s' % fname)
    fname = fname[0]

    cdef TTree* t = <TTree*> f.Get(tree)
    if t is NULL:
        raise IOError('Tree %s not found in %s' % (tree, fname))

    tmp = parse_tree_structure(t)
    return tmp


def list_branches(fname, tree=None):
    return list_structures(fname, tree).keys()


cdef parse_tree_structure(TTree* tree):
    cdef char* name
    cdef TBranch* thisBranch
    cdef TLeaf* thisLeaf
    cdef TObjArray* branches = tree.GetListOfBranches()
    cdef TObjArray* leaves
    ret = OrderedDict()
    if branches is NULL: return ret
    for ibranch in range(branches.GetEntries()):
        thisBranch = <TBranch*>(branches.At(ibranch))
        leaves = thisBranch.GetListOfLeaves()
        leaflist = []
        if leaves is not NULL:
            for ibranch in range(leaves.GetEntries()):
                thisLeaf = <TLeaf*>leaves.At(ibranch)
                lname = thisLeaf.GetName()
                ltype = thisLeaf.GetTypeName()
                leaflist.append((lname, ltype))
        ret[thisBranch.GetName()] = leaflist
    return ret

def unique(seq):
    seen = {}
    result = []
    for item in seq:
        marker = item
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

cdef class Converter:
    cdef int write(self,Column* col, void* buffer):
        pass
    cdef object get_nptype(self):
        pass

# create numpy array of given type code with
# given numelement and size of each element
# and write it to buffer
cdef inline int create_numpyarray(
        void* buffer, void* src, int typecode, int numele, int elesize):
    cdef np.npy_intp dims[1]
    dims[0]=numele;
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, typecode, 0)

    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # increase one since we are putting in buffer directly
    Py_INCREF(tmp)

    # copy to tmp.data
    cdef int nbytes = numele * elesize
    memcpy(tmp.data,src,nbytes)

    # now write PyObject* to buffer
    memcpy(buffer, &tmpobj, sizeof(PyObject*))

    return sizeof(tmpobj)


cdef class VaryArray_NumpyConverter(Converter):
    cdef BasicNumpy_Converter conv # converter for single element
    cdef public object nptype
    cdef int typecode
    cdef int elesize
    def __init__(self,Converter conv):
        self.conv = conv
        self.typecode = self.conv.get_nptypecode()
        self.elesize = conv.size
    cdef int write(self,Column* col, void* buffer):
        cdef int numele = col.getLen()
        cdef int elesize = self.elesize
        cdef void* src = col.GetValuePointer()
        cdef int typecode = self.typecode
        return create_numpyarray(buffer, src, typecode, numele, elesize)

    cdef object get_nptype(self):
        return np.object
    cdef object get_nptypecode(self):
        return np.NPY_OBJECT


cdef class FixedArray_NumpyConverter(Converter):
    cdef BasicNumpy_Converter conv # converter for single element
    cdef int L # numele
    def __init__(self, BasicNumpy_Converter conv, int L):
        self.conv = conv
        self.L = L
    cdef int write(self,Column* col, void* buffer):
        cdef void* src = col.GetValuePointer()
        cdef int nbytes = col.getSize()
        memcpy(buffer,src,nbytes)
        return nbytes
    cdef object get_nptype(self):
        return (self.conv.nptype, self.L)
    cdef int get_nptypecode(self):
        return self.conv.nptypecode


cdef class BasicNumpy_Converter(Converter):
    # cdef string rtype
    cdef public int size
    cdef public object nptype
    cdef int nptypecode
    def __init__(self, size, nptype, nptypecode):
        self.size = size
        self.nptype = nptype
        self.nptypecode = nptypecode
    cdef int write(self, Column* col, void* buffer):
        cdef void* src = col.GetValuePointer()
        memcpy(buffer, src, self.size)
        return self.size
    cdef object get_nptype(self):
        return self.nptype
    cdef int get_nptypecode(self):
        return self.nptypecode

cdef class VectorFloat_Converter(Converter):
    cdef int elesize
    cdef int nptypecode
    cdef Vector2Array[float] v2a
    def __init__(self):
        self.nptypecode = np.NPY_FLOAT32
        self.elesize = 4
    cdef int write(self,Column* col, void* buffer):
        cdef int elesize = self.elesize
        cdef int typecode = self.nptypecode
        cdef vector[float]* tmp = <vector[float]*> col.GetValuePointer()
        cdef int numele = tmp.size()
        # check cython auto generate code
        # if it really does &((*tmp)[0])
        cdef float* fa = self.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, typecode, numele, elesize)
    cdef object get_nptype(self):
        return np.object
    cdef object get_nptypecode(self):
        return np.NPY_OBJECT

cdef class VectorDouble_Converter(Converter):
    cdef int elesize
    cdef int nptypecode
    cdef Vector2Array[double] v2a
    def __init__(self):
        self.nptypecode = np.NPY_FLOAT64
        self.elesize = 8
    cdef int write(self,Column* col, void* buffer):
        cdef int elesize = self.elesize
        cdef int typecode = self.nptypecode
        cdef vector[double]* tmp = <vector[double]*> col.GetValuePointer()
        cdef int numele = tmp.size()
        # check cython auto generate code
        # if it really does &((*tmp)[0])
        cdef double* fa = self.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, typecode, numele, elesize)
    cdef object get_nptype(self):
        return np.object
    cdef object get_nptypecode(self):
        return np.NPY_OBJECT

cdef class VectorInt_Converter(Converter):
    cdef int elesize
    cdef int nptypecode
    cdef Vector2Array[int] v2a
    def __init__(self):
        self.nptypecode = np.NPY_INT32
        self.elesize = 4
    cdef int write(self,Column* col, void* buffer):
        cdef int elesize = self.elesize
        cdef int typecode = self.nptypecode
        cdef vector[int]* tmp = <vector[int]*> col.GetValuePointer()
        cdef int numele = tmp.size()
        # check cython auto generate code
        # if it really does &((*tmp)[0])
        cdef int* fa = self.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, typecode, numele, elesize)
    cdef object get_nptype(self):
        return np.object
    cdef object get_nptypecode(self):
        return np.NPY_OBJECT

cdef class VectorLong_Converter(Converter):
    cdef int elesize
    cdef int nptypecode
    cdef Vector2Array[long] v2a
    def __init__(self):
        self.nptypecode = np.NPY_INT64
        self.elesize = 8
    cdef int write(self,Column* col, void* buffer):
        cdef int elesize = self.elesize
        cdef int typecode = self.nptypecode
        cdef vector[long]* tmp = <vector[long]*> col.GetValuePointer()
        cdef int numele = tmp.size()
        # check cython auto generate code
        # if it really does &((*tmp)[0])
        cdef long* fa = self.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, typecode, numele, elesize)
    cdef object get_nptype(self):
        return np.object
    cdef object get_nptypecode(self):
        return np.NPY_OBJECT

cdef class VectorChar_Converter(Converter):
    cdef int elesize
    cdef int nptypecode
    cdef Vector2Array[char] v2a
    def __init__(self):
        self.nptypecode = np.NPY_INT8
        self.elesize = 1
    cdef int write(self,Column* col, void* buffer):
        cdef int elesize = self.elesize
        cdef int typecode = self.nptypecode
        cdef vector[char]* tmp = <vector[char]*> col.GetValuePointer()
        cdef int numele = tmp.size()
        # check cython auto generate code
        # if it really does &((*tmp)[0])
        cdef char* fa = self.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, typecode, numele, elesize)
    cdef object get_nptype(self):
        return np.object
    cdef object get_nptypecode(self):
        return np.NPY_OBJECT

converters = {
    'Char_t':       BasicNumpy_Converter(1, np.int8, np.NPY_INT8),
    'UChar_t':      BasicNumpy_Converter(1, np.uint8, np.NPY_UINT8),

    'Short_t':      BasicNumpy_Converter(2, np.int16, np.NPY_INT16),
    'UShort_t':     BasicNumpy_Converter(2, np.uint16, np.NPY_UINT8),

    'Int_t':        BasicNumpy_Converter(4, np.int32, np.NPY_INT32),
    'UInt_t':       BasicNumpy_Converter(4, np.uint32, np.NPY_UINT32),

    'Float_t':      BasicNumpy_Converter(4, np.float32, np.NPY_FLOAT32),
    'Double_t':     BasicNumpy_Converter(8, np.float64, np.NPY_FLOAT64),

    'Long64_t':     BasicNumpy_Converter(8, np.int64, np.NPY_INT64),
    'ULong64_t':    BasicNumpy_Converter(8, np.uint64, np.NPY_UINT64),

    'Bool_t':       BasicNumpy_Converter(1, np.bool, np.NPY_BOOL),

    'vector<float>':VectorFloat_Converter(),
    'vector<double>':VectorDouble_Converter(),
    'vector<int>':  VectorInt_Converter(),
    'vector<long>': VectorLong_Converter(),
    'vector<char>': VectorChar_Converter(),
}


cdef Converter find_converter(Column* col):
    cdef ColumnType ct = col.coltype
    if ct == SINGLE:
        return converters[col.GetTypeName()]
    elif ct == FIXED:
        return FixedArray_NumpyConverter(converters[col.GetTypeName()], col.countval)
    elif ct == VARY:
        return VaryArray_NumpyConverter(converters[col.GetTypeName()])


cdef np.ndarray initarray(vector[Column*] columns, int numEntries, list cv):
    cdef Column* thisCol
    cdef Converter thisConv

    nst = []
    for i in range(columns.size()):
        thisCol = columns[i]
        thisConv = find_converter(thisCol)
        nst.append((thisCol.colname, thisConv.get_nptype()))
        cv.append(thisConv)
    return np.empty(numEntries, dtype=nst)


cdef object root2array_fromTTree(TTree* tree, branches, entries, offset):
    # this is actually vector of pointers despite how it looks
    cdef vector[Column*] columns
    cdef Column* thisCol

    # make a better chain so we can register all columns
    cdef BetterChain* bc = new BetterChain(tree)
    cdef int numEntries = bc.GetEntries()

    cdef list cv=[] # list of converter in the same order
    cdef Converter thisCV
    cdef int numcol
    cdef int ientry
    cdef void* dataptr
    cdef np.ndarray arr
    cdef int nb
    cdef vector[Converter] cvarray
    try:
        # parse the tree structure to determine
        # whether to use shortname or long name
        # and loop through all leaves
        structure = parse_tree_structure(tree)
        if branches is None: branches = structure.keys()
        branches = unique(branches)

        for branch in branches:
            leaves = structure[branch]
            shortname = len(leaves) == 1
            for leaf,ltype in leaves:
                if ltype in converters:
                    colname = branch if shortname else '%s_%s' % (branch, leaf)
                    thisCol = bc.MakeColumn(branch, leaf, colname)
                    columns.push_back(thisCol)
                else:
                    msg = 'Cannot convert leaf %s of branch %s with type %s (skipping)'\
                        % (branch, leaf, ltype)
                    warnings.warn(msg, RootNumpyUnconvertibleWarning)

        # now we got all the columns time to make an appropriate array structure
        # first determine the correct size given tree size offset and entries
        if entries is None: entries = numEntries
        numEntries = min(max(numEntries - offset, 0), entries)
        # numEntries = min(entries, numEntries) if entries is not None else numEntries

        arr = initarray(columns, numEntries, cv)
        numcol = columns.size()
        ientry = 0
        bc.GetEntry(offset)
        # convert cv list to cvarray for speed (this PYINCREF and PYDECREF relies)
        # on cv list this is to optimize the tight loop
        for c in cv: cvarray.push_back(c)
        while bc.Next() != 0 and ientry < numEntries:
            dataptr = np.PyArray_GETPTR1(arr, ientry)
            for icol in range(numcol):
                thisCol = columns[icol]
                thisCV = cvarray[icol]
                nb = thisCV.write(thisCol, dataptr)
                dataptr = shift(dataptr, nb) # poorman pointer magic
            ientry += 1
    finally:
        del bc
    return arr


def root2array_fromFname(fnames, treename, branches, entries, offset):
    cdef TChain* ttree = NULL
    try:
        ttree = new TChain(treename)
        for fn in fnames:
            ttree.Add(fn)
        ret = root2array_fromTTree(<TTree*> ttree, branches,
                entries, offset)
    finally:
        del ttree
    return ret


def root2array_fromCObj(tree, branches, entries, offset):
    # this is not a safe method
    # provided here for convenience only
    # typecheck should be implemented for the wrapper
    if not PyCObject_Check(tree):
        raise ValueError('tree must be PyCObject')
    cdef TTree* chain = <TTree*> PyCObject_AsVoidPtr(tree)
    return root2array_fromTTree(chain, branches,
            entries, offset)
