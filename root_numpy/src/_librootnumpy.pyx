# cython: experimental_cpp_class_def=True
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp.pair cimport pair
from libcpp.string cimport string, const_char
from libc.string cimport memcpy

from cpython cimport array
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cpython cimport PyObject
from cpython.cobject cimport PyCObject_AsVoidPtr, PyCObject_Check

from cython.operator cimport dereference as deref, preincrement as inc

try:
    from collections import OrderedDict
except ImportError:
    # Fall back on drop-in
    from OrderedDict import OrderedDict

import atexit
from glob import glob
import warnings
from root_numpy_warnings import RootNumpyUnconvertibleWarning
include "all.pxi"
np.import_array()


def list_trees(fname):
    # Poor man support for globbing
    fname = glob(fname)
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
    # Support for automatically find tree
    if tree is None:
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


# create numpy array of given type code with
# given numelement and size of each element
# and write it to buffer
cdef inline int create_numpyarray(
        void* buffer, void* src, int typecode, int numele, int elesize):
    cdef np.npy_intp dims[1]
    dims[0] = numele;
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, typecode, 0)

    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # increase one since we are putting in buffer directly
    Py_INCREF(tmp)

    # copy to tmp.data
    cdef int nbytes = numele * elesize
    memcpy(tmp.data, src, nbytes)

    # now write PyObject* to buffer
    memcpy(buffer, &tmpobj, sizeof(PyObject*))

    return sizeof(tmpobj)


# special treatment for vector<bool>
cdef inline int create_numpyarray_vectorbool(void* buffer, vector[bool]* src):
    cdef int numele = src.size()
    cdef np.npy_intp dims[1]
    dims[0] = numele;
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, np.NPY_BOOL, 0)

    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # increase one since we are putting in buffer directly
    Py_INCREF(tmp)

    # can't use memcpy here...
    cdef int i
    for i from 0 <= i < numele:
        tmp[i] = src.at(i)

    # now write PyObject* to buffer
    memcpy(buffer, &tmpobj, sizeof(PyObject*))

    return sizeof(tmpobj)


cdef cppclass Converter:
    __init__():
        pass
    int write(Column* col, void* buffer):
        pass
    object get_nptype():
        pass


cdef cppclass BasicConverter(Converter):
    # cdef string rtype
    int size
    int nptypecode
    string nptype
    __init__(int size, string nptype, int nptypecode):
        this.size = size
        this.nptypecode = nptypecode
        this.nptype = nptype
    int write(Column* col, void* buffer):
        cdef void* src = col.GetValuePointer()
        memcpy(buffer, src, this.size)
        return this.size
    object get_nptype():
        return np.dtype(this.nptype)
    int get_nptypecode():
        return this.nptypecode


cdef cppclass VaryArrayConverter(Converter):
    BasicConverter* conv # converter for single element
    int typecode
    int elesize
    __init__(BasicConverter* conv):
        this.conv = conv
        this.typecode = conv.get_nptypecode()
        this.elesize = conv.size
    int write(Column* col, void* buffer):
        cdef int numele = col.getLen()
        cdef void* src = col.GetValuePointer()
        return create_numpyarray(buffer, src, this.typecode, numele, this.elesize)
    object get_nptype():
        return np.object
    object get_nptypecode():
        return np.NPY_OBJECT


cdef cppclass FixedArrayConverter(Converter):
    BasicConverter* conv # converter for single element
    int L # numele
    __init__(BasicConverter* conv, int L):
        this.conv = conv
        this.L = L
    int write(Column* col, void* buffer):
        cdef void* src = col.GetValuePointer()
        cdef int nbytes = col.getSize()
        memcpy(buffer, src, nbytes)
        return nbytes
    object get_nptype():
        return (np.dtype(this.conv.nptype), this.L)
    int get_nptypecode():
        return this.conv.nptypecode


cdef cppclass VectorConverterBase(Converter):
    object get_nptype():
        return np.object
    object get_nptypecode():
        return np.NPY_OBJECT


cdef cppclass VectorConverter[T](VectorConverterBase):
    int elesize
    int nptypecode
    Vector2Array[T] v2a
    __init__():
        cdef TypeName[T] ast = TypeName[T]()
        info = TYPES[ast.name]
        this.elesize = info[2].itemsize
        this.nptypecode = info[3]
    int write(Column* col, void* buffer):
        cdef vector[T]* tmp = <vector[T]*> col.GetValuePointer()
        cdef int numele = tmp.size()
        # check cython auto generate code
        # if it really does &((*tmp)[0])
        cdef T* fa = this.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, this.nptypecode, numele, this.elesize)


cdef cppclass VectorBoolConverter(VectorConverterBase):
    __init__():
        pass
    # Requires special treament since vector<bool> stores contents as bits...
    int write(Column* col, void* buffer):
        cdef vector[bool]* tmp = <vector[bool]*> col.GetValuePointer()
        return create_numpyarray_vectorbool(buffer, tmp)

ctypedef unsigned char unsigned_char
ctypedef unsigned short unsigned_short
ctypedef unsigned int unsigned_int
ctypedef unsigned long unsigned_long

TYPES = {
    TypeName[bool]().name:           ('bool', 'Bool_t', np.dtype(np.bool), np.NPY_BOOL),
    TypeName[char]().name:           ('char', 'Char_t', np.dtype(np.int8), np.NPY_INT8),
    TypeName[unsigned_char]().name:  ('unsigned char', 'UChar_t', np.dtype(np.uint8), np.NPY_UINT8),
    TypeName[short]().name:          ('short', 'Short_t', np.dtype(np.int16), np.NPY_INT16),
    TypeName[unsigned_short]().name: ('unsigned short', 'UShort_t', np.dtype(np.uint16), np.NPY_UINT8),
    TypeName[int]().name:            ('int', 'Int_t', np.dtype(np.int32), np.NPY_INT32),
    TypeName[unsigned_int]().name:   ('unsigned int', 'UInt_t', np.dtype(np.uint32), np.NPY_UINT32),
    TypeName[long]().name:           ('long', 'Long64_t', np.dtype(np.int64), np.NPY_INT64),
    TypeName[unsigned_long]().name:  ('unsigned long', 'ULong64_t', np.dtype(np.uint64), np.NPY_UINT64),
    TypeName[float]().name:          ('float', 'Float_t', np.dtype(np.float32), np.NPY_FLOAT32),
    TypeName[double]().name:         ('double', 'Double_t', np.dtype(np.float64), np.NPY_FLOAT64),
}

cdef cpp_map[string, Converter*] CONVERTERS
ctypedef pair[string, Converter*] CONVERTERS_ITEM

for ctypename, (ctype, roottype, dtype, dtypecode) in TYPES.items():
    CONVERTERS.insert(CONVERTERS_ITEM(
        roottype, new BasicConverter(
            dtype.itemsize, dtype.name, dtypecode)))

# special case for vector<bool>
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<bool>', new VectorBoolConverter()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<char>', new VectorConverter[char]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<unsigned char>', new VectorConverter[unsigned_char]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<short>', new VectorConverter[short]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<unsigned short>', new VectorConverter[unsigned_short]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<int>', new VectorConverter[int]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<unsigned int>', new VectorConverter[unsigned_int]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<long>', new VectorConverter[long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<unsigned long>', new VectorConverter[unsigned_long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<float>', new VectorConverter[float]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<double>', new VectorConverter[double]()))

cdef Converter* find_converter(Column* col):
    cdef ColumnType ct = col.coltype
    it = CONVERTERS.find(string(col.GetTypeName()))
    if it == CONVERTERS.end():
        return NULL
    cdef Converter* basic_conv = deref(it).second
    if ct == SINGLE:
        return basic_conv
    elif ct == FIXED:
        return new FixedArrayConverter(<BasicConverter*>basic_conv, col.countval)
    elif ct == VARY:
        return new VaryArrayConverter(<BasicConverter*>basic_conv)
    return NULL

cdef np.ndarray initarray(vector[Column*]& columns,
                          vector[Converter*]& cv,
                          int entries):
    cdef Column* this_col
    cdef Converter* this_conv
    nst = []
    for i in range(columns.size()):
        this_col = columns[i]
        this_conv = find_converter(this_col)
        if this_conv == NULL:
            raise ValueError("No converter for %s" % this_col.GetTypeName())
        nst.append((this_col.colname, this_conv.get_nptype()))
        cv.push_back(this_conv)
    return np.empty(entries, dtype=nst)


cdef object root2array_fromTTree(TTree* tree, branches,
                                 entries, offset, selection):
    # This is actually vector of pointers despite how it looks
    cdef vector[Column*] columns
    cdef Column* thisCol

    # Make a better chain so we can register all columns
    cdef BetterChain* bc = new BetterChain(tree)
    cdef TTreeFormula* formula = NULL
    cdef int num_entries = bc.GetEntries()

    # list of converter in the same order
    cdef Converter* thisCV
    cdef int numcol
    cdef int ientry
    cdef void* dataptr
    cdef np.ndarray arr
    cdef int nb
    cdef vector[Converter*] cvarray
    cdef bytes py_select_formula
    cdef char* select_formula

    try:
        # Setup the selection if we have one
        if selection:
            py_select_formula = str(selection)
            select_formula = py_select_formula
            formula = new TTreeFormula("selection", select_formula, bc.fChain)
            if formula == NULL or formula.GetNdim() == 0:
                del formula
                raise ValueError("could not compile selection formula")
            # The chain will take care of updating the formula leaves when
            # rolling over to the next tree.
            bc.AddFormula(formula)
        
        # Activate branches used by formulae and deactivate all others
        # MakeColumn will active the ones needed for the output array
        bc.InitBranches()

        # Parse the tree structure to determine
        # whether to use short or long name
        # and loop through all leaves
        structure = parse_tree_structure(tree)
        if branches is None:
            branches = structure.keys()
        branches = unique(branches)

        for branch in branches:
            try:
                leaves = structure[branch]
            except KeyError:
                raise ValueError(
                        'the branch %s is not present in the tree. '
                        'Call list_branches or appropriate ROOT methods '
                        'to see a list of available branches' % branch)
            shortname = len(leaves) == 1
            for leaf, ltype in leaves:
                if CONVERTERS.find(ltype) != CONVERTERS.end():
                    colname = branch if shortname else '%s_%s' % (branch, leaf)
                    thisCol = bc.MakeColumn(branch, leaf, colname)
                    columns.push_back(thisCol)
                else:
                    warnings.warn(
                        'Cannot convert leaf %s of branch %s '
                        'with type %s (skipping)' % (branch, leaf, ltype),
                        RootNumpyUnconvertibleWarning)
        
        # Now that we have all the columns we can
        # make an appropriate array structure
        # First determine the correct size given tree size, offset, and entries
        if entries is None:
            entries = num_entries
        num_entries = min(max(num_entries - offset, 0), entries)

        arr = initarray(columns, cvarray, num_entries)
        numcol = columns.size()
        ientry = 0
        ientry_selected = 0
        bc.GetEntry(offset)
        
        while bc.Next() != 0 and ientry < num_entries:
            ientry += 1
            # Following code in ROOT's tree/treeplayer/src/TTreePlayer.cxx
            if formula != NULL:
                ndata = formula.GetNdata()
                keep = False
                for current from 0 <= current < ndata:
                    keep |= (formula.EvalInstance(current) != 0)
                    if keep:
                        break
                if not keep:
                    continue
            dataptr = np.PyArray_GETPTR1(arr, ientry_selected)
            for icol in range(numcol):
                thisCol = columns[icol]
                thisCV = cvarray[icol]
                nb = thisCV.write(thisCol, dataptr)
                dataptr = shift(dataptr, nb) # poorman pointer magic
            ientry_selected += 1
    finally:
        del bc
    
    # If we selected less than the num_entries entries then shrink the array
    if ientry_selected < num_entries:
        arr.resize(ientry_selected)

    return arr


def root2array_fromFname(fnames, treename, branches, entries, offset, selection):
    cdef TChain* ttree = NULL
    try:
        ttree = new TChain(treename)
        for fn in fnames:
            ttree.Add(fn)
        ret = root2array_fromTTree(
                <TTree*> ttree, branches, entries, offset, selection)
    finally:
        del ttree
    return ret


def root2array_fromCObj(tree, branches, entries, offset, selection):
    # this is not a safe method
    # provided here for convenience only
    # typecheck should be implemented for the wrapper
    if not PyCObject_Check(tree):
        raise ValueError('tree must be PyCObject')
    cdef TTree* chain = <TTree*> PyCObject_AsVoidPtr(tree)
    return root2array_fromTTree(
            chain, branches, entries, offset, selection)


@atexit.register
def cleanup():
    
    it = CONVERTERS.begin()
    while it != CONVERTERS.end():
        del deref(it).second
        inc(it)
