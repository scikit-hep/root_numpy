# cython: experimental_cpp_class_def=True
import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport array
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cpython cimport PyObject
from cpython.cobject cimport (PyCObject_AsVoidPtr,
                              PyCObject_Check,
                              PyCObject_FromVoidPtr)
from cython.operator cimport dereference as deref, preincrement as inc

from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp.pair cimport pair
from libcpp.string cimport string, const_char
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free, realloc

try:
    from collections import OrderedDict
except ImportError:
    # Fall back on drop-in
    from .extern.ordereddict import OrderedDict

import atexit
from glob import glob
import warnings
from warnings import warn
from root_numpy_warnings import RootNumpyUnconvertibleWarning

include "ROOT.pxi"
include "root_numpy.pxi"


ctypedef unsigned char unsigned_char
ctypedef unsigned short unsigned_short
ctypedef unsigned int unsigned_int
ctypedef unsigned long unsigned_long
ctypedef long long long_long
ctypedef unsigned long long unsigned_long_long

TYPES = {
    TypeName[bool]().name:               ('bool',               np.dtype(np.bool),      np.NPY_BOOL),
    TypeName[char]().name:               ('char',               np.dtype(np.int8),      np.NPY_INT8),
    TypeName[unsigned_char]().name:      ('unsigned char',      np.dtype(np.uint8),     np.NPY_UINT8),
    TypeName[short]().name:              ('short',              np.dtype(np.int16),     np.NPY_INT16),
    TypeName[unsigned_short]().name:     ('unsigned short',     np.dtype(np.uint16),    np.NPY_UINT8),
    TypeName[int]().name:                ('int',                np.dtype(np.int32),     np.NPY_INT32),
    TypeName[unsigned_int]().name:       ('unsigned int',       np.dtype(np.uint32),    np.NPY_UINT32),
    TypeName[long]().name:               ('long',               np.dtype(np.int64),     np.NPY_INT64),
    TypeName[unsigned_long]().name:      ('unsigned long',      np.dtype(np.uint64),    np.NPY_UINT64),
    TypeName[long_long]().name:          ('long long',          np.dtype(np.longlong),  np.NPY_LONGLONG),
    TypeName[unsigned_long_long]().name: ('unsigned long long', np.dtype(np.ulonglong), np.NPY_ULONGLONG),
    TypeName[float]().name:              ('float',              np.dtype(np.float32),   np.NPY_FLOAT32),
    TypeName[double]().name:             ('double',             np.dtype(np.float64),   np.NPY_FLOAT64),
}

TYPES_NUMPY2ROOT = {
    np.dtype(np.bool):      (1, 'O'),
    #np.int8 from cython means something else
    np.dtype(np.int8):      (1, 'B'),
    np.dtype(np.int16):     (2, 'S'),
    np.dtype(np.int32):     (4, 'I'),
    np.dtype(np.int64):     (8, 'L'),
    np.dtype(np.uint8):     (1, 'b'),
    np.dtype(np.uint16):    (2, 's'),
    np.dtype(np.uint32):    (4, 'i'),
    np.dtype(np.uint64):    (8, 'l'),
    np.dtype(np.float):     (8, 'D'),
    np.dtype(np.float32):   (4, 'F'),
    np.dtype(np.float64):   (8, 'D'),
}


def list_trees(fname):
    
    # globbing support
    fname = glob(fname)
    if len(fname) == 0:
        raise IOError("file not found: %s" % fname)
    fname = fname[0]

    cdef TFile* f = new TFile(fname, 'read')
    if f is NULL:
        raise IOError("cannot read: %s" % fname)

    cdef TList* keys = f.GetListOfKeys()
    if keys is NULL:
        raise IOError("not a valid root file: %s" % fname)
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

    # automatically select single tree
    if tree is None:
        tree = list_trees(fname)
        if len(tree) != 1:
            raise ValueError("multiple trees found: %s" % (', '.join(tree)))
        else:
            tree = tree[0]
    
    # globbing support
    fname = glob(fname)
    if len(fname) == 0:
        raise IOError("file not found: %s" % fname)
    fname = fname[0]

    cdef TFile* f = new TFile(fname, 'read')
    
    cdef TTree* t = <TTree*> f.Get(tree)
    if t is NULL:
        raise IOError("tree %s not found in %s" % (tree, fname))

    return parse_tree_structure(t)


def list_branches(fname, tree=None):
    return list_structures(fname, tree).keys()


cdef parse_tree_structure(TTree* tree):
    cdef char* name
    cdef TBranch* thisBranch
    cdef TLeaf* thisLeaf
    cdef TObjArray* branches = tree.GetListOfBranches()
    cdef TObjArray* leaves
    ret = OrderedDict()
    if branches is NULL:
        return ret
    for ibranch in range(branches.GetEntries()):
        thisBranch = <TBranch*>(branches.At(ibranch))
        leaves = thisBranch.GetListOfLeaves()
        if leaves is NULL:
            raise RuntimeError("branch %s has no leaves" % thisBranch.GetName())
        leaflist = []
        for ibranch in range(leaves.GetEntries()):
            thisLeaf = <TLeaf*>leaves.At(ibranch)
            lname = thisLeaf.GetName()
            # resolve Float_t -> float, vector<Float_t> -> vector<float>, ..
            ltype = <bytes>ResolveTypedef(thisLeaf.GetTypeName(), True).c_str()
            leaflist.append((lname, ltype))
        if not leaflist:
            raise RuntimeError(
                "leaf list for branch %s is empty" %
                    thisBranch.GetName())
        ret[thisBranch.GetName()] = leaflist
    return ret


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
        this.elesize = info[1].itemsize
        this.nptypecode = info[2]
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


cdef cpp_map[string, Converter*] CONVERTERS
ctypedef pair[string, Converter*] CONVERTERS_ITEM

for ctypename, (ctype, dtype, dtypecode) in TYPES.items():
    CONVERTERS.insert(CONVERTERS_ITEM(
        ctype, new BasicConverter(
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
    cdef string typename = string(col.GetTypeName())
    cdef Converter* conv
    cdef Converter* basic_conv
    if ct == SINGLE:
        return find_converter_by_typename(typename)
    elif ct == FIXED:
        conv = find_converter_by_typename(typename + '[fixed]')
        if conv == NULL:
            basic_conv = find_converter_by_typename(typename)
            if basic_conv == NULL:
                return NULL
            conv = new FixedArrayConverter(
                    <BasicConverter*>basic_conv,
                    col.countval)
            CONVERTERS.insert(CONVERTERS_ITEM(
                typename + '[fixed]', conv))
        return conv
    elif ct == VARY:
        conv = find_converter_by_typename(typename + '[vary]')
        if conv == NULL:
            basic_conv = find_converter_by_typename(typename)
            if basic_conv == NULL:
                return NULL
            conv = new VaryArrayConverter(
                    <BasicConverter*>basic_conv)
            CONVERTERS.insert(CONVERTERS_ITEM(
                typename + '[vary]', conv))
        return conv
    return NULL


cdef Converter* find_converter_by_typename(string typename):
    it = CONVERTERS.find(ResolveTypedef(typename.c_str(), True))
    if it == CONVERTERS.end():
        return NULL
    return deref(it).second


cdef np.ndarray init_array(vector[Column*]& columns,
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


cdef object tree2array(TTree* tree, branches, selection, start, stop, step):

    # This is actually vector of pointers despite how it looks
    cdef vector[Column*] columns
    cdef Column* thisCol

    # Make a better chain so we can register all columns
    cdef BetterChain* bc = new BetterChain(tree)
    cdef TTreeFormula* formula = NULL
    cdef int num_entries = bc.GetEntries()
    cdef int num_entries_selected = 0
    cdef int ientry

    # list of converter in the same order
    cdef Converter* thisCV
    cdef int numcol
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
        elif len(branches) != len(set(branches)):
            raise ValueError("duplicate branches requested")

        for branch in branches:
            try:
                leaves = structure[branch]
            except KeyError:
                raise ValueError(
                        "branch %s is not present in the tree" % branch)
            shortname = len(leaves) == 1
            for leaf, ltype in leaves:
                if CONVERTERS.find(ltype) != CONVERTERS.end():
                    colname = branch if shortname else '%s_%s' % (branch, leaf)
                    thisCol = bc.MakeColumn(branch, leaf, colname)
                    columns.push_back(thisCol)
                else:
                    warnings.warn(
                        "cannot convert leaf %s of branch %s "
                        "with type %s (skipping)" % (branch, leaf, ltype),
                        RootNumpyUnconvertibleWarning)
        
        # Now that we have all the columns we can
        # make an appropriate array structure
        arr = init_array(columns, cvarray, num_entries)
        numcol = columns.size()
        
        indices = slice(start, stop, step).indices(num_entries)
        for ientry in xrange(*indices):
            if bc.GetEntry(ientry) == 0:
                break

            # Determine if this entry passes the selection,
            # similar to the code in ROOT's tree/treeplayer/src/TTreePlayer.cxx
            if formula != NULL:
                ndata = formula.GetNdata()
                keep = False
                for current from 0 <= current < ndata:
                    keep |= (formula.EvalInstance(current) != 0)
                    if keep:
                        break
                if not keep:
                    continue

            # Copy the values into the array
            dataptr = np.PyArray_GETPTR1(arr, num_entries_selected)
            for icol in xrange(numcol):
                thisCol = columns[icol]
                thisCV = cvarray[icol]
                nb = thisCV.write(thisCol, dataptr)
                # poorman pointer magic
                dataptr = shift(dataptr, nb)
            
            # Increment number of selected entries last
            num_entries_selected += 1
    finally:
        del bc
    
    # If we selected fewer than num_entries entries then shrink the array
    if num_entries_selected < num_entries:
        arr.resize(num_entries_selected)

    return arr


def root2array_fromFname(fnames, treename, branches, selection, start, stop, step):
    cdef TChain* ttree = NULL
    try:
        ttree = new TChain(treename)
        for fn in fnames:
            ttree.Add(fn)
        ret = tree2array(
                <TTree*> ttree, branches, selection, start, stop, step)
    finally:
        del ttree
    return ret


def root2array_fromCObj(tree, branches, selection, start, stop, step):
    # this is not a safe method
    # provided here for convenience only
    # typecheck should be implemented by the wrapper
    if not PyCObject_Check(tree):
        raise ValueError('tree must be PyCObject')
    cdef TTree* chain = <TTree*> PyCObject_AsVoidPtr(tree)
    return tree2array(
            chain, branches, selection, start, stop, step)


"""
array -> TTree conversion follows:
"""

cdef cppclass NP2CConverter:
    void fill_from(void* source):
        pass


cdef cppclass ScalarNP2CConverter(NP2CConverter):
    int nbytes
    string roottype
    string name
    void* value
    TBranch* branch
    # don't use copy constructor of this one since it will screw up
    # tree binding and/or ownership of value
    __init__(TTree* tree, string name, string roottype, int nbytes):
        cdef string leaflist
        this.nbytes = nbytes
        this.roottype = roottype
        this.name = name
        this.value = malloc(nbytes)
        this.branch = tree.GetBranch(this.name.c_str())
        if this.branch == NULL:
            leaflist = this.name + '/' + this.roottype
            this.branch = tree.Branch(this.name.c_str(), this.value, leaflist.c_str())
        else:
            # check type compatibility of existing branch
            existing_type = this.branch.GetTitle().rpartition('/')[-1]
            if str(roottype) != existing_type:
                raise TypeError(
                    "field `%s` of type `%s` is not compatible "
                    "with existing branch of type `%s`" % (
                        name, roottype, existing_type))
            this.branch.SetAddress(this.value)
        this.branch.SetStatus(1)

    __del__(self): # does this do what I want?
        free(this.value)

    void fill_from(void* source):
        memcpy(this.value, source, this.nbytes)
        this.branch.Fill()


cdef NP2CConverter* find_np2c_converter(TTree* tree, name, dtype, peekvalue=None):
    # TODO:
    # np.float16: #this needs special treatment root doesn't have 16 bit float?
    # np.object #this too should detect basic numpy array
    # How to detect fixed length array?
    if dtype in TYPES_NUMPY2ROOT:
        nbytes, roottype = TYPES_NUMPY2ROOT[dtype]
        return new ScalarNP2CConverter(tree, name, roottype, nbytes)
    elif dtype == np.dtype(np.object):
        warn('Converter for %r not implemented yet. Skip.' % dtype)
        return NULL
        #lets peek
        """
        if type(peekvalue) == type(np.array([])):
            ndim = peekvalue.ndim
            dtype = peekvalue.dtype
            #TODO finish this
        """
    else:
        warn('Converter for %r not implemented yet. Skip.' % dtype)
    return NULL


cdef TTree* array2tree(np.ndarray arr, name='tree', TTree* tree=NULL) except *:
    # hmm how do I catch all python exception
    # and clean up before throwing ?
    cdef vector[NP2CConverter*] cvarray
    cdef vector[int] posarray
    cdef vector[int] roffsetarray
    cdef int icol
    cdef auto_ptr[NP2CConverter] tmp
    cdef int icv = 0
    cdef int arr_len = 0
    cdef int pos_len = 0
    cdef void* source = NULL
    cdef void* thisrow = NULL
    cdef NP2CConverter* tmpcv
    
    try: 
        if tree == NULL:
            tree = new TTree(name, name)
        
        fieldnames = arr.dtype.names
        fields = arr.dtype.fields
        
        # figure out the structure
        for icol, fieldname in enumerate(fieldnames):
            # roffset is an offset of particular field in each record
            dtype, roffset = fields[fieldname] 
            cvt = find_np2c_converter(tree, fieldname, dtype, arr[0][fieldname])
            if cvt is not NULL:
                roffsetarray.push_back(roffset)
                cvarray.push_back(cvt)
                posarray.push_back(icol)

        # fill in data
        pos_len = posarray.size()
        for idata in xrange(len(arr)):
            thisrow = np.PyArray_GETPTR1(arr, idata)
            for ipos in range(pos_len):
                roffset = roffsetarray[ipos]
                source = shift(thisrow, roffset)
                cvarray[ipos].fill_from(source)
        
        # need to update the number of entries in the tree to match
        # the number in the branches since each branch is filled separately.
        tree.SetEntries(-1)
    
    except:
        raise
    
    finally:
        # how do I clean up TTree?
        # root has some global funny memory management...
        # need to make sure no double free
        for icv in range(cvarray.size()):
            tmpcv = cvarray[icv]
            del tmpcv

    return tree


def array2tree_toCObj(arr, name='tree', tree=None):
    
    cdef TTree* intree = NULL
    cdef TTree* outtree = NULL
    if tree is not None:
        # this is not a safe method
        # provided here for convenience only
        # typecheck should be implemented by the wrapper
        if not PyCObject_Check(tree):
            raise ValueError("tree must be PyCObject")
        intree = <TTree*> PyCObject_AsVoidPtr(tree)
    outtree = array2tree(arr, name=name, tree=intree)
    return PyCObject_FromVoidPtr(outtree, NULL)


def array2root(arr, filename, treename='tree', mode='update'):
    
    cdef TFile* file = new TFile(filename, mode)
    if file is NULL:
        raise IOError("cannot open file %s" % filename)
    if not file.IsWritable():
        raise IOError("file %s is not writable" % filename)
    cdef TTree* tree = array2tree(arr, name=treename)
    tree.Write()
    file.Close()
    # how to clean up TTree? Same question as above.
    del file


@atexit.register
def cleanup():
    # delete all allocated converters 
    it = CONVERTERS.begin()
    while it != CONVERTERS.end():
        del deref(it).second
        inc(it)
