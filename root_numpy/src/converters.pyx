import re

# match leaf_name[length_leaf][N][M]... or leaf_name[N][M]...
LEAF_PATTERN = re.compile('^[^\[]+((?:\[[^\s\]]+\])(?:\[[0-9]+\])*)?$')

TYPES = {
    TypeName[bool]().name:               ('bool',               np.dtype(np.bool),      np.NPY_BOOL),
    TypeName[char]().name:               ('char',               np.dtype(np.int8),      np.NPY_INT8),
    TypeName[unsigned_char]().name:      ('unsigned char',      np.dtype(np.uint8),     np.NPY_UINT8),
    TypeName[short]().name:              ('short',              np.dtype(np.int16),     np.NPY_INT16),
    TypeName[unsigned_short]().name:     ('unsigned short',     np.dtype(np.uint16),    np.NPY_UINT16),
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
    np.dtype(np.bool):    (1, 'O'),
    np.dtype(np.int8):    (1, 'B'),
    np.dtype(np.int16):   (2, 'S'),
    np.dtype(np.int32):   (4, 'I'),
    np.dtype(np.int64):   (8, 'L'),
    np.dtype(np.uint8):   (1, 'b'),
    np.dtype(np.uint16):  (2, 's'),
    np.dtype(np.uint32):  (4, 'i'),
    np.dtype(np.uint64):  (8, 'l'),
    np.dtype(np.float):   (8, 'D'),
    np.dtype(np.float32): (4, 'F'),
    np.dtype(np.float64): (8, 'D'),
}

SPECIAL_TYPEDEFS = {
    'Long64_t': 'long long',
    'ULong64_t': 'unsigned long long',
}


cdef inline unicode resolve_type(const char* typename):
    # resolve Float_t -> float, vector<Float_t> -> vector<float>, ...
    resolvedtype = <unicode>ResolveTypedef(typename, True).c_str()
    resolvedtype = <unicode>SPECIAL_TYPEDEFS.get(resolvedtype, resolvedtype)
    return resolvedtype


cdef inline int write_array(string name,
                            void* here, void* src, int typecode,
                            unsigned long numele, int elesize,
                            int ndim=1, SIZE_t* dims=NULL,
                            Selector* selector=NULL) except -1:
    """
    create numpy array of type typecode with numele elements and size of
    each element elesize and write it to the array
    """
    cdef unsigned long i = 0, j = 0
    cdef SIZE_t* _dims = dims
    cdef SIZE_t default_dims[1]
    if dims == NULL:
        _dims = default_dims
        _dims[0] = numele;
    if selector != NULL:
        # check that lengths match
        if selector.selected.size() != <unsigned_long> _dims[0]:
            raise RuntimeError("lengths of object selection '{0}' ({1}) "
                               "and object array '{2}' ({3}) are not equal".format(
                                   selector.selection.GetTitle(), selector.selected.size(),
                                   name, _dims[0]))
        _dims[0] = selector.num_selected
    cdef np.ndarray tmp = np.PyArray_EMPTY(ndim, _dims, typecode, 0)
    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # incref since we are writing directly in the array
    Py_INCREF(tmp)
    # copy to tmp.data
    cdef unsigned long nbytes = numele * elesize
    if selector != NULL:
        # copy with selection
        for i in range(selector.selected.size()):
            if selector.selected[i]:
                memcpy(static_cast['char*'](tmp.data) + j * elesize,
                       static_cast['char*'](src) + i * elesize,
                       elesize)
                j += 1
    else:
        # quick copy
        memcpy(tmp.data, src, nbytes)
    # now write PyObject* to the array
    memcpy(here, &tmpobj, sizeof(PyObject*))
    return sizeof(tmpobj)


# special treatment for vector<bool>
cdef inline int write_array_vectorbool(string name,
                                       void* here, vector[bool]* src,
                                       Selector* selector=NULL) except -1:
    cdef unsigned long i = 0, j = 0
    cdef unsigned long numele = src.size()
    cdef SIZE_t dims[1]
    dims[0] = numele;
    if selector != NULL:
        # check that lengths match
        if selector.selected.size() != <unsigned_long> dims[0]:
            raise RuntimeError("lengths of object selection '{0}' ({1}) "
                               "and object array '{2}' ({3}) are not equal".format(
                                   selector.selection.GetTitle(), selector.selected.size(),
                                   name, dims[0]))
        dims[0] = selector.num_selected
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, np.NPY_BOOL, 0)
    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # incref since we are writing directly in the array
    Py_INCREF(tmp)
    if selector != NULL:
        # copy with selection
        for i in range(selector.selected.size()):
            if selector.selected[i]:
                tmp[j] = deref(src)[i]
                j += 1
    else:
        # can't use memcpy here...
        for i in range(numele):
            tmp[i] = deref(src)[i]
    # now write PyObject* to the array
    memcpy(here, &tmpobj, sizeof(PyObject*))
    return sizeof(tmpobj)


cdef inline int write_array_vectorstring(string name,
                                         void* here, vector[string]* src,
                                         Selector* selector=NULL) except -1:
    cdef unsigned long i = 0, j = 0
    cdef unsigned long numele = src.size()
    cdef SIZE_t dims[1]
    dims[0] = numele;
    if selector != NULL:
        # check that lengths match
        if selector.selected.size() != <unsigned_long> dims[0]:
            raise RuntimeError("lengths of object selection '{0}' ({1}) "
                               "and object array '{2}' ({3}) are not equal".format(
                                   selector.selection.GetTitle(), selector.selected.size(),
                                   name, dims[0]))
        dims[0] = selector.num_selected
    cdef int objsize = np.dtype('O').itemsize
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, np.NPY_OBJECT, 0)
    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # incref since we are writing directly in the array
    Py_INCREF(tmp)
    cdef PyObject* tmpstrobj
    cdef char* dataptr = <char*> tmp.data
    if selector != NULL:
        # copy with selection
        for i in range(selector.selected.size()):
            if selector.selected[i]:
                py_bytes = str(deref(src)[i])
                Py_INCREF(py_bytes)
                tmpstrobj = <PyObject*> py_bytes
                memcpy(&dataptr[j*objsize], &tmpstrobj, sizeof(PyObject*))
                j += 1
    else:
        # can't use memcpy here...
        for i in range(numele):
            py_bytes = str(deref(src)[i])
            Py_INCREF(py_bytes)
            tmpstrobj = <PyObject*> py_bytes
            memcpy(&dataptr[i*objsize], &tmpstrobj, sizeof(PyObject*))
    # now write PyObject* to the array
    memcpy(here, &tmpobj, sizeof(PyObject*))
    return sizeof(tmpobj)


cdef cppclass Converter:

    int write(Column* col, void* here) except -1:
        pass

    object get_nptype(Column* col):
        pass

    int get_nptypecode():
        pass


cdef cppclass BasicConverter(Converter):
    int size
    int nptypecode
    string nptype

    __init__(int size, string nptype, int nptypecode):
        this.size = size
        this.nptypecode = nptypecode
        this.nptype = nptype

    int write(Column* col, void* here) except -1:
        cdef void* src = col.GetValuePointer()
        memcpy(here, src, this.size)
        return this.size

    object get_nptype(Column* col):
        return np.dtype(this.nptype)

    int get_nptypecode():
        return this.nptypecode


cdef cppclass ObjectConverterBase(Converter):

    object get_nptype(Column* col):
        return np.object

    int get_nptypecode():
        return np.NPY_OBJECT


cdef cppclass VaryArrayConverter(ObjectConverterBase):
    BasicConverter* conv # converter for single element
    SIZE_t* dims
    int ndim
    int typecode
    int elesize

    __init__(BasicConverter* conv, int ndim, SIZE_t* dims):
        this.conv = conv
        this.dims = dims
        this.ndim = ndim
        this.typecode = conv.get_nptypecode()
        this.elesize = conv.size

    __dealloc__():
        free(this.dims)

    int write(Column* col, void* here) except -1:
        # only the first dimension can vary in length
        this.dims[0] = col.GetCountLen()
        return write_array(col.name, here, col.GetValuePointer(),
                           this.typecode, col.GetLen(), this.elesize,
                           this.ndim, this.dims, col.selector)

    object get_nptype(Column* col):
        if col.max_length == 1:
            # Single value
            return np.dtype(this.conv.nptype)
        elif col.max_length:
            # Truncated
            return (np.dtype(this.conv.nptype), col.max_length)
        # Pointer to array
        return np.object


cdef cppclass FixedArrayConverter(Converter):
    BasicConverter* conv # converter for single element
    PyObject* shape

    __init__(BasicConverter* conv, PyObject* shape):
        Py_INCREF(<object> shape)
        this.conv = conv
        this.shape = shape

    __dealloc__():
        Py_XDECREF(this.shape)

    int write(Column* col, void* here) except -1:
        cdef int nbytes = col.GetSize()
        memcpy(here, col.GetValuePointer(), nbytes)
        return nbytes

    object get_nptype(Column* col):
        return (np.dtype(this.conv.nptype), <object> this.shape)

    int get_nptypecode():
        return this.conv.nptypecode


cdef cppclass CharArrayConverter(Converter):
    BasicConverter* conv # converter for single element
    int size

    __init__(int size):
        this.conv = <BasicConverter*> CONVERTERS['char']
        this.size = size

    int write(Column* col, void* here) except -1:
        cdef int nbytes = col.GetSize() - sizeof(char)  # exclude null-termination
        cdef int length = strlen(<char*> col.GetValuePointer())
        memcpy(here, col.GetValuePointer(), nbytes)
        if length < nbytes:
            memset((<char*> here) + length, '\0', nbytes - length)
        return nbytes

    object get_nptype(Column* col):
        return 'S{0:d}'.format(this.size)

    int get_nptypecode():
        return this.conv.nptypecode


cdef cppclass VectorConverter[T](ObjectConverterBase):
    int elesize
    int nptypecode
    Vector2Array[T] v2a

    __init__():
        cdef TypeName[T] ast = TypeName[T]()
        info = TYPES[ast.name]
        this.elesize = info[1].itemsize
        this.nptypecode = info[2]

    int write(Column* col, void* here) except -1:
        cdef vector[T]* tmp = <vector[T]*> col.GetValuePointer()
        cdef unsigned long numele = tmp.size()
        # check cython auto-generated code
        # if it really does &((*tmp)[0])
        cdef T* fa = this.v2a.convert(tmp)
        return write_array(col.name, here, fa, this.nptypecode,
                           numele, this.elesize, 1, NULL,
                           col.selector)


cdef cppclass VectorVectorConverter[T](ObjectConverterBase):
    int elesize
    int nptypecode
    Vector2Array[T] v2a

    __init__():
        cdef TypeName[T] ast = TypeName[T]()
        info = TYPES[ast.name]
        this.elesize = info[1].itemsize
        this.nptypecode = info[2]

    int write(Column* col, void* here) except -1:
        cdef vector[vector[T]]* tmp = <vector[vector[T]]*> col.GetValuePointer()
        # this will hold number of subvectors
        cdef unsigned long numele
        cdef T* fa
        # these are defined solely for the outer array wrapper
        cdef int objsize = np.dtype('O').itemsize
        cdef int objtypecode = np.NPY_OBJECT
        numele = tmp[0].size()
        # create an outer array container that dataptr points to,
        # containing pointers from write_array().
        # define an (numele)-dimensional outer array to hold our subvectors fa
        cdef SIZE_t dims[1]
        dims[0] = numele
        cdef np.ndarray outer = np.PyArray_EMPTY(1, dims, objtypecode, 0)
        cdef PyObject* outerobj = <PyObject*> outer # borrow ref
        # increase one since we are writing directly in the array
        Py_INCREF(outer)
        # now write PyObject* to the array
        memcpy(here, &outerobj, sizeof(PyObject*))
        # build a dataptr pointing to outer, so we can shift and write each
        # of the subvectors
        cdef char* dataptr = <char*> outer.data
        # loop through all subvectors
        cdef unsigned long i
        for i in range(numele):
            fa = this.v2a.convert(&tmp[0][i])
            write_array(col.name, &dataptr[i*objsize], fa, this.nptypecode,
                        tmp[0][i].size(), this.elesize)
        return sizeof(outerobj)


cdef cppclass VectorBoolConverter(ObjectConverterBase):
    # Requires special treament since vector<bool> stores contents as bits...
    int write(Column* col, void* here) except -1:
        cdef vector[bool]* tmp = <vector[bool]*> col.GetValuePointer()
        return write_array_vectorbool(col.name, here, tmp, col.selector)


cdef cppclass VectorVectorBoolConverter(ObjectConverterBase):
    # Requires special treament since vector<bool> stores contents as bits...
    int write(Column* col, void* here) except -1:
        cdef vector[vector[bool]]* tmp = <vector[vector[bool]]*> col.GetValuePointer()
        # this will hold number of subvectors
        cdef unsigned long numele
        # these are defined solely for the outer array wrapper
        cdef int objsize = np.dtype('O').itemsize
        cdef int objtypecode = np.NPY_OBJECT
        numele = tmp[0].size()
        # create an outer array container that dataptr points to,
        # containing pointers from write_array().
        # define an (numele)-dimensional outer array to hold our subvectors fa
        cdef SIZE_t dims[1]
        dims[0] = numele
        cdef np.ndarray outer = np.PyArray_EMPTY(1, dims, objtypecode, 0)
        cdef PyObject* outerobj = <PyObject*> outer # borrow ref
        # increase one since we are writing directly in the array
        Py_INCREF(outer)
        # now write PyObject* to the array
        memcpy(here, &outerobj, sizeof(PyObject*))
        # build a dataptr pointing to outer, so we can shift and write each
        # of the subvectors
        cdef char* dataptr = <char*> outer.data
        # loop through all subvectors
        cdef unsigned long i
        for i in range(numele):
            write_array_vectorbool(col.name, &dataptr[i*objsize], &tmp[0][i])
        return sizeof(outerobj)


cdef cppclass StringConverter(ObjectConverterBase):
    int write(Column* col, void* here) except -1:
        cdef string* s = <string*> col.GetValuePointer()
        py_bytes = str(s[0])
        cdef PyObject* tmpobj = <PyObject*> py_bytes # borrow ref
        # increase one since we are writing directly in the array
        Py_INCREF(py_bytes)
        # now write PyObject* to the array
        memcpy(here, &tmpobj, sizeof(PyObject*))
        return sizeof(tmpobj)


cdef cppclass VectorStringConverter(ObjectConverterBase):
    int write(Column* col, void* here) except -1:
        cdef vector[string]* tmp = <vector[string]*> col.GetValuePointer()
        return write_array_vectorstring(col.name, here, tmp, col.selector)


cdef cppclass VectorVectorStringConverter(ObjectConverterBase):
    int write(Column* col, void* here) except -1:
        cdef vector[vector[string]]* tmp = <vector[vector[string]]*> col.GetValuePointer()
        # this will hold number of subvectors
        cdef unsigned long numele
        # these are defined solely for the outer array wrapper
        cdef int objsize = np.dtype('O').itemsize
        cdef int objtypecode = np.NPY_OBJECT
        numele = tmp[0].size()
        # create an outer array container that dataptr points to,
        # containing pointers from write_array().
        # define an (numele)-dimensional outer array to hold our subvectors fa
        cdef SIZE_t dims[1]
        dims[0] = numele
        cdef np.ndarray outer = np.PyArray_EMPTY(1, dims, objtypecode, 0)
        cdef PyObject* outerobj = <PyObject*> outer # borrow ref
        # increase one since we are writing directly in the array
        Py_INCREF(outer)
        # now write PyObject* to the array
        memcpy(here, &outerobj, sizeof(PyObject*))
        # build a dataptr pointing to outer, so we can shift and write each
        # of the subvectors
        cdef char* dataptr = <char*> outer.data
        # loop through all subvectors
        cdef unsigned long i
        for i in range(numele):
            write_array_vectorstring(col.name, &dataptr[i*objsize], &tmp[0][i])
        return sizeof(outerobj)


ctypedef cpp_map[string, Converter*] CONVERTERS_TYPE
ctypedef pair[string, Converter*] CONVERTERS_ITEM_TYPE
cdef CONVERTERS_TYPE CONVERTERS

# basic type converters
for ctypename, (ctype, dtype, dtypecode) in TYPES.items():
    CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
        ctype, new BasicConverter(
            dtype.itemsize, dtype.name, dtypecode)))

# vector<> converters
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<bool>', new VectorBoolConverter()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<char>', new VectorConverter[char]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<unsigned char>', new VectorConverter[unsigned_char]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<short>', new VectorConverter[short]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<unsigned short>', new VectorConverter[unsigned_short]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<int>', new VectorConverter[int]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<unsigned int>', new VectorConverter[unsigned_int]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<long>', new VectorConverter[long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<unsigned long>', new VectorConverter[unsigned_long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<long long>', new VectorConverter[long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<unsigned long long>', new VectorConverter[unsigned_long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<float>', new VectorConverter[float]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<double>', new VectorConverter[double]()))

# vector<vector<> > converters
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<bool> >', new VectorVectorBoolConverter()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<char> >', new VectorVectorConverter[char]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<unsigned char> >', new VectorVectorConverter[unsigned_char]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<short> >', new VectorVectorConverter[short]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<unsigned short> >', new VectorVectorConverter[unsigned_short]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<int> >', new VectorVectorConverter[int]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<unsigned int> >', new VectorVectorConverter[unsigned_int]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<long> >', new VectorVectorConverter[long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<unsigned long> >', new VectorVectorConverter[unsigned_long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<long long> >', new VectorVectorConverter[long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<unsigned long long> >', new VectorVectorConverter[unsigned_long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<float> >', new VectorVectorConverter[float]()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<double> >', new VectorVectorConverter[double]()))

# string converters
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'string', new StringConverter()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<string>', new VectorStringConverter()))
CONVERTERS.insert(CONVERTERS_ITEM_TYPE(
    'vector<vector<string> >', new VectorVectorStringConverter()))


cdef Converter* find_converter_by_typename(string typename):
    cdef cpp_map[string, Converter*].iterator it = CONVERTERS.find(typename)
    if it == CONVERTERS.end():
        return NULL
    return deref(it).second


cdef Converter* get_array_converter(string typename, arraydef):
    # Determine shape ignoring possible variable first dimension
    arraytokens = arraydef.strip('[]')
    if arraytokens:
        arraytokens = arraytokens.split('][')
    shape = tuple([int(token) for token in arraytokens])

    # Variable-length array
    if arraydef.startswith('[]'):
        conv = find_converter_by_typename(typename + arraydef)
        if conv == NULL:
            # Create new converter on demand
            basic_conv = find_converter_by_typename(typename)
            if basic_conv == NULL:
                return NULL
            # the variable-length dimension is excluded from leaf_shape above
            # so add 1 here:
            ndim = len(shape) + 1
            dims = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
            if dims == NULL:
                raise MemoryError("could not allocate %d bytes" % (ndim * sizeof(SIZE_t)))
            # only the first dimension can vary in length
            # so dims[0] is set dynamically for each entry
            for idim from 1 <= idim < ndim:
                dims[idim] = shape[idim - 1]
            conv = new VaryArrayConverter(
                <BasicConverter*> basic_conv, ndim, dims)
            CONVERTERS.insert(CONVERTERS_ITEM_TYPE(typename + arraydef, conv))
        return conv

    # Fixed-length array
    conv = find_converter_by_typename(typename + arraydef)
    if conv == NULL:
        # Create new converter on demand
        basic_conv = find_converter_by_typename(typename)
        if basic_conv == NULL:
            return NULL
        conv = new FixedArrayConverter(
            <BasicConverter*> basic_conv, <PyObject*> shape)
        CONVERTERS.insert(CONVERTERS_ITEM_TYPE(typename + arraydef, conv))
    return conv


cdef Converter* get_converter(TLeaf* leaf, char type_code='\0') except *:
    # Find existing converter or attempt to create a new one
    cdef Converter* conv
    cdef Converter* basic_conv
    cdef TLeaf* leaf_count = leaf.GetLeafCount()
    cdef SIZE_t* dims
    cdef int ndim, idim, leaf_length

    leaf_name = leaf.GetName()
    leaf_title = leaf.GetTitle()
    leaf_type = resolve_type(leaf.GetTypeName())

    # Special case for null-terminated char array string
    if type_code == 'C':
        leaf_length = leaf.GetLenStatic()
        conv = find_converter_by_typename(leaf_type + '[{0:d}]/C'.format(leaf_length))
        if conv == NULL:
            conv = new CharArrayConverter(leaf_length - 1)  # exclude null-termination
            CONVERTERS.insert(CONVERTERS_ITEM_TYPE(leaf_type + '[{0:d}]/C'.format(leaf_length), conv))
        return conv

    match = re.match(LEAF_PATTERN, leaf_title)
    if match is not None:
        arraydef = match.group(1)
        if arraydef is not None:
            if leaf_count != NULL:
                # Ignore length-leaf name and use [] to denote variable-length first dimension
                arraydef = '[' + arraydef[arraydef.find(']'):]
            return get_array_converter(leaf_type, arraydef)
    return find_converter_by_typename(leaf_type)


@atexit.register
def cleanup():
    # Delete all converters when module is town down
    cdef cpp_map[string, Converter*].iterator it = CONVERTERS.begin()
    while it != CONVERTERS.end():
        del deref(it).second
        inc(it)


####################################
# array -> TTree conversion follows:
####################################

cdef cppclass NP2ROOTConverter:

    void fill_from(void* source):
        pass


cdef cppclass FixedNP2ROOTConverter(NP2ROOTConverter):
    int nbytes
    void* value
    TBranch* branch

    __init__(TTree* tree, string name, string roottype,
             int length, int elembytes,
             int ndim=0, SIZE_t* dims=NULL):
        cdef string leaflist
        cdef int axis
        this.nbytes = length * elembytes
        if roottype.compare('C') == 0:
            # include null-termination
            this.value = malloc(nbytes + 1)
            if this.value == NULL:
                raise MemoryError("could not allocate %d bytes" % (nbytes + 1))
            (<char*> this.value)[nbytes] = '\0'
        else:
            this.value = malloc(nbytes)
            if this.value == NULL:
                raise MemoryError("could not allocate %d bytes" % nbytes)
        # Construct leaflist name
        leaflist = name
        if ndim > 0 and roottype.compare('C') != 0:
            for axis in range(ndim):
                token = ('[{0:d}]'.format(dims[axis])).encode('utf-8')
                leaflist.append(<char*> token)
        leaflist.append(b'/')
        leaflist.append(roottype)
        this.branch = tree.GetBranch(name.c_str())
        if this.branch == NULL:
            this.branch = tree.Branch(name.c_str(), this.value, leaflist.c_str())
        else:
            # check type compatibility of existing branch
            if leaflist.compare(string(this.branch.GetTitle())) != 0:
                raise TypeError(
                    "field '{0}' of type '{1}' is not compatible "
                    "with existing branch of type '{2}'".format(
                        name, leaflist, str(this.branch.GetTitle())))
            this.branch.SetAddress(this.value)
        this.branch.SetStatus(1)

    __dealloc__():
        free(this.value)

    void fill_from(void* source):
        memcpy(this.value, source, this.nbytes)
        this.branch.Fill()


cdef NP2ROOTConverter* find_np2root_converter(TTree* tree, name, dtype):
    # TODO:
    # np.float16 needs special treatment. ROOT doesn't support 16-bit floats.
    # Handle np.object (array) columns
    cdef NP2ROOTConverter* conv = NULL
    cdef int axis, ndim = 0
    cdef int length = 1
    cdef SIZE_t* dims = NULL
    subdtype = dtype.subdtype
    if subdtype is not None:
        # Fixed-size subarray type
        dtype, shape = subdtype
        ndim = len(shape)
        dims = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
        if dims == NULL:
            raise MemoryError("could not allocate %d bytes" % (ndim * sizeof(SIZE_t)))
        for axis in range(ndim):
            dims[axis] = shape[axis]
            length *= dims[axis]
    if dtype in TYPES_NUMPY2ROOT:
        elembytes, roottype = TYPES_NUMPY2ROOT[dtype]
        conv = new FixedNP2ROOTConverter(tree, name, roottype, length, elembytes, ndim, dims)
    elif dtype.kind == 'S':
        conv = new FixedNP2ROOTConverter(tree, name, 'C', dtype.itemsize, 1)
    free(dims)
    return conv
