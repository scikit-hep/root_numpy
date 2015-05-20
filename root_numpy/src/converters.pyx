import re

# match leaf_name[length_leaf][N][M]...
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
    np.dtype(np.bool):      (1, 'O'),
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

SPECIAL_TYPEDEFS = {
    'Long64_t': 'long long',
    'ULong64_t': 'unsigned long long',
}


cdef inline unicode resolve_type(const char* typename):
    # resolve Float_t -> float, vector<Float_t> -> vector<float>, ...
    resolvedtype = <unicode>ResolveTypedef(typename, True).c_str()
    resolvedtype = <unicode>SPECIAL_TYPEDEFS.get(resolvedtype, resolvedtype)
    return resolvedtype


# create numpy array of given type code with
# given numelement and size of each element
# and write it to buffer
cdef inline int create_numpyarray(void* buffer, void* src, int typecode,
                                  unsigned long numele, int elesize,
                                  int ndim = 1, np.npy_intp* dims = NULL):
    cdef np.npy_intp* _dims = dims
    cdef np.npy_intp default_dims[1]
    if dims == NULL:
        _dims = default_dims
        _dims[0] = numele;
    cdef np.ndarray tmp = np.PyArray_EMPTY(ndim, _dims, typecode, 0)
    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # incref since we are placing in buffer directly
    Py_INCREF(tmp)
    # copy to tmp.data
    cdef unsigned long nbytes = numele * elesize
    memcpy(tmp.data, src, nbytes)
    # now write PyObject* to buffer
    memcpy(buffer, &tmpobj, sizeof(PyObject*))
    return sizeof(tmpobj)


# special treatment for vector<bool>
cdef inline int create_numpyarray_vectorbool(void* buffer, vector[bool]* src):
    cdef unsigned long numele = src.size()
    cdef np.npy_intp dims[1]
    dims[0] = numele;
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, np.NPY_BOOL, 0)
    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # incref since we are placing in buffer directly
    Py_INCREF(tmp)
    # can't use memcpy here...
    cdef unsigned long i
    for i in range(numele):
        tmp[i] = src.at(i)
    # now write PyObject* to buffer
    memcpy(buffer, &tmpobj, sizeof(PyObject*))
    return sizeof(tmpobj)


cdef inline int create_numpyarray_vectorstring(void* buffer, vector[string]* src):
    cdef unsigned long numele = src.size()
    cdef np.npy_intp dims[1]
    dims[0] = numele;
    cdef int objsize = np.dtype('O').itemsize
    cdef np.ndarray tmp = np.PyArray_EMPTY(1, dims, np.NPY_OBJECT, 0)
    cdef PyObject* tmpobj = <PyObject*> tmp # borrow ref
    # incref since we are placing in buffer directly
    Py_INCREF(tmp)
    cdef PyObject* tmpstrobj
    cdef char* dataptr = <char*> tmp.data
    # can't use memcpy here...
    cdef unsigned long i
    for i in range(numele):
        py_bytes = str(src.at(i))
        Py_INCREF(py_bytes)
        tmpstrobj = <PyObject*> py_bytes
        memcpy(&dataptr[i*objsize], &tmpstrobj, sizeof(PyObject*))
    # now write PyObject* to buffer
    memcpy(buffer, &tmpobj, sizeof(PyObject*))
    return sizeof(tmpobj)


cdef cppclass Converter:

    int write(Column* col, void* buffer):
        pass

    object get_nptype():
        pass


cdef cppclass BasicConverter(Converter):
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


cdef cppclass ObjectConverterBase(Converter):

    object get_nptype():
        return np.object

    object get_nptypecode():
        return np.NPY_OBJECT


cdef cppclass VaryArrayConverter(ObjectConverterBase):
    BasicConverter* conv # converter for single element
    np.npy_intp* dims
    int ndim
    int typecode
    int elesize

    __init__(BasicConverter* conv, int ndim, np.npy_intp* dims):
        this.conv = conv
        this.dims = dims
        this.ndim = ndim
        this.typecode = conv.get_nptypecode()
        this.elesize = conv.size

    __dealloc__():
        free(this.dims)

    int write(Column* col, void* buffer):
        this.dims[0] = col.GetCountLen()
        return create_numpyarray(buffer, col.GetValuePointer(),
                                 this.typecode, col.GetLen(), this.elesize,
                                 this.ndim, this.dims)


cdef cppclass FixedArrayConverter(Converter):
    BasicConverter* conv # converter for single element
    PyObject* shape

    __init__(BasicConverter* conv, PyObject* shape):
        Py_INCREF(<object> shape)
        this.conv = conv
        this.shape = shape

    __dealloc__():
        Py_XDECREF(this.shape)

    int write(Column* col, void* buffer):
        cdef int nbytes = col.GetSize()
        memcpy(buffer, col.GetValuePointer(), nbytes)
        return nbytes

    object get_nptype():
        return (np.dtype(this.conv.nptype), <object> this.shape)

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

    int write(Column* col, void* buffer):
        cdef vector[T]* tmp = <vector[T]*> col.GetValuePointer()
        cdef unsigned long numele = tmp.size()
        # check cython auto-generated code
        # if it really does &((*tmp)[0])
        cdef T* fa = this.v2a.convert(tmp)
        return create_numpyarray(buffer, fa, this.nptypecode, numele, this.elesize)


cdef cppclass VectorVectorConverter[T](ObjectConverterBase):
    int elesize
    int nptypecode
    Vector2Array[T] v2a

    __init__():
        cdef TypeName[T] ast = TypeName[T]()
        info = TYPES[ast.name]
        this.elesize = info[1].itemsize
        this.nptypecode = info[2]

    int write(Column* col, void* buffer):
        cdef vector[vector[T]]* tmp = <vector[vector[T]]*> col.GetValuePointer()
        # this will hold number of subvectors
        cdef unsigned long numele
        cdef T* fa
        # these are defined solely for the outer array wrapper
        cdef int objsize = np.dtype('O').itemsize
        cdef int objtypecode = np.NPY_OBJECT
        numele = tmp[0].size()
        # create an outer array container that dataptr points to,
        # containing pointers from create_numpyarray().
        # define an (numele)-dimensional outer array to hold our subvectors fa
        cdef np.npy_intp dims[1]
        dims[0] = numele
        cdef np.ndarray outer = np.PyArray_EMPTY(1, dims, objtypecode, 0)
        cdef PyObject* outerobj = <PyObject*> outer # borrow ref
        # increase one since we are putting in buffer directly
        Py_INCREF(outer)
        # now write PyObject* to buffer
        memcpy(buffer, &outerobj, sizeof(PyObject*))
        # build a dataptr pointing to outer, so we can shift and write each
        # of the subvectors
        cdef char* dataptr = <char*> outer.data
        # loop through all subvectors
        cdef unsigned long i
        for i in range(numele):
            fa = this.v2a.convert(&tmp[0][i])
            create_numpyarray(&dataptr[i*objsize], fa, this.nptypecode,
                              tmp[0][i].size(), this.elesize)
        return sizeof(outerobj)


cdef cppclass VectorBoolConverter(ObjectConverterBase):
    # Requires special treament since vector<bool> stores contents as bits...
    int write(Column* col, void* buffer):
        cdef vector[bool]* tmp = <vector[bool]*> col.GetValuePointer()
        return create_numpyarray_vectorbool(buffer, tmp)


cdef cppclass VectorVectorBoolConverter(ObjectConverterBase):
    # Requires special treament since vector<bool> stores contents as bits...
    int write(Column* col, void* buffer):
        cdef vector[vector[bool]]* tmp = <vector[vector[bool]]*> col.GetValuePointer()
        # this will hold number of subvectors
        cdef unsigned long numele
        # these are defined solely for the outer array wrapper
        cdef int objsize = np.dtype('O').itemsize
        cdef int objtypecode = np.NPY_OBJECT
        numele = tmp[0].size()
        # create an outer array container that dataptr points to,
        # containing pointers from create_numpyarray().
        # define an (numele)-dimensional outer array to hold our subvectors fa
        cdef np.npy_intp dims[1]
        dims[0] = numele
        cdef np.ndarray outer = np.PyArray_EMPTY(1, dims, objtypecode, 0)
        cdef PyObject* outerobj = <PyObject*> outer # borrow ref
        # increase one since we are putting in buffer directly
        Py_INCREF(outer)
        # now write PyObject* to buffer
        memcpy(buffer, &outerobj, sizeof(PyObject*))
        # build a dataptr pointing to outer, so we can shift and write each
        # of the subvectors
        cdef char* dataptr = <char*> outer.data
        # loop through all subvectors
        cdef unsigned long i
        for i in range(numele):
            create_numpyarray_vectorbool(&dataptr[i*objsize], &tmp[0][i])
        return sizeof(outerobj)


cdef cppclass StringConverter(ObjectConverterBase):
    int write(Column* col, void* buffer):
        cdef string* s = <string*> col.GetValuePointer()
        py_bytes = str(s[0])
        cdef PyObject* tmpobj = <PyObject*> py_bytes # borrow ref
        # increase one since we are putting in buffer directly
        Py_INCREF(py_bytes)
        # now write PyObject* to buffer
        memcpy(buffer, &tmpobj, sizeof(PyObject*))
        return sizeof(tmpobj)


cdef cppclass VectorStringConverter(ObjectConverterBase):
    int write(Column* col, void* buffer):
        cdef vector[string]* tmp = <vector[string]*> col.GetValuePointer()
        return create_numpyarray_vectorstring(buffer, tmp)


cdef cppclass VectorVectorStringConverter(ObjectConverterBase):
    int write(Column* col, void* buffer):
        cdef vector[vector[string]]* tmp = <vector[vector[string]]*> col.GetValuePointer()
        # this will hold number of subvectors
        cdef unsigned long numele
        # these are defined solely for the outer array wrapper
        cdef int objsize = np.dtype('O').itemsize
        cdef int objtypecode = np.NPY_OBJECT
        numele = tmp[0].size()
        # create an outer array container that dataptr points to,
        # containing pointers from create_numpyarray().
        # define an (numele)-dimensional outer array to hold our subvectors fa
        cdef np.npy_intp dims[1]
        dims[0] = numele
        cdef np.ndarray outer = np.PyArray_EMPTY(1, dims, objtypecode, 0)
        cdef PyObject* outerobj = <PyObject*> outer # borrow ref
        # increase one since we are putting in buffer directly
        Py_INCREF(outer)
        # now write PyObject* to buffer
        memcpy(buffer, &outerobj, sizeof(PyObject*))
        # build a dataptr pointing to outer, so we can shift and write each
        # of the subvectors
        cdef char* dataptr = <char*> outer.data
        # loop through all subvectors
        cdef unsigned long i
        for i in range(numele):
            create_numpyarray_vectorstring(&dataptr[i*objsize], &tmp[0][i])
        return sizeof(outerobj)


cdef cpp_map[string, Converter*] CONVERTERS
ctypedef pair[string, Converter*] CONVERTERS_ITEM

# basic type converters
for ctypename, (ctype, dtype, dtypecode) in TYPES.items():
    CONVERTERS.insert(CONVERTERS_ITEM(
        ctype, new BasicConverter(
            dtype.itemsize, dtype.name, dtypecode)))

# vector<> converters
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
    'vector<long long>', new VectorConverter[long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<unsigned long long>', new VectorConverter[unsigned_long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<float>', new VectorConverter[float]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<double>', new VectorConverter[double]()))

# vector<vector<> > converters
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<bool> >', new VectorVectorBoolConverter()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<char> >', new VectorVectorConverter[char]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<unsigned char> >', new VectorVectorConverter[unsigned_char]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<short> >', new VectorVectorConverter[short]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<unsigned short> >', new VectorVectorConverter[unsigned_short]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<int> >', new VectorVectorConverter[int]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<unsigned int> >', new VectorVectorConverter[unsigned_int]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<long> >', new VectorVectorConverter[long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<unsigned long> >', new VectorVectorConverter[unsigned_long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<long long> >', new VectorVectorConverter[long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<unsigned long long> >', new VectorVectorConverter[unsigned_long_long]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<float> >', new VectorVectorConverter[float]()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<double> >', new VectorVectorConverter[double]()))

# string converters
CONVERTERS.insert(CONVERTERS_ITEM(
    'string', new StringConverter()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<string>', new VectorStringConverter()))
CONVERTERS.insert(CONVERTERS_ITEM(
    'vector<vector<string> >', new VectorVectorStringConverter()))


cdef Converter* find_converter_by_typename(string typename):
    it = CONVERTERS.find(typename)
    if it == CONVERTERS.end():
        return NULL
    return deref(it).second


cdef enum LeafShapeType:
    SINGLE_VALUE = 1
    VARIABLE_LENGTH_ARRAY = 2
    FIXED_LENGTH_ARRAY = 3


cdef Converter* get_converter(TLeaf* leaf):
    # Find existing converter or attempt to create a new one
    cdef Converter* conv
    cdef Converter* basic_conv
    cdef TLeaf* leaf_count = leaf.GetLeafCount()
    cdef LeafShapeType leaf_shape_type = SINGLE_VALUE
    cdef SIZE_t* dims
    cdef int ndim, idim

    leaf_name = leaf.GetName()
    leaf_title = leaf.GetTitle()
    leaf_type = resolve_type(leaf.GetTypeName())
    leaf_shape = ()

    # Determine shape of this leaf
    match = re.match(LEAF_PATTERN, leaf_title)
    if match is not None:
        arraydef = match.group(1)
        if arraydef is not None:
            arraytokens = arraydef.strip('[]').split('][')
            shape = []
            # First group might be the name of the length-leaf
            if leaf_count != NULL:
                # Ignore leaf name token
                arraytokens.pop(0)
                leaf_shape_type = VARIABLE_LENGTH_ARRAY
            else:
                leaf_shape_type = FIXED_LENGTH_ARRAY
            shape.extend([int(token) for token in arraytokens])
            leaf_shape = tuple(shape)

    if leaf_shape_type == SINGLE_VALUE:
        return find_converter_by_typename(leaf_type)

    if leaf_shape_type == VARIABLE_LENGTH_ARRAY:
        conv = find_converter_by_typename(leaf_type + arraydef)
        if conv == NULL:
            # Create new converter on demand
            basic_conv = find_converter_by_typename(leaf_type)
            if basic_conv == NULL:
                return NULL
            ndim = len(leaf_shape) + 1
            dims = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
            for idim from 1 <= idim < ndim:
                dims[idim] = leaf_shape[idim - 1]
            conv = new VaryArrayConverter(
                <BasicConverter*> basic_conv, ndim, dims)
            CONVERTERS.insert(CONVERTERS_ITEM(leaf_type + arraydef, conv))
        return conv

    # Fixed-length array
    conv = find_converter_by_typename(leaf_type + arraydef)
    if conv == NULL:
        # Create new converter on demand
        basic_conv = find_converter_by_typename(leaf_type)
        if basic_conv == NULL:
            return NULL
        conv = new FixedArrayConverter(
            <BasicConverter*> basic_conv, <PyObject*> leaf_shape)
        CONVERTERS.insert(CONVERTERS_ITEM(leaf_type + arraydef, conv))
    return conv


@atexit.register
def cleanup():
    # Delete all converters when module is town down
    it = CONVERTERS.begin()
    while it != CONVERTERS.end():
        del deref(it).second
        inc(it)
