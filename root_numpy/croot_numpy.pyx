from cython.operator cimport dereference as deref, preincrement as inc
#from cpython.version cimport PY_VERSION_HEX
cdef void emit_ifdef "#if defined(_WIN32) //" ()
cdef void emit_else  "#else //" ()
cdef void emit_endif "#endif //" ()
cdef extern from *:
    cdef int PY_VERSION_HEX
#DEF HAVE_COBJ = ( (PY_VERSION_HEX <  0x03020000) )
#DEF HAVE_CAPSULE = ( ((PY_VERSION_HEX >=  0x02070000) && (PY_VERSION_HEX <  0x03000000)) || (PY_VERSION_HEX >=  0x03010000) )

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)
cdef extern from "TObject.h":
    cdef cppclass TObject:
        char* ClassName()
cdef extern from "TObjArray.h":
    cdef cppclass TObjArray:
        TObject* At(int)
        int GetEntries()
cdef extern from "TBranch.h":
    cdef cppclass TBranch:
        TObjArray* GetListOfLeaves()
        char* GetName()
        int GetEntry(int)
cdef extern from "TLeaf.h":
    cdef cppclass TLeaf:
        char* GetTypeName()
        TLeaf* GetLeafCounter(int&)

cdef extern from "TFile.h":
    cdef cppclass TFile:
        TFile(char*)
        TFile(char*, char*)
        int IsZombie()
        void Print()
cdef extern from "TTree.h":
    cdef cppclass TTree:
        void GetEntry(int i)
        int GetEntries()
        void SetBranchAddress(char* bname,void* addr)
        TObjArray* GetListOfBranches()
cdef extern from "TChain.h":
    cdef cppclass TChain(TTree):
        TChain(char*)
        int Add(char*)
        void Print()
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()

cdef void* get_ptr(object o):
    emit_ifdef
    emit_endif
    return NULL
root_type_map = {
    "Char_t":("i1",1),
    "UChar_t":("u1",1),
    "Short_t":("i2",2),
    "UShort_t":("u2",2),
    "Int_t":("i4",4),
    "UInt_t":("u4",4),
    "Float_t":("f4",4),
    "Double_t":("f8",8),
    "Long64_t":("i8",8),
    "ULong64_t":("u8",8),
    "Bool_t":("bool",1),
}
 
cdef enum col_type:
    SINGLE = 1
    FIXED = 2
    VARY = 3

cdef struct col_descr:
    char* colname #colname
    char* typename
    void* payload #payload
    int guard #=9999 to easily check buffer overflow from payload
    int payloadsize
    #keeping track of size of payload in case we need to increase size 
    #payloadsize is number of element not number of bytes
    int size #size in byte of 1 element
    int static_asize #in case asize is a constant it points to here
    int* asize #pointer to number of element in array should point to the payload field of the length col
    col_type coltype
    TBranch* branch #so that we can peek the length first
    vector[col_descr*]* child #columns that use this column as length

# cdef class TreeStructure:
#     cdef vector[col_descr*]* columns
#     cdef TTree* tree
#     def __init__(self,tree_):
#         tree = <TTree*>tree_
#         print 'hey'
#     cdef init(self):
#         return 1
          #   
          #     pass
          #     
          # cdef init(self, TTree* tree):
          #     cdef TObjArray* branches
          #     self.columns = new vector[col_descr*]()
          #     branches = tree.GetListOfBranches()
          #     numbr = branches.GetEntries()
          #     for ibr in xrange(numbr):
          #         branch = <TBranch*>branches.At(ibr)
          #         bname = branch.GetName()
          #         print s
          # 
          # def __del__(self):
          #     
          #     del self.columns
          #     pass

def test3():
    a = 'test'
    get_ptr(a)

# def test2():
#        chain = new TChain("emc")
#        chain.Add("test.root")
#        ts = new TreeStructure(chain)
#        del chain

#     
# #for column with 1 or fixed size element
# def _col_descr_init_fixed( bytes name, bytes typename, int asize, TBranch* branch):
#     cdef int size
#     col = col_descr()
#     col.colname=name
#     if typename not in root_type_map: 
#         print "Warning: unknown type %s for column %s skip"%(typename,name)
#         return None
#     size = root_type_map[typename][1]
#     col.payload=malloc(size*asize)
#     col.guard = 9999
#     col.payloadsize = asize
#     col.size = size
#     col.static_asize= asize
#     col.asize = &col.static_asize
#     if asize==1 :
#         col_type = col_type.SINGLE
#     else:
#         col_type = col_type.FIXED
#     col.branch = branch
#     return col
# 
# #caller is responsible to call instructure one collist again
# def _col_descr_init_variable( bytes name, bytes typename, TBranch* branch):
#     col = col_descr()
#     col.colname = name
#     if typename not in root_type_map: 
#         print "Warning: unknown type %s for column %s skip"%(typename,name)
#         return None
#     size = root_type_map[typename][1]
#     asize = 16 #initialize with this
#     col.payload=malloc(size*asize)
#     col.guard = 9999
#     col.payloadsize = asize
#     col.size = size
#     col.static_asize= -1
#     col.asize = NULL
#     col.col_type = col_type.VARIABLE
#     col.branch = branch
#     return col
#     
# def make_col_structure(TTree* tree):
#     tree.
# 
# #lcn = list of length column names
# def init_col_structure(colmap,lcn,collist):
#     #
#     return lengthcols
# 
# 
# #for column 
# 
# 
# #clean up
# def int _col_descr_cleanup(col_descr& col):#make sure you close the root file before you call this
#     if(col.payload != 0) free(col.payload)
#     return 1

def test(x,t):
    cdef char* fname = x
    cdef char* tname = t
    cdef TChain* chain = new TChain(tname)
    cdef int i = 0
    cdef int ientry=0
    cdef int numentry
    chain.Add(fname)
    chain.SetBranchAddress("i",<void**>&i)
    numentry = chain.GetEntries()
    print numentry
    chain.Print()
    for ientry in range(numentry):
        chain.GetEntry(ientry)
        print ientry, i
    del chain