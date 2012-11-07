from libcpp cimport bool
from libcpp.string cimport string, const_char

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef extern from "TObject.h":
    cdef cppclass TObject:
        TObject()
        const_char* GetName()
        const_char* ClassName()


cdef extern from "TObjArray.h":
    cdef cppclass TObjArray:
        TObject* At(int i)
        int GetSize()
        int GetEntries()

cdef extern from "TBranch.h":
    cdef cppclass TBranch:
        const_char* GetName()
        TObjArray* GetListOfLeaves()

cdef extern from "TLeaf.h":
    cdef cppclass TLeaf:
        const_char* GetTypeName()
        TLeaf* GetLeafCounter(int&)
        const_char* GetName()

cdef extern from "TFile.h":
    cdef cppclass TFile:
        TFile(const_char*)
        TFile(const_char*, const_char*)
        void Print()
        TList* GetListOfKeys()
        TObject* Get(const_char*)

cdef extern from "TTree.h":
    cdef cppclass TTree:
        TTree()
        void GetEntry(int i)
        int GetEntries()
        void SetBranchAddress(const_char* bname,void* addr)
        void Print()
        TObjArray* GetListOfBranches()

cdef extern from "TChain.h":
    cdef cppclass TChain(TTree):
        TChain()
        TChain(const_char*)
        int Add(const_char*)
        void Print()

cdef extern from "TList.h":
    cdef cppclass TList:
        TObject* list
        TObject* At(int idx)
        int GetEntries()

cdef extern from "Column.h":
    cdef enum ColumnType:
        SINGLE, FIXED, VARY
    cdef cppclass Column:
        TLeaf* leaf
        bool skipped
        ColumnType coltype
        string colname
        int countval
        string rttype
        int getLen()
        int getSize()
        void Print()
        void* GetValuePointer()
        const_char* GetTypeName()

cdef extern from "BetterChain.h":
    cdef cppclass BetterChain:
        BetterChain(TTree*)
        int Next()
        Column* MakeColumn(string bname, string lname, string colname)
        int GetEntries()
        int GetEntry(int i)

cdef extern from "util.h":
    cdef void* shift(void*, int)
    void printaddr(void* v)

cdef extern from "Vector2Array.h":
    cdef cppclass Vector2Array[T]:
        T* convert(vector[T]* v)
