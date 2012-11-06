from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef extern from "TObject.h":
    cdef cppclass TObject:
        TObject()

cdef extern from "TObjArray.h":
    cdef cppclass TObjArray:
        TObject* At(int i)
        int GetSize()
        int GetEntries()

cdef extern from "TBranch.h":
    cdef cppclass TBranch:
        char* GetName()
        TObjArray* GetListOfLeaves()

cdef extern from "TLeaf.h":
    cdef cppclass TLeaf:
        char* GetTypeName()
        TLeaf* GetLeafCounter(int&)
        char* GetName()

cdef extern from "TFile.h":
    cdef cppclass TFile:
        TFile(char*)
        TFile(char*, char*)
        void Print()

cdef extern from "TTree.h":
    cdef cppclass TTree:
        TTree()
        void GetEntry(int i)
        int GetEntries()
        void SetBranchAddress(char* bname,void* addr)
        void Print()
        TObjArray* GetListOfBranches()

cdef extern from "TChain.h":
    cdef cppclass TChain(TTree):
        TChain()
        TChain(char*)
        int Add(char*)
        void Print()

cdef extern from "Column.h":
    cdef enum ColumnType:
        SINGLE
        FIXED
        VARY
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
    
cdef extern from "BetterChain.h":
    cdef cppclass BetterChain:
        BetterChain(TTree*)
        int Next()
        Column* MakeColumn(string bname, string lname, string colname)
