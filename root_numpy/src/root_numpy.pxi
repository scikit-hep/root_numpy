from libcpp cimport bool
from libcpp.string cimport string, const_char

cdef extern from "Column.h":
    cdef enum ColumnType:
        SINGLE, FIXED, VARY
    cdef cppclass Column:
        bool skipped
        ColumnType coltype
        string colname
        int countval
        string rttype
        int GetLen()
        int GetSize()
        void* GetValuePointer()
        const_char* GetTypeName()
    cdef cppclass FormulaColumn(Column):
        FormulaColumn(string, TTreeFormula*)
        bool skipped
        ColumnType coltype
        string colname
        int countval
        string rttype
        int GetLen()
        int GetSize()
        void* GetValuePointer()
        const_char* GetTypeName()

cdef extern from "BetterChain.h":
    cdef cppclass BetterChain:
        BetterChain(TTree*)
        long Prepare()
        int Next()
        Column* MakeColumn(string bname, string lname, string colname)
        int GetEntries()
        int GetEntry(int i)
        double GetWeight()
        TTree* fChain
        void AddFormula(TTreeFormula* formula)
        void InitBranches()

cdef extern from "util.h":
    cdef void* shift(void*, int)
    void printaddr(void* v)
    cdef cppclass TypeName[T]:
        TypeName()
        const_char* name

cdef extern from "Vector2Array.h":
    cdef cppclass Vector2Array[T]:
        T* convert(vector[T]* v)

cdef extern from "<memory>" namespace "std": 
    cdef cppclass auto_ptr[T]:
        auto_ptr() 
        auto_ptr(T* ptr) 
        reset (T* p)
        T* get()
