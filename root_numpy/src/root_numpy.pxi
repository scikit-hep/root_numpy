
cdef extern from "2to3.h":
    pass 

cdef extern from "util.h":
    cdef void* shift(void*, int)
    void printaddr(void* v)
    cdef cppclass TypeName[T]:
        TypeName()
        const_char* name
    cdef cppclass Vector2Array[T]:
        T* convert(vector[T]* v)
 
cdef extern from "<memory>" namespace "std": 
    cdef cppclass auto_ptr[T]:
        auto_ptr() 
        auto_ptr(T* ptr) 
        reset (T* p)
        T* get()

cdef extern from "Column.h":
    cdef cppclass Column:
        string name
        string type
        int GetLen()
        int GetCountLen()
        int GetSize()
        void* GetValuePointer()
        const_char* GetTypeName()

    cdef cppclass FormulaColumn(Column):
        FormulaColumn(string, TTreeFormula*)
        string name
        string type
        int GetLen()
        int GetSize()
        void* GetValuePointer()
        const_char* GetTypeName()
    
    cdef cppclass BranchColumn(Column):
        BranchColumn(string, TLeaf*)
        string name
        string type
        int GetLen()
        int GetCountLen()
        int GetSize()
        void* GetValuePointer()
        const_char* GetTypeName()

cdef extern from "TreeChain.h":
    cdef cppclass TreeChain:
        TreeChain(TTree*)
        long Prepare()
        int Next()
        void AddColumn(string bname, string lname, BranchColumn* column)
        int GetEntry(int i)
        TTree* fChain
        void AddFormula(TTreeFormula* formula)
        void InitBranches()
