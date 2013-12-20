from libcpp cimport bool
from libcpp.string cimport string, const_char

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
        const_char* GetTitle()
        TObjArray* GetListOfLeaves()
        void SetAddress(void* addr)
        void SetStatus(bool status)
        int Fill()

cdef extern from "TLeaf.h":
    cdef cppclass TLeaf:
        const_char* GetTypeName()
        TLeaf* GetLeafCounter(int&)
        const_char* GetName()

cdef extern from "TFile.h":
    cdef cppclass TFile:
        TFile(const_char*, const_char*)
        void Print()
        TList* GetListOfKeys()
        TObject* Get(const_char*)
        void Close()
        bool IsOpen()
        bool IsWritable()

cdef extern from "TFile.h" namespace "TFile":
    TFile* Open(const_char*, const_char*)

cdef extern from "TTree.h":
    cdef cppclass TTree:
        TTree()
        TTree(const_char*,  const_char*)
        void GetEntry(int i)
        int GetEntries()
        void SetBranchAddress(const_char* bname, void* addr)
        void SetBranchStatus(const_char* bname, bool status)
        void Print()
        TBranch* Branch(const_char* name, void* address, const_char* leaflist)
        TBranch* GetBranch(const_char* name)
        TObjArray* GetListOfBranches()
        int Fill()
        int Scan()
        void Delete(void*)
        long SetEntries(long)
        int Write()

cdef extern from "TChain.h":
    cdef cppclass TChain(TTree):
        TChain()
        TChain(const_char*)
        int Add(const_char*, long)
        void Print()

cdef extern from "TList.h":
    cdef cppclass TList:
        TObject* list
        TObject* At(int idx)
        int GetEntries()

cdef extern from "TTreeFormula.h":
    cdef cppclass TTreeFormula:
        TTreeFormula(const_char*, const_char*, TTree*)
        int GetNdim()
        int GetNdata()
        double EvalInstance(int)

cdef extern from "TClassEdit.h" namespace "TClassEdit":
    string ResolveTypedef(const_char*, bool)

cdef extern from "TF1.h":
    cdef cppclass TF1:
        double GetRandom()

cdef extern from "TF2.h":
    cdef cppclass TF2:
        double GetRandom2(double& x, double& y)

cdef extern from "TF3.h":
    cdef cppclass TF3:
        double GetRandom3(double& x, double& y, double& z)

cdef extern from "TH1.h":
    cdef cppclass TH1:
        double GetRandom()
        int Fill(double x)
        int Fill(double x, double w)

cdef extern from "TH2.h":
    cdef cppclass TH2:
        double GetRandom2(double& x, double& y)
        int Fill(double x, double y)
        int Fill(double x, double y, double w)

cdef extern from "TH3.h":
    cdef cppclass TH3:
        double GetRandom3(double& x, double& y, double& z)
        int Fill(double x, double y, double z)
        int Fill(double x, double y, double z, double w)

cdef extern from "TGraph.h":
    cdef cppclass TGraph:
        void Set(int n)
        void SetPoint(int i, double x, double y)

cdef extern from "TGraph2D.h":
    cdef cppclass TGraph2D:
        void Set(int n)
        void SetPoint(int i, double x, double y, double z)

cdef extern from "TArrayD.h":
    cdef cppclass TArrayD:
        int GetSize()
        double* GetArray()

cdef extern from "TArrayF.h":
    cdef cppclass TArrayF:
        int GetSize()
        float* GetArray()

cdef extern from "TArrayL.h":
    cdef cppclass TArrayL:
        int GetSize()
        long* GetArray()

cdef extern from "TArrayI.h":
    cdef cppclass TArrayI:
        int GetSize()
        int* GetArray()

cdef extern from "TArrayS.h":
    cdef cppclass TArrayS:
        int GetSize()
        short* GetArray()

cdef extern from "TArrayC.h":
    cdef cppclass TArrayC:
        int GetSize()
        char* GetArray()

cdef extern from "TMatrixTBase.h":
    cdef cppclass TMatrixDBase:
        int GetNrows()
        int GetNcols()
        double get "operator()"(int rown, int coln)
    cdef cppclass TMatrixFBase:
        int GetNrows()
        int GetNcols()
        float get "operator()"(int rown, int coln)
