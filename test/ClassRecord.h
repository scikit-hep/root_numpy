//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Apr 12 02:37:29 2012 by ROOT version 5.32/00
// from TChain ClassRecord/
//////////////////////////////////////////////////////////

#ifndef ClassRecord_h
#define ClassRecord_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class ClassRecord {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Float_t         Vars_index;
   Float_t         Vars_classification;
   Float_t         Vars_weight;
   Float_t         Vars_p1cm;
   Float_t         Vars_p2cm;
   Float_t         Vars_emiss;
   Float_t         Vars_thrust;
   Float_t         Vars_sph;
   Float_t         Vars_costh12;
   Float_t         Vars_deltaT;
   Float_t         Vars_dtErr1;
   Float_t         Vars_nn;
   Float_t         Vars_pp;
   Float_t         Vars_ee;
   Float_t         Vars_em;
   Float_t         Vars_me;
   Float_t         Vars_mm;
   Float_t         Vars_evttype;
   Float_t         Vars_n771_20;
   Float_t         Vars_n771_200;
   Float_t         Vars_n771_50;

   // List of branches
   TBranch        *b_Vars;   //!

   ClassRecord(TTree *tree=0);
   virtual ~ClassRecord();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef ClassRecord_cxx
ClassRecord::ClassRecord(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {

#ifdef SINGLE_TREE
      // The following code should be used if you want this class to access
      // a single tree instead of a chain
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("Memory Directory");
      if (!f || !f->IsOpen()) {
         f = new TFile("Memory Directory");
      }
      f->GetObject("ClassRecord",tree);

#else // SINGLE_TREE

      // The following code should be used if you want this class to access a chain
      // of trees.
      TChain * chain = new TChain("ClassRecord","");
      chain->Add("sprtest.root/ClassRecord");
      tree = chain;
#endif // SINGLE_TREE

   }
   Init(tree);
}

ClassRecord::~ClassRecord()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t ClassRecord::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t ClassRecord::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void ClassRecord::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("Vars", &Vars_index, &b_Vars);
   Notify();
}

Bool_t ClassRecord::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void ClassRecord::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t ClassRecord::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef ClassRecord_cxx
