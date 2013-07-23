#include <cstdio>
#include <vector>

#include "TFile.h"
#include "TTree.h"


void hvector()
{

   TFile *f = TFile::Open("hvector.root","RECREATE");

   std::vector<int> v_i;
   std::vector<float> v_f;
   std::vector<Float_t> v_F;
   std::vector<double> v_d;
   std::vector<long> v_l;
   std::vector<char> v_c;
   std::vector<bool> v_b;

   // Create a TTree
   TTree *t = new TTree("tvec","Tree with vectors");
   t->Branch("v_i","std::vector<int>",&v_i);
   t->Branch("v_f","std::vector<float>",&v_f);
   t->Branch("v_F","std::vector<Float_t>",&v_F);
   t->Branch("v_d","std::vector<double>",&v_d);
   t->Branch("v_l","std::vector<long>",&v_l);
   t->Branch("v_c","std::vector<char>",&v_c);
   t->Branch("v_b","std::vector<bool>",&v_b);

   for (Int_t i = 0; i < 100; i++) {
      Int_t npx = i%10;

      v_i.clear();
      v_f.clear();
      v_F.clear();
      v_d.clear();
      v_l.clear();
      v_c.clear();
      v_b.clear();

      for (Int_t j = 0; j < npx; ++j) {

         v_i.push_back(i+j);
         v_f.push_back(2*i+j);
         v_F.push_back(2*i+j);
         v_d.push_back(3*i+j);
         v_l.push_back(4*i+j);
         v_c.push_back(i+j);
         v_b.push_back(j % 2 == 0);
      }
      t->Fill();
   }
   f->Write();
   f->Close();

   delete f;
}

void makesingle(int id)
{
    char buffer[255];
    sprintf(buffer,"single%d.root",id);
    TFile file(buffer,"RECREATE");
    TTree tree("tree","tree");
    int n; tree.Branch("n_int",&n,"n_int/I");
    float f; tree.Branch("f_float",&f,"f_float/F");
    double d; tree.Branch("d_double",&d,"d_double/D");
    for(int i=0;i<100;i++){
        n = i+id;
        f = i*2.0+id;
        d = i*3.0+id;
        tree.Fill();
    }
    tree.Write();
    file.Close();
}

void makefixed(int id)
{
    char buffer[255];
    sprintf(buffer,"fixed%d.root",id);
    TFile file(buffer,"RECREATE");
    TTree tree("tree","tree");

    int n[5]; tree.Branch("n_int",&n,"n_int[5]/I");
    float f[7]; tree.Branch("f_float",&f,"f_float[7]/F");
    double d[10]; tree.Branch("d_double",&d,"d_double[10]/D");
    for(int i=0;i<100;i++){
        for(int i_n=0;i_n<5;i_n++){n[i_n] = 5*i+i_n+id;}
        for(int i_f=0;i_f<7;i_f++){f[i_f] = 2*(5*i+i_f)+0.5+id;}
        for(int i_d=0;i_d<10;i_d++){d[i_d] = 3*(5*i+i_d)+0.5+id;}
        tree.Fill();
    }
    tree.Write();
}

void makevary(int id)
{
    int n[100];
    int len_n;
    float f[100];
    int len_f;
    double d[100];
    int len_d;
    char buffer[255];
    sprintf(buffer,"vary%d.root",id);
    TFile file(buffer,"RECREATE");
    TTree tree("tree","tree");
    tree.Branch("len_n",&len_n,"len_n/I");
    tree.Branch("len_f",&len_f,"len_f/I");
    tree.Branch("len_d",&len_d,"len_d/I");
    tree.Branch("n_int",&n,"n_int[len_n]/I");
    tree.Branch("f_float",&f,"f_float[len_f]/F");
    tree.Branch("d_double",&d,"d_double[len_d]/D");

    for(int i=0;i<20;i++){
        len_n = i*id;
        len_f = i*id+1;
        len_d = i*id+2;
        for(int i_n=0;i_n<len_n;i_n++){n[i_n]=20*i+i_n;}
        for(int i_f=0;i_f<len_f;i_f++){f[i_f]=20*i+2.*i_f+0.5;}
        for(int i_d=0;i_d<len_d;i_d++){d[i_d]=20*i+3.*i_d+0.25;}
        tree.Fill();
    }
    tree.Write();
}

void make2tree(int id)
{
    char buffer[255];
    sprintf(buffer,"doubletree%d.root",id);
    TFile file(buffer,"RECREATE");
    TTree tree("tree","tree");
    double x,y;
    tree.Branch("x",&x);
    tree.Branch("y",&y);
    for(int i=0;i<10;++i){
        x=i;y=2*i;
        tree.Fill();
    }
    tree.Write();
    TTree tree2("tree2","tree2");
    double x2,y2;
    tree2.Branch("x2",&x2);
    tree2.Branch("y2",&y2);
    for(int i=0;i<10;++i){
        x2=i;y2=2*i;
        tree2.Fill();
    }
    tree2.Write();
}

void makestruct()
{
    struct branchstruct {
        int intleaf;
        float floatleaf;
    };

    TFile f("structbranches.root", "RECREATE");
    TTree t("test", "Test tree with identical leaf names in different branches");

    branchstruct *br1 = new branchstruct;
    branchstruct *br2 = new branchstruct;

    br1->intleaf = 10;
    br1->floatleaf = 15.5;

    br2->intleaf = 20;
    br2->floatleaf = 781.2;

    t.Branch("branch1", br1, "intleaf/I:floatleaf/F");
    t.Branch("branch2", br2, "intleaf/I:floatleaf/F");

    t.Fill();

    t.Write();
    f.Close();

    delete br1;
    delete br2;
}

int main(void)
{
    makesingle(1);
    makesingle(2);
    makefixed(1);
    makefixed(2);
    makevary(1);
    makevary(2);
    make2tree(1);
    hvector();
    makestruct();
    return 0;
}
