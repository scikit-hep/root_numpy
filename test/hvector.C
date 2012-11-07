void write() 
{
  
   TFile *f = TFile::Open("hvector.root","RECREATE");
   
   if (!f) { return; }

   std::vector<int> v_i;
   std::vector<float> v_f;
   std::vector<double> v_d;
   std::vector<long> v_l;
   std::vector<char> v_c;

   // Create a TTree
   TTree *t = new TTree("tvec","Tree with vectors");
   t->Branch("v_i",&v_i);
   t->Branch("v_f",&v_f);
   t->Branch("v_d",&v_d);
   t->Branch("v_l",&v_l);
   t->Branch("v_c",&v_c);


   const Int_t kUPDATE = 1000;
   for (Int_t i = 0; i < 100; i++) {
      Int_t npx = i%10;

      v_i.clear();
      v_f.clear();
      v_d.clear();
      v_l.clear();
      v_c.clear();

      for (Int_t j = 0; j < npx; ++j) {
          
         v_i.push_back(i+j);
         v_f.push_back(2*i+j);
         v_d.push_back(3*i+j);
         v_l.push_back(4*i+j);
         v_c.push_back(i+j);
      }
      if (i && (i%kUPDATE) == 0) {
         if (i == kUPDATE) hpx->Draw();
         c1->Modified();
         c1->Update();
         if (gSystem->ProcessEvents())
            break;
      }
      t->Fill();
   }
   f->Write();
   
   delete f;
}


void hvector() 
{
   write();  
}
