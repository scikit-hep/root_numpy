void makesingle(){
    TFile file("single.root","RECREATE");
    TTree tree("tree","tree");
    int n; tree.Branch("n_int",&n,"n_int/I");
    float f; tree.Branch("f_float",&f,"f_float/F");
    double d; tree.Branch("d_double",&d,"d_double/D");
    for(int i=0;i<100;i++){
        n = i;
        f = i*2.0;
        d = i*3.0;
        tree.Fill();
    }
    tree.Write();
}

void makefixed(){
    TFile file("fixed.root","RECREATE");
    TTree tree("tree","tree");
    
    int n[5]; tree.Branch("n_int",&n,"n_int[5]/I");
    float f[7]; tree.Branch("f_float",&f,"f_float[7]/F");
    double d[10]; tree.Branch("d_double",&d,"d_double[10]/D");
    for(int i=0;i<100;i++){
        for(int i_n=0;i_n<5;i_n++){n[i_n] = 5*i+i_n;}
        for(int i_f=0;i_f<7;i_f++){f[i_f] = 2*(5*i+i_f)+0.5;}
        for(int i_d=0;i_d<10;i_d++){d[i_d] = 3*(5*i+i_d)+0.5;}
        tree.Fill();
    }
    tree.Write();
}

void makevary(){
    int n[100];
    int len_n;
    float f[100];
    int len_f;    
    double d[100];
    int len_d;
    TFile file("vary.root","RECREATE");
    TTree tree("tree","tree");
    tree.Branch("len_n",&len_n,"len_n/I");
    tree.Branch("len_f",&len_f,"len_f/I");
    tree.Branch("len_d",&len_d,"len_d/I");
    tree.Branch("n_int",&n,"n_int[len_n]/I");
    tree.Branch("f_float",&f,"f_float[len_f]/F");
    tree.Branch("d_double",&d,"d_double[len_d]/D");
    
    for(int i=0;i<20;i++){
        len_n = i;
        len_f = i+1;
        len_d = i+2;
        for(int i_n=0;i_n<len_n;i_n++){n[i_n]=20*i+i_n;}
        for(int i_f=0;i_f<len_f;i_f++){f[i_f]=20*i+2.*i_f+0.5;}
        for(int i_d=0;i_d<len_d;i_d++){d[i_d]=20*i+3.*i_d+0.25;}
        tree.Fill();
    }
    tree.Write();
}

void testsample(){
    makesingle();
    makefixed();
    makevary();
}