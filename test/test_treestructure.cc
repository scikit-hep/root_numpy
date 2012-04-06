#include "TreeStructure.h"

int test_single(){
    using namespace std;
    TChain chain("tree");
    chain.Add("single.root");
    
    vector<string> bnames = branch_names(&chain);
    for(int b=0;b<bnames.size();++b){cout << bnames[b] << " , "; }
    cout << endl;
    
    TreeStructure t(&chain,bnames);
    cout << t.to_str() << endl;
    int n = chain.GetEntries();
    //lets try read the payload
    for(int i=0;i<n;i++){
        chain.LoadTree(i);
        chain.GetEntry(i);
        t.print_current_value();
        cout << " " << endl;
    }
    
    return 0;
}

int test_fixed(){
    using namespace std;
    TChain chain("tree");
    chain.Add("fixed.root");
    
    vector<string> bnames = branch_names(&chain);
    for(int b=0;b<bnames.size();++b){cout << bnames[b] << " , "; }
    cout << endl;
    
    TreeStructure t(&chain,bnames);
    cout << t.to_str() << endl;
    int n = chain.GetEntries();
    //lets try read the payload
    for(int i=0;i<n;i++){
        chain.LoadTree(i);
        chain.GetEntry(i);
        t.print_current_value();
        cout << " " << endl;
    }
    
    return 0;
}

int test_vary(){
    using namespace std;
    TChain chain("tree");
    chain.Add("vary.root");
    
    vector<string> bnames = branch_names(&chain);
    for(int b=0;b<bnames.size();++b){cout << bnames[b] << " , "; }
    cout << endl;
    
    TreeStructure t(&chain,bnames);
    cout << t.to_str() << endl;
    int n = chain.GetEntries();
    //lets try read the payload
    for(int i=0;i<n;i++){
        chain.LoadTree(i);
        t.peek(i);
        chain.GetEntry(i);
        t.print_current_value();
        cout << " " << endl;
    }
    
    return 0;
}


int main(int argc,char* argv[]){
    
    init_roottypemap();
    Py_Initialize();
    //test_single();
    //test_fixed();
    test_vary();
    Py_Finalize();
}