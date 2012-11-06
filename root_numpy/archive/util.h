#ifndef __UTIL_H_
#define __UTIL_H_
#define RNDEBUG(s) std::cout << "DEBUG: " << __FILE__ << "(" <<__LINE__ << ") " << #s << " = " << s << std::endl;
#define RNHEXDEBUG(s) std::cout << "DEBUG: " << __FILE__ << "(" <<__LINE__ << ") " << #s << " = " << std::hex << s << std::dec << std::endl;


bool convertible(const string& rt){
    std::map<std::string, TypeInfo>::iterator it;
    it = root_typemap.find(rt);
    return it!=root_typemap.end();
}

vector<string> branch_names(TTree* tree){
     //first get list of branches
    vector<string> ret;
    TObjArray* branches = tree->GetListOfBranches();
    int numbranches = branches->GetEntries();
    for(int ib=0;ib<numbranches;++ib){
        TBranch* branch = dynamic_cast<TBranch*>(branches->At(ib));
        const char* bname = branch->GetName();
        ret.push_back(bname);
    }
    return ret;
}

inline std::vector<std::string> vector_unique(const std::vector<std::string>& org){
    using namespace std;
    set<string> myset;
    myset.insert(org.begin(),org.end());
    vector<string> ret;
    for(int i=0;i<org.size();i++){
        set<string>::iterator it = myset.find(org[i]);
        if(it!=myset.end()){
            myset.erase(it);
            ret.push_back(org[i]);
        }
    }
    return ret;
}

//convert list of string to vector of string
//if los is just a string vos will be filled with that string
//if los is null or PyNone it do nothing to vos and return OK;
int los2vos(PyObject* los, std::vector<std::string>& vos){
    int ret=1;
    if(los==NULL){
        //do nothing
    }
    else if(los==Py_None){
        //do nothing
    }
    else if(PyString_Check(los)){//passing string put that in to vector
        char* tmp = PyString_AsString(los);
        vos.push_back(tmp);
    }else if(PyList_Check(los)){//an actual list of string
        int len = PyList_Size(los);
        for(int i=0;i<len;i++){
            PyObject* s = PyList_GetItem(los,i);
            if(!s){return NULL;}
            char* tmp = PyString_AsString(s);
            if(!tmp){return NULL;}
            std::string str(tmp);
            vos.push_back(tmp);
        }
    }else{
        ret=NULL;
    }
    return ret;
}

//check if file exists
bool file_exists(std::string fname){
    std::ifstream my_file(fname.c_str());
    return my_file.good();
}

bool has_wildcard(std::string fname){
    return fname.find("*") != std::string::npos;
}

#endif