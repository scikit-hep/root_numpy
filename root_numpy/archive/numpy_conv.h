#include <numpy/arrayobject.h>

struct TypeInfo{
    string nptype;
    int size;//in bytes
    NPY_TYPES npt;
    TypeInfo(const TypeInfo& t):nptype(t.nptype),size(t.size),npt(t.npt){}
    TypeInfo(const std::string& nptype, int size, NPY_TYPES npt):nptype(nptype),size(size),npt(npt){}
    ~TypeInfo(){}
    void print(){
        cout << nptype << ":" << size;
    }
};


//missing string printf
//this is safe and convenient but not exactly efficient
inline std::string format(const char* fmt, ...){
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl,fmt);
    int nsize = vsnprintf(buffer,size,fmt,vl);
    if(size<=nsize){//fail delete buffer and try again
        delete buffer; buffer = 0;
        buffer = new char[nsize+1];//+1 for /0
        nsize = vsnprintf(buffer,size,fmt,vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete buffer;
    return ret;
}



//convert root type to numpy type
TypeInfo* rt2npt(const string& rt, bool must_found=false){
    std::map<std::string, TypeInfo>::iterator it;
    TypeInfo* ret=0;//float default
    it = root_typemap.find(rt);
    if(must_found){assert(it!=root_typemap.end());}
    if(it!=root_typemap.end()){
        ret = &(it->second);
    }
    return ret;
}

//copy to numpy element destination
    //and return number of byte written
    int copy_to(void* destination){
        if(skipped){
            if(coltype==FIXED){
                return countval*tinfo->size;
            }else if(coltype==SINGLE){
                return tinfo->size;
            }else if(coltype==VARY){
                //make empty array
                npy_intp dims[1];
                dims[0]=0;
                PyArrayObject* newobj = (PyArrayObject*)PyArray_EMPTY(1,dims,tinfo->npt,0);
                assert(newobj!=0);
                memcpy(destination,&newobj,sizeof(PyArrayObject*));
                return sizeof(PyObject*);
            }else{
                assert(false);//shouldn't reach here
                return 0;
            }
        }
        else{
            int ret;
            if(coltype==FIXED || coltype==SINGLE){
                assert(leaf!=NULL);
                void* src = leaf->GetValuePointer();
                assert(src!=NULL);
                ret = leaf->GetLenType()*leaf->GetLen();
                assert(ret>=0);
                memcpy(destination,src,ret);
            }else{//variable length array
                //build a numpy array of the length and put pyobject there
                void* src = leaf->GetValuePointer();
                int sizetocopy = leaf->GetLenType()*leaf->GetLen();
                npy_intp dims[1];
                dims[0]=leaf->GetLen();
                PyArrayObject* newobj = (PyArrayObject*)PyArray_EMPTY(1,dims,tinfo->npt,0);
                assert(newobj!=0);
                memcpy(newobj->data,src,sizetocopy);
                memcpy(destination,&newobj,sizeof(PyArrayObject*));
                ret = sizeof(PyObject*);
            }
            return ret;
        }
        assert(false);//shoudln't reach here
        return 0;
    }
    //convert to PyArray_Descr tuple
    PyObject* totuple(){
        //return ('col','f8')
        if(coltype==SINGLE){
            
            PyObject* pyname = PyString_FromString(colname.c_str());

            PyObject* pytype = PyString_FromString(tinfo->nptype.c_str());
            PyObject* nt_tuple = PyTuple_New(2);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);
            char* tmp = PyString_AsString(pytype);
            return nt_tuple;
        }else if(coltype==FIXED){//return ('col','f8',(10))
            PyObject* pyname = PyString_FromString(colname.c_str());

            PyObject* pytype = PyString_FromString(tinfo->nptype.c_str());

            PyObject* subsize = PyTuple_New(1);
            PyObject* pysubsize = PyInt_FromLong(countval);
            PyTuple_SetItem(subsize,0,pysubsize);

            PyObject* nt_tuple = PyTuple_New(3);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);
            PyTuple_SetItem(nt_tuple,2,subsize);

            return nt_tuple;
        }else if(coltype==VARY){//return ('col','object')
            PyObject* pyname = PyString_FromString(colname.c_str());

            PyObject* pytype = PyString_FromString("object");

            PyObject* nt_tuple = PyTuple_New(2);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);

            return nt_tuple;
        }else{
            assert(false);//shouldn't reach here
        }
        return NULL;
    }

static std::map<std::string, TypeInfo> root_typemap;

inline void init_roottypemap(){
    using std::make_pair;
    //TODO: correct this one so it doesn't depend on system
    // from TTree doc
    // - C : a character string terminated by the 0 character
    // - B : an 8 bit signed integer (Char_t)
    // - b : an 8 bit unsigned integer (UChar_t)
    // - S : a 16 bit signed integer (Short_t)
    // - s : a 16 bit unsigned integer (UShort_t)
    // - I : a 32 bit signed integer (Int_t)
    // - i : a 32 bit unsigned integer (UInt_t)
    // - F : a 32 bit floating point (Float_t)
    // - D : a 64 bit floating point (Double_t)
    // - L : a 64 bit signed integer (Long64_t)
    // - l : a 64 bit unsigned integer (ULong64_t)
    // - O : [the letter 'o', not a zero] a boolean (Bool_t)
    // from numericdtype.py
    // # b -> boolean
    // # u -> unsigned integer
    // # i -> signed integer
    // # f -> floating point
    // # c -> complex
    // # M -> datetime
    // # m -> timedelta
    // # S -> string
    // # U -> Unicode string
    // # V -> record
    // # O -> Python object

    root_typemap.insert(make_pair("Char_t",TypeInfo("i1",1,NPY_INT8)));
    root_typemap.insert(make_pair("UChar_t",TypeInfo("u1",1,NPY_UINT8)));

    root_typemap.insert(make_pair("Short_t",TypeInfo("i2",2,NPY_INT16)));
    root_typemap.insert(make_pair("UShort_t",TypeInfo("u2",2,NPY_UINT16)));

    root_typemap.insert(make_pair("Int_t",TypeInfo("i4",4,NPY_INT32)));
    root_typemap.insert(make_pair("UInt_t",TypeInfo("u4",4,NPY_UINT32)));

    root_typemap.insert(make_pair("Float_t",TypeInfo("f4",4,NPY_FLOAT32)));

    root_typemap.insert(make_pair("Double_t",TypeInfo("f8",8,NPY_FLOAT64)));

    root_typemap.insert(make_pair("Long64_t",TypeInfo("i8",8,NPY_INT64)));
    root_typemap.insert(make_pair("ULong64_t",TypeInfo("u8",8,NPY_UINT64)));

    root_typemap.insert(make_pair("Bool_t",TypeInfo("bool",1,NPY_BOOL)));
}


class TreeStructure{
public:
    vector<Column*> cols;//i don't own this
    BetterChain bc;
    bool good;
    vector<string> bnames;
    
    TreeStructure(TTree*tree,const vector<string>& bnames):bc(tree),bnames(bnames){
        good=false;
        init();
    }
    
    void init(){
        //TODO: refractor this
        //goal here is to fil cols array
        //map of name of len column and all the column that has length defined by the key
        for(int i=0;i<bnames.size();i++){
            string bname = bnames[i];
            TBranch* branch = bc.FindBranch(bname.c_str());
            if(branch==0){
                good=false;
                PyErr_SetString(PyExc_IOError,("Unable to get branch "+bname).c_str());
                return;
            }
            //now get the leaf the type info
            TObjArray* leaves = branch->GetListOfLeaves();
            int numleaves = leaves->GetEntries();
            bool shortname = numleaves==1;
            
            for(int ileaves=0;ileaves<numleaves;ileaves++){
                TLeaf* leaf = dynamic_cast<TLeaf*>(leaves->At(ileaves));
                if(leaf==0){
                    good=false;
                    PyErr_SetString(PyExc_IOError,format("Unable to get leaf %s for branch %s",leaf->GetName(),branch->GetName()).c_str());
                    return;
                }

                string rttype(leaf->GetTypeName());
                if(!convertible(rttype)){//no idea how to convert this
                    cerr << "Warning: unable to convert " << rttype << " for branch " << bname << ". Skip." << endl;
                    continue;
                }

                //figure out column name
                string colname;
                if(shortname){colname=bname;}
                else{colname=format("%s_%s",bname.c_str(),leaf->GetName());}
                
                Column* thisCol = bc.MakeColumn(bname,leaf->GetName(),colname);
                if(thisCol==0){return;}
                cols.push_back(thisCol);
            }//end for each laves
        }//end for each branch
        
        good=true;
    }

    //return list of tuple
    //[('col','f8'),('kkk','i4',(10)),('bbb','object')]
    PyObject* to_descr_list(){
        PyObject* mylist = PyList_New(0);
        for(int i=0;i<cols.size();++i){
            PyList_Append(mylist,cols[i]->totuple());
       }
       return mylist;
    }
    
    int copy_to(void* destination){
        char* current = (char*)destination;
        int total=0;
        for(int i=0;i<cols.size();++i){
            Column* thiscol = cols[i];
            int nbytes = thiscol->copy_to((void*)current);
            current += nbytes;
            total += nbytes;
        }
        return total;
    }

    //convert all leaf specified in lis to numpy structured array
    PyObject* build_array(){
        using namespace std;
        int numEntries = bc.GetEntries();
        PyObject* numpy_descr = to_descr_list();
        if(numpy_descr==0){return NULL;}
        //build the array

        PyArray_Descr* descr;
        int kkk = PyArray_DescrConverter(numpy_descr,&descr);
        Py_DECREF(numpy_descr);

        npy_intp dims[1];
        dims[0]=numEntries;

        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromDescr(1,dims,descr);

        //assume numpy array is contiguous
        char* current = NULL;
        //now put stuff in array
        for(int iEntry=0;iEntry<numEntries;++iEntry){
            int ilocal = bc.LoadTree(iEntry);
            assert(ilocal>=0);
            bc.GetEntry(iEntry);
            current = (char*)PyArray_GETPTR1(array, iEntry);
            int nbytes = copy_to((void*)current);
            current+=nbytes;
        }
        return (PyObject*)array;
    }
};