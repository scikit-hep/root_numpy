#ifndef __COLUMN_H_
#define __COLUMN_H_
#include <TLeaf.h>
#include <string>
#include <Python.h>
#include <iostream>
using namespace std;

enum ColumnType{SINGLE=1,FIXED=2,VARY=3};

//This describe the structure of the tree
//Converter should take this and make appropriate data structure
class Column{
public:

    TLeaf* leaf;
    bool skipped;
    ColumnType coltype;//single fixed vary?
    string colname;//column name
    int countval; //useful in case of fixed element
    string rttype;//name of the roottype

    static int find_coltype(TLeaf* leaf, ColumnType& coltype, int& countval ){
        //now check whether it's array if so of which type
        TLeaf* len_leaf = leaf->GetLeafCounter(countval);
        if(countval==1){
            if(len_leaf==0){//single element
                coltype = SINGLE;
            }
            else{//variable length          
                coltype = VARY;
            }
        }else if(countval>0){
            //fixed multiple array
            coltype = FIXED;
        }else{//negative
            string msg("Unable to understand the structure of leaf ");
            msg += leaf->GetName();
            PyErr_SetString(PyExc_IOError,msg.c_str());
            return 0;
        }
        return 1;
    }

    static Column* build(TLeaf* leaf,const string& colname){
        Column* ret = new Column();
        ret->leaf = leaf;
        ret->colname = colname;
        ret->skipped = false;
        int ok = find_coltype(leaf,ret->coltype,ret->countval);
        if(!ok){
            delete ret;
            return NULL;
        }
        ret->rttype = leaf->GetTypeName();
        return ret;
    }
    
    void SetLeaf(TLeaf* newleaf, bool paranoidmode=false){
        leaf = newleaf;
        if(paranoidmode){
            assert(leaf->GetTypeName() == rttype);
            int cv;
            ColumnType ct;
            int ok = find_coltype(leaf,ct,cv);
            assert(ok!=0);
            assert(ct==coltype);
            //if(ct==FIXED){assert(cv==countval);}
        }
    }
    //get len of this block(in unit of element)
    int getLen(){
        return leaf->GetLen();
    }
    //get size of this block in bytes
    int getSize(){
        int size = leaf->GetLenType()*leaf->GetLen();
        return size;
    }

    void* GetValuePointer(){
        return leaf->GetValuePointer();
    }
    int getintVal(){
        return *(int*)(leaf->GetValuePointer());
    }

    const char* GetTypeName(){
        return leaf->GetTypeName();
    }

    void Print(){
        cout << colname << " - " << rttype << endl;
    }

};
#endif
