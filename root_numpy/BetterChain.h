#ifndef __BETTER_CHAIN_H
#define __BETTER_CHAIN_H
#include <Python.h>
#include <string>
#include <iostream>
#include <TTree.h>
#include <TFile.h>
#include <TChain.h>
#include <TLeaf.h>
#include <map>

#include <cassert>
#include <set>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <TObject.h>

#include "Column.h"
#include "util.h"
using namespace std;

//correct TChain implementation with cache TLeaf*
class BetterChain{
public:
    class MiniNotify:public TObject{
    public:
        bool notified;
        TObject* oldnotify;
        MiniNotify(TObject* oldnotify):TObject(),notified(false),oldnotify(oldnotify){}
        virtual Bool_t Notify(){
            notified=true;
            if(oldnotify) oldnotify->Notify();
            return true;
        }
    };
    
    TTree* fChain;
    int fCurrent;
    int ientry;
    MiniNotify* notifier;
    BetterChain(TTree* fChain):fChain(fChain),ientry(0){
        fCurrent = -1;
        notifier = new MiniNotify(fChain->GetNotify());
        fChain->SetNotify(notifier);
        LoadTree(0);
        fChain->SetBranchStatus("*",0);//disable all branches
        //fChain->SetCacheSize(10000000);
    }
    
    ~BetterChain(){
        if (!fChain) return;//some how i need this(copy from make class)

        fChain->SetNotify(notifier->oldnotify);//do not switch these two lines!
        //delete fChain->GetCurrentFile();//root does something funny

        LeafCache::iterator it;
        for(it=leafcache.begin();it!=leafcache.end();++it){
            delete it->second;
        }

        delete notifier;
    }

    typedef pair<string,string> BL; //branch name to leaf name conversion
    typedef map<BL,Column*> LeafCache;
    //column pointer cache since the leaf inside needs to be updated
    //when new file is loaded in the chain
    LeafCache leafcache;

    int LoadTree(int entry){
        if (!fChain) return -5;
        //RNHEXDEBUG(fChain->FindBranch("mcLen")->FindLeaf("mcLen"));
        Long64_t centry = fChain->LoadTree(entry);
        //RNHEXDEBUG(fChain->FindBranch("mcLen")->FindLeaf("mcLen"));
        if (centry < 0) return centry;
        if (fChain->GetTreeNumber() != fCurrent) {
           fCurrent = fChain->GetTreeNumber();
        }
        if(notifier->notified){
            Notify();
            notifier->notified=false;
        }
        return centry;
    }

    int GetEntry(int entry){
        // Read contents of entry.
        if (!fChain) return 0;
        LoadTree(entry);
        ientry = entry;
        return fChain->GetEntry(ientry);
    }
    
    int Next(){
        int ret = GetEntry(ientry);
        ientry++;
        return ret;
    }

    void Notify(){
        //taking care of all the leaves
        //RNDEBUG("NOTIFY");
        LeafCache::iterator it;
        for(it=leafcache.begin();it!=leafcache.end();++it){
            string bname = it->first.first;
            string lname = it->first.second;
            TBranch* branch = fChain->FindBranch(bname.c_str());
            if(branch==0){
                cerr << "Warning cannot find branch " << bname << endl;
                it->second->skipped = true;
                continue;
            }
            TLeaf* leaf = branch->FindLeaf(lname.c_str());
            if(leaf==0){
                cerr << "Warning cannot find leaf " << lname << " for branch " << bname << endl;
                it->second->skipped = true;
                continue;
            }
            it->second->SetLeaf(leaf,true); 
            it->second->skipped = false;
        }
    }
    
    int GetEntries(){
        int ret = fChain->GetEntries();
        return ret;
    }

    TBranch* FindBranch(const char* bname){
        return fChain->FindBranch(bname);
    }
    
    Column* MakeColumn(const string& bname, const string& lname, const string& colname){
        //as bonus set branch status on all the active branch including the branch that define the length
        LoadTree(0);

        TBranch* branch = fChain->FindBranch(bname.c_str());
        if(branch==0){
            PyErr_SetString(PyExc_IOError,format("Cannot find branch %s",bname.c_str()).c_str());
            return 0;
        }
        
        TLeaf* leaf = fChain->FindLeaf(lname.c_str());
        if(leaf==0){
            PyErr_SetString(PyExc_IOError,format("Cannot find leaf %s for branch %s",lname.c_str(),bname.c_str()).c_str());
            return 0;
        }
        
        //make the branch active
        //and cache it
        fChain->SetBranchStatus(bname.c_str(),1);
        fChain->AddBranchToCache(branch,kTRUE);
        //and the length leaf as well

        //TODO Does it work if user dont' want length column in the structure?
        TLeaf* leafCount = leaf->GetLeafCount();
        if(leafCount != 0){
            fChain->SetBranchStatus(leafCount->GetBranch()->GetName(),1);
            fChain->AddBranchToCache(leafCount->GetBranch(),kTRUE);
        }
        
        BL bl = make_pair(bname,lname);
        Column* ret = Column::build(leaf,colname);
        if(ret==0){return 0;}
        leafcache.insert(make_pair(bl,ret));
        return ret;
    }
};

#endif
