#ifndef __BETTER_CHAIN_H
#define __BETTER_CHAIN_H

#include <string>
#include <iostream>
#include <TTree.h>
#include <TFile.h>
#include <TChain.h>
#include <TLeaf.h>
#include <TTreeFormula.h>
#include <TObject.h>

#include <map>
#include <vector>
#include <cassert>
#include <set>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>

#include "Column.h"
#include "util.h"


// Correct TChain implementation with cache TLeaf*
class BetterChain
{
    public:

        BetterChain(TTree* fChain):
            fChain(fChain),
            ientry(0)
        {
            fCurrent = -1;
            notifier = new MiniNotify(fChain->GetNotify());
            fChain->SetNotify(notifier);
        }

        ~BetterChain()
        {
            if (!fChain)
            {
                // This is somehow needed (copied from MakeClass)
                return;
            }

            // Revert branches to their original activated/deactivated state
            std::map<std::string, bool>::iterator status_it;
            for (status_it = original_branch_status.begin();
                 status_it != original_branch_status.end();
                 ++status_it)
            {
                fChain->SetBranchStatus(status_it->first.c_str(),
                                        status_it->second);
            }

            fChain->SetNotify(notifier->oldnotify); // Do not switch these two lines!
            //delete fChain->GetCurrentFile(); // ROOT does something funny

            LeafCache::iterator it;
            for(it = leafcache.begin(); it != leafcache.end(); ++it)
            {
                delete it->second;
            }

            // BetterChain owns the formulae and so we delete them here
            std::vector<TTreeFormula*>::iterator fit;
            for (fit = formulae.begin(); fit != formulae.end(); ++fit)
            {
                delete *fit;
            }

            delete notifier;
        }

        long Prepare()
        {
            long load = LoadTree(0);
            if (load < 0)
            {
                return load;
            }
            // Remember original branch status
            TObjArray* branches = fChain->GetListOfBranches();
            int ibranch, nbranches;
            TBranch* branch;
            nbranches = branches->GetEntries();
            for (ibranch = 0; ibranch < nbranches; ++ibranch)
            {
                branch = (TBranch*) branches->At(ibranch);
                original_branch_status[branch->GetName()] =
                    branch->TestBit(kDoNotProcess) == 0;
                // Only the required branches will be added to the cache later
                fChain->DropBranchFromCache(branch, kTRUE);
            }
            // Enable all branches since we don't know yet which branches are
            // required by the formulae. The branches must be activated when a
            // TTreeFormula is initially created. All branches will be
            // disabled in InitBranches() before only enabling the ones that are
            // actually required
            fChain->SetBranchStatus("*", 1);
            //fChain->SetCacheSize(10000000);
            return load;
        }

        long long LoadTree(long entry)
        {
            if (!fChain)
            {
                return -5;
            }
            long long load = fChain->LoadTree(entry);
            if (load < 0)
            {
                return load;
            }
            if (fChain->GetTreeNumber() != fCurrent)
            {
                fCurrent = fChain->GetTreeNumber();
            }
            if(notifier->notified)
            {
                Notify();
                notifier->notified = false;
            }
            return load;
        }

        void AddFormula(TTreeFormula* formula)
        {
            // The BetterChain will take ownership of the formula
            if (formula == NULL)
            {
                return;
            }
            formulae.push_back(formula);
        }

        void InitBranches()
        {
            // The branches must be activated when a TTreeFormula is initially created.
            TBranch* branch;
            TLeaf* leaf;
            std::string bname, lname;
            LeafCache::iterator it;

            // Disable all branches
            fChain->SetBranchStatus("*", 0);

            for (it=leafcache.begin(); it!=leafcache.end(); ++it)
            {
                bname = it->first.first;
                lname = it->first.second;
                branch = fChain->GetBranch(bname.c_str());
                leaf = branch->FindLeaf(lname.c_str());

                // Make the branch active and cache it
                fChain->SetBranchStatus(bname.c_str(), 1);
                fChain->AddBranchToCache(branch, kTRUE);
                // and the length leaf as well

                // TODO Does it work if user doesn't want the length column in the structure?
                TLeaf* leafCount = leaf->GetLeafCount();
                if (leafCount != NULL)
                {
                    fChain->SetBranchStatus(leafCount->GetBranch()->GetName(), 1);
                    fChain->AddBranchToCache(leafCount->GetBranch(), kTRUE);
                }
            }

            // Activate all branches used by the formulae
            int ncodes;
            std::vector<TTreeFormula*>::iterator fit;
            for (fit = formulae.begin(); fit != formulae.end(); ++fit)
            {
                ncodes = (*fit)->GetNcodes();
                for (int n = 0; n < ncodes; ++n)
                {
                    branch = (*fit)->GetLeaf(n)->GetBranch();
                    // Make the branch active and cache it
                    fChain->SetBranchStatus(branch->GetName(), 1);
                    fChain->AddBranchToCache(branch, kTRUE);
                }
            }
        }

        int GetEntry(long entry)
        {
            long load;
            // Read contents of entry.
            if (!fChain)
            {
                return 0;
            }
            load = LoadTree(entry);
            if (load < 0)
            {
                return (int)load;
            }
            ientry = entry;
            return fChain->GetEntry(ientry);
        }

        int Next()
        {
            int ret = GetEntry(ientry);
            ++ientry;
            return ret;
        }

        void Notify()
        {
            // Taking care of all the leaves
            LeafCache::iterator it;
            for(it = leafcache.begin(); it != leafcache.end(); ++it)
            {
                std::string bname = it->first.first;
                std::string lname = it->first.second;
                TBranch* branch = fChain->FindBranch(bname.c_str());
                if (branch==0)
                {
                    std::cerr << "WARNING cannot find branch " << bname
                              << std::endl;
                    it->second->skipped = true;
                    continue;
                }
                TLeaf* leaf = branch->FindLeaf(lname.c_str());
                if (leaf==0)
                {
                    std::cerr << "WARNING cannot find leaf " << lname
                              << " for branch " << bname << std::endl;
                    it->second->skipped = true;
                    continue;
                }
                it->second->SetLeaf(leaf, true);
                it->second->skipped = false;
            }

            // Update all formula leaves
            std::vector<TTreeFormula*>::iterator fit;
            for (fit = formulae.begin(); fit != formulae.end(); ++fit)
            {
                (*fit)->UpdateFormulaLeaves();
            }
        }

        long long GetEntries()
        {
            return fChain->GetEntries();
        }

        double GetWeight()
        {
            return fChain->GetWeight();
        }

        Column* MakeColumn(const std::string& bname,
                           const std::string& lname,
                           const std::string& colname)
        {
            TBranch* branch = fChain->GetBranch(bname.c_str());
            if (branch == NULL)
            {
                PyErr_SetString(PyExc_IOError,
                    format("cannot find branch %s", bname.c_str()).c_str());
                return NULL;
            }

            TLeaf* leaf = branch->FindLeaf(lname.c_str());
            if (leaf == NULL)
            {
                PyErr_SetString(PyExc_IOError,
                    format("cannot find leaf %s for branch %s", lname.c_str(),
                           bname.c_str()).c_str());
                return NULL;
            }

            BL bl = make_pair(bname, lname);
            BranchColumn* ret = BranchColumn::build(leaf, colname);
            if (ret == NULL)
            {
                return NULL;
            }
            leafcache.insert(make_pair(bl, ret));
            return ret;
        }

        class MiniNotify: public TObject
        {
            public:
                MiniNotify(TObject* oldnotify):
                    TObject(),
                    notified(false),
                    oldnotify(oldnotify){}

                virtual Bool_t Notify()
                {
                    notified = true;
                    if (oldnotify)
                    {
                        oldnotify->Notify();
                    }
                    return true;
                }

                bool notified;
                TObject* oldnotify;
        };

        TTree* fChain;
        int fCurrent;
        long ientry;
        MiniNotify* notifier;
        std::vector<TTreeFormula*> formulae;
        std::map<std::string, bool> original_branch_status;

        // Branch name to leaf name conversion
        typedef std::pair<std::string, std::string> BL;
        typedef std::map<BL, BranchColumn*> LeafCache;

        // Column pointer cache since the leaf inside needs to be updated
        // when new file is loaded in the chain
        LeafCache leafcache;
};

#endif
