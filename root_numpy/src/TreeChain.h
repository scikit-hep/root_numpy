#ifndef __TREE_CHAIN_H
#define __TREE_CHAIN_H

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


void activate_branch_recursive(TBranch* branch)
{
    if (branch == NULL)
    {
        return;
    }
    // Activate the branch
    branch->SetStatus(true);
    TObjArray* subbranches = branch->GetListOfBranches();
    // Loop on subbranches
    for (int i = 0; i < subbranches->GetEntries(); ++i)
    {
        activate_branch_recursive((TBranch*)subbranches->At(i));
    }
}


// Improved TChain implementation
class TreeChain
{
    public:

        TreeChain(TTree* fChain):
            fChain(fChain),
            ientry(0)
        {
            fCurrent = -1;
            notifier = new MiniNotify(fChain->GetNotify());
            fChain->SetNotify(notifier);
        }

        ~TreeChain()
        {
            if (!fChain)
            {
                // Copied from MakeClass
                return;
            }

            fChain->SetNotify(notifier->oldnotify);

            LeafCache::iterator it;
            for(it = leafcache.begin(); it != leafcache.end(); ++it)
            {
                delete it->second;
            }

            // TreeChain owns the formulae and so we delete them here
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
            // Enable all branches since we don't know yet which branches are
            // required by the formulae. The branches must be activated when a
            // TTreeFormula is initially created.
            fChain->SetBranchStatus("*", true);
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
            // The TreeChain will take ownership of the formula
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

            // Only the required branches will be added to the cache below
            fChain->DropBranchFromCache("*", true);

            for (it = leafcache.begin(); it != leafcache.end(); ++it)
            {
                bname = it->first.first;
                lname = it->first.second;
                branch = fChain->GetBranch(bname.c_str());
                leaf = branch->FindLeaf(lname.c_str());

                // Make the branch active and cache it
                branch->SetStatus(true);
                fChain->AddBranchToCache(branch, true);
                // and the length leaf as well

                // TODO: Does this work if user doesn't want the length column
                // in the output structure?
                TLeaf* leafCount = leaf->GetLeafCount();
                if (leafCount != NULL)
                {
                    branch = leafCount->GetBranch();
                    branch->SetStatus(true);
                    fChain->AddBranchToCache(branch, true);
                }
            }

            // Activate all branches used by the formulae
            int ncodes, n;
            std::vector<TTreeFormula*>::iterator fit;
            for (fit = formulae.begin(); fit != formulae.end(); ++fit)
            {
                ncodes = (*fit)->GetNcodes();
                for (n = 0; n < ncodes; ++n)
                {
                    branch = (*fit)->GetLeaf(n)->GetBranch();
                    // Branch may be a TObject split across multiple
                    // subbranches. These must be activated recursively.
                    activate_branch_recursive(branch);
                    fChain->AddBranchToCache(branch, true);
                }
            }
        }

        int GetEntry(long entry)
        {
            /*
            In order to get performance comparable to TTreeFormula, we manually
            iterate over the branches we need and call TBranch::GetEntry. This
            is effectively the same procedure as TTree::GetEntry, except
            TTree::GetEntry loops over ALL branches, not just those which are
            active, and calls TBranch::GetEntry. While TBranch::GetEntry is a
            no-op in the case that the branch is inactive, this iteration can
            be a HUGE performance hit for TTrees with many branches, which is
            why TTreeFormula doesn't use TTree::GetEntry and sees far better
            performance.

            Note: The code in tree.pyx expects the return value of this
            function to be non-0, because TTree::GetEntry normally picks up
            those branches which are activate due to their membership in
            formulae. In fact, it is perfectly legitimate for TTree::GetEntry
            to return 0 without indicating an error, but to appease the
            existing code, we'll call GetEntry on all branches with formula
            membership, and it won't cost us anything since TTreeFormula won't
            reload them.
            */
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
            int total_read = 0;
            int read, ncodes;
            LeafCache::iterator lit, lend = leafcache.end();
            for (lit = leafcache.begin(); lit != lend; ++lit)
            {
                read = lit->second->leaf->GetBranch()->GetEntry(load);
                if (read < 0)
                {
                    return read;
                }
                total_read += read;
            }
            std::vector<TTreeFormula*>::iterator fit, fend = formulae.end();
            for (fit = formulae.begin(); fit != fend; ++fit)
            {
                ncodes = (*fit)->GetNcodes();
                for (int n = 0; n < ncodes; ++n)
                {
                    read = (*fit)->GetLeaf(n)->GetBranch()->GetEntry(load);
                    if (read < 0)
                    {
                        return read;
                    }
                    total_read += read;
                }
            }
            return total_read;
        }

        int Next()
        {
            return GetEntry(ientry++);
        }

        void Notify()
        {
            TBranch* branch;
            TLeaf* leaf;

            // Handle all the leaves
            LeafCache::iterator it;
            for(it = leafcache.begin(); it != leafcache.end(); ++it)
            {
                std::string bname = it->first.first;
                std::string lname = it->first.second;
                branch = fChain->FindBranch(bname.c_str());
                if (branch == NULL)
                {
                    std::cerr << "WARNING: cannot find branch " << bname
                              << std::endl;
                    it->second->skipped = true;
                    continue;
                }
                leaf = branch->FindLeaf(lname.c_str());
                if (leaf == NULL)
                {
                    std::cerr << "WARNING: cannot find leaf " << lname
                              << " for branch " << bname << std::endl;
                    it->second->skipped = true;
                    continue;
                }
                it->second->SetLeaf(leaf, true);
                it->second->skipped = false;
            }

            // Update all formula leaves and activate all object subbranches
            // used by the formulae
            int ncodes, n;
            std::vector<TTreeFormula*>::iterator fit;
            for (fit = formulae.begin(); fit != formulae.end(); ++fit)
            {
                (*fit)->UpdateFormulaLeaves();
                ncodes = (*fit)->GetNcodes();
                for (n = 0; n < ncodes; ++n)
                {
                    branch = (*fit)->GetLeaf(n)->GetBranch();
                    // Branch may be a TObject split across multiple
                    // subbranches. These must be activated recursively.
                    activate_branch_recursive(branch);
                }
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

        // Branch name to leaf name conversion
        typedef std::pair<std::string, std::string> BL;
        typedef std::map<BL, BranchColumn*> LeafCache;

        // Column pointer cache since the leaf inside needs to be updated
        // when new file is loaded in the chain
        LeafCache leafcache;
};

#endif
