#ifndef __TREE_CHAIN_H
#define __TREE_CHAIN_H

#include <TObject.h>
#include <TObjArray.h>
#include <TTree.h>
#include <TFile.h>
#include <TChain.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TTreeFormula.h>
#include <TTreeCache.h>

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <utility>

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

    TreeChain(TTree* tree, bool ischain, long long cache_size):
        tree(tree),
        ischain(ischain),
        itree(-1),
        ientry(0),
        cache_size(cache_size)
    {
        notifier = new MiniNotify(tree->GetNotify());
        tree->SetNotify(notifier);
    }

    ~TreeChain()
    {
        tree->SetNotify(notifier->oldnotify);

        // Delete all TTreeFormula
        std::vector<TTreeFormula*>::iterator fit;
        for (fit = formulae.begin(); fit != formulae.end(); ++fit)
        {
            delete *fit;
        }

        delete notifier;
    }

    int Prepare()
    {
        long long load = LoadTree(0);
        if (load >= 0)
        {
            // Enable all branches since we don't know yet which branches are
            // required by the formulae. The branches must be activated when a
            // TTreeFormula is initially created.
            tree->SetBranchStatus("*", true);
        }
        return (int)load;
    }

    long long LoadTree(long long entry)
    {
        long long load = tree->LoadTree(entry);
        if (load < 0)
        {
            return load;
        }
        if (tree->GetTreeNumber() != itree)
        {
            itree = tree->GetTreeNumber();
        }
        if (notifier->notified)
        {
            Notify();
            notifier->notified = false;
        }
        return load;
    }

    void AddFormula(TTreeFormula* formula)
    {
        // The TreeChain will take ownership of the formula
        formulae.push_back(formula);
    }

    void InitBranches()
    {
        // Prepare() must be called before InitBranches()
        TFile* file = NULL;
        TTreeCache* cache = NULL;
        TBranch* branch;
        TLeaf* leaf;
        std::string bname, lname;
        LeafCache::iterator it;

        // Set the cache size. This should create a new cache if one does not
        // already exist.
        tree->SetCacheSize(cache_size);

        // Get and set up the cache
        file = tree->GetCurrentFile();
        if (file)
        {
            cache = dynamic_cast<TTreeCache*>(file->GetCacheRead(tree));
        }
        if (cache)
        {
            cache->ResetCache();
            cache->StartLearningPhase();
        }

        for (it = leafcache.begin(); it != leafcache.end(); ++it)
        {
            leaf = it->second->leaf;
            branch = leaf->GetBranch();

            // Make the branch active and cache it
            branch->SetStatus(true);
            if (cache)
            {
                tree->AddBranchToCache(branch, true);
            }

            // ... and the length leaf as well
            // TODO: Does this work if user doesn't want the length column
            // in the output structure?
            TLeaf* leafCount = leaf->GetLeafCount();
            if (leafCount != NULL)
            {
                branch = leafCount->GetBranch();
                branch->SetStatus(true);
                if (cache)
                {
                    tree->AddBranchToCache(branch, true);
                }
            }
        }

        // Activate all branches used by the formulae
        int ncodes, icode;
        std::vector<TTreeFormula*>::iterator fit;
        for (fit = formulae.begin(); fit != formulae.end(); ++fit)
        {
            ncodes = (*fit)->GetNcodes();
            for (icode = 0; icode < ncodes; ++icode)
            {
                branch = (*fit)->GetLeaf(icode)->GetBranch();
                // Branch may be a TObject split across multiple
                // subbranches. These must be activated recursively.
                activate_branch_recursive(branch);
                if (cache)
                {
                    tree->AddBranchToCache(branch, true);
                }
            }
        }
        if (cache)
        {
            // Stop the cache learning phase since we have included a fixed set
            // of branches
            cache->StopLearningPhase();
        }
    }

    int GetEntry(long long entry)
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
        long long load = LoadTree(entry);
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
        std::string bname, lname;

        // Update all BranchColumn leaves
        LeafCache::iterator it;
        for(it = leafcache.begin(); it != leafcache.end(); ++it)
        {
            bname = it->first.first;
            lname = it->first.second;
            branch = tree->FindBranch(bname.c_str());
            if (branch == NULL)
            {
                std::cerr << "WARNING: cannot find branch " << bname;
                if (ischain)
                {
                    std::cerr << " in file " << ((TChain*) tree)->GetFile()->GetName();
                }
                std::cerr << std::endl;
                continue;
            }
            leaf = branch->FindLeaf(lname.c_str());
            if (leaf == NULL)
            {
                std::cerr << "WARNING: cannot find leaf " << lname
                          << " for branch " << bname
                          << std::endl;
                continue;
            }
            it->second->SetLeaf(leaf, true);
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

    void AddColumn(const std::string& branch_name,
                   const std::string& leaf_name,
                   BranchColumn* column)
    {
        BranchLeaf bl = make_pair(branch_name, leaf_name);
        leafcache.insert(make_pair(bl, column));
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

    TTree* tree;
    bool ischain;
    int itree;
    long long ientry;
    long long cache_size;
    MiniNotify* notifier;
    std::vector<TTreeFormula*> formulae;

    // Branch name to leaf name association
    typedef std::pair<std::string, std::string> BranchLeaf;
    typedef std::map<BranchLeaf, BranchColumn*> LeafCache;

    // Column pointer cache to update leaves
    // when new file is loaded in the chain
    LeafCache leafcache;
};

#endif
