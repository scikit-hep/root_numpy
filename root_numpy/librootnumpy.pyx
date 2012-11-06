import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from pprint import pprint
from cython.operator cimport dereference as deref
include "all.pxi"

#numpy

def root2array_fromPyRoot(ttree, branches=None):
    pass

cdef parse_tree_structure(TTree* tree):
    cdef char* name
    cdef TBranch* thisBranch
    cdef TLeaf* thisLeaf
    cdef TObjArray* branches = tree.GetListOfBranches()
    cdef TObjArray* leaves
    ret = {}
    for ibranch in range(branches.GetEntries()):
        thisBranch = <TBranch*>(branches.At(ibranch))
        leaves = thisBranch.GetListOfLeaves()
        leaflist = []
        for ibranch in range(leaves.GetEntries()):
            thisLeaf = <TLeaf*>leaves.At(ibranch)
            leaflist.append(thisLeaf.GetName())
        ret[thisBranch.GetName()] = leaflist
    return ret

cdef root2array_fromTTree(TTree* tree,branches=None, N=None): #from CPP TTree
    #this is actually vector of pointers despite how it looks
    cdef vector[Column*] columns
    cdef Column* thisCol

    #make a better chain so we can register all columns
    cdef BetterChain* bc = new BetterChain(tree)

    #parse the tree structure to determine 
    #whether to use shortname or long name
    #and loop through all leaves
    structure = parse_tree_structure(tree)
    if branches is None: branches = structure.keys()
    for branch in branches:
        leaves = structure[branch]
        shortname = len(leaves)==1
        for leaf in leaves:
            colname = branch if shortname else '%s_%s'%(branch,leaf)
        thisCol = bc.MakeColumn(branch, leaf, colname)
        columns.push_back(thisCol)
        thisCol.Print()

    del bc



 # cdef BetterChain bc = BetterChain(tree)
 #    cdef vector[Column] columns #this is actually vector of pointer
    
 #    #get tree structure
 #    for b in branches:
 #        columns.append()
    #prepare numpy structure



def root2array_fromFname(fnames, treename, branches=None, N=None):
    cdef TChain *ttree = new TChain(treename)
    for fn in fnames:
        ttree.Add(fn)
    root2array_fromTTree(<TTree*>ttree,branches)
    del ttree
