include "converters.pyx"


def list_trees(fname):
    cdef TFile* rfile = Open(fname, 'read')
    if rfile == NULL:
        raise IOError("cannot read {0}".format(fname))
    cdef TList* keys = rfile.GetListOfKeys()
    if keys == NULL:
        raise IOError("unable to get keys in {0}".format(fname))
    ret = dict()
    cdef int nkeys = keys.GetEntries()
    cdef TKey* key
    for i in range(nkeys):
        key = <TKey*> keys.At(i)
        clsname = str(key.GetClassName())
        if clsname == 'TTree' or clsname == 'TNtuple':
            ret[str(key.GetName())] = None
    rfile.Close()
    del rfile
    return list(ret.keys())


def list_structures(fname, tree=None):
    if tree == None:
        # automatically select single tree
        tree = list_trees(fname)
        if len(tree) != 1:
            raise ValueError("multiple trees found: {0}".format(', '.join(tree)))
        tree = tree[0]
    cdef TFile* rfile = Open(fname, 'read')
    if rfile == NULL:
        raise IOError("cannot read {0}".format(fname))
    cdef TTree* rtree = <TTree*> rfile.Get(tree)
    if rtree == NULL:
        raise IOError("tree '{0}' not found in {1}".format(tree, fname))
    structure = get_tree_structure(rtree)
    rfile.Close()
    del rfile
    return structure


def list_branches(fname, tree=None):
    return list(list_structures(fname, tree).keys())


cdef get_branch_structure(TBranch* branch):
    cdef TObjArray* leaves
    cdef TLeaf* leaf
    cdef int ileaf
    leaves = branch.GetListOfLeaves()
    if leaves == NULL:
        raise RuntimeError("branch '{0}' has no leaves".format(branch.GetName()))
    leaflist = []
    for ileaf in range(leaves.GetEntries()):
        leaf = <TLeaf*>leaves.At(ileaf)
        leaflist.append((leaf.GetTitle(), resolve_type(leaf.GetTypeName())))
    if not leaflist:
        raise RuntimeError(
            "leaf list for branch '{0}' is empty".format(
                branch.GetName()))
    return leaflist


cdef get_tree_structure(TTree* tree, branches=None):
    cdef int ibranch
    cdef TBranch* branch
    ret = OrderedDict()
    if branches is not None:
        for branch_name in branches:
            branch = tree.GetBranch(branch_name)
            if branch == NULL:
                continue
            ret[branch.GetName()] = get_branch_structure(branch)
        return ret
    # all branches
    cdef TObjArray* all_branches = tree.GetListOfBranches()
    if all_branches == NULL:
        return ret
    for ibranch in range(all_branches.GetEntries()):
        branch = <TBranch*>(all_branches.At(ibranch))
        ret[branch.GetName()] = get_branch_structure(branch)
    return ret


cdef handle_load(int load, bool ignore_index=False):
    if load >= 0:
        return
    if load == -1:
        raise ValueError("chain is empty")
    elif load == -2:
        if ignore_index:
            return
        raise IndexError("tree index in chain is out of bounds")
    elif load == -3:
        raise IOError("cannot open current file")
    elif load == -4:
        raise IOError("cannot access tree in current file")
    raise RuntimeError("the chain is not initialized")


cdef object tree2array(TTree* tree, branches, string selection,
                       start, stop, step,
                       bool include_weight, string weight_name):

    if tree.GetNbranches() == 0:
        raise ValueError("tree has no branches")

    cdef int num_requested_branches = 0
    if branches is not None:
        num_requested_branches = len(branches)
        if num_requested_branches == 0:
            raise ValueError("branches is an empty list")

    cdef int num_entries = tree.GetEntries()
    cdef int num_entries_selected = 0

    cdef TreeChain* chain = new TreeChain(tree)
    handle_load(chain.Prepare(), True)

    cdef TObjArray* branch_array = tree.GetListOfBranches()
    cdef TObjArray* leaf_array
    cdef TBranch* tbranch
    cdef TLeaf* tleaf

    cdef Column* col
    cdef Converter* conv

    cdef vector[Column*] columns, columns_tmp
    cdef vector[Converter*] converters, converters_tmp
    # Used to preserve branch order if user specified branches:
    cdef vector[vector['Column*']] column_buckets
    cdef vector[vector['Converter*']] converter_buckets
    # Avoid calling FindBranch for each branch since that results in O(n^2)

    cdef TTreeFormula* selection_formula = NULL
    cdef TTreeFormula* formula = NULL

    cdef int ibranch, ileaf, ientry, branch_idx = 0
    cdef int num_branches = branch_array.GetEntries()
    cdef unsigned int icol, num_columns

    cdef np.ndarray arr
    cdef void* data_ptr
    cdef int num_bytes
    cdef int entry_size

    cdef char* c_string
    cdef bool shortname
    cdef string column_name
    cdef const_char* branch_name
    cdef const_char* leaf_name
    cdef string branch_title
    cdef int branch_title_size
    cdef char type_code

    if num_requested_branches > 0:
        columns.reserve(num_requested_branches)
        converters.reserve(num_requested_branches)
        column_buckets.assign(num_requested_branches, vector['Column*']())
        converter_buckets.assign(num_requested_branches, vector['Converter*']())
    else:
        columns.reserve(num_branches)
        converters.reserve(num_branches)

    try:
        # Set up the selection if we have one
        if selection.size():
            selection_formula = new TTreeFormula("selection", selection.c_str(), tree)
            if selection_formula == NULL or selection_formula.GetNdim() == 0:
                del selection_formula
                raise ValueError(
                    "could not compile selection expression '{0}'".format(selection))
            # The chain will take care of updating the formula leaves when
            # rolling over to the next tree.
            chain.AddFormula(selection_formula)

        branch_dict = None
        if num_requested_branches > 0:
            branch_dict = dict([(b, idx) for idx, b in enumerate(branches)])
            if len(branch_dict) != num_requested_branches:
                raise ValueError("duplicate branches requested")

        # Build vector of Converters for branches
        for ibranch in range(num_branches):
            tbranch = <TBranch*> branch_array.At(ibranch)
            branch_name = tbranch.GetName()
            if num_requested_branches > 0:
                if len(branch_dict) == 0:
                    # No more branches to consider
                    break
                branch_idx = branch_dict.pop(branch_name, -1)
                if branch_idx == -1:
                    # This branch was not selected by the user
                    continue

            branch_title = string(tbranch.GetTitle())
            branch_title_size = branch_title.size()
            if branch_title_size > 2 and branch_title[branch_title_size - 2] == '/':
                type_code = branch_title[branch_title_size - 1]
            else:
                type_code = '\0'
            leaf_array = tbranch.GetListOfLeaves()
            shortname = leaf_array.GetEntries() == 1

            for ileaf in range(leaf_array.GetEntries()):
                tleaf = <TLeaf*> leaf_array.At(ileaf)
                leaf_name = tleaf.GetName()
                conv = get_converter(tleaf, type_code)
                if conv != NULL:
                    # A converter exists for this leaf
                    column_name = string(branch_name)
                    if not shortname:
                        column_name.append(<string> '_')
                        column_name.append(leaf_name)
                    # Create a column for this branch/leaf pair
                    col = new BranchColumn(column_name, tleaf)

                    if num_requested_branches > 0:
                        column_buckets[branch_idx].push_back(col)
                        converter_buckets[branch_idx].push_back(conv)
                    else:
                        columns.push_back(col)
                        converters.push_back(conv)

                    chain.AddColumn(string(branch_name), string(leaf_name),
                                    <BranchColumn*> col)

                elif num_requested_branches > 0:
                    # User explicitly requested this branch but there is no
                    # converter to handle it
                    raise TypeError(
                        "cannot convert leaf '{0}' of branch '{1}' "
                        "with type '{2}'".format(
                            branch_name, leaf_name,
                            resolve_type(tleaf.GetTypeName())))
                else:
                    # Just warn that this branch cannot be converted
                    warnings.warn(
                        "cannot convert leaf '{0}' of branch '{1}' "
                        "with type '{2}' (skipping)".format(
                            branch_name, leaf_name,
                            resolve_type(tleaf.GetTypeName())),
                        RootNumpyUnconvertibleWarning)

        if num_requested_branches > 0:
            # Attempt to interpret remaining "branches" as expressions
            for expression in branch_dict.keys():
                branch_idx = branch_dict[expression]
                c_string = expression
                formula = new TTreeFormula(c_string, c_string, tree)
                if formula == NULL or formula.GetNdim() == 0:
                    del formula
                    raise ValueError(
                        "the branch or expression '{0}' "
                        "is not present or valid".format(expression))
                # The chain will take care of updating the formula leaves when
                # rolling over to the next tree.
                chain.AddFormula(formula)
                col = new FormulaColumn(expression, formula)
                conv = find_converter_by_typename('double')
                if conv == NULL:
                    # Oops, this should never happen
                    raise AssertionError(
                        "could not find double converter for formula")

                column_buckets[branch_idx].push_back(col)
                converter_buckets[branch_idx].push_back(conv)

            # Flatten buckets into 1D vectors, thus preserving branch order
            for branch_idx in range(num_requested_branches):
                columns.insert(columns.end(),
                               column_buckets[branch_idx].begin(),
                               column_buckets[branch_idx].end())
                converters.insert(converters.end(),
                                  converter_buckets[branch_idx].begin(),
                                  converter_buckets[branch_idx].end())

        elif columns.size() == 0:
            raise RuntimeError("unable to convert any branches in this tree")

        # Activate branches used by formulae and columns
        # and deactivate all others
        chain.InitBranches()

        # Now that we have all the columns we can
        # make an appropriate array structure
        dtype = []
        for icol in range(columns.size()):
            this_col = columns[icol]
            this_conv = converters[icol]
            dtype.append((this_col.name, this_conv.get_nptype()))
        if include_weight:
            dtype.append((weight_name, np.dtype('d')))

        # Initialize the array
        arr = np.empty(num_entries, dtype=dtype)

        # Exclude weight column in num_columns
        num_columns = columns.size()

        # Loop on entries in the tree and write the data in the array
        indices = slice(start, stop, step).indices(num_entries)
        for ientry in xrange(*indices):
            entry_size = chain.GetEntry(ientry)
            handle_load(entry_size)
            if entry_size == 0:
                raise IOError("read failure in current tree")

            # Determine if this entry passes the selection,
            # similar to the code in ROOT's tree/treeplayer/src/TTreePlayer.cxx
            if selection_formula != NULL:
                selection_formula.GetNdata() # required, as in TTreePlayer
                if selection_formula.EvalInstance(0) == 0:
                    continue

            # Copy the values into the array
            data_ptr = np.PyArray_GETPTR1(arr, num_entries_selected)
            for icol in range(num_columns):
                col = columns[icol]
                conv = converters[icol]
                num_bytes = conv.write(col, data_ptr)
                data_ptr = shift(data_ptr, num_bytes)
            if include_weight:
                (<double*> data_ptr)[0] = tree.GetWeight()

            # Increment number of selected entries last
            num_entries_selected += 1

    finally:
        # Delete TreeChain
        del chain
        # Delete Columns
        for icol in range(columns.size()):
            del columns[icol]

    # Shrink the array if we selected fewer than num_entries entries
    if num_entries_selected < num_entries:
        arr.resize(num_entries_selected)

    return arr


def root2array_fromFname(fnames, string treename, branches,
                         selection, start, stop, step,
                         bool include_weight, string weight_name):
    cdef TChain* ttree = NULL
    try:
        ttree = new TChain(treename.c_str())
        for fn in fnames:
            if ttree.Add(fn, -1) == 0:
                raise IOError("unable to access tree '{0}' in {1}".format(
                    treename, fn))
        ret = tree2array(
            <TTree*> ttree, branches,
            selection or '', start, stop, step,
            include_weight, weight_name)
    finally:
        del ttree
    return ret


def root2array_fromCObj(tree, branches, selection,
                        start, stop, step,
                        bool include_weight, string weight_name):
    cdef TTree* chain = <TTree*> PyCObject_AsVoidPtr(tree)
    return tree2array(
        chain, branches,
        selection or '', start, stop, step,
        include_weight, weight_name)


cdef TTree* array2tree(np.ndarray arr, string name='tree', TTree* tree=NULL) except *:
    cdef vector[NP2ROOTConverter*] converters
    cdef vector[int] posarray
    cdef vector[int] roffsetarray
    cdef unsigned int icv
    cdef int icol
    cdef long arr_len = arr.shape[0]
    cdef long idata
    cdef unsigned long pos_len
    cdef unsigned long ipos
    cdef void* source = NULL
    cdef void* thisrow = NULL

    try:
        if tree == NULL:
            tree = new TTree(name.c_str(), name.c_str())

        fieldnames = arr.dtype.names
        fields = arr.dtype.fields

        # Determine the structure
        for icol in range(len(fieldnames)):
            fieldname = fieldnames[icol]
            # roffset is an offset of particular field in each record
            dtype, roffset = fields[fieldname]
            cvt = find_np2root_converter(tree, fieldname, dtype)
            if cvt != NULL:
                roffsetarray.push_back(roffset)
                converters.push_back(cvt)
                posarray.push_back(icol)

        # Fill the data
        pos_len = posarray.size()
        for idata in range(arr_len):
            thisrow = np.PyArray_GETPTR1(arr, idata)
            for ipos in range(pos_len):
                roffset = roffsetarray[ipos]
                source = shift(thisrow, roffset)
                converters[ipos].fill_from(source)

        # Need to update the number of entries in the tree to match
        # the number in the branches since each branch is filled separately.
        tree.SetEntries(-1)

    except:
        raise

    finally:
        for icv in range(converters.size()):
            del converters[icv]
        # TODO: clean up tree

    return tree


def array2tree_toCObj(arr, name='tree', tree=None):
    cdef TTree* intree = NULL
    cdef TTree* outtree = NULL
    if tree is not None:
        intree = <TTree*> PyCObject_AsVoidPtr(tree)
    outtree = array2tree(arr, name=name, tree=intree)
    return PyCObject_FromVoidPtr(outtree, NULL)


def array2root(arr, filename, treename='tree', mode='update'):
    cdef TFile* rfile = Open(filename, mode)
    if rfile == NULL:
        raise IOError("cannot open file {0}".format(filename))
    if not rfile.IsWritable():
        raise IOError("file {0} is not writable".format(filename))
    # If a tree with that name exists, we want to update it
    cdef TTree* tree = <TTree*> rfile.Get(treename)
    tree = array2tree(arr, name=treename, tree=tree)
    tree.Write(treename, 2) # TObject::kOverwrite
    rfile.Close()
    # TODO: clean up tree
    del rfile
