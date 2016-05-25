include "converters.pyx"


cdef list_objects_recursive(TDirectory* rdir, objects, vector[TClass*]& classes, path=""):
    cdef TList* keys = rdir.GetListOfKeys()
    if keys == NULL:
        raise IOError("unable to get keys in {0}".format(path))
    cdef TClass* tclass
    cdef vector[TClass*].iterator it
    cdef int nkeys = keys.GetEntries()
    cdef TKey* key
    for i in range(nkeys):
        key = <TKey*> keys.At(i)
        clsname = str(key.GetClassName())
        if not classes.empty():
            tclass = GetClass(clsname, True, True)
            if tclass != NULL:
                it = classes.begin()
                while it != classes.end():
                    if tclass.InheritsFrom(deref(it)):
                        objects.append(path + str(key.GetName()))
                        break
                    inc(it)
        else:
            objects.append(path + str(key.GetName()))
        if clsname == "TDirectoryFile":
            # recursively enter lower directory levels
            list_objects_recursive(<TDirectory*> rdir.Get(key.GetName()),
                                   objects, classes,
                                   path=path + key.GetName() + "/")


def list_objects(fname, types=None):
    cdef TClass* tclass
    # ROOT owns these pointers
    cdef vector[TClass*] classes
    if types is not None:
        for clsname in types:
            tclass = GetClass(clsname, True, True)
            if tclass == NULL:
                raise ValueError("'{0}' is not a ROOT class".format(clsname))
            classes.push_back(tclass)
    cdef TFile* rfile = Open(fname, 'read')
    if rfile == NULL:
        raise IOError("cannot read {0}".format(fname))
    objects = []
    list_objects_recursive(rfile, objects, classes)
    rfile.Close()
    del rfile
    return objects


def list_trees(fname):
    return list_objects(fname, types=['TTree'])


def list_directories(fname):
    return list_objects(fname, types=['TDirectoryFile'])


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


cdef humanize_bytes(long value, int precision=1):
    abbrevs = (
        (1<<50, 'PB'),
        (1<<40, 'TB'),
        (1<<30, 'GB'),
        (1<<20, 'MB'),
        (1<<10, 'kB'),
        (1, 'bytes'))
    if value == 1:
        return '1 byte'
    for factor, suffix in abbrevs:
        if value >= factor:
            break
    return '%.*f %s' % (precision, value / float(factor), suffix)


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


cdef object tree2array(TTree* tree, bool ischain, branches, string selection,
                       start, stop, step,
                       bool include_weight, string weight_name,
                       long cache_size):

    if tree.GetNbranches() == 0:
        raise ValueError("tree has no branches")

    cdef int num_requested_branches = 0
    if branches is not None:
        num_requested_branches = len(branches)
        if num_requested_branches == 0:
            raise ValueError("branches is an empty list")

    cdef long long num_entries = tree.GetEntries()
    cdef long long num_entries_selected = 0
    cdef long long ientry

    cdef TreeChain* chain = new TreeChain(tree, ischain, cache_size)
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
    cdef int instance
    cdef bool keep

    cdef int ibranch, ileaf, branch_idx = 0
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

        seen_branches = set()

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
            elif branch_name in seen_branches:
                warnings.warn("ignoring duplicate branch named '{0}'".format(branch_name),
                              RuntimeWarning)
                # Ignore duplicate branches
                continue
            else:
                seen_branches.add(branch_name)

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
                if formula.GetMultiplicity() > 0:
                    col = new MultiFormulaColumn(expression, formula)
                    conv = get_array_converter('double', '[]')
                else:
                    col = new FormulaColumn(expression, formula)
                    conv = find_converter_by_typename('double')
                if conv == NULL:
                    # Oops, this should never happen
                    raise AssertionError(
                        "could not find formula converter")
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
        chain.InitBranches()

        # Now that we have all the columns we can
        # make an appropriate array structure
        dtype_fields = []
        for icol in range(columns.size()):
            this_col = columns[icol]
            this_conv = converters[icol]
            dtype_fields.append((this_col.name, this_conv.get_nptype()))
        if include_weight:
            dtype_fields.append((weight_name, np.dtype('d')))
        dtype = np.dtype(dtype_fields)

        # Determine indices in slice
        indices = xrange(*(slice(start, stop, step).indices(num_entries)))
        num_entries = len(indices)

        # Initialize the array
        try:
            arr = np.empty(num_entries, dtype=dtype)
        except MemoryError:
            # Raise a more informative exception
            raise MemoryError("failed to allocate memory for {0} array of {1} records with {2} fields".format(
                humanize_bytes(dtype.itemsize * num_entries), num_entries, len(dtype_fields)))

        # Exclude weight column in num_columns
        num_columns = columns.size()

        # Loop on entries in the tree and write the data in the array
        for ientry in indices:
            entry_size = chain.GetEntry(ientry)
            handle_load(entry_size)
            if entry_size == 0:
                raise IOError("read failure in current tree or requested entry "
                              "does not exist (branches have different lengths?)")

            # Determine if this entry passes the selection,
            # similar to the code in ROOT's tree/treeplayer/src/TTreePlayer.cxx
            if selection_formula != NULL:
                keep = False
                for instance in range(selection_formula.GetNdata()):
                    if selection_formula.EvalInstance(instance) != 0:
                        keep = True
                        break
                if not keep:
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


def root2array_fromfile(fnames, string treename, branches,
                        selection, start, stop, step,
                        bool include_weight, string weight_name,
                        long cache_size, bool warn_missing_tree):
    cdef TChain* chain = NULL
    cdef TFile* file = NULL
    cdef TTree* tree = NULL
    try:
        chain = new TChain(treename.c_str())
        for fn in fnames:
            if warn_missing_tree:
                file = Open(fn, 'read')
                if file == NULL:
                    raise IOError("cannot open file {0}".format(fn))
                tree = <TTree*> file.Get(treename.c_str())
                if tree == NULL:
                    # skip this file
                    warnings.warn("tree '{0}' not found in {1}".format(treename, fn),
                                  RuntimeWarning)
                    file.Close()
                    continue
                del tree
                file.Close()
            if chain.Add(fn, -1) == 0:
                raise IOError("unable to access tree '{0}' in {1}".format(
                    treename, fn))
        if chain.GetNtrees() == 0:
            raise IOError("none of the input files contain "
                          "the requested tree '{0}'".format(treename))
        ret = tree2array(
            <TTree*> chain, True, branches,
            selection or '', start, stop, step,
            include_weight, weight_name, cache_size)
    finally:
        del chain
    return ret


def root2array_fromtree(tree, branches, selection,
                        start, stop, step,
                        bool include_weight, string weight_name,
                        long cache_size):
    cdef TTree* rtree = <TTree*> PyCObject_AsVoidPtr(tree)
    return tree2array(
        rtree, False, branches,
        selection or '', start, stop, step,
        include_weight, weight_name, cache_size)


cdef TTree* array2tree(np.ndarray arr, string name='tree', TTree* tree=NULL) except *:
    cdef vector[NP2ROOTConverter*] converters
    cdef NP2ROOTConverter* cvt
    cdef vector[int] roffsetarray
    cdef int roffset
    cdef unsigned int icol
    cdef unsigned int num_cols
    cdef SIZE_t arr_len = arr.shape[0]
    cdef SIZE_t idata
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
            else:
                warnings.warn("converter for {!r} is not "
                              "implemented (skipping)".format(dtype))

        # Fill the data
        num_cols = converters.size()
        for idata in range(arr_len):
            thisrow = np.PyArray_GETPTR1(arr, idata)
            for icol in range(num_cols):
                roffset = roffsetarray[icol]
                source = shift(thisrow, roffset)
                converters[icol].fill_from(source)

        # Need to update the number of entries in the tree to match
        # the number in the branches since each branch is filled separately.
        tree.SetEntries(-1)

    except:
        raise

    finally:
        for icol in range(converters.size()):
            del converters[icol]
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
    tree.Write(treename, kOverwrite)
    rfile.Close()
    # TODO: clean up tree
    del rfile
