from glob import glob
import numpy as np
from numpy.lib import recfunctions

import _librootnumpy
import _libnumpyhist


__all__ = [
    'root2array',
    'root2rec',
    'list_trees',
    'list_branches',
    'lt',
    'lst',
    'lb',
    'tree2array',
    'tree2rec',
    'array2tree',
    'array2root',
    'fill_array',
]


def _add_weight_field(arr,
                      tree,
                      weight_name='weight',
                      weight_dtype='f4'):
    """Add a new column containing the tree weight.

    """
    weights = np.empty(arr.shape[0], dtype=weight_dtype)
    weights.fill(tree.GetWeight())
    return recfunctions.rec_append_fields(
        arr, names=weight_name,
        data=weights,
        dtypes=weight_dtype)


def list_trees(filename):
    """List the trees in a ROOT file.

    """
    return _librootnumpy.list_trees(filename)


def lt(filename):
    """Shorthand for :func:`list_trees`

    """
    return _librootnumpy.list_trees(filename)


def list_branches(filename, treename=None):
    """Get a list of branches for trees in a ROOT file.

    Parameters
    ----------
    filename : str
        Path to ROOT file.
    treename : str, optional (default=None)
        Name of tree in the ROOT file.
        (optional if the ROOT file has only one tree).
    """
    return _librootnumpy.list_branches(filename, treename)


def lb(filename, treename=None):
    """Shorthand for :func:`list_branches`

    """
    return list_branches(filename, treename)


def lst(filename, treename=None):
    """Return tree structures.

    Parameters
    ----------
    filename : str
        Path to ROOT file.
    treename : str, optional (default=None)
        Name of tree in the ROOT file
        (optional if the ROOT file has only one tree).
    """
    return _librootnumpy.list_structures(filename, treename)


def root2array(filenames,
               treename=None,
               branches=None,
               entries=None,
               offset=0,
               selection=None):
    """
    Convert trees in ROOT files into a numpy structured array.
    Refer to the type conversion table :ref:`here <conversion_table>`.

    Parameters
    ----------
    filenames : str or list
        ROOT file name pattern or list of patterns. Wildcarding is
        supported by Python globbing.
    treename : str, optional (default=None)
        Name of the tree to convert (optional if each file contains exactly one
        tree).
    branches : list of str, optional (default=None)
        List of branch names to include as columns of the array.
        If None or empty then include all branches than can be converted in the
        first tree.
        If branches contains duplicate branches, only the first one is used.
    entries : int, optional (default=None)
        Maximum number of entries that will be converted from the chained
        trees. If None then convert all entries. If a selection is applied then
        fewer entries may be converted.
    offset : int, optional (default=0):
        Offset from the beginning of the chained trees where conversion will
        begin.
    selection : str, optional (default=None)
        Only include entries passing a cut expression.

    Examples
    --------

    Read all branches from the tree named ``mytree`` in ``a.root``
    Remember that ``mytree`` is optional if ``a.root`` has one tree::

        root2array('a.root', 'mytree')

    Read all branches starting from entry 5 and include 10 entries or up to the
    end of the file::

        root2array('a.root', 'mytree', entries=10, offset=5)

    Read all branches from the tree named ``mytree`` in ``a*.root``::

        root2array('a*.root', 'mytree')

    Read all branches from the tree named ``mytree`` in ``a*.root`` and
    ``b*.root``::

        root2array(['a*.root', 'b*.root'], 'mytree')

    Read branch ``x`` and ``y`` from the tree named ``mytree`` in ``a.root``::

        root2array('a.root', 'mytree', ['x', 'y'])

    Notes
    -----

    Due to the way TChain works, if the trees specified in the input files have
    different structures, only the branch in the first tree will be
    automatically extracted. You can work around this by either reordering the
    input file or specifying the branches manually.

    """
    matched_filenames = []
    if isinstance(filenames, basestring):
        matched_filenames = glob(filenames)
    else:
        for fn in filenames:
            tmp = glob(fn)
            if len(tmp) == 0:
                raise IOError('%s does not match any readable file.' % fn)
            matched_filenames.extend(tmp)

    if len(matched_filenames) == 0:
        raise IOError('pattern given does not match any file %s' % filenames)

    if treename is None:
        trees = list_trees(matched_filenames[0])
        if len(trees) != 1:
            raise ValueError('treename needs to be specified if the file '
                             'contains more than one tree. Your choices are:'
                             + str(trees))
        else:
            treename = trees[0]

    return _librootnumpy.root2array_fromFname(
        matched_filenames, treename, branches, entries, offset, selection)


def root2rec(filenames,
             treename=None,
             branches=None,
             entries=None,
             offset=0,
             selection=None):
    """
    View the result of :func:`root2array` as a record array.

    Notes
    -----
    This is equivalent to::

        root2array(filenames, treename, branches).view(np.recarray)

    See Also
    --------
    root2array
    """
    return root2array(filenames, treename, branches,
                      entries, offset, selection).view(np.recarray)


def tree2array(tree,
               branches=None,
               entries=None,
               offset=0,
               selection=None,
               include_weight=False,
               weight_name='weight',
               weight_dtype='f4'):
    """
    Convert a tree into a numpy structured array.
    Refer to the type conversion table :ref:`here <conversion_table>`.

    Parameters
    ----------
    treename : str
        Name of the tree to convert.
    branches : list of str, optional (default=None)
        List of branch names to include as columns of the array.
        If None or empty then include all branches than can be converted in the
        first tree.
        If branches contains duplicate branches, only the first one is used.
    entries : int, optional (default=None)
        Maximum number of entries that will be converted from the chained
        trees. If None then convert all entries. If a selection is applied then
        fewer entries may be converted.
    offset : int, optional (default=0):
        Offset from the beginning of the chained trees where conversion will
        begin.
    selection : str, optional (default=None)
        Only include entries passing a cut expression.
    include_weight : bool, optional (default=False)
        Include a column containing the tree weight.
    weight_name : str, optional (default='weight')
        The field name for the weight column if ``include_weight=True``.
    weight_dtype : NumPy dtype, optional (default='f4')
        The datatype to use for the weight column if ``include_weight=True``.

    See Also
    --------
    root2array

    """
    import ROOT
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")
    if hasattr(ROOT, 'AsCapsule'):
        #o = ROOT.AsCapsule(tree)
        # this will cause tons of compilation issue
        raise NotImplementedError()
        #return _librootnumpy.root2array_from_capsule(o, branches)
    cobj = ROOT.AsCObject(tree)
    arr = _librootnumpy.root2array_fromCObj(
        cobj, branches, entries, offset, selection)
    if include_weight:
        arr = _add_weight_field(arr, tree, weight_name, weight_dtype)
    return arr


def tree2rec(tree,
             branches=None,
             entries=None,
             offset=0,
             selection=None,
             include_weight=False,
             weight_name='weight',
             weight_dtype='f4'):
    """
    View the result of :func:`tree2array` as a record array.

    Notes
    -----
    This is equivalent to::

        tree2array(treename, branches).view(np.recarray)

    See Also
    --------
    tree2array

    """
    return tree2array(tree,
                      branches,
                      entries,
                      offset,
                      selection,
                      include_weight=include_weight,
                      weight_name=weight_name,
                      weight_dtype=weight_dtype).view(np.recarray)


def array2tree(arr, name='tree', tree=None):
    """
    Convert a numpy structured array into a ROOT TTree.

    .. warning::
       This function is experimental. Please report problems.
       Not all data types are supported (``np.object`` and ``np.float16``).

    Parameters
    ----------
    arr : array
        A numpy structured array
    name : str (optional, default='tree')
        Name of the created ROOT TTree if ``tree`` is None.
    tree : existing ROOT TTree (optional, default=None)
        Any branch with the same name as a field in the
        numpy array will be extended as long as the types are compatible,
        otherwise a TypeError is raised. New branches will be created
        and filled for all new fields.

    Returns
    -------
    root_tree : a ROOT TTree

    See Also
    --------
    array2root

    """
    import ROOT
    if tree is not None:
        if not isinstance(tree, ROOT.TTree):
            raise TypeError("tree must be a ROOT.TTree")
        incobj = ROOT.AsCObject(tree)
    else:
        incobj = None
    cobj = _librootnumpy.array2tree_toCObj(arr, name=name, tree=incobj)
    return ROOT.BindObject(cobj, 'TTree')


def array2root(arr, filename, treename='tree', mode='update'):
    """
    Convert a numpy structured array into a ROOT TTree and save directly in a
    ROOT TFile.

    .. warning::
       This function is experimental. Please report problems.
       Not all data types are supported (``np.object`` and ``np.float16``).

    Parameters
    ----------
    arr : array
        A numpy structured array
    filename : str
        Name of the output ROOT TFile. A new file will be created if it
        doesn't already exist.
    treename : str (optional, default='tree')
        Name of the created ROOT TTree.
    mode : str (optional, default='update')
        Mode used to open the ROOT TFile ('update' or 'recreate').

    See Also
    --------
    array2tree

    """
    _librootnumpy.array2root(arr, filename, treename, mode)


def fill_array(hist, array, weights=None):
    """Fill a ROOT histogram with a NumPy array.

    """
    import ROOT
    if not isinstance(hist, ROOT.TH1):
        raise TypeError("``hist`` must be a subclass of ROOT.TH1")
    hist = ROOT.AsCObject(hist)
    if weights is not None:
        _libnumpyhist.fill_hist_with_ndarray(
            hist, array, weights)
    else:
        _libnumpyhist.fill_hist_with_ndarray(
            hist, array)
