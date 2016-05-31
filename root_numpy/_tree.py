import warnings
from glob import glob
import numpy as np

from .extern.six import string_types
from . import _librootnumpy


__all__ = [
    'root2array',
    'root2rec',
    'list_trees',
    'list_branches',
    'list_structures',
    'list_directories',
    'tree2array',
    'tree2rec',
    'array2tree',
    'array2root',
]


def _glob(filenames):
    """Glob a filename or list of filenames but always return the original
    string if the glob didn't match anything so URLs for remote file access
    are not clobbered.
    """
    if isinstance(filenames, string_types):
        filenames = [filenames]
    matches = []
    for name in filenames:
        matched_names = glob(name)
        if not matched_names:
            # use the original string
            matches.append(name)
        else:
            matches.extend(matched_names)
    return matches


def list_trees(filename):
    """Get list of the tree names in a ROOT file.

    Parameters
    ----------
    filename : str
        Path to ROOT file.

    Returns
    -------
    trees : list
        List of tree names

    """
    return _librootnumpy.list_trees(filename)


def list_branches(filename, treename=None):
    """Get a list of the branch names of a tree in a ROOT file.

    Parameters
    ----------
    filename : str
        Path to ROOT file.
    treename : str, optional (default=None)
        Name of tree in the ROOT file.
        (optional if the ROOT file has only one tree).

    Returns
    -------
    branches : list
        List of branch names

    """
    return _librootnumpy.list_branches(filename, treename)


def list_directories(filename):
    """Get a list of the directories in a ROOT file.

    Parameters
    ----------
    filename : str
        Path to ROOT file.

    Returns
    -------
    directories : list
        List of directory names.

    """
    return _librootnumpy.list_directories(filename)


def list_structures(filename, treename=None):
    """Get a dictionary mapping branch names to leaf structures.

    .. warning:: ``list_structures`` is deprecated and will be removed in
       release 5.0.0.

    Parameters
    ----------
    filename : str
        Path to ROOT file.
    treename : str, optional (default=None)
        Name of tree in the ROOT file
        (optional if the ROOT file has only one tree).

    Returns
    -------
    structures : OrderedDict
        An ordered dictionary mapping branch names to leaf structures.

    """
    warnings.warn("list_structures is deprecated and will be "
                  "removed in 5.0.0.", DeprecationWarning)
    return _librootnumpy.list_structures(filename, treename)


def root2array(filenames,
               treename=None,
               branches=None,
               selection=None,
               start=None,
               stop=None,
               step=None,
               include_weight=False,
               weight_name='weight',
               cache_size=-1,
               warn_missing_tree=False):
    """Convert trees in ROOT files into a numpy structured array.

    Refer to the documentation of :func:`tree2array`.

    Parameters
    ----------
    filenames : str or list
        ROOT file name pattern or list of patterns. Wildcarding is supported by
        Python globbing.
    treename : str, optional (default=None)
        Name of the tree to convert (optional if each file contains exactly one
        tree).
    branches : list of strings or single string, optional (default=None)
        List of branch names and expressions to include as columns of the
        array or a single branch name or expression to convert into a
        one-dimensional array. If None then include all branches that can be
        converted.
    selection : str, optional (default=None)
        Only include entries fulfilling this condition.
    start, stop, step: int, optional (default=None)
        The meaning of the ``start``, ``stop`` and ``step`` parameters is the
        same as for Python slices. If a range is supplied (by setting some of
        the ``start``, ``stop`` or ``step`` parameters), only the entries in
        that range and fulfilling the ``selection`` condition (if defined) are
        used.
    include_weight : bool, optional (default=False)
        Include a column containing the tree weight.
    weight_name : str, optional (default='weight')
        The field name for the weight column if ``include_weight=True``.
    cache_size : int, optional (default=-1)
        Set the size (in bytes) of the TTreeCache used while reading a TTree. A
        value of -1 uses ROOT's default cache size. A value of 0 disables the
        cache.
    warn_missing_tree : bool, optional (default=False)
        If True, then warn when a tree is missing from an input file instead of
        raising an IOError.

    Notes
    -----
    * Refer to the :ref:`type conversion table <conversion_table>`.

    See Also
    --------
    tree2array
    array2tree
    array2root

    """
    filenames = _glob(filenames)

    if not filenames:
        raise ValueError("specify at least one filename")

    if treename is None:
        trees = list_trees(filenames[0])
        if len(trees) > 1:
            raise ValueError(
                "treename must be specified if the file "
                "contains more than one tree")
        elif not trees:
            raise IOError(
                "no trees present in {0}".format(filenames[0]))
        treename = trees[0]

    if isinstance(branches, string_types):
        # single branch selected
        branches = [branches]
        flatten = True
    else:
        flatten = False

    arr = _librootnumpy.root2array_fromfile(
        filenames, treename, branches,
        selection,
        start, stop, step,
        include_weight,
        weight_name,
        cache_size,
        warn_missing_tree)

    if flatten:
        # select single column
        return arr[branches[0]]
    return arr


def root2rec(filenames,
             treename=None,
             branches=None,
             selection=None,
             start=None,
             stop=None,
             step=None,
             include_weight=False,
             weight_name='weight',
             cache_size=-1,
             warn_missing_tree=False):
    """View the result of :func:`root2array` as a record array.

    .. warning:: ``root2rec`` is deprecated and will be removed in
       release 5.0.0. Instead use ``root2array(...).view(np.recarray)``.

    Notes
    -----
    * This is equivalent to::

        root2array(filenames, treename, branches).view(np.recarray)

    * Refer to the :ref:`type conversion table <conversion_table>`.

    See Also
    --------
    root2array

    """
    warnings.warn("root2rec is deprecated and will be removed in 5.0.0. "
                  "Instead use root2array(...).view(np.recarray)",
                  DeprecationWarning)
    return root2array(filenames, treename,
                      branches, selection,
                      start, stop, step,
                      include_weight,
                      weight_name,
                      cache_size,
                      warn_missing_tree).view(np.recarray)


def tree2array(tree,
               branches=None,
               selection=None,
               start=None,
               stop=None,
               step=None,
               include_weight=False,
               weight_name='weight',
               cache_size=-1):
    """Convert a tree into a numpy structured array.

    Convert branches of strings and basic types such as bool, int, float,
    double, etc. as well as variable-length and fixed-length multidimensional
    arrays and 1D or 2D vectors of basic types and strings. ``tree2array`` can
    also create columns in the output array that are expressions involving the
    TTree branches (i.e. ``'vect.Pt() / 1000'``) similar to ``TTree::Draw()``.
    See the notes below for important details.

    Parameters
    ----------
    tree : ROOT TTree instance
        The ROOT TTree to convert into an array.
    branches : list of strings or single string, optional (default=None)
        List of branch names and expressions to include as columns of the
        array or a single branch name or expression to convert into a
        one-dimensional array. If None then include all branches that can be
        converted.
    selection : str, optional (default=None)
        Only include entries fulfilling this condition.
    start, stop, step: int, optional (default=None)
        The meaning of the ``start``, ``stop`` and ``step`` parameters is the
        same as for Python slices. If a range is supplied (by setting some of
        the ``start``, ``stop`` or ``step`` parameters), only the entries in
        that range and fulfilling the ``selection`` condition (if defined) are
        used.
    include_weight : bool, optional (default=False)
        Include a column containing the tree weight.
    weight_name : str, optional (default='weight')
        The field name for the weight column if ``include_weight=True``.
    cache_size : int, optional (default=-1)
        Set the size (in bytes) of the TTreeCache used while reading a TTree. A
        value of -1 uses ROOT's default cache size. A value of 0 disables the
        cache.

    Notes
    -----
    Types are converted according to the following table:

    .. _conversion_table:

    ========================  ===============================
    ROOT                      NumPy
    ========================  ===============================
    ``Bool_t``                ``np.bool``
    ``Char_t``                ``np.int8``
    ``UChar_t``               ``np.uint8``
    ``Short_t``               ``np.int16``
    ``UShort_t``              ``np.uint16``
    ``Int_t``                 ``np.int32``
    ``UInt_t``                ``np.uint32``
    ``Float_t``               ``np.float32``
    ``Double_t``              ``np.float64``
    ``Long64_t``              ``np.int64``
    ``ULong64_t``             ``np.uint64``
    ``<type>[2][3]...``       ``(<nptype>, (2, 3, ...))``
    ``<type>[nx][2]...``      ``np.object``
    ``string``                ``np.object``
    ``vector<t>``             ``np.object``
    ``vector<vector<t> >``    ``np.object``
    ========================  ===============================

    * Variable-length arrays (such as ``x[nx][2]``) and vectors (such as
      ``vector<int>``) are converted to NumPy arrays of the corresponding
      types.

    * Fixed-length arrays are converted to fixed-length NumPy array fields.

    **Branches with different lengths:**

    Note that when converting trees that have branches of different lengths
    into numpy arrays, the shorter branches will be extended to match the
    length of the longest branch by repeating their last values. If all
    requested branches are shorter than the longest branch in the tree, this
    will result in a "read failure" since beyond the end of the longest
    requested branch no additional bytes will be read from the file and
    root_numpy is unable to distinguish this from other ROOT errors that result
    in no bytes being read. In this case, explicitly set the ``stop`` argument
    to the length of the longest requested branch.


    See Also
    --------
    root2array
    array2root
    array2tree

    """
    import ROOT
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")
    cobj = ROOT.AsCObject(tree)

    if isinstance(branches, string_types):
        # single branch selected
        branches = [branches]
        flatten = True
    else:
        flatten = False

    arr = _librootnumpy.root2array_fromtree(
        cobj, branches, selection,
        start, stop, step,
        include_weight,
        weight_name,
        cache_size)

    if flatten:
        # select single column
        return arr[branches[0]]
    return arr


def tree2rec(tree,
             branches=None,
             selection=None,
             start=None,
             stop=None,
             step=None,
             include_weight=False,
             weight_name='weight',
             cache_size=-1):
    """View the result of :func:`tree2array` as a record array.

    .. warning:: ``tree2rec`` is deprecated and will be removed in
       release 5.0.0. Instead use ``tree2array(...).view(np.recarray)``.

    Notes
    -----
    * This is equivalent to::

        tree2array(treename, branches).view(np.recarray)

    * Refer to the :ref:`type conversion table <conversion_table>`.

    See Also
    --------
    tree2array

    """
    warnings.warn("tree2rec is deprecated and will be removed in 5.0.0. "
                  "Instead use tree2array(...).view(np.recarray)",
                  DeprecationWarning)
    return tree2array(tree,
                      branches=branches,
                      selection=selection,
                      start=start,
                      stop=stop,
                      step=step,
                      include_weight=include_weight,
                      weight_name=weight_name,
                      cache_size=cache_size).view(np.recarray)


def array2tree(arr, name='tree', tree=None):
    """Convert a numpy structured array into a ROOT TTree.

    Fields of basic types, strings, and fixed-size subarrays of basic types are
    supported. ``np.object`` and ``np.float16`` are currently not supported.

    Parameters
    ----------
    arr : array
        A numpy structured array
    name : str (optional, default='tree')
        Name of the created ROOT TTree if ``tree`` is None.
    tree : ROOT TTree (optional, default=None)
        An existing ROOT TTree to be extended by the numpy array. Any branch
        with the same name as a field in the numpy array will be extended as
        long as the types are compatible, otherwise a TypeError is raised. New
        branches will be created and filled for all new fields.

    Returns
    -------
    root_tree : a ROOT TTree

    Notes
    -----
    When using the ``tree`` argument to extend and/or add new branches to an
    existing tree, note that it is possible to create branches of different
    lengths. This will result in a warning from ROOT when root_numpy calls the
    tree's ``SetEntries()`` method. Beyond that, the tree should still be
    usable. While it might not be generally recommended to create branches with
    differing lengths, this behaviour could be required in certain situations.
    root_numpy makes no attempt to prevent such behaviour as this would be more
    strict than ROOT itself. Also see the note about converting trees that have
    branches of different lengths into numpy arrays in the documentation of
    :func:`tree2array`.

    See Also
    --------
    array2root
    root2array
    tree2array

    Examples
    --------

    Convert a numpy array into a tree:

    >>> from root_numpy import array2tree
    >>> import numpy as np
    >>>
    >>> a = np.array([(1, 2.5, 3.4),
    ...               (4, 5, 6.8)],
    ...              dtype=[('a', np.int32),
    ...                     ('b', np.float32),
    ...                     ('c', np.float64)])
    >>> tree = array2tree(a)
    >>> tree.Scan()
    ************************************************
    *    Row   *         a *         b *         c *
    ************************************************
    *        0 *         1 *       2.5 *       3.4 *
    *        1 *         4 *         5 *       6.8 *
    ************************************************

    Add new branches to an existing tree (continuing from the example above):

    >>> b = np.array([(4, 10),
    ...               (3, 5)],
    ...              dtype=[('d', np.int32),
    ...                     ('e', np.int32)])
    >>> array2tree(b, tree=tree)
    <ROOT.TTree object ("tree") at 0x1449970>
    >>> tree.Scan()
    ************************************************************************
    *    Row   *         a *         b *         c *         d *         e *
    ************************************************************************
    *        0 *         1 *       2.5 *       3.4 *         4 *        10 *
    *        1 *         4 *         5 *       6.8 *         3 *         5 *
    ************************************************************************

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
    """Convert a numpy array into a ROOT TTree and save it in a ROOT TFile.

    Fields of basic types, strings, and fixed-size subarrays of basic types are
    supported. ``np.object`` and ``np.float16`` are currently not supported.

    Parameters
    ----------
    arr : array
        A numpy structured array
    filename : str
        Name of the output ROOT TFile. A new file will be created if it doesn't
        already exist.
    treename : str (optional, default='tree')
        Name of the ROOT TTree that will be created. If a TTree with the same
        name already exists in the TFile, it will be extended as documented in
        :func:`array2tree`.
    mode : str (optional, default='update')
        Mode used to open the ROOT TFile ('update' or 'recreate').

    See Also
    --------
    array2tree
    tree2array
    root2array

    Examples
    --------

    >>> from root_numpy import array2root, root2array
    >>> import numpy as np
    >>>
    >>> a = np.array([(1, 2.5, 3.4),
    ...               (4, 5, 6.8)],
    ...              dtype=[('a', np.int32),
    ...                     ('b', np.float32),
    ...                     ('c', np.float64)])
    >>> array2root(a, 'test.root', mode='recreate')
    >>> root2array('test.root')
    array([(1, 2.5, 3.4), (4, 5.0, 6.8)],
          dtype=[('a', '<i4'), ('b', '<f4'), ('c', '<f8')])
    >>>
    >>> a = np.array(['', 'a', 'ab', 'abc', 'xyz', ''],
    ...              dtype=[('string', 'S3')])
    >>> array2root(a, 'test.root', mode='recreate')
    >>> root2array('test.root')
    array([('',), ('a',), ('ab',), ('abc',), ('xyz',), ('',)],
          dtype=[('string', 'S3')])
    >>>
    >>> a = np.array([([1, 2, 3],),
    ...               ([4, 5, 6],)],
    ...              dtype=[('array', np.int32, (3,))])
    >>> array2root(a, 'test.root', mode='recreate')
    >>> root2array('test.root')
    array([([1, 2, 3],), ([4, 5, 6],)],
          dtype=[('array', '<i4', (3,))])

    """
    _librootnumpy.array2root(arr, filename, treename, mode)
