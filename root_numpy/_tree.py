from glob import glob
import numpy as np

import _librootnumpy


__all__ = [
    'root2array',
    'root2rec',
    'list_trees',
    'list_branches',
    'list_structures',
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
    if isinstance(filenames, basestring):
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


def list_structures(filename, treename=None):
    """Get a dictionary mapping branch names to leaf structures.

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
    return _librootnumpy.list_structures(filename, treename)


def root2array(filenames,
               treename=None,
               branches=None,
               selection=None,
               start=None,
               stop=None,
               step=None,
               include_weight=False,
               weight_name='weight'):
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

    selection : str, optional (default=None)
        Only include entries fulfilling this condition.

    start, stop, step: int, optional (default=None)
        The meaning of the ``start``, ``stop`` and ``step``
        parameters is the same as for Python slices.
        If a range is supplied (by setting some of the
        ``start``, ``stop`` or ``step`` parameters), only the entries in that
        range and fulfilling the ``selection`` condition (if defined) are used.

    include_weight : bool, optional (default=False)
        Include a column containing the tree weight.

    weight_name : str, optional (default='weight')
        The field name for the weight column if ``include_weight=True``.

    Examples
    --------

    Read all branches from the tree named ``mytree`` in ``a.root``
    Remember that ``mytree`` is optional if ``a.root`` has one tree::

        root2array('a.root', 'mytree')

    Read all branches starting from entry 5 and include 10 entries or up to the
    end of the file::

        root2array('a.root', 'mytree', start=5, stop=11)

    Read all branches in reverse order::

        root2array('a.root', 'mytree', step=-1)

    Read every second entry::

        root2array('a.root', 'mytree', step=2)

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
        else:
            treename = trees[0]

    return _librootnumpy.root2array_fromFname(
        filenames, treename, branches,
        selection,
        start, stop, step,
        include_weight,
        weight_name)


def root2rec(filenames,
             treename=None,
             branches=None,
             selection=None,
             start=None,
             stop=None,
             step=None,
             include_weight=False,
             weight_name='weight'):
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
    return root2array(filenames, treename,
                      branches, selection,
                      start, stop, step,
                      include_weight,
                      weight_name).view(np.recarray)


def tree2array(tree,
               branches=None,
               selection=None,
               start=None,
               stop=None,
               step=None,
               include_weight=False,
               weight_name='weight'):
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

    selection : str, optional (default=None)
        Only include entries fulfilling this condition.

    start, stop, step: int, optional (default=None)
        The meaning of the ``start``, ``stop`` and ``step``
        parameters is the same as for Python slices.
        If a range is supplied (by setting some of the
        ``start``, ``stop`` or ``step`` parameters), only the entries in that
        range and fulfilling the ``selection`` condition (if defined) are used.

    include_weight : bool, optional (default=False)
        Include a column containing the tree weight.

    weight_name : str, optional (default='weight')
        The field name for the weight column if ``include_weight=True``.

    See Also
    --------
    root2array

    """
    import ROOT
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")
    # will need AsCapsule for Python 3
    cobj = ROOT.AsCObject(tree)
    arr = _librootnumpy.root2array_fromCObj(
        cobj, branches, selection,
        start, stop, step,
        include_weight,
        weight_name)
    return arr


def tree2rec(tree,
             branches=None,
             selection=None,
             start=None,
             stop=None,
             step=None,
             include_weight=False,
             weight_name='weight'):
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
                      branches=branches,
                      selection=selection,
                      start=start,
                      stop=stop,
                      step=step,
                      include_weight=include_weight,
                      weight_name=weight_name).view(np.recarray)


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
