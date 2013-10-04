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
    'list_structures',
    'tree2array',
    'tree2rec',
    'array2tree',
    'array2root',
    'fill_array',
    'random_sample',
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
               step=None):
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
        filenames, treename, branches, selection, start, stop, step)


def root2rec(filenames,
             treename=None,
             branches=None,
             selection=None,
             start=None,
             stop=None,
             step=None):
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
    return root2array(filenames, treename, branches, selection,
                      start, stop, step).view(np.recarray)


def tree2array(tree,
               branches=None,
               selection=None,
               start=None,
               stop=None,
               step=None,
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
        cobj, branches, selection, start, stop, step)
    if include_weight:
        arr = _add_weight_field(arr, tree, weight_name, weight_dtype)
    return arr


def tree2rec(tree,
             branches=None,
             selection=None,
             start=None,
             stop=None,
             step=None,
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
                      branches=branches,
                      selection=selection,
                      start=start,
                      stop=stop,
                      step=step,
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
    """
    Fill a ROOT histogram with a NumPy array.

    Parameters
    ----------

    hist : a ROOT TH1, TH2, or TH3
        The ROOT histogram to fill.

    array : numpy array of shape [n_samples, n_dimensions]
        The values to fill the histogram with. The number of columns
        must match the dimensionality of the histogram. Supply a flat
        numpy array when filling a 1D histogram.

    weights : numpy array
        A flat numpy array of weights for each sample in ``array``.

    """
    import ROOT
    if not isinstance(hist, ROOT.TH1):
        raise TypeError(
            "hist must be an instance of ROOT.TH1, ROOT.TH2, or ROOT.TH3")
    hist = ROOT.AsCObject(hist)
    if weights is not None:
        _libnumpyhist.fill_hist_with_ndarray(
            hist, array, weights)
    else:
        _libnumpyhist.fill_hist_with_ndarray(
            hist, array)


def random_sample(func, n_samples, seed=None):
    """
    Construct a NumPy array from a random sampling of a ROOT function.

    Parameters
    ----------

    func : a ROOT TF1, TF2, or TF3
        The ROOT function to sample.

    n_samples : positive int
        The number of random samples to generate.

    seed : None, positive int or 0, optional (default=None)
        The random seed, set via ROOT.gRandom.SetSeed(seed):
        http://root.cern.ch/root/html/TRandom3.html#TRandom3:SetSeed
        If 0, the seed will be random. If None (the default),
        ROOT.gRandom will not be touched and the current seed will be used.

    Returns
    -------

    array : a numpy array
        A numpy array with a shape corresponding to the dimensionality
        of the function. A flat array is returned when sampling TF1.
        An array with shape [n_samples, n_dimensions] is returned when
        sampling TF2 or TF3.

    Examples
    --------

    >>> from root_numpy import random_sample
    >>> from ROOT import TF1, TF2, TF3
    >>> random_sample(TF1("f1", "TMath::DiLog(x)"), 1E4, seed=1)
    array([ 0.68307934,  0.9988919 ,  0.87198158, ...,  0.50331049,
            0.53895257,  0.57576984])
    >>> random_sample(TF2("f2", "sin(x)*sin(y)/(x*y)"), 1E4, seed=1)
    array([[ 0.93425084,  0.39990616],
           [ 0.00819315,  0.73108525],
           [ 0.00307176,  0.00427081],
           ...,
           [ 0.66931215,  0.0421913 ],
           [ 0.06469985,  0.10253632],
           [ 0.31059832,  0.75892702]])
    >>> random_sample(TF3("f3", "sin(x)*sin(y)*sin(z)/(x*y*z)"), 1E4, seed=1)
    array([[ 0.03323949,  0.95734415,  0.39775191],
           [ 0.07093748,  0.01007775,  0.03330135],
           [ 0.80786963,  0.13641129,  0.14655269],
           ...,
           [ 0.96223632,  0.43916482,  0.05542078],
           [ 0.06631163,  0.0015063 ,  0.46550416],
           [ 0.88154752,  0.24332142,  0.66746564]])

    """
    import ROOT
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be positive or 0")
        ROOT.gRandom.SetSeed(seed)
    if isinstance(func, ROOT.TF3):
        return _librootnumpy.sample_f3(ROOT.AsCObject(func), n_samples)
    elif isinstance(func, ROOT.TF2):
        return _librootnumpy.sample_f2(ROOT.AsCObject(func), n_samples)
    elif isinstance(func, ROOT.TF1):
        return _librootnumpy.sample_f1(ROOT.AsCObject(func), n_samples)
    else:
        raise TypeError(
            "func must be an instance of ROOT.TF1, ROOT.TF2 or ROOT.TF3")
