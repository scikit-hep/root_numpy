from glob import glob
import numpy as np

import _librootnumpy

__all__ = [
    'root2array',
    'root2rec',
    'list_trees',
    'list_branches',
    'lt',
    'lst',
    'lb',
    'tree2array',
    'tree2rec']
__version__ = '2.00'


def list_trees(fname):
    """list trees in rootfile *fname*"""
    return _librootnumpy.list_trees(fname)


def lt(fname):
    """shorthand for :func:`list_trees`"""
    return _librootnumpy.list_trees(fname)


def list_branches(fname, treename=None):
    """
    get a list of branches for given *fname* and *treename*.
    *treename* is optional if fname has only one tree
    """
    return _librootnumpy.list_branches(fname, treename)


def lb(fname, treename=None):
    """shorthand for :func:`list_branches`"""
    return list_branches(fname, treename)


def lst(fname, treename=None):
    """
    return tree structures.
    *treename* is optional if fname has only one tree
    """
    return _librootnumpy.list_structures(fname, treename)


def root2array(fnames, treename=None, branches=None, N=None, offset=0):

    """
    convert tree *treename* in root files specified in *fnames* to
    numpy structured array. Type conversion table
    is given :ref:`here <conversion_table>`

    Arguments:

        *fnames*: Root file name pattern. Wildcard is also supported by
        Python glob (not ROOT semi-broken wildcard)
        fnames can be string or list of string.

        *treename*: name of tree to convert to numpy array.
        This is optional if the file contains exactly 1 tree.

        *branches(optional)*: list of string for branch name to be
        extracted from tree.

        * If branches is not specified or is None or is empty,
          all from the first treebranches are extracted
        * If branches contains duplicate branches, only the first one is used.

        *N(optional)*: maximum number of data that it should load
        useful for testing out stuff

        *offset(optional)*: start index (first one is 0)

    **Example**

    ::

        # read all branches from tree named mytree from a.root
        # remember that 'mytree' is optional if a.root has 1 tree
        root2array('a.root', 'mytree')

    ::

        #read all branches starting from record 5 for 10 records
        #or the end of file.
        root2array('a.root', 'mytree',offset=5,N=10)

    ::

        # read all branches from tree named mytree from a*.root
        root2array('a*.root', 'mytree')

    ::

        # read all branches from tree named mytree from a*.root and b*.root
        root2array(['a*.root', 'b*.root'], 'mytree')

    ::

        #read branch x and y from tree named mytree from a.root
        root2array('a.root', 'mytree', ['x', 'y'])


    .. note::

        Due to
        the way TChain works, if the trees specified
        in the input files have different structures, only the
        branch in the first tree will be automatically extracted.
        You can work around this by either reordering the input
        file or specifying the branches manually.

    """
    if treename is None:
        afname = None
        if isinstance(fnames, basestring):
            afname = glob(fnames)
        else:
            afname = glob(fnames[0])
        trees = list_trees(afname[0])
        if len(trees) != 1:
            raise ValueError('treename need to be specified if the file '
                             'contains more than 1 tree. Your choices are:'
                             + str(trees))
        else:
            treename = trees[0]

    filenames = []
    if isinstance(fnames, basestring):
        filenames = glob(fnames)
    else:
        for fn in fnames:
            tmp = glob(fn)
            if len(tmp) == 0:
                raise IOError('%s does not match any readble file.' % tmp)
            filenames.extend(tmp)
    return _librootnumpy.root2array_fromFname(
        filenames, treename, branches, N, offset)


def root2rec(fnames, treename=None, branches=None, N=None, offset=0):
    """
    read branches in tree treename in file(s) given by fnames can
    convert it to numpy recarray

    This is equivalent to
    root2array(fnames, treename, branches).view(np.recarray)

    .. seealso::
        :func:`root2array`
    """
    return root2array(fnames, treename, branches, N, offset).view(np.recarray)


def tree2array(tree, branches=None, N=None, offset=0):
    """
    convert PyROOT TTree *tree* to numpy structured array
    see :func:`root2array` for details on parameter
    """
    import ROOT
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")

    if hasattr(ROOT, 'AsCapsule'):
        #o = ROOT.AsCapsule(tree)
        # this will cause tons of compilation issue
        raise NotImplementedError()
        #return _librootnumpy.root2array_from_capsule(o, branches)
    else:
        o = ROOT.AsCObject(tree)
        return _librootnumpy.root2array_fromCObj(o, branches, N, offset)


def tree2rec(tree, branches=None, N=None, offset=0):
    """
    convert PyROOT TTree *tree* to numpy structured array
    see :func:`root2array` for details on parameters.
    """
    return tree2array(tree, branches, N, offset).view(np.recarray)
