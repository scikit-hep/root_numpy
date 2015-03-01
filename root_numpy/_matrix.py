from . import _librootnumpy


__all__ = [
    'matrix',
]


def matrix(mat):
    """Convert a ROOT TMatrix into a NumPy matrix.

    Parameters
    ----------
    mat : ROOT TMatrixT
        A ROOT TMatrixD or TMatrixF

    Returns
    -------
    mat : numpy.matrix
        A NumPy matrix

    Examples
    --------
    >>> from root_numpy import matrix
    >>> from ROOT import TMatrixD
    >>> a = TMatrixD(4, 4)
    >>> a[1][2] = 2
    >>> matrix(a)
    matrix([[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  2.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]])

    """
    import ROOT
    if isinstance(mat, (ROOT.TMatrixD, ROOT.TMatrixDSym)):
        return _librootnumpy.matrix_d(ROOT.AsCObject(mat))
    elif isinstance(mat, (ROOT.TMatrixF, ROOT.TMatrixFSym)):
        return _librootnumpy.matrix_f(ROOT.AsCObject(mat))
    raise TypeError(
        "unable to convert object of type {0} "
        "into a numpy matrix".format(type(mat)))
