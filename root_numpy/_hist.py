import numpy as np
import _librootnumpy


__all__ = [
    'fill_hist',
]


def fill_hist(hist, array, weights=None, return_indices=False):
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

    return_indices : bool, optional (default=False)
        If True then return an array of the bin indices filled for each element
        in ``array``.

    Returns
    -------

    indices : numpy array or None
        If ``return_indices`` is True, then return an array of the bin indices
        filled for each element in ``array`` otherwise return None.

    """
    import ROOT
    array = np.asarray(array, dtype=np.double)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.double)
        if weights.shape[0] != array.shape[0]:
            raise ValueError("array and weights must have the same length")
        if weights.ndim != 1:
            raise ValueError("weight must be 1-dimensional")
    if isinstance(hist, ROOT.TH3):
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != 3:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the histogram")
        return _librootnumpy.fill_h3(
            ROOT.AsCObject(hist), array, weights, return_indices)
    elif isinstance(hist, ROOT.TH2):
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != 2:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the histogram")
        return _librootnumpy.fill_h2(
            ROOT.AsCObject(hist), array, weights, return_indices)
    elif isinstance(hist, ROOT.TH1):
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.fill_h1(
            ROOT.AsCObject(hist), array, weights, return_indices)
    else:
        raise TypeError(
            "hist must be an instance of ROOT.TH1, ROOT.TH2, or ROOT.TH3")
