import numpy as np
from . import _librootnumpy


__all__ = [
    'fill_graph',
]


def fill_graph(graph, array):
    """Fill a ROOT graph with a NumPy array.

    Parameters
    ----------
    hist : a ROOT TGraph or TGraph2D
        The ROOT graph to fill.
    array : numpy array of shape [n_samples, n_dimensions]
        The values to fill the graph with. The number of columns must match the
        dimensionality of the graph.

    """
    import ROOT
    array = np.asarray(array, dtype=np.double)
    if isinstance(graph, ROOT.TGraph):
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != 2:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the graph")
        return _librootnumpy.fill_g1(
            ROOT.AsCObject(graph), array)
    elif isinstance(graph, ROOT.TGraph2D):
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != 3:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the graph")
        return _librootnumpy.fill_g2(
            ROOT.AsCObject(graph), array)
    else:
        raise TypeError(
            "hist must be an instance of ROOT.TGraph or ROOT.TGraph2D")
