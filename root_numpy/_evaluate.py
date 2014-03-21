import numpy as np
import _librootnumpy


__all__ = [
    'evaluate',
]


def evaluate(root_object, array):
    """
    Evaluate a ROOT histogram, function, graph, or spline at all values
    in a NumPy array and return the resulting array.

    Parameters
    ----------
    root_object : TH[1|2|3], TF[1|2|3], TGraph, TSpline
        A ROOT histogram, function, graph, or spline
    array : ndarray
        An array containing the values to evaluate the ROOT object on.
        The shape must match the dimensionality of the ROOT object.

    Returns
    -------
    y : array
        An array containing the values of the ROOT object evaluated at each
        value in the input array.

    Raises
    ------
    TypeError
        If the ROOT object is not a histogram, function, graph, or spline
    ValueError
        If the shape of the array is not compatible with the dimensionality
        of the ROOT object being evaluated.
    """
    import ROOT
    array = np.asarray(array, dtype=np.double)
    if isinstance(root_object, ROOT.TH1):
        if isinstance(root_object, ROOT.TH3):
            if array.ndim != 2:
                raise ValueError("array must be 2-dimensional")
            if array.shape[1] != 3:
                raise ValueError(
                    "length of the second dimension must equal "
                    "the dimension of the histogram")
            return _librootnumpy.evaluate_h3(ROOT.AsCObject(root_object), array)
        elif isinstance(root_object, ROOT.TH2):
            if array.ndim != 2:
                raise ValueError("array must be 2-dimensional")
            if array.shape[1] != 2:
                raise ValueError(
                    "length of the second dimension must equal "
                    "the dimension of the histogram")
            return _librootnumpy.evaluate_h2(ROOT.AsCObject(root_object), array)
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.evaluate_h1(ROOT.AsCObject(root_object), array)
    elif isinstance(root_object, ROOT.TF1):
        if isinstance(root_object, ROOT.TF3):
            if array.ndim != 2:
                raise ValueError("array must be 2-dimensional")
            if array.shape[1] != 3:
                raise ValueError(
                    "length of the second dimension must equal "
                    "the dimension of the function")
            return _librootnumpy.evaluate_f3(ROOT.AsCObject(root_object), array)
        elif isinstance(root_object, ROOT.TF2):
            if array.ndim != 2:
                raise ValueError("array must be 2-dimensional")
            if array.shape[1] != 2:
                raise ValueError(
                    "length of the second dimension must equal "
                    "the dimension of the function")
            return _librootnumpy.evaluate_f2(ROOT.AsCObject(root_object), array)
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.evaluate_f1(ROOT.AsCObject(root_object), array)
    elif isinstance(root_object, ROOT.TGraph):
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.evaluate_graph(ROOT.AsCObject(root_object), array)
    elif isinstance(root_object, ROOT.TSpline):
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.evaluate_spline(ROOT.AsCObject(root_object), array)
    raise TypeError(
        "root_object is not a ROOT histogram, function, graph, or spline")
