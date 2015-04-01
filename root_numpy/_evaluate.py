import uuid
import numpy as np

from .extern.six import string_types
from . import _librootnumpy


__all__ = [
    'evaluate',
]


def evaluate(root_object, array):
    """Evaluate a ROOT histogram, function, graph, or spline over an array.

    Parameters
    ----------
    root_object : TH[1|2|3], TF[1|2|3], TFormula, TGraph, TSpline, or string
        A ROOT histogram, function, formula, graph, spline, or string. If a
        string is specified, a TFormula is created.
    array : ndarray
        An array containing the values to evaluate the ROOT object on. The
        shape must match the dimensionality of the ROOT object.

    Returns
    -------
    y : array
        An array containing the values of the ROOT object evaluated at each
        value in the input array.

    Raises
    ------
    TypeError
        If the ROOT object is not a histogram, function, graph, or spline.
    ValueError
        If the shape of the array is not compatible with the dimensionality of
        the ROOT object being evaluated. If the string expression does not
        compile to a valid TFormula expression.

    Examples
    --------
    >>> from root_numpy import evaluate
    >>> from ROOT import TF1, TF2
    >>> func = TF1("f1", "x*x")
    >>> evaluate(func, [1, 2, 3, 4])
    array([  1.,   4.,   9.,  16.])
    >>> func = TF2("f2", "x*y")
    >>> evaluate(func, [[1, 1], [1, 2], [3, 1]])
    array([ 1.,  2.,  3.])
    >>> evaluate("x*y", [[1, 1], [1, 2], [3, 1]])
    array([ 1.,  2.,  3.])

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
    elif isinstance(root_object, (string_types, ROOT.TFormula)):
        if isinstance(root_object, string_types):
            # attempt to create a formula
            root_object = ROOT.TFormula(uuid.uuid4().hex, root_object)
        ndim = root_object.GetNdim()
        if ndim == 0:
            raise ValueError("invalid formula expression")
        if ndim == 1:
            if array.ndim != 1:
                raise ValueError("array must be 1-dimensional")
            return _librootnumpy.evaluate_formula_1d(
                ROOT.AsCObject(root_object), array)
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if array.shape[1] != ndim:
            raise ValueError(
                "length of the second dimension must equal "
                "the dimension of the function")
        if ndim == 2:
            return _librootnumpy.evaluate_formula_2d(
                ROOT.AsCObject(root_object), array)
        elif ndim == 3:
            return _librootnumpy.evaluate_formula_3d(
                ROOT.AsCObject(root_object), array)
        # 4d
        return _librootnumpy.evaluate_formula_4d(
            ROOT.AsCObject(root_object), array)
    elif isinstance(root_object, ROOT.TGraph):
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.evaluate_graph(ROOT.AsCObject(root_object), array)
    elif isinstance(root_object, ROOT.TSpline):
        if array.ndim != 1:
            raise ValueError("array must be 1-dimensional")
        return _librootnumpy.evaluate_spline(ROOT.AsCObject(root_object), array)
    raise TypeError(
        "root_object is not a ROOT histogram, function, formula, "
        "graph, spline or string")
