import numpy as np
import ROOT
from ROOT import TMVA
from . import _libtmvanumpy


__all__ = [
    'evaluate_reader',
    'evaluate_method',
]


def evaluate_reader(reader, name, events):
    """Evaluate a TMVA::Reader over a NumPy array.

    Parameters
    ----------
    reader : TMVA::Reader
        A TMVA::Factory instance with variables booked in
        exactly the same order as the columns in ``events``.
    name : string
        The method name.
    events : numpy array of shape [n_events, n_variables]
        A two-dimensional NumPy array containing the rows of events
        and columns of variables.

    Returns
    -------
    output : numpy array of shape [n_events]
        The method output value for each event

    See Also
    --------
    evaluate_method

    """
    if not isinstance(reader, TMVA.Reader):
        raise TypeError("reader must be a TMVA.Reader instance")
    method = reader.FindMVA(name)
    if not method:
        raise ValueError(
            "method '{0}' is not booked in this reader".format(name))
    return evaluate_method(method, events)


def evaluate_method(method, events):
    """Evaluate a TMVA::MethodBase over a NumPy array.

    Parameters
    ----------
    method : TMVA::MethodBase
        A TMVA::MethodBase instance with variables booked in
        exactly the same order as the columns in ``events``.
    events : numpy array of shape [n_events, n_variables]
        A two-dimensional NumPy array containing the rows of events
        and columns of variables.

    Returns
    -------
    output : numpy array of shape [n_events]
        The method output value for each event

    See Also
    --------
    evaluate_reader

    """
    if not isinstance(method, TMVA.MethodBase):
        raise TypeError("reader must be a TMVA.MethodBase instance")
    events = np.ascontiguousarray(events, dtype=np.float64)
    if events.ndim == 1:
        # convert to 2D
        events = events[:, np.newaxis]
    elif events.ndim != 2:
        raise ValueError(
            "events must be a two-dimensional array "
            "with one event per row")
    if events.shape[1] != method.GetNVariables():
        raise ValueError(
            "this method was trained with events containing "
            "{0} variables, but these events contain {1} variables".format(
                method.GetNVariables(), events.shape[1]))
    analysistype = method.GetAnalysisType()
    if analysistype == TMVA.Types.kClassification:
        return _libtmvanumpy.evaluate_twoclass(
            ROOT.AsCObject(method), events)
    elif analysistype == TMVA.Types.kMulticlass:
        n_classes = method.DataInfo().GetNClasses()
        if n_classes < 2:
            raise AssertionError("there must be at least two classes")
        return _libtmvanumpy.evaluate_multiclass(
            ROOT.AsCObject(method), events, n_classes)
    elif analysistype == TMVA.Types.kRegression:
        n_targets = method.DataInfo().GetNTargets()
        if n_targets < 1:
            raise AssertionError("there must be at least one regression target")
        output = _libtmvanumpy.evaluate_regression(
            ROOT.AsCObject(method), events, n_targets)
        if n_targets == 1:
            return np.ravel(output)
        return output
    raise AssertionError("the analysis type of this method is not supported")
