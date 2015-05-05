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
    events = np.ascontiguousarray(events, dtype=np.float64)
    if events.ndim == 1:
        # convert to 2D
        events = events[:, np.newaxis]
    elif events.ndim != 2:
        raise ValueError(
            "events must be a two-dimensional array "
            "with one event per row")
    return _libtmvanumpy.evaluate_reader(ROOT.AsCObject(reader), name, events)


def evaluate_method(method, events):
    """Evaluate a TMVA::MethodBase over a NumPy array.

    .. warning:: TMVA::Reader has known problems with thread safety in versions
       of ROOT earlier than 6.03. There will potentially be a crash if you call
       ``method = reader.FindMVA(name)`` in Python and then pass this
       ``method`` here. Consider using ``evaluate_reader`` instead if you are
       affected by this crash.

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
    return _libtmvanumpy.evaluate_method(ROOT.AsCObject(method), events)
