import numpy as np
import ROOT
from ROOT import TMVA
from . import _libtmvanumpy


__all__ = [
    'reader_evaluate',
]


def reader_evaluate(reader, name, events):
    """Evaluate a TMVA::Reader over a NumPy array.

    Parameters
    ----------
    reader : TMVA::Reader
        A TMVA::Factory instance with variables already booked in
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
    factory_add_events

    Examples
    --------
    .. literalinclude:: /examples/tmva.py

    """
    if not isinstance(reader, TMVA.Reader):
        raise TypeError("reader must be a TMVA.Reader instance")
    events = np.ascontiguousarray(events, dtype=np.float64)
    if events.ndim != 2:
        raise ValueError(
            "events must be a two-dimensional array "
            "with one event per row")
    return _libtmvanumpy.reader_evaluate(ROOT.AsCObject(reader), name, events)
