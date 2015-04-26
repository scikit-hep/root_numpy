import numpy as np
import ROOT
from ROOT import TMVA
from . import _libtmvanumpy


__all__ = [
    'factory_add_events',
]


def factory_add_events(factory, events, labels,
                       signal_label=None, weights=None,
                       test=False):
    """Add training or test events to a TMVA::Factory from a NumPy array.

    Parameters
    ----------
    factory : TMVA::Factory
        A TMVA::Factory instance with variables already booked in
        exactly the same order as the columns in ``events``.
    events : numpy array of shape [n_events, n_variables]
        A two-dimensional NumPy array containing the rows of events
        and columns of variables.
    labels : numpy array of shape [n_events]
        The class labels (signal or background) corresponding to each event
        in ``events``.
    signal_label : float or int, optional (default=None)
        The value in ``labels`` for signal events.
    weights : numpy array of shape [n_events], optional
        Event weights.
    test : bool, optional (default=False)
        If True, then the events will be added as test events, otherwise
        they are added as training events by default.

    See Also
    --------
    reader_evaluate

    Examples
    --------
    .. literalinclude:: /examples/tmva.py

    """
    if not isinstance(factory, TMVA.Factory):
        raise TypeError("factory must be a TMVA.Factory instance")
    events = np.ascontiguousarray(events, dtype=np.float64)
    if events.ndim != 2:
        raise ValueError(
            "events must be a two-dimensional array "
            "with one event per row")
    class_labels, class_idx = np.unique(labels, return_inverse=True)
    if class_idx.shape[0] != events.shape[0]:
        raise ValueError("numbers of events and labels do not match")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != events.shape[0]:
            raise ValueError("numbers of events and weights do not match")
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
    if class_labels.shape[0] != 2:
        raise ValueError(
            "there must be exactly two classes "
            "present (found {0})".format(len(class_labels)))
    if signal_label is None:
        signal_label = class_labels[1]
    signal_label = np.where(class_labels == signal_label)[0][0]
    _libtmvanumpy.factory_add_events(
        ROOT.AsCObject(factory), events, class_idx,
        signal_label, weights, test)
