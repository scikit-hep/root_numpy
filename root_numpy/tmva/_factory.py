import numpy as np
import ROOT
from ROOT import TMVA
from . import _libtmvanumpy


__all__ = [
    'add_classification_events',
    'add_regression_events',
]


def add_classification_events(factory, events, labels, signal_label=None,
                              weights=None, test=False):
    """Add classification events to a TMVA::Factory from NumPy arrays.

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
        The value in ``labels`` for signal events, if ``labels`` contains only
        two classes. If None, the highest value in ``labels`` is used.
    weights : numpy array of shape [n_events], optional
        Event weights.
    test : bool, optional (default=False)
        If True, then the events will be added as test events, otherwise
        they are added as training events by default.

    """
    if not isinstance(factory, TMVA.Factory):
        raise TypeError("factory must be a TMVA.Factory instance")
    events = np.ascontiguousarray(events, dtype=np.float64)
    if events.ndim == 1:
        # convert to 2D
        events = events[:, np.newaxis]
    elif events.ndim != 2:
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
    n_classes = class_labels.shape[0]
    if n_classes > 2:
        # multiclass classification
        _libtmvanumpy.factory_add_events_multiclass(
            ROOT.AsCObject(factory), events, class_idx,
            weights, test)
    elif n_classes == 2:
        # binary classification
        if signal_label is None:
            signal_label = class_labels[1]
        signal_label = np.where(class_labels == signal_label)[0][0]
        _libtmvanumpy.factory_add_events_twoclass(
            ROOT.AsCObject(factory), events, class_idx,
            signal_label, weights, test)
    else:
        raise ValueError("labels must contain at least two classes")


def add_regression_events(factory, events, targets, weights=None, test=False):
    """Add regression events to a TMVA::Factory from NumPy arrays.

    Parameters
    ----------
    factory : TMVA::Factory
        A TMVA::Factory instance with variables already booked in
        exactly the same order as the columns in ``events``.
    events : numpy array of shape [n_events, n_variables]
        A two-dimensional NumPy array containing the rows of events
        and columns of variables.
    targets : numpy array of shape [n_events] or [n_events, n_targets]
        The target value(s) for each event in ``events``.
    weights : numpy array of shape [n_events], optional
        Event weights.
    test : bool, optional (default=False)
        If True, then the events will be added as test events, otherwise
        they are added as training events by default.

    """
    if not isinstance(factory, TMVA.Factory):
        raise TypeError("factory must be a TMVA.Factory instance")
    events = np.ascontiguousarray(events, dtype=np.float64)
    if events.ndim == 1:
        # convert to 2D
        events = events[:, np.newaxis]
    elif events.ndim != 2:
        raise ValueError(
            "events must be a two-dimensional array "
            "with one event per row")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape[0] != events.shape[0]:
        raise ValueError("the lengths of events and targets do not match")
    if targets.ndim == 1:
        # convert to 2D
        targets = targets[:, np.newaxis]
    elif targets.ndim > 2:
        raise ValueError("targets can not have more than two dimensions")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != events.shape[0]:
            raise ValueError("numbers of events and weights do not match")
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
    _libtmvanumpy.factory_add_events_regression(
        ROOT.AsCObject(factory), events, targets, weights, test)
