try:
    from . import _libtmvanumpy

except ImportError:  # pragma: no cover
    import warnings
    warnings.warn(
        "root_numpy.tmva requires that you install root_numpy with "
        "the tmva interface enabled", ImportWarning)
    __all__ = []

else:

    from ._data import add_classification_events, add_regression_events
    from ._evaluate import evaluate_reader, evaluate_method


    __all__ = [
        'add_classification_events',
        'add_regression_events',
        'evaluate_reader',
        'evaluate_method',
    ]
