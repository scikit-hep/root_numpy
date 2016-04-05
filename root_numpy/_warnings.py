import warnings


__all__ = [
    'RootNumpyUnconvertibleWarning',
]


class RootNumpyUnconvertibleWarning(RuntimeWarning):
    pass

warnings.simplefilter('always', RootNumpyUnconvertibleWarning)
