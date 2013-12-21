import warnings


__all__ = [
    'RootNumpyWarning',
    'RootNumpyUnconvertibleWarning',
]


class RootNumpyWarning(RuntimeWarning):
    pass


class RootNumpyUnconvertibleWarning(RootNumpyWarning):
    pass

warnings.simplefilter('always', RootNumpyUnconvertibleWarning)
warnings.simplefilter('always', RootNumpyWarning)
