__all__ = ['RootNumpyWarning',
           'RootNumpyUnconvertibleWarning']
import warnings

class RootNumpyWarning(RuntimeWarning):
    pass

class RootNumpyUnconvertibleWarning(RootNumpyWarning):
    pass

warnings.simplefilter('always', RootNumpyUnconvertibleWarning)
warnings.simplefilter('always', RootNumpyWarning)
