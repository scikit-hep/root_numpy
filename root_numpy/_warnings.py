import warnings


__all__ = [
    'RootNumpyUnconvertibleWarning',
]


class RootNumpyUnconvertibleWarning(RuntimeWarning):
    """
    This warning is raised when root_numpy is unable to convert a branch into a
    column of a NumPy array because there is no converter for the type. If the
    user explicitly requests a branch that cannot be converted, an error is
    raised. If the user does not specify a list of branches in an attempt to
    convert all branches, then this warning is raised for each branch that
    cannot be converted and these branches are merely skipped.
    """

warnings.simplefilter('always', RootNumpyUnconvertibleWarning)
