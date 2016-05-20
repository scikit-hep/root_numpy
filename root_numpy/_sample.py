from . import _librootnumpy


__all__ = [
    'random_sample',
]


def random_sample(obj, n_samples, seed=None):
    """Create a random array by sampling a ROOT function or histogram.

    Parameters
    ----------
    obj : TH[1|2|3] or TF[1|2|3]
        The ROOT function or histogram to sample.
    n_samples : positive int
        The number of random samples to generate.
    seed : None, positive int or 0, optional (default=None)
        The random seed, set via ROOT.gRandom.SetSeed(seed):
        http://root.cern.ch/root/html/TRandom3.html#TRandom3:SetSeed
        If 0, the seed will be random. If None (the default), ROOT.gRandom will
        not be touched and the current seed will be used.

    Returns
    -------
    array : a numpy array
        A numpy array with a shape corresponding to the dimensionality of the
        function or histogram. A flat array is returned when sampling TF1 or
        TH1. An array with shape [n_samples, n_dimensions] is returned when
        sampling TF2, TF3, TH2, or TH3.

    Examples
    --------
    >>> from root_numpy import random_sample
    >>> from ROOT import TF1, TF2, TF3
    >>> random_sample(TF1("f1", "TMath::DiLog(x)"), 10000, seed=1)
    array([ 0.68307934,  0.9988919 ,  0.87198158, ...,  0.50331049,
            0.53895257,  0.57576984])
    >>> random_sample(TF2("f2", "sin(x)*sin(y)/(x*y)"), 10000, seed=1)
    array([[ 0.93425084,  0.39990616],
           [ 0.00819315,  0.73108525],
           [ 0.00307176,  0.00427081],
           ...,
           [ 0.66931215,  0.0421913 ],
           [ 0.06469985,  0.10253632],
           [ 0.31059832,  0.75892702]])
    >>> random_sample(TF3("f3", "sin(x)*sin(y)*sin(z)/(x*y*z)"), 10000, seed=1)
    array([[ 0.03323949,  0.95734415,  0.39775191],
           [ 0.07093748,  0.01007775,  0.03330135],
           [ 0.80786963,  0.13641129,  0.14655269],
           ...,
           [ 0.96223632,  0.43916482,  0.05542078],
           [ 0.06631163,  0.0015063 ,  0.46550416],
           [ 0.88154752,  0.24332142,  0.66746564]])

    """
    import ROOT
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be positive or 0")
        ROOT.gRandom.SetSeed(seed)
    # functions
    if isinstance(obj, ROOT.TF1):
        if isinstance(obj, ROOT.TF3):
            return _librootnumpy.sample_f3(
                ROOT.AsCObject(obj), n_samples)
        elif isinstance(obj, ROOT.TF2):
            return _librootnumpy.sample_f2(
                ROOT.AsCObject(obj), n_samples)
        return _librootnumpy.sample_f1(ROOT.AsCObject(obj), n_samples)
    # histograms
    elif isinstance(obj, ROOT.TH1):
        if isinstance(obj, ROOT.TH3):
            return _librootnumpy.sample_h3(
                ROOT.AsCObject(obj), n_samples)
        elif isinstance(obj, ROOT.TH2):
            return _librootnumpy.sample_h2(
                ROOT.AsCObject(obj), n_samples)
        return _librootnumpy.sample_h1(ROOT.AsCObject(obj), n_samples)
    raise TypeError(
        "obj must be a ROOT function or histogram")
