import ROOT
import root_numpy as rnp
from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_equal, assert_raises


def check_random_sample(obj):
    sample = rnp.random_sample(obj, 100)
    ndim = getattr(obj, 'GetDimension',
                   getattr(obj, 'GetNdim', None))()
    if ndim > 1:
        assert_equal(sample.shape, (100, ndim))
    else:
        assert_equal(sample.shape, (100,))
    a = rnp.random_sample(obj, 10, seed=1)
    b = rnp.random_sample(obj, 10, seed=1)
    c = rnp.random_sample(obj, 10, seed=2)
    assert_array_equal(a, b)
    assert_true((a != c).any())


def test_random_sample():
    funcs = [
        ROOT.TF1("f1", "TMath::DiLog(x)"),
        ROOT.TF2("f2", "sin(x)*sin(y)/(x*y)"),
        ROOT.TF3("f3", "sin(x)*sin(y)*sin(z)/(x*y*z)"),
    ]
    hists = [
        ROOT.TH1D("h1", "h1", 10, -3, 3),
        ROOT.TH2D("h2", "h2", 10, -3, 3, 10, -3, 3),
        ROOT.TH3D("h3", "h3", 10, -3, 3, 10, -3, 3, 10, -3, 3),
    ]
    for i, hist in enumerate(hists):
        hist.FillRandom(funcs[i].GetName())
    for obj in funcs + hists:
        yield check_random_sample, obj


def test_random_sample_bad_input():
    func = ROOT.TF1("f1", "TMath::DiLog(x)")
    assert_raises(ValueError, rnp.random_sample, func, 0)
    assert_raises(ValueError, rnp.random_sample, func, 10, seed=-1)
    assert_raises(TypeError, rnp.random_sample, object(), 10)
