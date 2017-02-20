import ROOT
import root_numpy as rnp
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
from . import RNG


def make_histogram(hist_type, shape, fill=True):
    # shape=([[z_bins,] y_bins,] x_bins)
    ndim = len(shape)
    hist_cls = getattr(ROOT, 'TH{0}{1}'.format(ndim, hist_type))
    if ndim == 1:
        hist = hist_cls(hist_cls.__name__, '',
                        shape[0], 0, shape[0])
        func = ROOT.TF1('func', 'x')
    elif ndim == 2:
        hist = hist_cls(hist_cls.__name__, '',
                        shape[1], 0, shape[1],
                        shape[0], 0, shape[0])
        func = ROOT.TF2('func', 'x*y')
    elif ndim == 3:
        hist = hist_cls(hist_cls.__name__, '',
                        shape[2], 0, shape[2],
                        shape[1], 0, shape[1],
                        shape[0], 0, shape[0])
        func = ROOT.TF3('func', 'x*y*z')
    else:
        raise ValueError("ndim must be 1, 2, or 3")  # pragma: no cover
    if fill:
        hist.FillRandom('func')
    return hist


def check_hist2array(hist, include_overflow, copy):
    array = rnp.hist2array(hist, include_overflow=include_overflow, copy=copy)
    assert_equal(hist.GetDimension(), array.ndim)
    for iaxis, axis in enumerate('XYZ'[:array.ndim]):
        if include_overflow:
            assert_equal(array.shape[iaxis],
                         getattr(hist, 'GetNbins{0}'.format(axis))() + 2)
        else:
            assert_equal(array.shape[iaxis],
                         getattr(hist, 'GetNbins{0}'.format(axis))())
    hist_sum = hist.Integral()
    assert_true(hist_sum > 0)
    assert_equal(hist_sum, np.sum(array))


def check_hist2array_THn(hist):
    hist_thn = ROOT.THn.CreateHn("", "", hist)
    array = rnp.hist2array(hist)
    array_thn = rnp.hist2array(hist_thn)
    # non-zero elements
    assert_true(np.any(array_thn))
    # arrays should be identical
    assert_array_equal(array, array_thn)


def check_hist2array_THnSparse(hist):
    hist_thnsparse = ROOT.THnSparse.CreateSparse("", "", hist)
    array = rnp.hist2array(hist)
    array_thnsparse = rnp.hist2array(hist_thnsparse)
    # non-zero elements
    assert_true(np.any(array_thnsparse))
    # arrays should be identical
    assert_array_equal(array, array_thnsparse)


def test_hist2array():
    assert_raises(TypeError, rnp.hist2array, object())
    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            yield check_hist2array, hist, False, False
            yield check_hist2array, hist, False, True
            yield check_hist2array, hist, True, False
            yield check_hist2array, hist, True, True
            yield check_hist2array_THn, hist
            # check that the memory was copied
            arr = rnp.hist2array(hist, copy=True)
            hist_sum = hist.Integral()
            assert_true(hist_sum > 0)
            hist.Reset()
            assert_equal(np.sum(arr), hist_sum)
            # check that the memory is shared
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            arr = rnp.hist2array(hist, copy=False)
            hist_sum = hist.Integral()
            assert_true(hist_sum > 0)
            assert_true(np.sum(arr) == hist_sum)
            hist.Reset()
            assert_true(np.sum(arr) == 0)


def test_hist2array_THn():
    assert_raises(TypeError, rnp.hist2array, object())
    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            yield check_hist2array_THn, hist


def test_hist2array_THnSparse():
    assert_raises(TypeError, rnp.hist2array, object())
    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim)
            yield check_hist2array_THnSparse, hist


def check_hist2array_edges(hist, ndim, bins):
    _, edges = rnp.hist2array(hist, return_edges=True)
    assert_equal(len(edges), ndim)
    for axis_edges in edges:
        assert_array_equal(axis_edges, np.arange(bins + 1, dtype=np.double))


def test_hist2array_edges():
    for ndim in (1, 2, 3):
        for bins in (1, 2, 5):
            hist = make_histogram('D', shape=(bins,) * ndim)
            yield check_hist2array_edges, hist, ndim, bins
            hist = ROOT.THn.CreateHn("", "", make_histogram('D', shape=(bins,) * ndim))
            yield check_hist2array_edges, hist, ndim, bins
            hist = ROOT.THnSparse.CreateSparse("", "", make_histogram('D', shape=(bins,) * ndim))
            yield check_hist2array_edges, hist, ndim, bins


def check_array2hist(hist):
    shape = np.array([hist.GetNbinsX(), hist.GetNbinsY(), hist.GetNbinsZ()])
    shape = shape[:hist.GetDimension()]
    arr = RNG.randint(0, 10, size=shape)
    rnp.array2hist(arr, hist)
    arr_hist = rnp.hist2array(hist)
    assert_array_equal(arr_hist, arr)

    # Check behaviour if errors are supplied
    errors = arr * 0.1
    _hist = hist.Clone()
    _hist.Reset()
    rnp.array2hist(arr, _hist, errors=errors)
    arr_hist = rnp.hist2array(_hist)
    assert_array_equal(arr_hist, arr)
    if hist.GetDimension() == 1:
        errors_from_hist = np.array([_hist.GetBinError(ix)
                                     for ix in range(1, _hist.GetNbinsX() + 1)])
    if hist.GetDimension() == 2:
        errors_from_hist = np.array([[_hist.GetBinError(ix, iy)
                                      for iy in range(1, _hist.GetNbinsY() + 1)]
                                     for ix in range(1, _hist.GetNbinsX() + 1)])
    if hist.GetDimension() == 3:
        errors_from_hist = np.array([[[_hist.GetBinError(ix, iy, iz)
                                       for iz in range(1, _hist.GetNbinsZ() + 1)]
                                      for iy in range(1, _hist.GetNbinsY() + 1)]
                                     for ix in range(1, _hist.GetNbinsX() + 1)])
    assert_array_equal(errors, errors_from_hist)

    shape_overflow = shape + 2
    arr_overflow = RNG.randint(0, 10, size=shape_overflow)
    hist_overflow = hist.Clone()
    hist_overflow.Reset()
    rnp.array2hist(arr_overflow, hist_overflow)
    arr_hist_overflow = rnp.hist2array(hist_overflow, include_overflow=True)
    assert_array_equal(arr_hist_overflow, arr_overflow)

    if len(shape) == 1:
        return

    # overflow not specified on all axes
    arr_overflow2 = arr_overflow[1:-1]
    hist_overflow2 = hist.Clone()
    hist_overflow2.Reset()
    rnp.array2hist(arr_overflow2, hist_overflow2)
    arr_hist_overflow2 = rnp.hist2array(hist_overflow2, include_overflow=True)
    assert_array_equal(arr_hist_overflow2[1:-1], arr_overflow2)


def test_array2hist():
    # wrong type
    assert_raises(TypeError, rnp.array2hist,
                  object(), ROOT.TH1D('test', '', 10, 0, 1))
    # wrong type
    assert_raises(TypeError, rnp.array2hist,
                  np.array([1, 2]), object())
    # dimensions don't match
    assert_raises(ValueError, rnp.array2hist,
                  np.arange(4).reshape(2, 2), ROOT.TH1D('test', '', 10, 0, 1))
    # shape not compatible
    assert_raises(ValueError, rnp.array2hist,
                  np.arange(4).reshape(2, 2),
                  ROOT.TH2D('test', '', 4, 0, 1, 3, 0, 1))
    # shape of errors and content array does not match
    assert_raises(ValueError, rnp.array2hist,
                  np.arange(4).reshape(2, 2),
                  ROOT.TH2D('test', '', 4, 0, 1, 4, 0, 1),
                  np.arange(6).reshape(2, 3))

    for ndim in (1, 2, 3):
        for hist_type in 'DFISC':
            hist = make_histogram(hist_type, shape=(5,) * ndim, fill=False)
            yield check_array2hist, hist

    # Check for histograms with unequal dimensions (reveals issues with transposing)
    hist = make_histogram(hist_type, shape=(5, 6, 7), fill=False)
    check_array2hist(hist)


def test_fill_hist():
    n_samples = 1000
    data1D = RNG.randn(n_samples)
    w1D = np.empty(n_samples)
    w1D.fill(2.)
    data2D = RNG.randn(n_samples, 2)
    data3D = RNG.randn(n_samples, 3)

    a = ROOT.TH1D('th1d', 'test', 100, -5, 5)
    rnp.fill_hist(a, data1D)
    assert_almost_equal(a.Integral(), n_samples)

    a_w = ROOT.TH1D('th1dw', 'test', 100, -5, 5)
    rnp.fill_hist(a_w, data1D, w1D)
    assert_almost_equal(a_w.Integral(), n_samples * 2)

    b = ROOT.TH2D('th2d', 'test', 100, -5, 5, 100, -5, 5)
    rnp.fill_hist(b, data2D)
    assert_almost_equal(b.Integral(), n_samples)

    c = ROOT.TH3D('th3d', 'test', 10, -5, 5, 10, -5, 5, 10, -5, 5)
    rnp.fill_hist(c, data3D)
    assert_almost_equal(c.Integral(), n_samples)

    # array and weights lengths do not match
    assert_raises(ValueError, rnp.fill_hist, c, data3D, np.ones(10))

    # weights is not 1D
    assert_raises(ValueError, rnp.fill_hist, c, data3D,
        np.ones((data3D.shape[0], 1)))

    # array not 2-d when filling 2D/3D histogram
    for h in (b, c):
        assert_raises(ValueError, rnp.fill_hist, h, RNG.randn(10))

    # length of second axis does not match dimensionality of histogram
    for h in (a, b, c):
        assert_raises(ValueError, rnp.fill_hist, h, RNG.randn(10, 4))

    # wrong type
    h = list()
    a = RNG.randn(10)
    assert_raises(TypeError, rnp.fill_hist, h, a)


def test_fill_profile():
    n_samples = 1000
    w1D = np.empty(n_samples)
    w1D.fill(2.)
    data1D = RNG.randn(n_samples, 2)
    data2D = RNG.randn(n_samples, 3)
    data3D = RNG.randn(n_samples, 4)

    a = ROOT.TProfile('th1d', 'test', 100, -5, 5)
    rnp.fill_profile(a, data1D)
    assert_true(a.Integral() != 0)

    a_w = ROOT.TProfile('th1dw', 'test', 100, -5, 5)
    rnp.fill_profile(a_w, data1D, w1D)
    assert_true(a_w.Integral() != 0)
    assert_equal(a_w.Integral(), a.Integral())

    b = ROOT.TProfile2D('th2d', 'test', 100, -5, 5, 100, -5, 5)
    rnp.fill_profile(b, data2D)
    assert_true(b.Integral() != 0)

    c = ROOT.TProfile3D('th3d', 'test', 10, -5, 5, 10, -5, 5, 10, -5, 5)
    rnp.fill_profile(c, data3D)
    assert_true(c.Integral() != 0)

    # array and weights lengths do not match
    assert_raises(ValueError, rnp.fill_profile, c, data3D, np.ones(10))

    # weights is not 1D
    assert_raises(ValueError, rnp.fill_profile, c, data3D,
                  np.ones((data3D.shape[0], 1)))

    # array is not 2D
    assert_raises(ValueError, rnp.fill_profile, c, np.ones(10))

    # length of second axis is not one more than dimensionality of the profile
    for h in (a, b, c):
        assert_raises(ValueError, rnp.fill_profile, h, RNG.randn(10, 5))

    # wrong type
    assert_raises(TypeError, rnp.fill_profile,
                  ROOT.TH1D("test", "test", 1, 0, 1), data1D)


def test_fill_graph():
    n_samples = 1000
    data2D = RNG.randn(n_samples, 2)
    data3D = RNG.randn(n_samples, 3)

    graph = ROOT.TGraph()
    rnp.fill_graph(graph, data2D)

    graph2d = ROOT.TGraph2D()
    rnp.fill_graph(graph2d, data3D)

    # array not 2-d
    for g in (graph, graph2d):
        assert_raises(ValueError, rnp.fill_graph, g, RNG.randn(10))

    # length of second axis does not match dimensionality of histogram
    for g in (graph, graph2d):
        assert_raises(ValueError, rnp.fill_graph, g, RNG.randn(10, 4))

    # wrong type
    h = list()
    a = RNG.randn(10)
    assert_raises(TypeError, rnp.fill_graph, h, a)
