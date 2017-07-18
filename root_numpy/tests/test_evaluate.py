import ROOT
import root_numpy as rnp
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from . import RNG, silence_serr


def test_evaluate_func():
    f1 = ROOT.TF1("f1", "x")
    f2 = ROOT.TF2("f2", "x*y")
    f3 = ROOT.TF3("f3", "x*y*z")

    # generate random arrays
    arr_1d = RNG.rand(5)
    arr_2d = RNG.rand(5, 2)
    arr_3d = RNG.rand(5, 3)
    arr_4d = RNG.rand(5, 4)

    assert_array_equal(rnp.evaluate(f1, arr_1d),
                       [f1.Eval(x) for x in arr_1d])
    assert_array_equal(rnp.evaluate(f1.GetTitle(), arr_1d),
                       [f1.Eval(x) for x in arr_1d])
    assert_array_equal(rnp.evaluate(f2, arr_2d),
                       [f2.Eval(*x) for x in arr_2d])
    assert_array_equal(rnp.evaluate(f2.GetTitle(), arr_2d),
                       [f2.Eval(*x) for x in arr_2d])
    assert_array_equal(rnp.evaluate(f3, arr_3d),
                       [f3.Eval(*x) for x in arr_3d])
    assert_array_equal(rnp.evaluate(f3.GetTitle(), arr_3d),
                       [f3.Eval(*x) for x in arr_3d])
    # 4d formula
    f4 = ROOT.TFormula('test', 'x*y+z*t')
    assert_array_equal(rnp.evaluate(f4, arr_4d),
                       [f4.Eval(*x) for x in arr_4d])

    assert_raises(ValueError, rnp.evaluate, f1, arr_2d)
    assert_raises(ValueError, rnp.evaluate, f2, arr_3d)
    assert_raises(ValueError, rnp.evaluate, f2, arr_1d)
    assert_raises(ValueError, rnp.evaluate, f3, arr_1d)
    assert_raises(ValueError, rnp.evaluate, f3, arr_2d)

    with silence_serr():  # silence cling error
        assert_raises(ValueError, rnp.evaluate, "f", arr_1d)

    assert_raises(ValueError, rnp.evaluate, "x*y", arr_1d)
    assert_raises(ValueError, rnp.evaluate, "x", arr_2d)
    assert_raises(ValueError, rnp.evaluate, "x*y", arr_3d)


def test_evaluate_hist():
    h1 = ROOT.TH1D("h1", "", 10, 0, 1)
    h1.FillRandom("f1")
    h2 = ROOT.TH2D("h2", "", 10, 0, 1, 10, 0, 1)
    h2.FillRandom("f2")
    h3 = ROOT.TH3D("h3", "", 10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3.FillRandom("f3")

    arr_1d = RNG.rand(5)
    arr_2d = RNG.rand(5, 2)
    arr_3d = RNG.rand(5, 3)

    assert_array_equal(rnp.evaluate(h1, arr_1d),
                       [h1.GetBinContent(h1.FindBin(x)) for x in arr_1d])
    assert_array_equal(rnp.evaluate(h2, arr_2d),
                       [h2.GetBinContent(h2.FindBin(*x)) for x in arr_2d])
    assert_array_equal(rnp.evaluate(h3, arr_3d),
                       [h3.GetBinContent(h3.FindBin(*x)) for x in arr_3d])

    assert_raises(ValueError, rnp.evaluate, h1, arr_2d)
    assert_raises(ValueError, rnp.evaluate, h2, arr_3d)
    assert_raises(ValueError, rnp.evaluate, h2, arr_1d)
    assert_raises(ValueError, rnp.evaluate, h3, arr_1d)
    assert_raises(ValueError, rnp.evaluate, h3, arr_2d)


def test_evaluate_graph():
    g = ROOT.TGraph(2)
    g.SetPoint(0, 0, 1)
    g.SetPoint(1, 1, 2)
    assert_array_equal(rnp.evaluate(g, [0, .5, 1]), [1, 1.5, 2])
    s = ROOT.TSpline3("spline", g)
    assert_array_equal(rnp.evaluate(s, [0, .5, 1]),
                       [s.Eval(x) for x in [0, .5, 1]])
    # test exceptions
    arr_2d = RNG.rand(5, 2)
    assert_raises(TypeError, rnp.evaluate, object(), [1, 2, 3])
    assert_raises(ValueError, rnp.evaluate, g, arr_2d)
    assert_raises(ValueError, rnp.evaluate, s, arr_2d)
