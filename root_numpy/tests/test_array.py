import ROOT
import root_numpy as rnp
from nose.tools import assert_raises, assert_equal


def check_array(cls, copy):
    a = cls(10)
    a[2] = 2
    b = rnp.array(a, copy=copy)
    assert_equal(b[2], 2)
    assert_equal(b.shape[0], 10)


def test_array():
    for copy in (True, False):
        for cls in (getattr(ROOT, 'TArray{0}'.format(atype))
                for atype in 'DFLIS'):
            yield check_array, cls, copy
        a = ROOT.TArrayC(10)
        b = rnp.array(a, copy=copy)
        assert_equal(b.shape[0], 10)
    assert_raises(TypeError, rnp.array, object)


def check_matrix(cls):
    mat = cls(5, 5)
    mat[1][2] = 2
    np_mat = rnp.matrix(mat)
    assert_equal(np_mat[1, 2], 2)


def check_matrix_sym(cls):
    mat = cls(5)
    mat[2][2] = 2
    np_mat = rnp.matrix(mat)
    assert_equal(np_mat[2, 2], 2)


def test_matrix():
    for cls in (getattr(ROOT, 'TMatrix{0}'.format(atype)) for atype in 'DF'):
        yield check_matrix, cls

    for cls in (getattr(ROOT, 'TMatrix{0}Sym'.format(atype)) for atype in 'DF'):
        yield check_matrix_sym, cls

    assert_raises(TypeError, rnp.matrix, object)
