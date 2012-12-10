import uuid
import unittest
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_raises
import ROOT
from root_numpy.hist import hist2array, array2hist, fill_hist_from_array

def _get_hist(dim, hist_class):
    name = uuid.uuid4().hex
    if dim == 1:
        return hist_class(name, '', 5, 0, 1)
    elif dim == 2:
        return hist_class(name, '', 5, 0, 1, 6, 0, 2)
    elif dim == 3:
        return hist_class(name, '', 5, 0, 1, 6, 0, 2, 7, 0, 3)

class TestHist2Array(unittest.TestCase):
    # List of ROOT hist types taken from
    # http://root.cern.ch/root/html/TH1.html
    dims = [1, 2, 3]
    # TODO: for hist_dtype = 'C' I get this error:
    # TypeError: buffer is too small for requested array
    hist_dtypes = 'S I F D'.split()
    #hists = [(dim, type, eval('ROOT.TH{0}{1}'.format(dim, type)))
    #         for (dim, type) in itertools.product(dims, dtypes)] 

    def test_all(self):
        for dim in self.dims: 
            for hist_dtype in self.hist_dtypes:
                hist_class = eval('ROOT.TH{0}{1}'.format(dim, hist_dtype))
                hist = _get_hist(dim, hist_class)
                array = hist2array(hist)
                # hist was initialized to zero, and summing zeros gives zero:
                assert_equal(np.sum(np.abs(array)), 0)

    def test_invalid_inputs(self):
        """Check that input we can't handle raises a proper exception"""
        inputs = []
        # Apparently THnD is not there in ROOT 5.32:
        # inputs.append(ROOT.THnD())
        inputs.append(42)
        inputs.append(np.arange(10))
        for input in inputs:
            assert_raises(TypeError, hist2array, input)

class TestArray2HistRountTripping(unittest.TestCase):

    numpy_dtypes = 'b i1 u1 i2 u2 i4 u4 i8 u8 f4 f8'

    def _test_all(self):
        for shape in [5, (5, 6), (5, 6, 7)]:
            for numpy_dtype in numpy_dtypes:
                array = np.random.random(5, dtype=numpy_dtype)
                hist = array2hist(array)
                array2 = hist2array(hist)
                np.assert_array_equal(array, array2)


class TestFillHistFromArray(unittest.TestCase):

    # TODO: 3D case doesn't work for now
    dims = [1, 2]
    hist_dtypes = 'S I F D'.split()
    numpy_dtypes = 'b i1 u1 i2 u2 i4 u4 i8 u8 f4 f8'.split()

    def test_all(self):
        nevents = 100
        for dim in self.dims:
            for hist_dtype in self.hist_dtypes:
                for numpy_dtype in self.numpy_dtypes:
                    hist_class = eval('ROOT.TH{0}{1}'.format(dim, hist_dtype))
                    hist = _get_hist(dim, hist_class)
                    array = np.ones(shape=(nevents, dim), dtype=numpy_dtype)
                    if dim == 1:
                        array = array.flatten()
                    fill_hist_from_array(hist, array)
                    #fill_hist_from_array(hist, array, weights=array)
                    

def _test_quickly():
    from rootpy.plotting import Hist, Hist2D, Hist3D
    #h1 = Hist(5, 0, 1, type='d')
    h1 = ROOT.TH1S('h', 'h', 5, 0, 1)
    h2 = Hist2D(5, 0, 1, 3, 0, 1, type='d')
    h3 = Hist3D(5, 0, 1, 3, 0, 1, 2, 0, 1, type='d')
    
    fill_hist_from_array(h1, np.random.randn(1E6))
    
    print hist2array(h1)
    print hist2array(h2)
    print hist2array(h3)


# TODO: Add more tests for invalid inputs and check that
# the right exceptions are raised.

if __name__ == '__main__':
    unittest.main()