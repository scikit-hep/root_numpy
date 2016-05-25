.. -*- mode: rst -*-

`[see full documentation] <http://rootpy.github.com/root_numpy/>`_

root_numpy: The interface between ROOT and NumPy
================================================

.. image:: https://img.shields.io/pypi/v/root_numpy.svg
   :target: https://pypi.python.org/pypi/root_numpy
.. image:: https://travis-ci.org/rootpy/root_numpy.png
   :target: https://travis-ci.org/rootpy/root_numpy
.. image:: https://coveralls.io/repos/rootpy/root_numpy/badge.svg?branch=master
   :target: https://coveralls.io/r/rootpy/root_numpy?branch=master
.. image:: https://landscape.io/github/rootpy/root_numpy/master/landscape.svg?style=flat
   :target: https://landscape.io/github/rootpy/root_numpy/master
.. image:: https://zenodo.org/badge/14091/rootpy/root_numpy.svg
   :target: https://zenodo.org/badge/latestdoi/14091/rootpy/root_numpy

root_numpy is a Python extension module that provides an efficient interface
between `ROOT <http://root.cern.ch/>`_ and `NumPy <http://www.numpy.org/>`_.
root_numpy's internals are compiled C++ and can therefore handle large amounts
of data much faster than equivalent pure Python implementations.

With your ROOT data in NumPy form, make use of NumPy's `broad library
<http://docs.scipy.org/doc/numpy/reference/>`_, including fancy indexing,
slicing, broadcasting, random sampling, sorting, shape transformations, linear
algebra operations, and more. See this `tutorial
<https://docs.scipy.org/doc/numpy-dev/user/quickstart.html>`_ to get started.
NumPy is the fundamental library of the scientific Python ecosystem. Using
NumPy arrays opens up many new possibilities beyond what ROOT offers. Convert
your TTrees into NumPy arrays and use `SciPy <http://www.scipy.org/>`_ for
numerical integration and optimization, `matplotlib <http://matplotlib.org/>`_
for plotting, `pandas <http://pandas.pydata.org/>`_ for data analysis,
`statsmodels <http://statsmodels.sourceforge.net/>`_ for statistical modelling,
`scikit-learn <http://scikit-learn.org/>`_ for machine learning, and perform
quick exploratory analysis in a `Jupyter notebook <https://jupyter.org/>`_.

At the core of root_numpy are powerful and flexible functions for converting
`ROOT TTrees <https://root.cern.ch/doc/master/classTTree.html>`_ into
`structured NumPy arrays
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ as well as converting
NumPy arrays back into ROOT TTrees. root_numpy can convert branches of strings
and basic types such as bool, int, float, double, etc. as well as
variable-length and fixed-length multidimensional arrays and 1D or 2D vectors
of basic types and strings. root_numpy can also create columns in the output
array that are expressions involving the TTree branches similar to
``TTree::Draw()``.

For example, get a structured NumPy array from a TTree (copy and paste the
following examples into your Python prompt):

.. code-block:: python

   from root_numpy import root2array, tree2array
   from root_numpy.testdata import get_filepath

   filename = get_filepath('test.root')

   # Convert a TTree in a ROOT file into a NumPy structured array
   arr = root2array(filename, 'tree')
   # The TTree name is always optional if there is only one TTree in the file

   # Or first get the TTree from the ROOT file
   import ROOT
   rfile = ROOT.TFile(filename)
   intree = rfile.Get('tree')

   # and convert the TTree into an array
   array = tree2array(intree)

Include specific branches or expressions and only entries passing a selection:

.. code-block:: python

   array = tree2array(intree,
       branches=['x', 'y', 'sqrt(y)', 'TMath::Landau(x)', 'cos(x)*sin(y)'],
       selection='z > 0',
       start=0, stop=10, step=2)

The above conversion creates an array with five columns from the branches
x and y where z is greater than zero and only looping on the first ten entries
in the tree while skipping every second entry.

Now convert our array back into a TTree:

.. code-block:: python

   from root_numpy import array2tree, array2root

   # Rename the fields
   array.dtype.names = ('x', 'y', 'sqrt_y', 'landau_x', 'cos_x_sin_y')

   # Convert the NumPy array into a TTree
   tree = array2tree(array, name='tree')

   # Or write directly into a ROOT file without using PyROOT
   array2root(array, 'selected_tree.root', 'tree')

root_numpy also provides a function for filling a ROOT histogram from a NumPy
array:

.. code-block:: python

   from ROOT import TH2D
   from root_numpy import fill_hist
   import numpy as np

   # Fill a ROOT histogram from a NumPy array
   hist = TH2D('name', 'title', 20, -3, 3, 20, -3, 3)
   fill_hist(hist, np.random.randn(1000000, 2))
   hist.Draw('LEGO2')

and a function for creating a random NumPy array by sampling a ROOT function
or histogram:

.. code-block:: python

   from ROOT import TF2, TH1D
   from root_numpy import random_sample

   # Sample a ROOT function
   func = TF2('func', 'sin(x)*sin(y)/(x*y)')
   arr = random_sample(func, 1000000)

   # Sample a ROOT histogram
   hist = TH1D('hist', 'hist', 10, -3, 3)
   hist.FillRandom('gaus')
   arr = random_sample(hist, 1000000)
