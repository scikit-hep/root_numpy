.. -*- mode: rst -*-

`[see full documentation] <http://rootpy.github.com/root_numpy/>`_

root_numpy
==========

.. image:: https://travis-ci.org/rootpy/root_numpy.png
   :target: https://travis-ci.org/rootpy/root_numpy
.. image:: https://coveralls.io/repos/rootpy/root_numpy/badge.png
   :target: https://coveralls.io/r/rootpy/root_numpy
.. image:: https://pypip.in/v/root_numpy/badge.png
   :target: https://pypi.python.org/pypi/root_numpy
.. image:: https://pypip.in/d/root_numpy/badge.png
   :target: https://crate.io/packages/root_numpy/

root_numpy is a Python extension module that provides an efficient interface
between `ROOT <http://root.cern.ch/>`_ and `NumPy <http://www.numpy.org/>`_.
root_numpy's internals are compiled C++ and can therefore handle
large amounts of data much faster than equivalent pure Python implementations.

With your ROOT data in NumPy form, make use of NumPy's
`broad library <http://docs.scipy.org/doc/numpy/reference/>`_, including
fancy indexing, slicing, broadcasting, random sampling, sorting,
shape transformations, linear algebra operations, and more.
See this introductory
`tutorial <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_ to get started.
NumPy is the fundamental library of the scientific Python ecosystem.
Using NumPy arrays opens up many new possibilities beyond what ROOT
offers. Convert your TTrees into NumPy arrays and use
`SciPy <http://www.scipy.org/>`_ for numerical integration and optimization,
`matplotlib <http://matplotlib.org/>`_ for plotting,
`pandas <http://pandas.pydata.org/>`_ for data analysis,
`statsmodels <http://statsmodels.sourceforge.net/>`_ for statistical modelling,
`scikit-learn <http://scikit-learn.org/>`_ for machine learning,
and perform quick exploratory analysis in interactive environments like
`IPython <http://ipython.org/>`_, especially IPython's popular
`notebook <http://ipython.org/ipython-doc/dev/interactive/notebook.html>`_
feature.

At the core of root_numpy are powerful and flexible functions for converting
`ROOT TTrees <http://root.cern.ch/root/html/TTree.html>`_ into NumPy
`recarrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_
or `structured arrays <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
as well as converting NumPy arrays back into ROOT TTrees.
root_numpy can convert branches of basic types such as bool, int, float,
double, etc. as well as variable and fixed-length 1D arrays and vectors
of basic types.

For example, get a NumPy structured or record array from a TTree
(copy and paste the following examples into your Python prompt):

.. code-block:: python

   import ROOT
   from root_numpy import root2array, root2rec, tree2rec
   from root_numpy.testdata import get_filepath

   filename = get_filepath('test.root')

   # Convert a TTree in a ROOT file into a NumPy structured array
   arr = root2array(filename, 'tree')
   # The TTree name is always optional if there is only one TTree in the file

   # Convert a TTree in a ROOT file into a NumPy record array
   rec = root2rec(filename, 'tree')

   # Get the TTree from the ROOT file
   rfile = ROOT.TFile(filename)
   intree = rfile.Get('tree')

   # Convert the TTree into a NumPy record array
   rec = tree2rec(intree)

Include specific branches or expressions and only entries passing a selection:

.. code-block:: python

   rec = tree2rec(intree,
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
   rec.dtype.names = ('x', 'y', 'sqrt_y', 'landau_x', 'cos_x_sin_y')

   # Convert the NumPy record array into a TTree
   tree = array2tree(rec, name='tree')

   # Dump directly into a ROOT file without using PyROOT
   array2root(rec, 'selected_tree.root', 'tree')

root_numpy also provides a function for filling a ROOT histogram from a NumPy
array:

.. code-block:: python

   from ROOT import TH2D, TCanvas
   from root_numpy import fill_hist
   import numpy as np

   # Fill a ROOT histogram from a NumPy array
   hist = TH2D('name', 'title', 20, -3, 3, 20, -3, 3)
   fill_hist(hist, np.random.randn(1E6, 2))
   canvas = TCanvas(); hist.Draw('LEGO2')

and a function for creating a random NumPy array by sampling a ROOT function
or histogram:

.. code-block:: python

   from ROOT import TF2, TH1D
   from root_numpy import random_sample

   # Sample a ROOT function
   func = TF2('func', 'sin(x)*sin(y)/(x*y)')
   arr = random_sample(func, 1E6)

   # Sample a ROOT histogram
   hist = TH1D('hist', 'hist', 10, -3, 3)
   hist.FillRandom('gaus')
   arr = random_sample(hist, 1E6)

Also see the `root2hdf5 <http://www.rootpy.org/commands/root2hdf5.html>`_
script in the `rootpy <https://github.com/rootpy/rootpy>`_
package that uses root_numpy and `PyTables <http://www.pytables.org>`_ to
convert all TTrees in a ROOT file into the
`HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.
