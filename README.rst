.. -*- mode: rst -*-

`[see full documentation] <http://rootpy.github.com/root_numpy/>`_

root_numpy
==========

.. image:: https://travis-ci.org/rootpy/root_numpy.png
   :target: https://travis-ci.org/rootpy/root_numpy
.. image:: https://pypip.in/v/root_numpy/badge.png
   :target: https://pypi.python.org/pypi/root_numpy
.. image:: https://pypip.in/d/root_numpy/badge.png
   :target: https://crate.io/packages/root_numpy/

root_numpy is a Python extension module that provides an efficient interface
between `ROOT <http://root.cern.ch/>`_ and `NumPy <http://www.numpy.org/>`_.
root_numpy's internals are written in compiled C++ and can therefore handle
large amounts of data much faster than equivalent pure Python implementations.

With your ROOT data in NumPy form, make use of NumPy's
`broad library <http://docs.scipy.org/doc/numpy/reference/>`_, including
fancy indexing, slicing, broadcasting, random sampling, sorting,
shape transformations, linear algebra operations, and more.
See this introductory
`tutorial <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_ to get started.
NumPy is the fundamental library of the scientific Python ecosystem.
Using NumPy arrays opens up many new possibilities beyond what ROOT
offers. For example, convert your TTrees into NumPy arrays and use
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
double, etc. as well as variable and fixed-length arrays of basic types.
``std::vector`` of basic types are also supported.

For example, get a structured or record array from a TTree in a ROOT file
(you should be able to copy and paste the following examples into a Python
session):

.. code-block:: python

   from root_numpy import root2array, root2rec
   from root_numpy.testdata import get_filepath

   filename = get_filepath('test.root')

   # Convert a tree into a numpy structured array
   arr = root2array(filename, 'tree')
   # The tree name is always optional if there is only one tree in the file

   # Convert a tree into a numpy record array
   rec = root2rec(filename, 'tree')

or directly from a TTree:

.. code-block:: python

   import ROOT
   from root_numpy import tree2rec

   file = ROOT.TFile(filename)
   intree = file.Get('tree')
   rec = tree2rec(intree)

Include only certain branches and entries:

.. code-block:: python

   rec = tree2rec(intree, branches=['x', 'y'], selection='z > 0',
                  start=0, stop=10, step=2)

The above conversion creates an array with two columns from the branches
x and y where z is greater than zero and only looping on the first ten entries
in the original tree while skipping every second entry.

Now convert our array back into a TTree:

.. code-block:: python

   from root_numpy import array2tree, array2root

   tree = array2tree(rec, name='tree')

   # or dump directly into a ROOT file without using PyROOT
   array2root(rec, 'selected_tree.root', 'tree')

root_numpy also provides a function for filling a ROOT histogram from a NumPy
array:

.. code-block:: python

   from ROOT import TH2D, TCanvas
   from root_numpy import fill_hist
   import numpy as np

   hist = TH2D('name', 'title', 20, -3, 3, 20, -3, 3)
   fill_hist(hist, np.random.randn(1000000, 2))
   canvas = TCanvas()
   hist.Draw('LEGO2')

and a function for creating a random NumPy array by sampling a ROOT function:

.. code-block:: python

   from ROOT import TF2
   from root_numpy import random_sample

   func = TF2('f2', 'sin(x)*sin(y)/(x*y)')
   arr = random_sample(func, 1E6)

Also see the `root2hdf5 <http://www.rootpy.org/commands/root2hdf5.html>`_
script in the `rootpy <https://github.com/rootpy/rootpy>`_
package that uses root_numpy and `PyTables <http://www.pytables.org>`_ to
convert all TTrees in a ROOT file into the
`HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.
