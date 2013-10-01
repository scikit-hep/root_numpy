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

root_numpy is a Python extension for converting
`ROOT TTrees <http://root.cern.ch/root/html/TTree.html>`_ into NumPy
`recarrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_
or `structured arrays <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
as well as converting NumPy arrays back into ROOT TTrees.
With the core internals written in C++, root_numpy can efficiently handle large
amounts of data (limited only by the available memory).
Now that your ROOT data is in NumPy form, you can make use of the many powerful
scientific Python packages or perform quick exploratory data analysis in
interactive environments like `IPython <http://ipython.org/>`_ (especially
IPython's popular `notebook <http://ipython.org/ipython-doc/dev/interactive/notebook.html>`_
feature).

root_numpy currently supports basic types such as bool, int, float,
double, etc. and arrays of basic types (both variable and fixed-length).
Vectors of basic types are also supported.

See the root2hdf5 script in the `rootpy <https://github.com/rootpy/rootpy>`_
package that uses root_numpy and `PyTables <http://www.pytables.org>`_ to
convert all TTrees in a ROOT file into the
`HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.

.. code-block:: python

   import ROOT
   from root_numpy import root2array, root2rec, tree2rec

   # convert into a numpy structured array
   # treename is always optional if there is only one tree in the file
   arr = root2array('a.root', 'treename')

   # convert into a numpy record array
   rec = root2rec('a.root', 'treename')

   # or directly convert a tree
   rfile = ROOT.TFile('a.root')
   tree = rfile.Get('treename')
   rec = tree2rec(tree)

