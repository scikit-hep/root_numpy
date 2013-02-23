.. -*- mode: rst -*-

.. image:: https://travis-ci.org/rootpy/root_numpy.png
   :target: https://travis-ci.org/rootpy/root_numpy

root_numpy
----------

root_numpy is a Python extension for converting
`ROOT TTrees <http://root.cern.ch/root/html/TTree.html>`_ into NumPy
`recarrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_
or `structured arrays <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_.
With the core internals written in C++, root_numpy can efficiently handle large
amounts of data (limited only by the available memory).
Now that your ROOT data is in NumPy form, you can make use of the many powerful
scientific Python packages or perform quick exploratory data analysis in
interactive environments like `IPython <http://ipython.org/>`_ (especially
IPython's popular `notebook <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_
feature).

root_numpy currently supports basic types like Bool_t, Int_t, Float_t,
Double_t, etc. and arrays of basic types (both variable and fixed-length).
Vectors of basic types are also supported.

Tab completion for numpy.recarray column names (yourdata.<TAB>)
is also available with this `numpy extension <https://github.com/piti118/inumpy>`_.

The `rootpy <http://rootpy.org>`_ package also provides a script that uses
root_numpy and `PyTables <http://www.pytables.org>`_ to convert all TTrees
in a ROOT file into the `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.

Requirements
------------

* `ROOT <http://root.cern.ch/>`_
* `NumPy <http://numpy.scipy.org/>`_

root_numpy is tested with ROOT 5.32, NumPy 1.6.1, Python 2.7.1 but it should
work in most places.

Installation
------------

python setup.py install

Documentation
-------------

See http://rootpy.github.com/root_numpy/
