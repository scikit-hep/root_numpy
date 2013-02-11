root_numpy
==========

**root_numpy** is a python library for converting ROOT_  TTree to numpy_
`structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ or
`record array <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_.
The library is written in C++ and `Cython <http://cython.org/>`_ and is much
faster than equivalent pure-Python implementations. It supports scalars, arrays
of scalars and vectors of basic types (int, float, double etc.).

.. _ROOT: http://root.cern.ch/
.. _numpy: http://numpy.scipy.org/

Download and Install
^^^^^^^^^^^^^^^^^^^^

    ``easy_install root_numpy``

or

    ``pip install root_numpy``

Obtaining Source Code
^^^^^^^^^^^^^^^^^^^^^

    Download it from http://pypi.python.org/pypi/root_numpy/ or

    Clone it from github ``git clone https://github.com/rootpy/root_numpy``.

    Then do the usual ``python setup.py install``

Typical Usage
^^^^^^^^^^^^^

	::

		from root_numpy import *
		#load to numpy structured array
		#treename is always optional if there is 1 tree in the file
		a = root2array('a.root','treename')
		#load to numpy
		a_rec = root2rec('a.root','treename')


Full Documentation
==================

.. automodule:: root_numpy.root_numpy
    :members:

.. automodule:: root_numpy.util
    :members:

.. _conversion_table:

Type Conversion Table
=====================

List of primitive type converion is given this table:

===========  =========================
ROOT type    numpy type
===========  =========================
Char_t       np.int8
UChar_t      np.uint8
Short_t      np.int16
UShort_t     np.uint16
Int_t        np.int32
UInt_t       np.uint32
Float_t      np.float32
Double_t     np.float64
Long64_t     np.int64
ULong64_t    np.uint64
Bool_t       np.bool
x[10]        (np.primitivetype, (10,))
x[nx]        np.object
vector<t>    np.object
===========  =========================

Variable length array (`particletype[nparticle]`) and vector (`vector<int>`)
are converted to object of numpy array of corresponding types. Fixed length
array is converted to fixed length array field in numpy.
