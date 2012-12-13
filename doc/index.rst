root_numpy
==========

**root_numpy** is a python library for converting ROOT_  TTree to numpy_ `structure array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ or `record array <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_. The library is written in C++ and `Cython <http://cython.org/>`_ and is much faster than PyROOT. It supports scalar, array of scalar and vector of basic types(int float double etc.).

.. _ROOT: http://root.cern.ch/
.. _numpy: http://numpy.scipy.org/

Download
^^^^^^^^

	http://pypi.python.org/pypi/root_numpy/

	or
	
	``git clone https://github.com/rootpy/root_numpy``

Install
^^^^^^^

	``python setup.py install``

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

.. automodule:: root_numpy
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

Variable length array (`particletype[nparticle]`) and vector (`vector<int>`) are converted to object of numpy array of corresponding types. Fixed length array is converted to fixed length array field in numpy.
