.. _reference:

####################
root_numpy Reference
####################

:Release: |version|
:Date: |today|

.. module:: root_numpy

This reference manual details the functions included in root_numpy, describing
what they are and what they do.

.. autosummary::
   :toctree: generated

   array
   matrix
   root2array
   root2rec
   tree2array
   tree2rec
   array2tree
   array2root
   fill_hist
   random_sample
   list_trees
   list_branches
   list_structures
   stretch
   blockwise_inner_join

.. _conversion_table:

Type Conversion Table
---------------------

The primitive types are converted according to this table:

===========  =========================
ROOT         NumPy
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

Variable length arrays (such as ``particletype[nparticle]``) and vectors
(such as ``vector<int>``) are converted to NumPy arrays of the corresponding
types. Fixed length arrays are converted to fixed length NumPy array fields.

.. note::
    Tab completion for numpy.recarray column names (yourdata.<TAB>)
    is also available with this `numpy extension <https://github.com/piti118/inumpy>`_.
