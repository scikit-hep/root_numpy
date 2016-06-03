.. _reference:

root_numpy Reference
====================

:Release: |version|
:Date: |today|

This reference manual details the functions included in root_numpy, describing
what they are and what they do.

root_numpy
----------

.. currentmodule:: root_numpy

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
   hist2array
   array2hist
   fill_hist
   fill_profile
   fill_graph
   random_sample
   evaluate
   list_trees
   list_branches
   list_directories
   list_structures
   rec2array
   stack
   stretch
   dup_idx
   blockwise_inner_join

root_numpy.tmva
---------------

.. warning:: The interface of TMVA has changed in ROOT 6.07. So building
   root_numpy there will fail until we can handle their new DataLoader
   interface. In the meantime disable the TMVA interface with the following
   if you must use ROOT 6.07 with TMVA enabled::

      NOTMVA=1 pip install --upgrade --user root_numpy

   Note that if TMVA is not enabled in the ROOT build, root_numpy will anyway
   not attempt to build the TMVA interface.

.. currentmodule:: root_numpy.tmva

.. autosummary::
   :toctree: generated

   add_classification_events
   add_regression_events
   evaluate_reader
   evaluate_method
