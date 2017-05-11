---
title: 'root_numpy: The interface between ROOT and NumPy'
tags:
  - Python
  - C++
  - ROOT
  - NumPy
  - CERN
authors:
 - name: Edmund Noel Dawe
   orcid: 0000-0003-0202-3284
   affiliation: 1
 - name: Piti Ongmongkolkul
   affiliation: 2
 - name: Giordon Stark
   orcid: 0000-0001-6616-3433
   affiliation: 3
affiliations:
 - name: The University of Melbourne
   index: 1
 - name: Mahidol University International College
   index: 2
 - name: The University of Chicago
   index: 3
date: 11 May 2017
bibliography: paper.bib
---

# Summary

root_numpy [@root_numpy_repo] is a Python extension module that provides an
efficient interface between CERN's ROOT software framework [@ROOT] and NumPy
[@NumPy]. root_numpy's internals are compiled C++ and can therefore handle
large amounts of data much faster than equivalent pure Python implementations.

At the core of root_numpy are powerful and flexible functions for converting
ROOT TTrees into structured NumPy arrays as well as converting NumPy arrays
back into ROOT TTrees. root_numpy can convert branches of strings and basic
types such as bool, int, float, double, etc. as well as variable-length and
fixed-length multidimensional arrays and 1D or 2D vectors of basic types and
strings. root_numpy can also create columns in the output array that are
expressions involving the TTree branches similar to ROOT's `TTree::Draw()`.

-![root_numpy benchmark](../benchmarks/bench_tree2array.png)

# References
