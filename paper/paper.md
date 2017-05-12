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

root_numpy [@root_numpy_repo] is a Python extension module providing an
interface between CERN's ROOT software framework [@ROOT] and NumPy [@NumPy].
root_numpy enables researchers typically operating within the C++ domain of
ROOT to analyse data in ROOT format within the broad and growing ecosystem of
scientific Python packages.

At the core of root_numpy are functions for converting a ROOT `TTree` into a
structured NumPy array as well as converting a NumPy array back into a ROOT
`TTree`. root_numpy can convert `TTree` branches (columns) of basic types such
as bool, int, float, double, etc. and strings, as well as variable-length and
fixed-length multidimensional arrays and 1D or 2D (nested) `std::vector<>`s of
basic types and strings. root_numpy can also create columns in the output NumPy
array that are mathematical expressions involving the `TTree` branches in the
same way as ROOT's `TTree::Draw()`. root_numpy's internals are compiled C++ and
can therefore handle large amounts of data much faster than equivalent pure
Python implementations.

![root_numpy benchmark](../benchmarks/bench_tree2array.png)

# References
