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

root_numpy interfaces NumPy [@NumPy] with CERN's ROOT [@ROOT] software
framework, providing the ability to analyse ROOT data within the broad
ecosystem of scientific Python packages.

At its core are functions for converting between ROOT `TTree`s and structured
NumPy arrays. root_numpy can convert `TTree` branches (columns) of fundamental
types and strings, as well as variable-length and fixed-length multidimensional
arrays and (nested) `std::vector`s. root_numpy can also create columns in the
output NumPy array from mathematical expressions like ROOT's `TTree::Draw()`.
root_numpy's internals are written in Cython [@Cython], installed as compiled
C++ extensions, and can handle data with comparable speed to ROOT as shown in
Figure \ref{benchmark}.

root_numpy also provides functions for converting between ROOT histograms and
NumPy arrays, sampling or evaluating ROOT functions as NumPy arrays, and an
interface to TMVA [@TMVA], ROOT's machine learning toolkit.

![Benchmarking root_numpy's `tree2array()` function against ROOT's `TTree::Draw()`\label{benchmark}](../benchmarks/bench_tree2array.pdf)

# References
