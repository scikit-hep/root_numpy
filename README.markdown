root_numpy
----------

Python Extension for converting root files to numpy
[recarray](http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html)
or [structure array](http://docs.scipy.org/doc/numpy/user/basics.rec.html). This
is very useful for using in interactive data exploration environment like
[ipython](http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html)
(especially notebook).

Written in C++ with lots of pointer+memcpy magic and it doesn't call PyRoot so
it's much faster especially if you are trying to read a large file in to memory
(100MB+ or even GB's of Data).

Currently only support basic types like Float_t Int_t Double_t Bool_t etc. and
array of basic types both variable and fixed length. vector of basic type (int,
float, double, char, long) is also supported.

Tab completion for numpy.recarray column name (yourdata.<TAB> showing the column
names so you don't have to remember it) is also available with this
[numpy patch](https://github.com/piti118/numpy/commit/a996292238ab98dcf53f2d48476d637eab9f1a72)

Requirements
------------

[Root](http://root.cern.ch/) installed

[numpy](http://numpy.scipy.org/) installed

Tested with Root 5.32, numpy 1.6.1, Python 2.7.1 but it should work in most
places.

Installation
------------

python setup.py install

Documentation
-------------

See http://piti118.github.com/root_numpy/
