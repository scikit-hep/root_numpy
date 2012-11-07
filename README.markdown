root_numpy
----------

Python Extension for converting root files to numpy [recarray](http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) or [structure array](http://docs.scipy.org/doc/numpy/user/basics.rec.html). This is very useful for using in interactive data exploration environment like [ipython](http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html) (especially notebook).

Written in C++ with lots of pointer+memcpy magic and it doesn't call PyRoot so it's much faster especially if you are trying to read a large file in to memory (100MB+ or even GB's of Data).

Currently only support basic types like Float_t Int_t Double_t Bool_t etc. and array of basic types both variable and fixed length. vector of basic type (int, float, double, char, long) is also supported.

Tab completion for numpy.recarray column name (yourdata.<TAB> showing the column names so you don't have to remember it) is also available with this [numpy patch](https://github.com/piti118/numpy/commit/a996292238ab98dcf53f2d48476d637eab9f1a72)

Requirements
------------

[Root](http://root.cern.ch/) installed

[numpy](http://numpy.scipy.org/) installed

Tested with Root 5.32, numpy 1.6.1, Python 2.7.1 but it should work in most places.

Installation
------------
python setup.py install

Short Tutorial
--------------

Basically it let you do things like this very fast and very efficiently memory wise.

```
import numpy as np
from root_numpy import *

a = root2rec('test/test.root','tree')
print a.i
print a.f
#which you can then plot with matplotlib using
#plot(a.i,a.f)

a = root2array('test/test.root','tree')
print a['i']
print a['f']
```

fore more information see tutorial.ipynb if you have ipython notebook
 
or tutorial.pdf if you don't have ipython notebook

Docstring
---------
<pre>
    root2array(fnames, treename, branches=None,N=None,offset=0)
    convert tree treename in root files specified in fnames to
    numpy structured array
    ------------------
    return numpy structure array
    fnames: list of string or string. Root file name patterns.
    Anything that works with TChain.Add is accepted
    treename: name of tree to convert to numpy array.
    This is optional if the file contains exactly 1 tree.
    branches(optional): list of string for branch name to be
    extracted from tree.
    * If branches is not specified or is None or is empty,
      all from the first treebranches are extracted
    * If branches contains duplicate branches, only the first one is used.
    N(optional): maximum number of data that it should load
    useful for testing out stuff
    offset(optional): start index (first one is 0)

    Caveat: This should not matter for most use cases. But, due to
    the way TChain works, if the trees specified
    in the input files have different structures, only the
    branch in the first tree will be automatically extracted.
    You can work around this by either reordering the input
    file or specifying the branches manually.
    ------------------
    Ex:
    # read all branches from tree named mytree from a.root
    root2array('a.root', 'mytree')

    #read all branches starting from record 5 for 10 records
    #or the end of file.
    root2array('a.root', 'mytree',offset=5,N=10)
    
    # read all branches from tree named mytree from a*.root
    # this is done using python glob not ROOT semi-broken glob
    root2array('a*.root', 'mytree')
    
    # read all branches from tree named mytree from a*.root and b*.root
    root2array(['a*.root', 'b*.root'], 'mytree')
    
    #read branch x and y from tree named mytree from a.root
    root2array('a.root', 'mytree', ['x', 'y'])
</pre>

<pre>
root2rec(fnames, treename, branches=None,N=None,offset=0)
read branches in tree treename in file(s) given by fnames can convert it to numpy recarray
---------------
This is equivalent to root2array(fnames,treename,branches).view(np.recarray)
---------------
see root2array for more details
</pre>
