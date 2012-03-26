root_numpy
----------

Python Extension for converting root files to numpy [recarray](http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) or [structure array](http://docs.scipy.org/doc/numpy/user/basics.rec.html). This is very useful for using in interactive data exploration environment like [ipython](http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html) (especially notebook).

Written in C++ with lots of pointer+memcpy magic and it doesn't call PyRoot so it's much faster especially if you are trying to read a large file in to memory (100MB+ or even GB's of Data).

Currently only support basic types like Float_t Int_t Double_t Bool_t etc. No array support yet. This should cover a large number of use cases already.
If you are trying to convert some other type, it will throw RunTimeError Unknown root type typename.

Tab completion for numpy.recarray column name is also available with this [numpy patch](https://github.com/piti118/numpy/commit/a996292238ab98dcf53f2d48476d637eab9f1a72)

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
from numpy import root2arry, root2rec

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
root2array(fnames,treename,branches=None)
convert tree treename in root files specified in fnames to numpy structured array
------------------
return numpy structure array
fnames: list of string or string. Root file name patterns. Anything that works with TChain.Add is accepted
treename: name of tree to convert to numpy array
branches(optional): list of string for branch name to be extracted from tree.
\tIf branches is not specified or is none or is empty, all from the first treebranches are extracted
\tIf branches contains duplicate branches, only the first one is used.

Caveat: This should not matter for most use cases. But, due to the way TChain works, if the trees specified 
in the input files have different structures, only the branch in the first tree will be automatically extracted. 
You can work around this by either reordering the input file or specifying the branches manually.
------------------
Ex:
root2array('a.root','mytree')#read all branches from tree named mytree from a.root
root2array('a*.root','mytree')#read all branches from tree named mytree from a*.root
root2array(['a*.root','b*.root'],'mytree')#read all branches from tree named mytree from a*.root and b*.root
root2array('a.root','mytree','x')#read branch x from tree named mytree from a.root(useful if memory usage matters)
root2array('a.root','mytree',['x','y'])#read branch x and y from tree named mytree from a.root
</pre>

<pre>
root2rec(fnames, treename, branches=None)
read branches in tree treename in file(s) given by fnames can convert it to numpy recarray
---------------
This is equivalent to root2array(fnames,treename,branches).view(np.recarray)
---------------
see root2array for more details
</pre>