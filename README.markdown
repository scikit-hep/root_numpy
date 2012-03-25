root_numpy
----------

Python Extension for converting root files to numpy structure array. This is very useful for using in interactive data exploration environment like ipython (especially notebook) or sage.

Written in C++ with lots of pointer+memcpy magic and it doesn't call PyRoot so it's much faster if you are trying to read a large file.

Requirements
------------

Root installed http://root.cern.ch/

numpy installed http://numpy.scipy.org/

Tested with Root 5.32, numpy 1.6.1, Python 2.7.1 but it should work in most places.

Installation
------------
python setup.py install

Short Tutorial
--------------

see tutorial.ipynb if you have ipython notebook with (open ipython notebook --pylab inline)
 
or tutorial.pdf if you don't have ipynb

Doc string
----------
<pre>
read(fnames,treename,branches=None)
convert tree treename in root files specified in fnames to numpy structured array
------------------
return numpy array
fnames: list of string or string. Root file name patterns. Anything that works with TChain.Add is accepted
treename: name of tree to convert to numpy array
branches(optional): list of string for branch name to be extracted from tree.
        If branches is not specified or is none or is empty, all from the first treebranches are extracted
        If branches contains duplicate branches, only the first one is used.

Caveat: This should not matter for most use cases. But, due to the way TChain works, if the trees specified in the input files have different
structure, only the branch in the first tree will be automatically extracted. You can work around this by either reordering the input file or
specify the branches manually.
------------------
Ex:
root_numpy.read('a.root','mytree')#read all branches from tree named mytree from a.root

root_numpy.read('a*.root','mytree')#read all branches from tree named mytree from a*.root

root_numpy.read(['a*.root','b*.root'],'mytree')#read all branches from tree named mytree from a*.root and b*.root

root_numpy.read('a.root','mytree','x')#read branch x from tree named mytree from a.root(useful if memory usage matters)

root_numpy.read('a.root','mytree',['x','y'])#read branch x and y from tree named mytree from a.root
</pre>