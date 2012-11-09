# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from root_numpy import root2rec, root2array

# <codecell>

#root2rec
ar = root2rec('../test/test.root','tree')
print ar.i
print ar.f
#ipython autocomplete columnname patch is available with this numpy patch
#https://github.com/piti118/numpy/commit/a996292238ab98dcf53f2d48476d637eab9f1a72
ar.i[0] #ar[0].i won't work
ar[0][0]

# <codecell>

ar.f[ar.i>5]

# <codecell>

#root2array is available if you don't like recarray
a=root2array('../test/test.root','tree')
#this tree has two column i and integer and f as float
a #you will see that a is a structure array

# <codecell>

#access whole column
print a['i']
print a['f']

# <codecell>

#access 0th record
print a[0]
#and the first record
print a[1]

# <codecell>

#access 1st record column i
print a[1]['i']
#and this may confuse you but
print a['i'][1]
#there is a tiny different here a[some string] will return numpy array of that 
#column which you can index it again while a[integer] will return the that structure 
#which you can index it again
print a[1][0] #this one works too

