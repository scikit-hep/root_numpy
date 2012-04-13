from root_numpy import *

#print 'hhhhh'

#a = root2array('test/single*.root','tree')
#a = root2array('test/fixed*.root','tree')
a = root2array('test/vary*.root','tree')
#a = root2array('test/sprtest.root','ClassRecord')
print a
print a.dtype
#print 'kkkk'
