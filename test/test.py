from root_numpy import root2rec, root2array

a = root2rec('test/test.root','tree')
print a
a = root2array('test/test.root','tree')
print a