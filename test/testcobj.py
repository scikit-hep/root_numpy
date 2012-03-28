import ROOT
from root_numpy.croot_numpy import *
f = ROOT.TFile('test/test.root')
t = f.Get('tree')

a = None
if 'AsCapsule' in dir(ROOT):
    #for the future cobj is deprecated in 2.7 and root doesn't provide AsCapsule yet
    o = ROOT.AsCapsule(t)
    a = root2array_from_capsule(o)
else:
    o = ROOT.AsCObject(t)
    a = root2array_from_cobj(o)

print a
