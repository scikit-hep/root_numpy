import ROOT
from root_numpy import *
f = ROOT.TFile('test/test.root')
t = f.Get('tree')
a =  pyroot2array(t)
print a
