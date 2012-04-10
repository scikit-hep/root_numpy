import ROOT
from root_numpy import *
f = ROOT.TFile('test/test.root')
t = f.Get('tree')
a =  tree2array(t)
print a
