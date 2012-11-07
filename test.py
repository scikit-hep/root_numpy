from root_numpy import root2array
from ROOT import TChain
import ROOT

chain = TChain('tree')
chain.Add('test/vary1.root')

#librootnumpy.root2array_fromCObj(ROOT.AsCObject(chain))

#librootnumpy.root2array_fromFname(['test/single1.root'],'tree')

#librootnumpy.root2array_fromFname(['test/single2.root'],'tree')

#librootnumpy.root2array_fromFname(['test/fixed1.root'],'tree')

#librootnumpy.root2array_fromFname(['test/fixed2.root'],'tree')

#librootnumpy.root2array_fromFname(['test/vary1.root'],'tree')

#librootnumpy.root2array_fromFname(['test/vary2.root'],'tree')

#print librootnumpy.root2array_fromFname(['test/sprtest.root'],'ClassRecord')

#print librootnumpy.list_tree('test/sprtest.root')

print root2array('test/vary1.*',offset=5,N=2)