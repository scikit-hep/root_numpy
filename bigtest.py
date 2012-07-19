from root_numpy import*
mcblock = ['mcLen','mcLund','dauLen','dauIdx','mcp3','mccosth','runNumber']
pat = 'SP6431_Run5*.root'
#fnames = glob('/Volumes/WD3TB/data/V4/SJM_SP6431_Run%d/*/SJM*.root'%runno)
fnames = glob('/Volumes/WD3TB/optdata/SJM_SP6431_Run[56].root')
a =root2rec(fnames,branches=mcblock)