getwd()
DATA = "/Users/hanxi/Documents/Research/local_runs/PhylogenicSP_runs/Data/simTrees/"
library('ape')

#Generate test files with:
ntip = 100
set.seed(0)
write.tree(rtree(ntip), #paste("L", 1:ntip, sep="")), 
           paste(DATA, 'testTree', ntip, '.newick', sep=''))
