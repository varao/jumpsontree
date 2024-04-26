#install.packages('coda')
library(coda)
x <- read.csv('/Users/hanxi/Documents/Research/local_runs/PhylogenicSP_runs/pmr_022421_inhomo/UtoAztecan_0224_130000_00/jrs.csv', header = FALSE)
effectiveSize(x) / 1044.1800141334534

x <- read.csv('/Users/hanxi/Documents/Research/local_runs/PhylogenicSP_runs/pmr_022421_inhomo/UtoAztecan_0224_124252_93/total_jps.csv', header = FALSE)
effectiveSize(x) / 978.0955650806427

x <- read.csv('/Users/hanxi/Documents/Research/local_runs/PhylogenicSP_runs/hla_031021/jrs.csv', header = F)
effectiveSize(x) #/ 2098.2846958637238
