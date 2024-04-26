"""

phyloHPYP: hla_data.py

Created on 2019-03-05 17:12

@author: Hanxi Sun

========================================================================================================================
cleaning data for HLA
Data: https://www.hiv.lanl.gov/content/immunology/hlatem/study5/index.html
Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2447109/ and http://www.genetics.org/content/204/1/89

"""

import pandas as pd

from src import utils, plots
from src.restFranchise import RestFranchise

DATA = utils.LOCAL_DATA + "hla/"

# ======================================================================================================================
# clean newick
newick_file = DATA + "1-1000.txt"

newick0 = open(newick_file, "r").read().replace('\n', '')
tree = RestFranchise(newick=newick0, disc=.5, labels=[0, 1])

names = [""] * tree.nleaf
for i, n in enumerate(list(tree.leaves)):
    name = n.name
    name = name[(name.find('_') + 1):]
    name = name[:name.find('_')]
    while name in names:  # duplications in leave names
        name += "_"
    names[i] = name
    n.name = name

for i, n in enumerate(tree.traverse()):
    if not n.is_root() and not n.is_leaf():
        n.name = f"n{i}"

newick0 = tree.write()
open(DATA + "1-1000.newick", 'w').write(newick0)

# ======================================================================================================================
# clean data
data_file = DATA + "hla.txt"
raw_data = pd.read_csv(data_file)  # raw_data.columns
data0 = pd.DataFrame(columns=["node_name", "obs"])
data0.node_name = raw_data.Accession
data0.obs = raw_data.HLA.apply(lambda x: int(str(x).find("B*57") > -1))  # raw_data.HLA.apply(print)

data = data0[data0.node_name.apply(lambda x: x in tree.leaf_names)]
data.reset_index(drop=True, inplace=True)

data.to_csv(DATA + "hla_B57.csv", index=False)

# data = pd.read_csv(DATA + "hla_B57.csv")

plots.annotated_tree(tree, data=data, circular=True, file=DATA + "1-1000.pdf")


# ======================================================================================================================
# adjust leaf branch length
total_bl = tree.tl
nb = tree.nb
nl = tree.nleaf
leaf_bl = 0.
for n in tree.root.get_leaves():
    leaf_bl += n.dist
avg = (total_bl - leaf_bl) / (nb - nl)
rho = nl * avg / leaf_bl
for n in tree.root.get_leaves():
    n.dist *= rho

open(DATA + "1-1000_adj.newick", "w").write(tree.write())
plots.annotated_tree(tree, data=data, circular=True, file=DATA + "1-1000_adj.pdf")





