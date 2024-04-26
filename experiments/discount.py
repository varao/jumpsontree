"""

phyloHPYP: discount.py

Created on 12/27/20 1:45 PM

@author: Hanxi Sun

# study discount parameter behavior
"""

import os
import sys
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from src import evals, utils, plots
from src.restFranchise import RestFranchise

# RUNS = utils.setwd(utils.LOCAL_RUNS + f"discount_{utils.now()}/")
RUNS = utils.setwd(utils.LOCAL_RUNS + "discount/")
CIRCULAR = True

results = pd.DataFrame(columns=["ds", "K", "tv"])
n_rep = 100
ds_list = [.1, .3, .5, .7, .9]
K_list = [2, 5, 10]

# ======================================================================================================================
for _ in tqdm(range(n_rep)):
    for ds in ds_list:  # discount parameter
        for K in K_list:  # number of categories
            nobs = 100 * K

            # base measure: uniform distribution
            tree = RestFranchise(newick='(leaf:1.)root;', disc=.1, labels=[i for i in range(K)])
            # tree.init_rests()
            tree.root._observed = True
            tree.jps = [1, 1]

            root = tree.root
            leaf = tree.root.children[0]

            data = pd.DataFrame(columns=['node_name', 'obs'])
            for n in [root, leaf]:
                for _ in range(nobs):
                    data = data.append({'node_name': n.name, 'obs': n.seat_new_obs(depend=True)}, ignore_index=True)

            results = results.append({"ds": ds, "K": K, "tv": evals.empirical_total_variation(data, tree)},
                                     ignore_index=True)
results.K = results['K'].astype(int)
results.to_csv(RUNS + "results.csv", index=False)

# ======================================================================================================================
# results = pd.read_csv(RUNS + "results.csv")
for K in K_list:
    plt.scatter(results.ds[results.K == K], results.tv[results.K == K], label=f"{K:d}", alpha=.2)
plt.ylabel("total variation")
plt.xlabel("discount parameter")
plt.legend(title="number of categories")
plt.savefig(RUNS + "results_scatter.pdf")
plt.close()

# sns.boxplot(x="ds", y="tv", hue="K", data=results)
sns.violinplot(x="ds", y="tv", hue="K", inner="stick", data=results, bw=.05)
plt.ylabel("total variation")
plt.xlabel("discount parameter")
plt.legend(title="number of categories")
plt.savefig(RUNS + "results_violin.pdf")
plt.close()

tolerance = 0.05
grouped = results.groupby(["ds", "K"]).agg({"tv": lambda x: np.mean(x < tolerance)}).reset_index()
for K in K_list:
    plt.plot(grouped.ds[grouped.K == K], grouped.tv[grouped.K == K], label=f"{K:d}", alpha=.2)
plt.ylabel("P(total variation = 0)")
plt.xlabel("discount parameter")
plt.legend(title="number of categories")
plt.savefig(RUNS + "results_tv0.pdf")
plt.close()

results_positive = results[results.tv > tolerance]
# sns.boxplot(x="ds", y="tv", hue="K", data=results)
sns.violinplot(x="ds", y="tv", hue="K", inner="stick", data=results_positive)
plt.ylabel("total variation (> 0)")
plt.xlabel("discount parameter")
plt.legend(title="number of categories")
plt.savefig(RUNS + "results_violin_tv1.pdf")
plt.close()

grouped = results.groupby(["ds", "K"]).agg({"tv": lambda x: np.mean(x[x > tolerance])}).reset_index()
for K in K_list:
    plt.plot(grouped.ds[grouped.K == K], grouped.tv[grouped.K == K], label=f"{K:d}", alpha=.2)
plt.ylabel("mean total variation (> 0)")
plt.xlabel("discount parameter")
plt.legend(title="number of categories")
plt.savefig(RUNS + "results_tv_m0.pdf")
plt.close()

# ======================================================================================================================
# Simulate trees with multiple jumps
utils.setwd(RUNS + f"sim_ds_data_prior")
DATA = utils.LOCAL_DATA + "simTrees/"
ds_list = [.1, .3, .5, .7, .9]
TREE_SIZE = 100
LABELS = [0, 1]
NJUMPS = 2
N_REP = 20

for ds in ds_list:  # discount parameter  => 0.7 is better
    tree = RestFranchise(newick=utils.newick_from_file(DATA + f"testTree{TREE_SIZE:d}.newick"),
                         disc=ds, labels=LABELS)
    for i in tqdm(range(N_REP), desc=f"discount={ds:.1f}"):
        data = tree.simulate_prior(njumps=NJUMPS, not_on_same_branch=True, each_size=1,
                                   min_affected_leaves=.1, max_affected_leaves=.8)
        plots.annotated_tree(tree=tree, data=data, differentiate_jump_bg=True, circular=CIRCULAR, K=len(LABELS),
                             mark_branches=true_jump_node,
                             file=RUNS + f"sim_ds_data_prior/sim_ds_{ds:.1f}_data{i:03d}.pdf")

# ======================================================================================================================
# Sketch board
utils.setwd(RUNS + f"sim_2jps_data")
DATA = utils.LOCAL_DATA + "simTrees/"
TREE_SIZE = 100
LABELS = [0, 1]
N_REP = 20

tree = RestFranchise(newick=utils.newick_from_file(DATA + f"testTree{TREE_SIZE:d}.newick"), labels=LABELS)
data = tree.simulate_two_jumps(min_affected_leaves=.2, max_affected_leaves=.8, single_jump_total_variation=.3)[0]
plots.annotated_tree(tree=tree, data=data, differentiate_jump_bg=True, circular=CIRCULAR,
                     file=RUNS + f"sim_2jps_data/sim_2jps.pdf")

evals.empirical(data, tree)


for i in range(N_REP):
    tree = RestFranchise(newick=utils.newick_from_file(DATA + f"testTree{TREE_SIZE:d}.newick"), labels=LABELS)
    data = tree.simulate_two_jumps(min_affected_leaves=.2, max_affected_leaves=.8, single_jump_total_variation=.3)[0]
    plots.annotated_tree(tree=tree, data=data, differentiate_jump_bg=True, circular=CIRCULAR,
                         file=RUNS + f"sim_2jps_data/sim_2jps_{i:03d}.pdf")




