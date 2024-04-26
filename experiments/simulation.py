"""

phyloHPYP: simulation.py

Created on 12/10/20 12:00 AM

@author: Hanxi Sun

"""

import os
import sys
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from src import evals, utils, plots
from src.restFranchise import RestFranchise

DATA = utils.LOCAL_DATA + "simTrees/"
RUNS = utils.setwd(utils.LOCAL_RUNS + f"simulation_{utils.now()}/")

LABELS = [0, 1]  # COLORS = ["royalblue", "firebrick"]
CIRCULAR = True
# ======================================================================================================================
# tree
treeSize = 100  # number of tips
discount = .5

tree = RestFranchise(newick=utils.newick_from_file(DATA + f"testTree{treeSize:d}.newick"), disc=discount,
                     labels=LABELS)
# plots.annotated_tree(tree=tree, circular=CIRCULAR, file=RUNS+"tree.pdf")


# ======================================================================================================================
# jump + simulation
each_size = 1
# print("\n".join([f"{n.name}: {n.nleaf}" for n in tree.traverse() if (not n.is_root()) and (n.nleaf > treeSize * .2)]))

# data = tree.simulate(njumps=0, min_affected_leaves=.1, not_on_same_branch=True, each_size=each_size)
# data = tree.simulate_data(each_size=each_size)  # no jump
# data = tree.simulate_one_jump(affected_leaves=30, tolerance=10, total_variation=.3, each_size=each_size)[0]
data, nleaf = tree.simulate_one_jump(min_affected_leaves=.2, max_affected_leaves=.5,
                                     total_variation=.3, each_size=each_size)
print(nleaf, evals.empirical_total_variation(data, tree))

ps = evals.empirical(data, tree)
data.to_csv(RUNS + "data.csv", index=False)
data.to_csv(RUNS + "data_treeBreaker.csv", index=False, sep="\t", header=False)

jump_node_id = np.where(tree.jps)[0][1]
results = {"empirical_probs": ps.tolist(), "nleaf_affected": nleaf,
           "true_njumps": np.sum(tree.jps) - 1, "true_jps": tree.jps}
plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=RUNS+"data.pdf")


# ======================================================================================================================
# MCMC
n_iter = 50000
prior_mean_njumps = 1
fix_jump_rate = False

time0 = time.time()
tree.jps = {}
res = tree.particleMCMC(data, num_particles=3, n_iter=n_iter, if_switch=True, fix_jump_rate=fix_jump_rate,
                        init_jr=prior_mean_njumps / tree.tl, prior_mean_njumps=prior_mean_njumps)
elapse = time.time() - time0

log, jrs, jps, Zs = res["log"], res["jump_rate"], res["jumps"], res["partitions"]
log["elapse_time"] = time

open(RUNS + "log.txt", 'w').write(log.__str__())
np.save(RUNS + "jrs.npy", jrs)
np.save(RUNS + "jps.npy", jps)
np.save(RUNS + "Zs.npy", Zs)


# RUNS = utils.LOCAL_RUNS + f"simulation_{7}/"
# data,jrs,jps,Zs = pd.read_csv(RUNS+"data.csv"),np.load(RUNS+"jrs.npy"),np.load(RUNS+"jps.npy"),np.load(RUNS+"Zs.npy")

# ======================================================================================================================
# analysis
plots.jps_trajectory(jps, jrs, tree, file=RUNS + "jump_traj.pdf")

# test total number of jumps
plots.njps_histogram(jps, fix_jump_rate=fix_jump_rate, njumps=prior_mean_njumps, file=RUNS + "total_jumps_hist.pdf",
                     burn_in=n_iter//2)

results["bayes_factor"] = evals.bayes_factor(jps, fix_jump_rate, njumps=prior_mean_njumps, burn_in=n_iter//2)

# ======================================================================================================================
# minimizing the Binder's loss

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=False)
results["estimated_jps"] = tree.jps
plots.annotated_tree(tree=tree, data=data, mark_branch=jump_node_id,
                     show_jumps_background=True, circular=CIRCULAR, file=RUNS+"inferred.pdf")

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
results["estimated_jps1"] = tree.jps
plots.annotated_tree(tree=tree, data=data, mark_branch=jump_node_id,
                     show_jumps_background=True, circular=CIRCULAR, file=RUNS+"inferred1.pdf")

open(RUNS + "results.txt", 'w').write(results.__str__())

