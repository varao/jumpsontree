"""

phyloHPYP: hla.py

Created on 12/27/20 4:15 PM

@author: Hanxi Sun

"""

import os
import time
import numpy as np
import pandas as pd

from src import utils, plots, evals
from src.restFranchise import RestFranchise

# DATE = "030921"
DATE = "031021"
# RUNS = utils.setwd(utils.LOCAL_RUNS + f"hla_{utils.now()}")
RUNS = utils.setwd(utils.LOCAL_RUNS + f"hla_{DATE}")
DATA = utils.LOCAL_DATA + "hla/"

CIRCULAR = True
SCALE_LENGTH = .05

adjust_leaves = False
inhomo = True
inhomo_style = "delta time"
discount = .5
prior_njumps = 1
fix_jump_rate = False
n_iter = 10000

# init results
results = {'parameters': (f'adjust_leaves = {adjust_leaves}, discount = {discount:.2f}, ' +
                          f'prior_njumps = {prior_njumps}, fix_jump_rate = {fix_jump_rate}' +
                          f'inhomo = {inhomo}, inhomo_style = "{inhomo_style}"')}


# ======================================================================================================================
# load data
data = pd.read_csv(DATA + "hla_B57.csv")

# tree
newick = open(DATA + f"1-1000{'_adj' if adjust_leaves else ''}.newick", "r").read().replace('\n', '')
tree = RestFranchise(newick=newick, disc=discount, labels=data.obs.unique())

plots.annotated_tree(tree=tree, data=data, file=RUNS+"data.pdf",
                     circular=CIRCULAR, scale_length=SCALE_LENGTH)

# ======================================================================================================================
# mcmc
if inhomo:
    print("Adjusting the tree branches to achieve inhomogeneous Poisson jumps...")
    tree.inhomogeneous(style=inhomo_style)
    plots.annotated_tree(tree=tree, data=data, file=RUNS + "data_inhomo.pdf",
                         circular=CIRCULAR, scale_length=SCALE_LENGTH)

time0 = time.time()
res = tree.particleMCMC(data, num_particles=3, n_iter=n_iter, if_switch=True, fix_jump_rate=fix_jump_rate,
                        prior_mean_njumps=prior_njumps)
elapse = time.time() - time0

log, jrs, jps, Zs = res["log"], res["jump_rate"], res["jumps"], res["partitions"]

results["elapse_time"] = log["elapse_time"] = elapse

open(RUNS + "log.txt", 'w').write(log.__str__())
np.save(RUNS + "jrs.npy", jrs)
np.save(RUNS + "jps.npy", jps)
np.save(RUNS + "Zs.npy", Zs)

# RUNS = utils.LOCAL_RUNS + f"hla_{}/"
# jrs,jps,Zs = np.load(RUNS+"jrs.npy"),np.load(RUNS+"jps.npy"),np.load(RUNS+"Zs.npy")
# ======================================================================================================================
# plots

tree = RestFranchise(newick=newick, disc=discount, labels=data.obs.unique())
results["bayes_factor"] = evals.bayes_factor(jps, fix_jump_rate, njumps=prior_njumps, burn_in=n_iter//2)

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=False)

results["est_jps"] = tree.jps
results["est_probs"] = evals.empirical(data, tree).tolist()
tree.jps = results["est_jps"]
plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, file=RUNS + "inferred.pdf",
                     circular=CIRCULAR, scale_length=SCALE_LENGTH)

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
results["est_jps1"] = tree.jps
results["est_probs1"] = evals.empirical(data, tree).tolist()
plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, file=RUNS + "inferred1.pdf",
                     circular=CIRCULAR, scale_length=SCALE_LENGTH)

plots.jps_trajectory(jps, jrs, tree, file=RUNS + "jumps_traj.pdf")

plots.njps_histogram(jps, fix_jump_rate=fix_jump_rate, njumps=prior_njumps, burn_in=n_iter//2,
                     file=RUNS + "total_jumps_hist.pdf")

open(RUNS + "results.txt", 'w').write(results.__str__())

# ======================================================================================================================
# paper results
OUT = "/Users/hanxi/Documents/Research/Genetic/PhylogenicSP/phyloHPYP/tex/img/"

jrs, jps, Zs = np.load(RUNS+"jrs.npy"), np.load(RUNS+"jps.npy"), np.load(RUNS+"Zs.npy")
tree = RestFranchise(newick=newick, disc=discount, labels=data.obs.unique())

print(evals.bayes_factor(jps, fix_jump_rate, njumps=prior_njumps, burn_in=n_iter//2))

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
# tree["DQ396400"].njump = 0
# tree.update_jps()

plots.annotated_tree(tree=tree, data=data, show_node_name=False, show_leaf_name=False, data_color_leaf=True,
                     colors=["lightgrey", "black"], special_hzline_width=1, line_width=1,
                     show_jumps_background=True, jump_bg_color="yellow",
                     file=OUT + "hla.pdf",
                     circular=CIRCULAR, scale_length=SCALE_LENGTH)

results = eval(open(RUNS + "results.txt", "r").read())
print(results["elapse_time"])

np.savetxt(RUNS + "jrs.csv", jrs, delimiter=",")
np.savetxt(RUNS + "total_jps.csv", np.sum(jps, axis=1)-1, delimiter=",")
