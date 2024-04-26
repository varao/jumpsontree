"""

phyloHPYP: sim1_script.py

Created on 12/26/20 1:39 PM

@author: Hanxi Sun

"""

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")  # phyloHPYP

import argparse
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')

from src import evals, utils, plots
from src.restFranchise import RestFranchise


def parse_args():  # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n-iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--tree-size', type=int, default=100, help='simulation tree size')
    parser.add_argument('--affected-leaves', type=int, default=None,
                        help='#leaves affected by the jump. Randomly choose within appropriate jump locations if None')
    parser.add_argument('--discount', type=float, default=.5, help='discount parameter')
    parser.add_argument('--prior-njumps', type=float, default=1.0, help='prior number of jumps')
    parser.add_argument('--total-variation', type=float, default=0., help='prior mean number of jumps')
    parser.add_argument('--fix-jump-rate', action='store_true', help="whether the jump rate is fixed")
    parser.add_argument('--inhomo', action='store_true', help="inhomogeneous jump rate")
    parser.add_argument('--inhomo-style', type=str,
                        default="number of leaves", choices=["delta time", "number of leaves"],
                        help="type of inhomogeneous jump rate")
    parser.add_argument('--output', type=str, help='output directory')
    parser.add_argument('--summary', type=str, default=None, help='summary directory (prefix)')
    parser.add_argument('--no-pbar', action='store_true', help="not show the progress bar")
    return parser.parse_args()


args = parse_args()

DATA = utils.SERVER_DATA + "simTrees/"
wd = utils.setwd(args.output)

LABELS = [0, 1]
EACH_SIZE = 1

# ======================================================================================================================
# tree
tree = RestFranchise(newick=utils.newick_from_file(DATA + f"testTree{args.tree_size:d}.newick"),
                     disc=args.discount, labels=LABELS)
# plots.annotated_tree(tree=tree, circular=CIRCULAR, file=RUNS+"tree.pdf")

# ======================================================================================================================
# data
if not args.no_pbar:
    print("Generating data", utils.now())

data, nleaf = tree.simulate_one_jump(min_affected_leaves=.1, max_affected_leaves=.5,
                                     affected_leaves=args.affected_leaves, total_variation=args.total_variation,
                                     each_size=EACH_SIZE)
ps = evals.empirical(data, tree)
data.to_csv(wd + "data.csv", index=False)
data.to_csv(wd + "data_treeBreaker.csv", index=False, sep="\t", header=False)

results = {"empirical_probs": ps.tolist(), "empirical_total_variation": evals.empirical_total_variation(data, tree),
           "nleaf_affected": nleaf, "true_jps": tree.jps}  # "true_njumps": np.sum(tree.jps) - 1}
# plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=wd + "data.pdf")

open(wd + "results.txt", 'w').write(results.__str__())

# ======================================================================================================================
# MCMC
time0 = time.time()

if args.inhomo:
    print("Adjusting the tree branches to achieve inhomogeneous Poisson jumps...")
    tree.inhomogeneous(style=args.inhomo_style)

tree.jps = {}
res = tree.particleMCMC(data, num_particles=3, n_iter=args.n_iter, if_switch=True, fix_jump_rate=args.fix_jump_rate,
                        prior_mean_njumps=args.prior_njumps, progress_bar=not args.no_pbar)
elapse = time.time() - time0

log, jrs, jps, Zs = res["log"], res["jump_rate"], res["jumps"], res["partitions"]

results["elapse_time"] = log["elapse_time"] = elapse

open(wd + "log.txt", 'w').write(log.__str__())
np.save(wd + "jrs.npy", jrs)
np.save(wd + "jps.npy", jps)
np.save(wd + "Zs.npy", Zs)

# ======================================================================================================================
# analysis
results["bayes_factor"] = evals.bayes_factor(jps, args.fix_jump_rate, njumps=args.prior_njumps, burn_in=args.n_iter//2)

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=False)
results["estimated_jps"] = tree.jps
results["estimated_probs"] = evals.empirical(data, tree).tolist()
# plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=wd + "inferred.pdf")

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
results["estimated_jps1"] = tree.jps
results["estimated_probs1"] = evals.empirical(data, tree).tolist()
# plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=wd + "inferred1.pdf")

open(wd + "results.txt", 'w').write(results.__str__())

if args.summary is not None:
    data.to_csv(args.summary + "data.csv", index=False)
    plots.jps_trajectory(jps, jrs, tree, file=args.summary + "jumps_traj.pdf")
    plots.njps_histogram(jps, fix_jump_rate=args.fix_jump_rate, njumps=args.prior_njumps, burn_in=args.n_iter//2,
                         file=args.summary + "total_jumps_hist.pdf")
    open(args.summary + "results.txt", 'w').write(results.__str__())

