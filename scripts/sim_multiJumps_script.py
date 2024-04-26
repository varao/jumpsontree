"""

phyloHPYP: sim_multiJumps_script.py  (old sim3_script.py)

Created on 12/31/20 1:25 PM

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
# from matplotlib import pyplot as plt
# from sklearn.metrics import roc_curve, auc

from src import evals, utils, plots
from src.restFranchise import RestFranchise


def parse_args():  # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n-iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--prior-njumps', type=float, default=1.0, help='prior number of jumps')
    parser.add_argument('--fix-jump-rate', action='store_true', help="whether the jump rate is fixed")
    parser.add_argument('--data', type=str, help='data file')
    parser.add_argument('--tree', type=str, help='tree newick file')
    parser.add_argument('--true-jumps', type=str, help='true jump location file')
    parser.add_argument('--output', type=str, help='output directory')
    parser.add_argument('--treeBreaker-output', type=str, default=None, help='treeBreaker output file')
    parser.add_argument('--inhomo', action='store_true', help="inhomogeneous jump rate")
    parser.add_argument('--inhomo-style', type=str,
                        default="number of leaves", choices=["delta time", "number of leaves"],
                        help="type of inhomogeneous jump rate")
    parser.add_argument('--summary', type=str, default=None, help='summary directory (prefix of output files)')
    parser.add_argument('--no-pbar', action='store_true', help="not show the progress bar")
    return parser.parse_args()


args = parse_args()
wd = utils.setwd(args.output)

DISCOUNT = 0.5
LABELS = [0, 1]

# ======================================================================================================================
# tree
tree = RestFranchise(newick=utils.newick_from_file(args.tree), disc=DISCOUNT, labels=LABELS)
# plots.annotated_tree(tree=tree, circular=CIRCULAR, file=RUNS+"tree.pdf")

# ======================================================================================================================
# data
data = pd.read_csv(args.data)

true_jumps = eval(open(args.true_jumps, 'r').read())
tree.jps = true_jumps
results = {"empirical_probs": evals.empirical(data, tree).tolist()}
# plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=wd + "data.pdf")


# ======================================================================================================================
# MCMC
time0 = time.time()
tree.jps = {}
if args.inhomo:
    print("Adjusting the tree branches to achieve inhomogeneous Poisson jumps...")
    tree.inhomogeneous(style=args.inhomo_style)

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
burn_in = args.n_iter//2  # lag = max(1, (args.n_iter - burn_in) // 500)
results["bayes_factor"] = evals.bayes_factor(jps, args.fix_jump_rate, njumps=args.prior_njumps, burn_in=burn_in)

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=False)
results["ejps"] = tree.jps
results["pjps"] = np.mean(jps[burn_in:] > 0, axis=0).tolist()
results["emp"] = evals.empirical(data, tree).tolist()
# plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=wd + "inferred.pdf")

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
results["ejps1"] = tree.jps
results["ejps1"] = evals.empirical(data, tree).tolist()
# plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, file=wd + "inferred1.pdf")

open(wd + "results.txt", 'w').write(results.__str__())

if args.summary is not None:
    data.to_csv(args.summary + "data.csv", index=False)
    plots.jps_trajectory(jps, jrs, tree, file=args.summary + "jumps_traj.pdf")
    plots.njps_histogram(jps, fix_jump_rate=args.fix_jump_rate, njumps=args.prior_njumps, burn_in=burn_in,
                         file=args.summary + "total_jumps_hist.pdf")
    open(args.summary + "results.txt", 'w').write(results.__str__())

    if args.treeBreaker_output is not None and not os.path.exists(args.summary + "tB_results.txt"):
        tB_out = evals.treeBreaker_results(args.treeBreaker_output)
        np.save(wd + "tB_Zs.npy", tB_out[-3])
        tB_results = {"ejps": tB_out[-2], "pjps": tB_out[-1]}
        open(wd + "tB_results.txt", 'w').write(tB_results.__str__())
        open(args.summary + "tB_results.txt", 'w').write(tB_results.__str__())
