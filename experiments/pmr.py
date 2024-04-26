"""

phyloHPYP: pmr.py

Created on 12/28/20 9:19 AM

@author: Hanxi Sun

"""

import os
import time
import numpy as np
import pandas as pd

from src import utils, plots, evals
from src.restFranchise import RestFranchise

DATA = utils.LOCAL_DATA + "pmr/"
COLORS = ["purple", "royalblue", "firebrick", "darkgreen"]

CLADES = ["UtoAztecan", "IndoEuropean", "PamaNyungan", "Bantu", "Austronesian"]
CLADE = CLADES[0]
# CLADE = CLADES[1]
# CLADE = CLADES[3]
print(CLADE)
INHOMO = True
INHOMO_STYLE = "delta time"

discount = .5
prior_njumps = 1
fix_jump_rate = False
# fix_jump_rate = True
n_iter = 10000

results = {'parameters': (f'discount = {discount:.2f}, ' +
                          f'prior_njumps = {prior_njumps}, fix_jump_rate = {fix_jump_rate}')}

# RUNS = utils.setwd(utils.LOCAL_RUNS + f"pmr_021021/{CLADE}_0210_201117_15")

# RUNS = utils.setwd(utils.LOCAL_RUNS + f"pmr_{utils.today()}/{CLADE}_{utils.now()}")
RUNS = utils.setwd(utils.LOCAL_RUNS + f"pmr_022421_inhomo/UtoAztecan_0224_124252_93")
# RUNS = utils.setwd(utils.LOCAL_RUNS + f"pmr_022421_inhomo/UtoAztecan_0224_130000_00")


# ======================================================================================================================
# load data
data = pd.read_csv(DATA + CLADE + ".csv")

newick = open(DATA + CLADE + ".newick", "r").read().replace('\n', '')
tree = RestFranchise(newick=newick, disc=discount, labels=data.obs.unique())

CIRCULAR = tree.nleaf > 50
scale_length = tree.tl/tree.nleaf
plots.annotated_tree(tree=tree, data=data, circular=CIRCULAR, colors=COLORS, file=RUNS + "data.pdf",
                     scale_length=scale_length)

# ======================================================================================================================
# mcmc

if INHOMO:
    print("Adjusting the tree branches to achieve inhomogeneous Poisson jumps...")
    tree.inhomogeneous(style=INHOMO_STYLE)
    plots.annotated_tree(tree=tree, data=data, circular=CIRCULAR, colors=COLORS, file=RUNS + "data_inhomo.pdf",
                         scale_length=scale_length)

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


np.savetxt(RUNS + "jrs.csv", jrs, delimiter=",")
np.savetxt(RUNS + "total_jps.csv", np.sum(jps, axis=1)-1, delimiter=",")


# ======================================================================================================================
# plots & analysis

if INHOMO:
    tree = RestFranchise(newick=newick, disc=discount, labels=data.obs.unique())


results["bayes_factor"] = evals.bayes_factor(jps, fix_jump_rate, njumps=prior_njumps, burn_in=n_iter//2)
print(f"Bayes_Factor = {results['bayes_factor']:.2f}")

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=False)
results["est_jps"] = tree.jps
results["est_probs"] = evals.empirical(data, tree).tolist()
plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, colors=COLORS,
                     file=RUNS + "inferred.pdf")

plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=False, colors=COLORS,
                     differentiate_jump_bg=True, jump_bg_cmap="autumn_r",
                     data_color_leaf=True, show_node_name=False, special_hzline_width=2, line_width=2,
                     scale_length=0.1, file=RUNS + CLADE + ".pdf",
                     title=f"{CLADE}: Bayes Factor = {results['bayes_factor']:.2f}", title_fsize=10)

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
results["est_jps1"] = tree.jps
results["est_probs1"] = evals.empirical(data, tree).tolist()
plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=CIRCULAR, colors=COLORS,
                     file=RUNS + "inferred1.pdf")

plots.jps_trajectory(jps, jrs, tree, file=RUNS + "jumps_traj.pdf")

plots.njps_histogram(jps, fix_jump_rate=fix_jump_rate, njumps=prior_njumps, burn_in=n_iter//2,
                     file=RUNS + "total_jumps_hist.pdf")

open(RUNS + "results.txt", 'w').write(results.__str__())

# jrs, jps, Zs = np.load(RUNS + "jrs.npy"), np.load(RUNS + "jps.npy"), np.load(RUNS + "Zs.npy")
# inf = np.inf; results = eval(open(RUNS + "results.txt", 'r').read())
# tree.jps = results["est_jps"]

tree.jps = results["est_jps"]
plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=False, colors=COLORS,
                     differentiate_jump_bg=True, jump_bg_cmap=plots.truncate_colormap("autumn_r", 0., .3),
                     data_color_leaf=True, show_node_name=False, special_hzline_width=2, line_width=2,
                     scale_length=0.1, file=RUNS + CLADE + ".pdf",
                     title=f"{CLADE}: Bayes Factor = {results['bayes_factor']:.2f}", title_fsize=10)


# ======================================================================================================================
# paper results
OUT = "/Users/hanxi/Documents/Research/Genetic/PhylogenicSP/phyloHPYP/tex/img/"

jrs, jps, Zs = np.load(RUNS+"jrs.npy"), np.load(RUNS+"jps.npy"), np.load(RUNS+"Zs.npy")

inf = np.inf
results = eval(open(RUNS + "results.txt", "r").read())
print(results["elapse_time"])

data = pd.read_csv(DATA + CLADE + ".csv")
tree = RestFranchise(newick=newick, disc=discount, labels=data.obs.unique())

print(evals.bayes_factor(jps, fix_jump_rate, njumps=prior_njumps, burn_in=n_iter//2))

tree.jps = evals.estimate_jps(jps, Zs, at_least_one_jump=True)
# n1 = tree["Hopi-16"].up
# n1.njump = 1
# n1.children[0].njump = 0
# n1.children[1].njump = 0
# tree.update_jps()

for n in tree.leaves:
    n.name = n.name[:n.name.rfind('-')]
    if n.name.find("-") > -1:
        n.name = n.name.replace("-", "_")

node_names = data.node_name.apply(lambda x: x[:x.rfind('-')])
node_names = node_names.apply(lambda x: x if x.find("-") == -1 else x.replace("-", "_"))
data.node_name = node_names

plots.annotated_tree(tree=tree, data=data, show_jumps_background=True, circular=False, colors=COLORS,
                     differentiate_jump_bg=True, jump_bg_cmap=plots.truncate_colormap("autumn_r", 0., .3),
                     # jump_bg_color="yellow",
                     data_color_leaf=True, show_node_name=False, special_hzline_width=1, line_width=1,
                     scale_length=0.05, file=OUT + "pmr1.pdf")


