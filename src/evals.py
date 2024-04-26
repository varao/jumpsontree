"""

phyloHPYP: evals.py

Created on 2019-02-22 15:06

@author: Hanxi Sun

"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from src import utils
from src.restFranchise import RestFranchise
from src.categorical import Categorical
from scipy.stats import poisson, chisquare


def empirical(data, tree: RestFranchise):
    jp_tree, ref_names = tree.jps_prune()
    obs = np.array(pd.crosstab(data.node_name.map(ref_names), data.obs))
    return obs / np.sum(obs, axis=1).reshape(-1, 1)


def empirical_total_variation(data, tree: RestFranchise):
    ps = empirical(data, tree)
    assert ps.shape[0] == 2, "only work for tree with exactly one jump"
    return np.sum(np.abs(ps[0] - ps[1])) / 2.


def estimate_jps(post_jps, post_Zs, at_least_one_jump: bool = False, iter2estC: int = None):
    """
    Cluster assignment is acquired by minimizing the Binder loss
    see
    - Lau, John W., and Peter J. Green. "Bayesian model-based clustering procedures." Journal of Computational
      and Graphical Statistics 16.3 (2007): 526-558.
    - BINDER, D. A. (1978). Bayesian cluster analysis. Biometrika, 65(1), 31–38. doi:10.1093/biomet/65.1.31
    :param post_jps:
    :param post_Zs:
    :param at_least_one_jump:
    :param iter2estC:   number of iterations used to estimate coincidence matrix C, default to be the first half.
    :return: Zh
    """
    K = 0.5  # K = a/(a+b). K=0.5 <=> posterior median estimation of Z. See (4.1) in Lau and Green, 2007.
    if at_least_one_jump:
        indices = [i for i in range(post_Zs.shape[0]) if np.sum(post_jps[i]) > 1]
        post_jps, post_Zs = post_jps[indices], post_Zs[indices]
    N = post_jps.shape[0]
    if N == 0:  # no post_jps available
        return {}  # empty jump (except 1 at the root)
    if iter2estC is None:
        iter2estC = N // 2
    # coincidence matrix
    Cs = np.stack([Z.reshape(-1, 1) == Z for Z in post_Zs])
    rho = np.mean(Cs[:iter2estC], axis=0)
    loss = np.zeros(N - iter2estC)

    for i in range(iter2estC, N):  # i, C = 0, Cs[0]
        loss[i - iter2estC] = np.sum(np.triu(Cs[i] * (rho - K), k=1))
    argmax = np.argmax(loss) + iter2estC
    return post_jps[argmax].tolist()


def posterior_total_njumps(njps, burn_in: int = None):
    n_iter = njps.shape[0]
    if burn_in is None:
        burn_in = n_iter // 2
    return np.mean(njps[burn_in:] == 0)


def prior_expected(NJ: int, fix_jump_rate: bool, jump_rate=None, njumps=None, tree: RestFranchise = None):
    if njumps is None:
        assert jump_rate is not None and tree is not None
        njumps = jump_rate * tree.tl

    expected = np.zeros(NJ + 1)

    if fix_jump_rate:  # prior is poisson
        expected[:-1] = poisson.pmf(np.arange(NJ), njumps)
    else:  # prior is geometric (poisson | exponential)
        p0 = 1. / (njumps + 1)
        expected[:-1] = (1 - p0) * (p0 ** np.arange(NJ))
    expected[-1] = 1 - np.sum(expected)
    return expected


def posterior_total_njumps_vs_prior(njps, fix_jump_rate: bool, jump_rate=None, njumps=None, tree: RestFranchise = None,
                                    burn_in: int = None, lag: int = None):
    n_iter = njps.shape[0]
    if burn_in is None:
        burn_in = n_iter // 2
    if lag is None:
        lag = max(1, (n_iter - burn_in) // 500)

    observed = np.bincount(njps[burn_in::lag])
    expected = prior_expected(np.max(njps[burn_in::lag]), fix_jump_rate, jump_rate, njumps, tree)
    return chisquare(observed, expected)[1]  # p value


def bayes_factor(jps, fix_jump_rate: bool, jump_rate=None, njumps=None, tree: RestFranchise = None,
                 burn_in: int = None):
    if burn_in is None:
        burn_in = jps.shape[0] // 2
    njps = np.sum(jps[burn_in:], axis=1) - 1
    post0 = np.mean(njps == 0)
    prior0 = prior_expected(1, fix_jump_rate, jump_rate, njumps, tree)[0]
    return (1. - post0) / post0 / ((1. - prior0) / prior0) if post0 > 0 else np.Inf


# ##################
# treeBreaker
def treeBreaker_results(file, est_jps=True, pbar=True):
    # file = utils.LOCAL_RUNS + "sim3_123120/rep000_treeBreaker_out.txt"
    tb = open(file, "r").read().split("\n")
    tree = RestFranchise(tb[-2])
    pjps, indices = [], []
    for n in tree.traverse():
        values = [s.split('=')[1] for s in n.name[n.name.find("{"):-1].split('|')]
        indices.append(int(values[0]))
        pjps.append(float(values[-1]))

    post = np.stack([np.fromstring(s[1:], sep="\t") for s in tb[:-2]], axis=0)
    jrs = post[:, -1]
    post[:, -1] = 1  # root indexed the last
    jps = post[:, indices].astype(np.int)
    # pjps = np.array(pjps)

    ejps = None
    Zs = []
    if est_jps:
        for jp in tqdm(jps, desc="TreeBreaker: Estimating Jumps", disable=not pbar):
            tree.jps = jp.tolist()
            Zs.append(tree.partition_by_jps()[0])
        Zs = np.array(Zs, dtype=np.int)
        ejps = estimate_jps(jps, Zs)
    return jrs, jps, Zs, ejps, pjps

# def treeBreaker_rank_results(val=None, file=None):
#     if val is None:
#         val = treeBreaker_results(file)
#     return np.argsort(np.argsort(-val))


# ######################################################################################################################
# from scipy.stats import percentileofscore
# def priDiv_quantile(value, lbd, disc, div_name="KLcb"):
#     # div_name, value, lbd, disc = "KLcb", 0.06, 0.02, 0.5
#     div1jp = pd.read_csv(os.getcwd() + "/runs/" + "priDivs1jp.csv")
#     if value == 0:
#         return 0
#     else:
#         p0 = poisson.pmf(0, mu=lbd)
#         d = div1jp[div_name][div1jp.discount == disc]
#         p1 = percentileofscore(d, value) / 100
#         return p0 + (1-p0) * p1
#
#
#
# def div_emp_distr(div_name, discount=None, max_jps=None, tree=None, jump_rate=None, base=Categorical.uniform(2),
#                   n_iter=1, n_rep=1000, accuracy=.05):
#     branch = RestFranchise(newick="(Leaf)Root;", base=base, disc=.5)  # todo
#     root = branch.root
#     leaf = branch.root.children[0]
#
#     if discount is None:
#         if tree is None:
#             raise ValueError("When discount is None, tree should be given. ")
#         discount = tree.disc
#     if max_jps is None:
#         if tree is None or jump_rate is None:
#             raise ValueError("When avg_branch_njp is None, both tree and jump_rate should be given. ")
#         max_jps = max(tree.bls) * jump_rate
#
#     ds0 = discount  # ds in studyJumpRate  make these whole thing running everytime when inferring
#     p = 1. - poisson.pmf(0, max_jps)
#     Njp = 1
#     while p >= accuracy:
#         p -= poisson.pmf(Njp, max_jps)
#         Njp += 1
#
#     emp_distr = np.zeros((n_rep, Njp))
#
#     cnt = 0
#     branch.disc = ds0
#     for j in range(Njp):
#         njp = j + 1
#         # print("njp={}, start:".format(njp), now_str())
#         leaf.njump = njp
#         branch.init_rests()
#         for i in range(n_rep):
#             ss = 0.
#             for _ in range(n_iter):
#                 root_pm = root.rest.post_measure(n_atom=50)
#                 leaf_pm = leaf.rest.post_measure(n_atom=50)
#                 ss += utils.div(root_pm, leaf_pm, div_name)
#             emp_distr[i, j] = ss / n_iter
#             if ss == 1.:
#                 cnt += 1
#                 # print(pm)
#     # print(cnt)
#     return emp_distr
#
#
# def div_p_value(value, lbd, emp_distr, pp0=None, pps_stack=None, vs_emp=False):
#     if value == 0:
#         return 1.
#     elif value == np.inf:
#         return 0.
#     else:
#         if not vs_emp:
#             n_rep, Njp = emp_distr.shape
#             if pp0 is None:
#                 pp0 = poisson.pmf(0, lbd)
#             if pps_stack is None:
#                 pps = poisson.pmf(np.arange(1, Njp+1), lbd)  # poisson probabilities
#                 pps_stack = np.repeat(pps.reshape((1, -1)), n_rep, axis=0)
#             pge = np.sum(pps_stack[value > emp_distr]) / n_rep  # all the probabilities that div > emp_distr
#             p = 1. - pge - pp0
#         else:
#             p = np.mean(value <= emp_distr)
#         return p
#
#
# def div_overlap(divs, lbd, emp_distr, vs_emp=False, mann_test=False, mann_test_lag=None):
#     if np.sum(divs > 0) == 0:
#         return 1.
#     elif np.sum(divs) == np.inf:
#         return 0.
#     else:
#         if not vs_emp:
#             n_rep, Njp = emp_distr.shape
#             emp_distr = np.concatenate((np.zeros((n_rep, 1)), emp_distr), axis=1)
#             pps = poisson.pmf(np.arange(Njp+1), lbd)
#             weights = np.repeat(pps.reshape((1, -1)), n_rep, axis=0).reshape(-1)
#             weights = weights / np.sum(weights)
#             p = 1. - utils.overlap_area(emp_distr.reshape(-1), divs, arr1_weights=weights)
#         elif mann_test:
#             if np.sum(divs[::mann_test_lag] > 0.) == 0:
#                 p = 1.
#             else:
#                 p = mannwhitneyu(divs[::mann_test_lag], emp_distr[::mann_test_lag], alternative="greater")[1]
#         else:
#             p = 1. - utils.overlap_area(emp_distr, divs)
#         return p
#
#
# def _process(tree, jps=None, divs=None, statistic="mean", unit_length=False):
#     if jps is not None and jps.shape[1] == tree.nb + 1:
#         jps = jps[:, 1:]
#
#     if jps is None and divs is None:
#         raise ValueError("One metric (jumps or divergences) should be provided! ")
#     if jps is not None and divs is not None:
#         raise ValueError("Only handles one metric (jumps or divergences).")
#     if_divs = jps is None
#     samples = divs if if_divs else jps
#
#     if unit_length:
#         bls = np.array(tree.bls[1:]).reshape(1, -1)
#         samples = samples / bls
#
#     if statistic == "mean":
#         val = np.mean(samples, axis=0)  # estimate
#     elif statistic == "median":
#         val = np.median(samples, axis=0)
#     elif statistic[-1] == "%":  # bottom a% percentile
#         a = float(statistic[:-1])
#         val = np.percentile(samples, a, axis=0)
#     else:
#         val = None
#         # raise ValueError("Statistic (" + statistic + ") not recognized. " +
#         #                  "Should be one of [\"mean\", \"median\", \"*%\" (* being a number)].")
#
#     return val, samples, if_divs, jps
#
#
# def p_value(tree: RestFranchise, jps=None, divs=None,
#             statistic="mean", jump_rate=None, div_name=None,
#             vs_prior=False, vs_emp=False, unit_length=False, overlap=False, mann_test=False, mann_test_lag=100,
#             prior_div_distr=None, emp_div_distr=None):
#
#     p = np.zeros(tree.nb)
#
#     if not vs_prior and not vs_emp:
#         raise ValueError("p-value is calculated w.r.t. prior or null?")
#
#     if vs_emp:
#         unit_length = vs_prior = False
#         if emp_div_distr is None:
#             raise ValueError("emp_div_distr is required when comparing with empirical distributions. ")
#
#     val, samples, if_divs, jps = _process(tree, jps=jps, divs=divs, statistic=statistic, unit_length=unit_length)
#
#     if if_divs and vs_prior:
#         if prior_div_distr is None:
#             prior_div_distr = div_emp_distr(div_name, tree=tree, jump_rate=jump_rate, n_rep=1000, n_iter=1)
#
#     if vs_emp and type(emp_div_distr) == str:
#         emp_div_distr = np.load(emp_div_distr)
#
#     for i, n in enumerate(tree.root.traverse('preorder')):
#         if n.is_root():
#             continue
#         jr = jump_rate if unit_length else n.dist * jump_rate
#
#         if if_divs:
#             distr = emp_div_distr[:, i-1] if vs_emp else prior_div_distr
#             if overlap or mann_test:
#                 p[i-1] = div_overlap(samples[:, i-1], jr, distr, vs_emp=vs_emp,
#                                      mann_test=mann_test, mann_test_lag=mann_test_lag)
#             else:
#                 p[i-1] = div_p_value(val[i-1], jr, distr, vs_emp=vs_emp)
#         else:
#             p[i-1] = 1. - poisson.cdf(val[i-1], jr)
#
#     return p
#
#
# def L1(tree: RestFranchise, data):
#     tr = tree.deep_copy()
#     l1 = np.zeros(tr.nb)
#     for i, n in enumerate(tr.root.traverse('preorder')):
#         if n.is_root():
#             continue
#         tr.jps = {tr.root.name: 1, n.name: 1}
#         _, _l1 = data_L1(data, tr)
#         l1[i-1] = _l1
#     return l1
#
#
# def get_results(tree, data, jps=None, divs=None, statistic="mean", unit_length=False, return_L1=True, **kwargs):
#     val, samples, if_divs, jps = _process(tree, jps=jps, divs=divs, statistic=statistic, unit_length=unit_length)
#     std = None if val is None else np.std(samples, axis=0)
#     p = p_value(tree, jps=jps, divs=divs, statistic=statistic, unit_length=unit_length, **kwargs)
#     l1 = L1(tree, data) if return_L1 else None
#     return {"val": val, "std": std, "pvalue": p, "L1": l1}
#
#
# def rank(val):
#     return np.argsort(np.argsort(val))
#
#
# def rank_results(res):
#     res = {"val": rank(res["pvalue"]),
#            "direction": "l", "metric": "val", "int_value": True, "value_name": "rank"}
#     return res




