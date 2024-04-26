"""

phyloHPYP: plots.py

Created on 2019-01-08 10:05

@author: Hanxi Sun

"""

import numpy as np
from ete3 import TreeStyle, NodeStyle, TextFace
import matplotlib.pyplot as plt
import matplotlib.colors as Colors
from matplotlib.colors import to_hex

from matplotlib import cm
from src.restFranchise import RestFranchise
from src.evals import prior_expected, bayes_factor


# ==================================================================================================================== #

def jps_trajectory(jps, jrs, tree: RestFranchise, file):  # trajectory jump rate
    plt.plot(jrs * tree.tl, label="jump rate (per tree)", alpha=.5)
    plt.plot(np.sum(jps, axis=1) - 1, label="total #jump", alpha=.5)
    plt.plot(np.arange(jps.shape[0])+10, np.sum(jps > 0, axis=1) - 1.05, label="total #branches with jump", alpha=.5)
    plt.title("Trajectory of Jumps")
    plt.xlabel("iterations")
    plt.ylabel("number of jumps")
    plt.legend()
    plt.savefig(file, format="pdf")
    plt.close()


def njps_histogram(jps, file, fix_jump_rate: bool, jump_rate=None, njumps=None, tree: RestFranchise = None,
                   burn_in: int = None):  # total number of jumps
    if burn_in is None:
        burn_in = jps.shape[0] // 2
    njps = np.sum(jps[burn_in:], axis=1) - 1
    NJ = np.max(njps)
    plt.hist(njps, bins=np.arange(NJ+2)-0.5, label="Posterior", density=True)
    plt.plot(np.arange(NJ+1), prior_expected(NJ, fix_jump_rate, jump_rate, njumps, tree), label="Prior")
    plt.title("Posterior Total Number of Jumps. " +
              f"Bayes Factor = {bayes_factor(jps, fix_jump_rate, jump_rate, njumps, tree):.2f}")
    plt.legend()
    plt.savefig(file)
    plt.close()


# ==================================================================================================================== #

def rgb_scale(n, gray_scale=False, CMAP="Paired", light_bg=False):
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    # gray: 1. = black; 0. = white
    # light_bg (for reverse cmaps): 1. = lightest; 0.= darkest
    grid = np.linspace(.2 if gray_scale else (0. if not light_bg else .5), 1., n)
    if type(CMAP) == str:
        cmap = cm.get_cmap("Greys" if gray_scale else CMAP)
    else:
        cmap = CMAP
    return cmap(grid)


def signif_rgb(v, if_less, criteria, MAX=1., MIN=0.):
    cmap = cm.get_cmap("Reds")
    if criteria is None:
        signif = 1.
    elif if_less:
        signif = (v - MIN) / (criteria - MIN) if criteria > MIN else 1.
    else:
        signif = (MAX - v) / (MAX - criteria) if criteria < MAX else 1.
    return to_hex(cmap(signif * .5 + .5))


# ==================================================================================================================== #

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=200):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = Colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def annotated_tree(tree: RestFranchise, file=None, title=None, title_fsize=20,
                   val=None, std=None, L1=None, pvalue=None, value_name=None, int_value=False,
                   data=None, K=None, colors=None, data_color_leaf=False,
                   criteria=None, vs_node=None, direction='less', metric='p-value',
                   show_branch_length=False, show_jumps=True, show_jumps_background=False,
                   show_node_name=True, show_leaf_name=True,
                   mark_branch=None, mark_branches=None,
                   special_node_size=5, special_hzline_width=3, line_width=1,
                   jump_bg_color='lightyellow', jump_bg_cmap='autumn', differentiate_jump_bg=False,
                   scale_length=1, circular=False):
    tr = tree.deep_copy()
    ts = TreeStyle()
    ts.show_leaf_name = show_leaf_name
    if scale_length is not None:
        ts.scale = 50 / scale_length
    if circular:
        ts.mode = 'c'

    data_column = (not data_color_leaf) and (show_leaf_name or show_node_name or data is None or
                                             data.shape[0] > len(tree.leaves))
    if data is not None:
        if colors is None:
            K = len(np.unique(data.obs)) if K is None else K
            colors = rgb_scale(K, gray_scale=(K <= 2))

        if data_column:
            for leaf in tr.root.iter_leaves():
                obs = data.obs[data.node_name == leaf.name].sort_values()
                d = TextFace(" " * 1)
                d.background.color = to_hex("white")
                leaf.add_face(d, column=0, position="aligned")
                for i, label in enumerate(obs):
                    d = TextFace(" ")
                    d.background.color = to_hex(colors[label])
                    leaf.add_face(d, column=i+1, position="aligned")

    if val is None and pvalue is None:
        metric = None

    if vs_node is not None:
        if type(vs_node) == str:
            i = tr[vs_node].id
        elif type(vs_node) == int:
            i = vs_node
        else:
            i = vs_node.id
        criteria = pvalue[i - 1] if metric == "p-value" else val[i - 1]

    if direction == "less" or direction == "l":
        if_less = True
    elif direction == "greater" or direction == "g":
        if_less = False
    else:
        raise ValueError("direction not recognized. Should be \"less\" or \"greater\". ")

    jump_counter = 0
    if differentiate_jump_bg:
        show_jumps_background = True
        jump_bg_color = sum(int(njump > 0) for njump in tree.jps) - 1
    show_jumps = show_jumps if not show_jumps_background else True
    if show_jumps_background and type(jump_bg_color) == int and jump_bg_color > 0:
        jump_bg_color = [to_hex(c) for c in rgb_scale(jump_bg_color, CMAP=jump_bg_cmap, light_bg=True)]

    for i, node in enumerate(tr.traverse()):

        nstyle = NodeStyle()
        nstyle['size'] = 1
        nstyle["vt_line_width"] = line_width
        nstyle["hz_line_width"] = line_width

        if node.is_root():
            node.set_style(nstyle)
            continue

        if mark_branch is not None and (node.id == mark_branch or node.name == mark_branch):
            nstyle['size'] = 5
            nstyle['fgcolor'] = "red"
        if mark_branches is not None and ((node.id in mark_branches) or (node.name in mark_branches)):
            nstyle['size'] = 5
            nstyle['fgcolor'] = "red"

        if show_branch_length:
            node.add_face(TextFace(" length = {:.03f}".format(node.dist)),
                          column=0, position="branch-bottom")

        if L1 is not None:
            node.add_face(TextFace(" L1(data) = {:.03f}".format(L1[i - 1])),
                          column=0, position="branch-bottom")

        if show_node_name or val is not None:
            msg = node.name if not node.is_leaf() else ""
            if val is not None:
                if not node.is_leaf():
                    msg += ": "
                msg += (((value_name + " = ") if value_name is not None else "") +
                        ("{:d}".format(val[i-1]) if int_value else "{:.3f}".format(val[i-1])) +
                        (" (±{:.3f})".format(std[i-1]) if std is not None else ""))
            node.add_face(TextFace(msg), column=0, position="branch-top")

        if pvalue is not None:
            P = TextFace(" p-value = {:.04f}".format(pvalue[i-1]))
            node.add_face(P, column=0, position="branch-bottom")

        if metric is not None:
            v = pvalue[i-1] if metric == "p-value" else val[i-1]
            signif = True if criteria is None else ((v <= criteria) if if_less else (v >= criteria))
            nstyle['hz_line_color'] = signif_rgb(v, if_less=if_less, criteria=criteria) if signif else to_hex("black")
            nstyle['hz_line_width'] = special_hzline_width if signif else 1

        if node.njump > 0 and show_jumps:
            if show_jumps_background:
                if type(jump_bg_color) == str:
                    col = jump_bg_color
                else:
                    if differentiate_jump_bg:
                        col = jump_bg_color[jump_counter]
                        jump_counter += 1
                    else:
                        col = jump_bg_color[node.njump-1]
                nstyle['bgcolor'] = col
            else:
                nstyle['fgcolor'] = "goldenrod"
                nstyle['size'] = special_node_size

        if data is not None and data_color_leaf and node.is_leaf():
            obs = data.obs[data.node_name == node.name]
            if len(obs) == 1:
                c = colors[int(obs)]
                nstyle['fgcolor'] = to_hex(c)
                nstyle["hz_line_color"] = to_hex(c)
                nstyle['size'] = special_node_size
                nstyle['hz_line_width'] = special_hzline_width
                # print(node.name, c)

        node.set_style(nstyle)

    if title is not None:
        ts.title.add_face(TextFace(title, fsize=title_fsize), column=0)

    if file is None:
        tr.root.show(tree_style=ts)
    else:
        tr.root.render(file, tree_style=ts)
    del tr


# ==================================================================================================================== #

# def circular_tree(tree: RestFranchise, file=None,
#                   val=None, std=None, L1=None, pvalue=None, value_name=None, int_value=False,
#                   data=None, colors=None,
#                   criteria=None, vs_node=None, direction="less", metric="p-value",
#                   show_branch_length=False, show_jumps=True,  show_jumps_background=False,
#                   show_node_name=True, show_leaf_name=True,
#                   special_node_size=5, special_hzline_width=3, jump_bg_color="lightyellow", scale=50):
#     tr = tree.deep_copy()
#     ts = TreeStyle()
#     ts.show_leaf_name = show_leaf_name
#     if scale is not None:
#         ts.scale = scale
#     ts.mode = 'c'
#
#     if data is not None:
#         if colors is None:
#             K = len(np.unique(data.obs))
#             colors = rgb_scale(K, gray_scale=(K <= 2))
#
#         for leaf in tr.root.iter_leaves():
#             obs = data.obs[data.node_name == leaf.name].sort_values()
#             d = TextFace(" " * 1)
#             d.background.color = to_hex("white")
#             leaf.add_face(d, column=0, position="aligned")
#             for i, label in enumerate(obs):
#                 d = TextFace(" ")
#                 d.background.color = to_hex(colors[label])
#                 leaf.add_face(d, column=i+1, position="aligned")
#
#     if val is None and pvalue is None:
#         metric = None
#
#     if vs_node is not None:
#         if type(vs_node) == str:
#             i = tr[vs_node].id
#         elif type(vs_node) == int:
#             i = vs_node
#         else:
#             i = vs_node.id
#         criteria = pvalue[i - 1] if metric == "p-value" else val[i - 1]
#
#     if direction == "less" or direction == "l":
#         if_less = True
#     elif direction == "greater" or direction == "g":
#         if_less = False
#     else:
#         raise ValueError("direction not recognized. Should be \"less\" or \"greater\". ")
#
#     jump_count = 0
#     show_jumps = show_jumps if not show_jumps_background else True
#     if show_jumps_background and type(jump_bg_color) == int:
#         jump_bg_color = [to_hex(c) for c in rgb_scale(jump_bg_color)]
#
#     for i, node in enumerate(tr.root.traverse('preorder')):
#         nstyle = NodeStyle()
#         nstyle['fgcolor'] = "black"
#
#         if node.is_root():
#             continue
#
#         if show_branch_length:
#             node.add_face(TextFace(" length = {:.03f}".format(node.dist)),
#                           column=0, position="branch-bottom")
#
#         if L1 is not None:
#             node.add_face(TextFace(" L1(data) = {:.03f}".format(L1[i - 1])),
#                           column=0, position="branch-bottom")
#
#         if show_node_name or val is not None:
#             msg = node.name
#             if val is not None:
#                 msg += (": " + ((value_name + " = ") if value_name is not None else "") +
#                         ("{:d}".format(val[i-1]) if int_value else "{:.3f}".format(val[i-1])) +
#                         (" (±{:.3f})".format(std[i-1]) if std is not None else ""))
#             node.add_face(TextFace(msg), column=0, position="branch-top")
#
#         if pvalue is not None:
#             P = TextFace(" p-value = {:.04f}".format(pvalue[i-1]))
#             node.add_face(P, column=0, position="branch-bottom")
#
#         if metric is not None:
#             v = pvalue[i-1] if metric == "p-value" else val[i-1]
#             signif = (v <= criteria) if if_less else (v >= criteria)
#             nstyle['hz_line_color'] = signif_rgb(v, if_less=if_less, criteria=criteria) if signif else to_hex("black")
#             nstyle['hz_line_width'] = special_hzline_width if signif else 1
#
#         if node.njump > 0 and show_jumps:
#
#             if show_jumps_background:
#                 nstyle['bgcolor'] = jump_bg_color if type(jump_bg_color) == str else jump_bg_color[jump_count]
#                 jump_count += 1
#             else:
#                 nstyle['fgcolor'] = "goldenrod"
#                 nstyle['size'] = special_node_size
#
#         node.set_style(nstyle)
#
#     if file is None:
#         tr.root.show(tree_style=ts)
#     else:
#         tr.root.render(file, tree_style=ts)
#     del tr


# ==================================================================================================================== #

# def prior_hist(emp_distr=None, avg_njp=None, div_name=None, njp=None, tree=None,
#                plot=False, plot_title=None, plot_file=None, posterior=None):  # , p_value=False):
#     if emp_distr is None:
#         if njp is None or tree is None or div_name is None:
#             raise ValueError("njp, tree and div_name are all required to generate " +
#                              "empirical prior distribution (emp_distr). ")
#         jump_rate = njp / tree.tl
#         emp_distr = evals.div_emp_distr(div_name, tree=tree, jump_rate=jump_rate)
#
#     if avg_njp is None:
#         if njp is None or tree is None:
#             raise ValueError("njp and tree are required to obtain the average number " +
#                              "of jumps per branch (emp_distr). ")
#         avg_njp = njp / tree.nb
#
#     N, Njp = emp_distr.shape
#     pps = poisson.pmf(np.arange(Njp+1), avg_njp)  # poisson probabilities
#     sim = np.concatenate((np.zeros((N, 1)), emp_distr), 1).reshape(-1)
#     weights = np.repeat(pps.reshape(1, -1), N, axis=0).reshape(-1)
#
#     if plot:
#         plt.hist(sim, bins=100, density=True, weights=weights)
#         if posterior is not None:
#             plt.axvline(x=posterior, color='r')
#             # if p_value:
#             #     if plot_title is None:
#             #         plot_title = ""
#             #     plot_title += " p={:.04f}".format(np.sum(weights[sim > posterior]) / np.sum(weights))
#         if plot_title is not None:
#             plt.title(plot_title)
#         if plot_file is None:
#             plt.show()
#         else:
#             plt.savefig(plot_file)
#         plt.close()
#
#     return sim, weights
#
#
#
#
# def plot_sig(tree, values, data, metric=None, criteria=None, vs_node=None, jump_rate=None,
#              statistic="mean",
#              unit_length=False, vs_prior=True, overlap=False, mann_test=False, vs_null=False, vs_null_p=False,
#              show_value=True, show_std=False, show_emp_L1=True, show_jumps=True, show_p=True, show_leaf_name=False,
#              sig_branch=True, sig_node=False,
#              emp_div_distr=None, div_name=None, each_size=None,
#              null_div_distr=None, null_p_value_distr=None,
#              file=None, colors=None, labels=None,
#              special_node_size=5, special_hzline_width=3, scale=None):
#     """
#     Plot the tree with significance measurement on each node.
#     :param tree: A RestFranchise object representing the tree of interests
#     :param values: The values of shape (n_iter, n_branch), if it not matches with tree.nb, will throw an error
#     :param data: The data at each node
#     :param file: The file to save the figure. If None, show the figure directly.
#     :param criteria: the criteria to give red marks (significant ones) on branches, will be ignored if vs_node is
#                 given (the direction is determined by metric, overlap, vs_prior etc, no need to specify here).
#     :param vs_node: compare with a certain node. If provided, will ignore criteria.
#     :param metric: The metric of value. Can be "njumps", "div" or "p-value". If None, throw an error. If is "p-value",
#                 then vs_prior, overlap and unit_length will all be ignored.
#     :param jump_rate: The jump rate in the tree. Not required if metric == "njumps" and vs_prior == True.
#     :param statistic: The statistic to use, can be "mean", "median", "10" (bottom 10 percentile), "90" (top 10%).
#     :param vs_prior: Values are normalized with branch length by comparing with the prior. Conflicts with unit_length
#     :param overlap: Whether consider the overlapping between the empirical distribution with the prior distribution.
#                 If False, then compare the posterior mean with the prior. Will be ignored if vs_prior == False.
#                 Also not applicable when metric="njumps"  # todo make it works for "njumps"
#     :param vs_null: If the p-value is resulting from comparing posterior divergence
#                 with the null model (no jump), requires null_div_distr
#     :param vs_null_p: If the p-value is comparing with the bootstrapped p-values from the null model, requires
#                 null_p_value_distr
#     :param null_div_distr: The .npy file (str) or the np.array object that contains the same divergence posterior
#                 samples from running the algorithm on datasets that generated from the null model.
#     :param null_p_value_distr: The .npy file (str) or the np.array object that contains the bootstrapped p-values
#                 from the null.
#     :param unit_length: Values are normalized with branch length directly. (i.e. v/bl). Conflicts with vs_prior
#     :param show_std: Whether to show std along the estimate
#     :param show_emp_L1: Whether to show the empirical L1 divergence between data above and below each branch.
#     :param show_jumps: Whether to show the jumps (blue dot) in tree
#     :param emp_div_distr: estimates of prior distribution on the divergence. Will be ignored if metric == "njumps"
#     :param div_name: the name of the divergence. Will be ignored if emp_div_distr is provided.
#     :param each_size: the (max) number of data points at each leaf node.
#     :param labels: labels for showing the data
#     :param colors: colors for showing the data
#     :return:
#     """
#
#     tr = tree.deep_copy()
#
#     if np.ndim(values) != 2:
#         raise ValueError("values should be of shape (n_iter, n_branch). ")
#     if values.shape[1] != tr.nb:
#         raise ValueError("values should be of shape (n_iter, n_branch). " +
#                          "When working with njumps, please remove column 0 in jps (the jump at root) first. ")
#
#     if metric == "p-value":
#         unit_length = vs_prior = overlap = False
#     if vs_null:
#         unit_length = vs_prior = False
#         if null_div_distr is None:
#             raise ValueError("null_div_distr is required when comparing with null model. ")
#     if vs_null_p:
#         # unit_length = vs_prior = overlap = False
#         if null_p_value_distr is None:
#             raise ValueError("null_p_value_distr is required to compare with bootstrapped p-values from the null. ")
#         raise ValueError("Not supported in current version")
#
#     if unit_length:
#         bls = np.array(tr.bls[1:]).reshape(1, -1)
#         values = values / bls
#
#     if statistic == "mean":
#         est = np.mean(values, axis=0)  # estimate
#     elif statistic == "median":
#         est = np.median(values, axis=0)
#     elif statistic == "10":  # bottom 10 %
#         est = np.percentile(values, 10, axis=0)
#     elif statistic == "90":  # bottom 10 %
#         est = np.percentile(values, 90, axis=0)
#     else:
#         raise ValueError("Statistic (" + statistic + ") not recognized. ")
#     std = np.std(values, axis=0) if show_std else np.zeros(len(est))  # standard deviation (if need to be provided)
#
#     max_scale = np.max(est)
#
#     ts = TreeStyle()
#     ts.show_leaf_name = show_leaf_name
#     if scale is not None:
#         ts.scale = scale
#
#     es = max(data.groupby("node_name").size()) if each_size is None else each_size
#
#     K = max(max(data.obs) + 1, 0 if colors is None else len(colors))
#     binary = (K <= 2)
#
#     if colors is None or es > 1:  # todo
#         if binary:
#             RGBs = rgb_scale(es)
#             colors = rgb_scale(1)
#         else:
#             raise ValueError("No colors provided to plot the data. ")
#     else:
#         RGBs = [to_hex(c) for c in colors]
#
#     if metric not in ["njumps", "div", "p-value"]:
#         raise ValueError("metric not recognized. Should be one of [\"njumps\", \"div\", \"p-value\"].")
#     # if jump_rate is None:
#     #     if not (metric == "njumps" and vs_prior):
#     #         raise ValueError("jump_rate is required unless (metric = \"njumps\" and vs_prior = True) or " +
#     #                          "metric == \"p-value\". ")
#     if metric == "div" and vs_prior:
#         if emp_div_distr is None:
#             emp_div_distr = evals.div_emp_distr(div_name, tree=tr, jump_rate=jump_rate, n_rep=1000, n_iter=1)
#
#     res = np.zeros((tr.nb, 5))  # n.dist, est, std (if applicable, else 0.), data_l1, p_value
#
#     null = (np.load(null_div_distr) if type(null_div_distr) == str else null_div_distr) if vs_null else None
#
#     def get_p_value(node, node_i, _v, _values):  # v_ <= est[i]; values_ <= values[:, i]
#         jr = jump_rate if unit_length else node.dist * jump_rate
#
#         if metric == "njumps":
#             p = 1. - poisson.cdf(_v, jr)
#         elif metric == "div":
#             emp_distr = null[:, node_i] if vs_null else emp_div_distr
#             if overlap:
#                 p = evals.div_overlap(_values, jr, emp_distr, vs_emp=vs_null)
#             elif mann_test:
#                 p = evals.div_overlap(_values, jr, emp_distr, vs_emp=vs_null, mann_test=True)
#             else:
#                 p = evals.div_p_value(_v, jr, emp_distr, vs_emp=vs_null)
#         else:  # metric == "p-value"
#             raise ValueError("no need to calculate p-value when metric == \"p-value\". ")
#
#         return p
#
#     if vs_node is not None:
#         if np.issubdtype(type(vs_node), np.integer):
#             n = tr.root.get_node_by_i(vs_node+1)  # true_jp_id does not include the root and starts with 0
#             nid = vs_node  # true_jp_id
#         else:
#             n = tr.nodes[vs_node] if type(vs_node) == str else vs_node
#             nid = tr.root.get_node_i_by_name(n.name) - 1
#
#         criteria = get_p_value(n, nid, est[nid], values[:, nid]) if (vs_prior or vs_null) else est[nid]
#         if vs_prior or metric == "p-value":  # i.e. in p-value cases
#             # max_scale = .5 if 1. - criteria < .5 else (1. - criteria)  # ======================================== todo
#             max_scale = criteria
#         print("criteria = {}".format(criteria))
#
#     i = 0
#     for n in tr.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             obs = data.obs[data.node_name == n.name]
#             # print(n.name)
#             if binary:
#                 count = sum(obs)
#                 nstyle["fgcolor"] = RGBs[count]
#                 if es == 1:
#                     d = TextFace(" ")
#                     d.background.color = RGBs[count]
#                 else:
#                     d = TextFace(" " + "■" * count + "□" * (es - count))
#                     # for i in range(es):
#                     #     d = TextFace(" ")
#                     #     d.background.color = colors[1] if i < count else colors[0]
#                     #     n.add_face(d, column=i, position="aligned")
#             else:
#                 nstyle['fgcolor'] = RGBs[int(obs)] if es == 1 else to_hex("black")
#                 # nstyle['fgcolor'] = "black"
#                 if es == 1:
#                     d = TextFace(" ")
#                     d.background.color = colors[int(obs)]
#                 else:
#                     counts = obs.groupby(obs).size()
#                     for idx in np.delete(np.arange(K), counts.index):
#                         counts[idx] = 0
#                     d = TextFace("".join([str(labels[k]) * counts[k] for k in range(K)]))
#
#             n.add_face(d, column=0, position="aligned")
#             # nstyle['size'] = special_node_size
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             v, s = est[i], std[i]
#
#             # msg: the msg at the top
#             if show_value:
#                 msg = ("{:3d}: ".format(i+1) + metric + ("/l" if unit_length else "") +
#                        " = {:6.3f}".format(v) + (" (±{:6.3f})".format(s) if show_std else ""))
#                 n.add_face(TextFace(msg), column=0, position="branch-top")
#
#             # show_p: whether p_value will be showed
#             # l1: data l1 difference at the given node
#             p_value, l1 = 1., 0.
#
#             if vs_prior or vs_null:
#                 p_value = get_p_value(n, i, v, values[:, i])
#
#             # mark: whether to mark this branch red (with different shades) and thicken it
#             # col: the mark color, 0. = black, the larger the more red
#             mark, col = False, signif_rgb(0.)
#
#             if criteria is not None:
#                 if vs_prior or vs_null:
#                     mark = (p_value <= criteria)
#                     if mark:
#                         # col = pvalue2rgb(1.-p_value, MAX=2*max_scale)  # ===================================== todo
#                         col = signif_rgb(2 * criteria - p_value, MAX=2 * criteria)
#                 else:
#                     if metric == "p-value":
#                         mark = (v <= criteria)  # v is a statistic of p-values
#                     else:
#                         mark = (v >= criteria)
#                     if mark:
#                         if metric == "p-value":
#                             # col = measure2rgb(1.-v, MAX=2*max_scale)  # ====================================== todo
#                             col = signif_rgb(2 * criteria - v, MAX=2 * criteria)
#                         else:
#                             col = signif_rgb(v + max_scale, MAX=2 * max_scale)
#
#             if sig_branch:
#                 nstyle['hz_line_color'] = col
#                 nstyle['hz_line_width'] = special_hzline_width if mark else 1
#             elif sig_node:
#                 nstyle['size'] = special_node_size if mark else 1
#                 nstyle['fgcolor'] = col
#
#             if show_emp_L1:
#                 tr1 = tr.deep_copy()
#                 tr1.jps = {tr.root.name: 1, n.name: 1}
#                 _, l1 = evals.data_L1(data, tr1)
#                 L = TextFace(" L1(data) = {:.03f}".format(l1))
#                 n.add_face(L, column=0, position="branch-bottom")
#
#             if show_p:
#                 P = TextFace(" p-value = {:.04f}".format(p_value))
#                 n.add_face(P, column=0, position="branch-bottom")
#
#             if n.njump > 0 and show_jumps:
#                 nstyle['fgcolor'] = "goldenrod"
#                 nstyle['size'] = special_node_size
#
#             n.set_style(nstyle)
#
#             # n.dist, est, std (if applicable, else 0.), data_l1, p_value
#             res[i] = [n.dist, v, s, l1, p_value]
#
#             i += 1
#
#     if file is None:
#         tr.root.show(tree_style=ts)
#     else:
#         tr.root.render(file, tree_style=ts)
#
#     return res


# def plot_analysis(tree, data=None, val=None, std=None, p_value=None, l1=None,
#                   criteria=None, direction="less",
#                   show_jumps=True, show_leaf_name=False,
#                   file=None, colors=None, labels=None,
#                   special_node_size=5, special_hzline_width=3, scale=None):
#     tr = tree.deep_copy()
#
#     ts = TreeStyle()
#     ts.show_leaf_name = show_leaf_name
#     if scale is not None:
#         ts.scale = scale
#
#     if data is not None:
#         es = max(data.groupby("node_name").size())
#
#         K = max(max(data.obs) + 1, 0 if colors is None else len(colors))
#         binary = (K <= 2)
#
#         if colors is None or es > 1:  # todo
#             if binary:
#                 RGBs = rgb_scale(es)
#                 colors = rgb_scale(1)
#             else:
#                 raise ValueError("No colors provided to plot the data. ")
#         else:
#             RGBs = [to_hex(c) for c in colors]
#     else:
#         RGBs = binary = es = None
#
#     i = 0
#     for n in tr.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             if data is not None:
#                 obs = data.obs[data.node_name == n.name]
#                 if binary:
#                     count = sum(obs)
#                     nstyle["fgcolor"] = RGBs[count]
#                     if es == 1:
#                         d = TextFace(" ")
#                         d.background.color = RGBs[count]
#                     else:
#                         d = TextFace(" " + "■" * count + "□" * (es - count))
#                         # for i in range(es):
#                         #     d = TextFace(" ")
#                         #     d.background.color = colors[1] if i < count else colors[0]
#                         #     n.add_face(d, column=i, position="aligned")
#                 else:
#                     nstyle['fgcolor'] = RGBs[int(obs)] if es == 1 else to_hex("black")
#                     # nstyle['fgcolor'] = "black"
#                     if es == 1:
#                         d = TextFace(" ")
#                         d.background.color = colors[int(obs)]
#                     else:
#                         counts = obs.groupby(obs).size()
#                         for idx in np.delete(np.arange(K), counts.index):
#                             counts[idx] = 0
#                         d = TextFace("".join([str(labels[k]) * counts[k] for k in range(K)]))
#
#                 n.add_face(d, column=0, position="aligned")
#             # nstyle['size'] = special_node_size
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             v, s = est[i], std[i]
#
#             # msg: the msg at the top
#             if show_value:
#                 msg = ("{:3d}: ".format(i+1) + metric + ("/l" if unit_length else "") +
#                        " = {:6.3f}".format(v) + (" (±{:6.3f})".format(s) if show_std else ""))
#                 n.add_face(TextFace(msg), column=0, position="branch-top")
#
#             # show_p: whether p_value will be showed
#             # l1: data l1 difference at the given node
#             p_value, l1 = 1., 0.
#
#             if vs_prior or vs_null:
#                 p_value = get_p_value(n, i, v, values[:, i])
#
#             # mark: whether to mark this branch red (with different shades) and thicken it
#             # col: the mark color, 0. = black, the larger the more red
#             mark, col = False, measure2rgb(0.)
#
#             if criteria is not None:
#                 if vs_prior or vs_null:
#                     mark = (p_value <= criteria)
#                     if mark:
#                         # col = measure2rgb(1.-p_value, MAX=2*max_scale)  # ==================================== todo
#                         col = measure2rgb(2*criteria-p_value, MAX=2*criteria)
#                 else:
#                     if metric == "p-value":
#                         mark = (v <= criteria)  # v is a statistic of p-values
#                     else:
#                         mark = (v >= criteria)
#                     if mark:
#                         if metric == "p-value":
#                             # col = measure2rgb(1.-v, MAX=2*max_scale)  # ====================================== todo
#                             col = measure2rgb(2*criteria-v, MAX=2*criteria)
#                         else:
#                             col = measure2rgb(v+max_scale, MAX=2*max_scale)
#
#             if sig_branch:
#                 nstyle['hz_line_color'] = col
#                 nstyle['hz_line_width'] = special_hzline_width if mark else 1
#             elif sig_node:
#                 nstyle['size'] = special_node_size if mark else 1
#                 nstyle['fgcolor'] = col
#
#             if show_emp_L1:
#                 tr1 = tr.deep_copy()
#                 tr1.jps = {tr.root.name: 1, n.name: 1}
#                 _, l1 = evals.data_L1(data, tr1)
#                 L = TextFace(" L1(data) = {:.03f}".format(l1))
#                 n.add_face(L, column=0, position="branch-bottom")
#
#             if show_p:
#                 P = TextFace(" p-value = {:.04f}".format(p_value))
#                 n.add_face(P, column=0, position="branch-bottom")
#
#             if n.njump > 0 and show_jumps:
#                 nstyle['fgcolor'] = "goldenrod"
#                 nstyle['size'] = special_node_size
#
#             n.set_style(nstyle)
#
#             # n.dist, est, std (if applicable, else 0.), data_l1, p_value
#             res[i] = [n.dist, v, s, l1, p_value]
#
#             i += 1
#
#     if file is None:
#         tr.root.show(tree_style=ts)
#     else:
#         tr.root.render(file, tree_style=ts)


# def labelled_tree(tree, values, data, es=None, file=None, label_name=None, show_jump=True,
#                   show_emp_L1=True, MAX=None, MIN=None, direction="greater", integer=False,
#                   colors=None, labels=None, special_node_size=5, special_hzline_width=3):
#     tr = tree.deep_copy()
#
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if label_name is None:
#         label_name = ""
#     elif label_name[-1] != "=":
#         label_name += "="
#
#     if es is None:
#         es = max(data.groupby("node_name").size())
#
#     if MAX is None:
#         MAX = np.max(values)
#     if MIN is None:
#         MIN = np.min(values)
#
#     K = max(max(data.obs) + 1, 0 if colors is None else len(colors))
#     binary = (K <= 2)
#
#     if colors is None or es > 1:
#         if binary:
#             RGBs = rgb_scale(es)
#         else:
#             raise ValueError("No colors provided to plot the data. ")
#     else:
#         RGBs = [to_hex(c) for c in colors]
#
#     i = 0
#     for n in tr.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             obs = data.obs[data.node_name == n.name]
#             if binary:
#                 count = sum(obs)
#                 nstyle["fgcolor"] = RGBs[count]
#                 if es == 1:
#                     d = TextFace(" ")
#                     d.background.color = RGBs[count]
#                 else:
#                     d = TextFace(" " + "■" * count + "□" * (es - count))
#             else:
#                 nstyle['fgcolor'] = RGBs[int(obs)] if es == 1 else to_hex("black")
#                 counts = obs.groupby(obs).size()
#                 for idx in np.delete(np.arange(K), counts.index):
#                     counts[idx] = 0
#                 d = TextFace("".join([str(labels[k]) * counts[k] for k in range(K)]))
#
#             n.add_face(d, column=0, position="aligned")
#             nstyle['size'] = special_node_size
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             v = values[i]
#             if MIN <= v <= MAX:
#                 if direction == "greater" or direction == "g":
#                     nstyle['hz_line_color'] = signif_rgb(v, MAX=MAX, MIN=MIN)
#                 elif direction == "lesser" or direction == "l":
#                     nstyle['hz_line_color'] = signif_rgb(MAX - v, MAX=MAX - MIN, MIN=0)
#                 else:
#                     raise ValueError("direction not recognized. ")
#
#                 nstyle['hz_line_width'] = special_hzline_width
#             else:
#                 nstyle['hz_line_width'] = 1
#
#             str_value = "{:.03f}".format(values[i]) if not integer else "{:d}".format(values[i])
#             d = TextFace("{:2d}: ".format(i+1) + label_name + str_value)
#             n.add_face(d, column=0, position="branch-top")
#
#             if show_emp_L1:
#                 tr1 = tr.deep_copy()
#                 tr1.jps = {"Root": 1, n.name: 1}
#                 _, l1 = evals.data_L1(data, tr1)
#                 L = TextFace("L1(data)={:.03f}".format(l1))
#                 n.add_face(L, column=0, position="branch-bottom")
#
#             # if div_name is not None:
#             #     p = TextFace(" p={:.04f}".format(p_value))
#             #     n.add_face(p, column=0, position="branch-bottom")
#
#             # ret[i - 1] = [i, n.dist, div, l1, p_value]
#             i += 1
#
#             if n.njump > 0 and show_jump:
#                 nstyle['fgcolor'] = "goldenrod"
#                 nstyle['size'] = 5
#
#         n.set_style(nstyle)
#
#     if file is not None:
#         tr.root.render(file, tree_style=ts)


# def sig_jps(tree0, jps, data, criteria, file, show_std=False, jump_rate=None, es=None, vs_prior=True):
#     # tree, p_jps, data, criteria, file, es = tree, [1.]*(tree.nb+1), data, 2., files['jb']+"data.pdf", 1
#
#     tree = tree0.deep_copy()
#     p_jps = np.mean(jps, axis=0)
#     std_jps = np.std(jps, axis=0) if show_std else None
#
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if es is None:
#         es = max(data.groupby("node_name").size())
#     RGBs = rgb_scale(es)
#
#     if vs_prior:
#         if jump_rate is None:
#             raise ValueError("jump_rate should be provided when compare with the prior. ")
#
#     ret = np.zeros((tree.nb, 5))  # id, dist, p_div, l1, p_value
#
#     i = 1
#     for n in tree.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             count = sum(data.obs[data.node_name == n.name])
#             nstyle["fgcolor"] = RGBs[count]
#             nstyle["size"] = 5
#             d = TextFace("■" * count + "□" * (es - count))
#             n.add_face(d, column=0, position="aligned")
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         max_scale = np.max(p_jps)
#         if not n.is_root():
#             jp = p_jps[i]
#             if not vs_prior:
#                 p_value = show = (jp >= criteria)
#                 col = measure2rgb(jp if show else 0, MAX=max_scale)
#             else:
#                 p_value = 1. - poisson.cdf(jp, n.dist * jump_rate)
#                 show = p_value <= criteria
#                 col = measure2rgb(p_value if show else 0, MAX=1.)
#
#             nstyle['hz_line_color'] = col
#             nstyle['hz_line_width'] = 3 if show else 1
#
#             tree1 = tree.deep_copy()
#             tree1.jps = {"Root": 1, n.name: 1}
#             _, l1 = evals.data_L1(data, tree1)
#
#             d = TextFace("{:2d}: jp={:.03f}".format(i, jp) +
#                          ("" if not show_std else " ({:.03f})".format(std_jps[i])))
#             L = TextFace(" l1={:.03f}".format(l1))
#             n.add_face(d, column=0, position="branch-top")
#             n.add_face(L, column=0, position="branch-bottom")
#             if vs_prior:
#                 p = TextFace(" p={:.04f}".format(p_value))
#                 n.add_face(p, column=0, position="branch-bottom")
#
#             ret[i - 1] = [i, n.dist, jp, l1, p_value]
#             i += 1
#
#             if n.njump > 0:
#                 nstyle['fgcolor'] = "blue"
#                 nstyle['size'] = 5
#
#         n.set_style(nstyle)
#
#     nstyle = NodeStyle()
#     nstyle['fgcolor'] = "blue"
#     nstyle['size'] = 5
#
#     # tree.root.show(tree_style=ts)
#     tree.root.render(file, tree_style=ts)
#
#     return ret
#
#
# def sig_mean_divs(tree0, divs, jump_rate, data, criteria, div_name, file, show_std=False,
#                   emp_distr=None, ds=None, es=None):
#     # trajectory=True, jps=None, file_trj=None):
#
#     tree = tree0.deep_copy()
#     p_divs = np.mean(divs, axis=0)
#     std_divs = np.zeros(len(p_divs)) if not show_std else np.std(divs, axis=0)
#
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if ds is None:
#         ds = tree.disc
#     if es is None:
#         es = max(data.groupby("node_name").size())
#     if emp_distr is None:
#         emp_distr = evals.div_emp_distr(discount=ds, max_jps=tree.tl * jump_rate / tree.nb,
#                                         div_name=div_name)
#
#     RGBs = rgb_scale(es)
#
#     ret = np.zeros((tree.nb, 5))  # id, dist, p_div, l1, p_value
#
#     i = 1
#     ids = []
#     p_values = []
#     for n in tree.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             count = sum(data.obs[data.node_name == n.name])
#             nstyle["fgcolor"] = RGBs[count]
#             nstyle["size"] = 5
#             d = TextFace("■" * count + "□" * (es - count))
#             n.add_face(d, column=0, position="aligned")
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             div = p_divs[i-1]  # posterior measure
#             p_value = evals.div_p_value(div, n.dist * jump_rate, emp_distr)
#
#             nstyle['hz_line_color'] = measure2rgb(div if p_value <= criteria else 0)
#             nstyle['hz_line_width'] = 3 if p_value <= criteria else 1
#
#             ids += [i-1]
#             p_values += [p_value]
#
#             tree1 = tree.deep_copy()
#             tree1.jps = {"Root": 1, n.name: 1}
#             _, l1 = evals.data_L1(data, tree1)
#
#             d = TextFace("{:2d}: div={:.03f}".format(i, div) +
#                          ("" if not show_std else "({:.03f})".format(std_divs[i-1])))
#             L = TextFace(" l1={:.03f}".format(l1))
#             n.add_face(d, column=0, position="branch-top")
#             n.add_face(L, column=0, position="branch-bottom")
#             if div_name is not None:
#                 p = TextFace(" p={:.04f}".format(p_value))
#                 n.add_face(p, column=0, position="branch-bottom")
#
#             ret[i - 1] = [i, n.dist, div, l1, p_value]
#             i += 1
#
#             if n.njump > 0:
#                 nstyle['fgcolor'] = "blue"
#                 nstyle['size'] = 5
#
#         n.set_style(nstyle)
#
#     tree.root.render(file, tree_style=ts)
#     return ret
#
#
# def sig_overlap_divs(tree0, divs, jump_rate, data, criteria, div_name, file,
#                      emp_distr=None, ds=None, es=None):
#     # trajectory=True, jps=None, file_trj=None):
#
#     tree = tree0.deep_copy()
#
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if ds is None:
#         ds = tree.disc
#     if es is None:
#         es = max(data.groupby("node_name").size())
#     if emp_distr is None:
#         emp_distr = evals.div_emp_distr(discount=ds, max_jps=tree.tl * jump_rate / tree.nb,
#                                         div_name=div_name)
#
#     RGBs = rgb_scale(es)
#
#     ret = np.zeros((tree.nb, 5))  # id, dist, p_div, l1, p_value
#
#     i = 1
#     ids = []
#     p_values = []
#     for n in tree.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             count = sum(data.obs[data.node_name == n.name])
#             nstyle["fgcolor"] = RGBs[count]
#             nstyle["size"] = 5
#             d = TextFace("■" * count + "□" * (es - count))
#             n.add_face(d, column=0, position="aligned")
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             div1 = divs[:, i-1]  # posterior measure
#             p_value = evals.div_overlap(div1, n.dist * jump_rate, emp_distr)
#
#             nstyle['hz_line_color'] = measure2rgb(1 - p_value if p_value <= criteria else 0)
#             nstyle['hz_line_width'] = 3 if p_value <= criteria else 1
#
#             ids += [i-1]
#             p_values += [p_value]
#
#             tree1 = tree.deep_copy()
#             tree1.jps = {"Root": 1, n.name: 1}
#             _, l1 = evals.data_L1(data, tree1)
#
#             d = TextFace("{:2d}: div={:.03f}".format(i, np.mean(div1)))
#             L = TextFace(" l1={:.03f}".format(l1))
#             n.add_face(d, column=0, position="branch-top")
#             n.add_face(L, column=0, position="branch-bottom")
#             if div_name is not None:
#                 p = TextFace(" p={:.04f}".format(p_value))
#                 n.add_face(p, column=0, position="branch-bottom")
#
#             ret[i - 1] = [i, n.dist, np.mean(div1), l1, p_value]
#             i += 1
#
#             if n.njump > 0:
#                 nstyle['fgcolor'] = "blue"
#                 nstyle['size'] = 5
#
#         n.set_style(nstyle)
#
#     nstyle = NodeStyle()
#     nstyle['fgcolor'] = "blue"
#     nstyle['size'] = 5
#
#     tree.root.render(file, tree_style=ts)
#     # tree.root.show(tree_style=ts)
#     # if trajectory:
#     #     if jps is None or file_trj is None:
#     #         raise ValueError("Both jps and file_trj are needed to plot trajectory. ")
#     #     for nid, p_value in zip(ids, p_values):
#     #         # print(nid, p_value)
#     #         jps1 = [0] * (tree.nb + 1)
#     #         jps1[0], jps1[nid + 1] = 1, 1
#     #         tree.jps = jps1
#     #         _, l1 = data_L1(data, tree)
#     #
#     #         jps_trajectory(jps[:, nid + 1], file=file_trj + "_trj_{}.png".format(nid),
#     #                        title="trajectory of njumps at {}: ".format(nid) +
#     #                              "div={:.03f}, p-value={:.03f}, L1={:.03f}".format(pms[nid], p_value, l1))
#     return ret
#

# for ds in [.1, .5, .7, .9, .95, .99, .999]:
#     # ds = .999
#     emp_distr = plots.div_emp_distr(div_name, discount=ds, tree=tree, jump_rate=jump_rate)
#     # emp_distr = div_emp_distr(div_name, tree=tree, jump_rate=jump_rate)
#     print(emp_distr.shape)
#     plots.prior_hist(emp_distr, njp=njp, tree=tree, div_name="L1", plot=True,
#                      plot_title="prior of L1 divergence with discount {:.03f}".format(ds))


# tree.disc

# def sig_divs_overlap(tree, divs, jump_rate, data, criteria, div_name, ds, file, es=None):
#
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if es is None:
#         es = data.shape[0] // len(tree.leaves)
#     RGBs = rgb_scale(es)
#
#     i = 1
#     # ids = []
#     # p_values = []
#
#     div1jp = pd.read_csv(os.getcwd() + "/runs/" + "priDivs1jp.csv")
#     d0 = div1jp[div_name][div1jp.discount == ds]
#
#     for n in tree.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             count = sum(data.obs[data.node_name == n.name])
#             nstyle["fgcolor"] = RGBs[count]
#             nstyle["size"] = 5
#             d = TextFace("■" * count + "□" * (es - count))
#             n.add_face(d, column=0, position="aligned")
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             # print(i)
#             pms = divs[:, i-1]  # posterior measure
#             jp_rt = n.dist * jump_rate
#             p_value = div_overlap(pms, jp_rt, d=d0)
#             nstyle['hz_line_color'] = div2rgb(1.-p_value, MAX=1.)
#             nstyle['hz_line_width'] = 3 if p_value <= criteria else 1
#
#             # ids += [i-1]
#             # p_values += [p_value]
#
#             tree1 = tree.deep_copy()
#             tree1.jps = {"Root": 1, n.name: 1}
#             _, l1 = data_L1(data, tree1)
#
#             d = TextFace("{:2d}: p={:.03f}".format(i, p_value))
#             L = TextFace(" l1={:.03f}".format(l1))
#             n.add_face(d, column=0, position="branch-top")
#             n.add_face(L, column=0, position="branch-bottom")
#             # if div_name is not None:
#             #     p = TextFace(" p={:.03f}".format(p_value))
#             #     n.add_face(p, column=0, position="branch-bottom")
#             i += 1
#
#             if n.njump > 0:
#                 nstyle['fgcolor'] = "blue"
#                 nstyle['size'] = 5
#
#         n.set_style(nstyle)
#
#     nstyle = NodeStyle()
#     nstyle['fgcolor'] = "blue"
#     nstyle['size'] = 5
#
#     # tree.root.show(tree_style=ts)
#     tree.root.render(file, tree_style=ts)


# def all_div(tree, p_divs, data, file, es=None):
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if es is None:
#         es = data.shape[0] // len(tree.leaves)
#     RGBs = rgb_scale(es)
#
#     i = 0
#     for n in tree.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             count = sum(data.obs[data.node_name == n.name])
#             nstyle["fgcolor"] = RGBs[count]
#             nstyle["size"] = 5
#             d = TextFace("■" * count + "□" * (es - count))
#             n.add_face(d, column=0, position="aligned")
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#             nstyle['hz_line_color'] = div2rgb(p_divs[i])
#             if p_divs[i] > 0:
#                 nstyle['hz_line_width'] = 3
#                 d = TextFace("{:.03f}".format(p_divs[i]))
#                 n.add_face(d, column=0, position="branch-top")
#             i += 1
#
#             if n.njump > 0:
#                 nstyle['fgcolor'] = "blue"
#                 nstyle['size'] = 5
#
#         n.set_style(nstyle)
#
#     nstyle = NodeStyle()
#     nstyle['fgcolor'] = "blue"
#     nstyle['size'] = 5
#
#     # tree.root.show(tree_style=ts)
#     tree.root.render(file, tree_style=ts, units='mm', w=1000)


# def plot_data(tr, data, file=None, es=None,
#               show_node_id=False, show_node_name=False, show_l1=False, labels=None, colors=None, bgcolor=False,
#               special_node_size=5):
#
#     # tree, p_jps, data, criteria, file, es = tree, [1.]*(tree.nb+1), data, 2., files['jb']+"data.pdf", 1
#     tree = tr.deep_copy()
#     ts = TreeStyle()
#     ts.show_leaf_name = True
#
#     if es is None:  # each_size: (max) number of obs on each leaf
#         es = max(data.groupby("node_name").size())
#
#     K = max(data.obs) + 1
#     binary = (K == 2)
#
#     if colors is None:
#         if binary:
#             RGBs = rgb_scale(es)
#         else:
#             raise ValueError("No colors provided to plot the data. ")
#     else:
#         RGBs = [to_hex(c) for c in colors]
#
#     i = 1
#     for n in tree.root.traverse('preorder'):
#         nstyle = NodeStyle()
#
#         if n.is_leaf():
#             obs = data.obs[data.node_name == n.name]
#             if binary:
#                 count = sum(obs)
#                 nstyle["fgcolor"] = RGBs[count]
#                 d = TextFace("■" * count + "□" * (es - count))
#             else:
#                 nstyle['fgcolor'] = RGBs[int(obs)] if es == 1 else to_hex("black")
#                 counts = obs.groupby(obs).size()
#                 # # counts.append(pd.DataFrame(0, index=np.delete(np.arange(K), counts.index)))
#                 # # counts[np.delete(np.arange(K), counts.index)] = 0
#                 for idx in np.delete(np.arange(K), counts.index):
#                     counts[idx] = 0
#                 d = TextFace("".join([labels[k] * counts[k] for k in range(K)]))
#
#             n.add_face(d, column=0, position="aligned")
#             nstyle["size"] = special_node_size
#
#         else:
#             nstyle['fgcolor'] = "black"
#
#         if not n.is_root():
#
#             if not n.is_leaf():
#                 if show_node_name or show_node_id:
#                     names = (["{}".format(i) if show_node_id else []]) + ([n.name] if show_node_name else [])
#                     d = TextFace(" " + ": ".join(names))
#                     n.add_face(d, column=0, position="branch-top")
#                 if show_l1:
#                     tree1 = tree.deep_copy()
#                     tree1.jps = {"Root": 1, n.name: 1}
#                     _, l1 = evals.data_L1(data, tree1)
#                     d = TextFace(" l1={:.03f}".format(l1))
#                     n.add_face(d, column=0, position="branch-bottom")
#
#             if n.njump > 0:
#                 nstyle['fgcolor'] = "goldenrod"
#                 nstyle['size'] = special_node_size
#                 print("Target:", i)
#
#             i += 1
#
#         n.set_style(nstyle)
#
#     # nstyle = NodeStyle()
#     # nstyle['fgcolor'] = "blue"
#     # nstyle['size'] = 5
#
#     if file is None:
#         tree.root.show(tree_style=ts)
#     else:
#         tree.root.render(file, tree_style=ts)
#
#
#
# def show_proposal(info, jps, branches=None, file=None, switch_to=False, njps=True):
#     new_jump_sampled = info['sampled'][1::2, 0][1:]
#     switch_from_sampled = info['sampled'][0::2, 0]
#     switch_to_sampled = info['sampled'][0::2, 1]
#     proposed = info['proposed']
#
#     msk_proposed = proposed.copy()
#     msk_proposed[:, 0] = 0
#     accepted_jps = np.vstack(np.where(msk_proposed > 0))
#
#     plt.axhline(0)
#     plt.ylim(top=jps.shape[1]+1)
#     if branches is None:
#         plt.scatter([2 * i for i in range(new_jump_sampled.shape[0])],
#                     new_jump_sampled, s=1, alpha=0.5, label="proposed new jump")
#         plt.scatter([2 * i + 1 for i in range(switch_from_sampled.shape[0])],
#                     switch_from_sampled, s=10, label="selected to switch")
#         if switch_to:
#             plt.scatter([2 * i + 1 for i in range(switch_to_sampled.shape[0])],
#                         switch_to_sampled, s=5, alpha=0.5, label="switch to")
#
#         plt.scatter(accepted_jps[0], accepted_jps[1], s=1, alpha=0.5, label="jump locations")
#     else:
#         new_jump_sampled_id = np.array([i for i in range(new_jump_sampled.shape[0])
#                                         if new_jump_sampled[i] in branches])
#         switch_from_sampled_id = np.array([i for i in range(switch_from_sampled.shape[0])
#                                            if switch_from_sampled[i] in branches])
#         plt.scatter([2 * i for i in new_jump_sampled_id],
#                     new_jump_sampled[new_jump_sampled_id], s=1, alpha=0.5, label="proposed new jump")
#         plt.scatter([2 * i + 1 for i in switch_from_sampled_id],
#                     switch_from_sampled[switch_from_sampled_id], s=10, label="selected to switch")
#         if switch_to:
#             switch_to_sampled_id = np.array([i for i in range(switch_to_sampled.shape[0])
#                                              if switch_to_sampled[i] in branches])
#             plt.scatter([2 * i + 1 for i in switch_to_sampled_id],
#                         switch_to_sampled[switch_to_sampled_id], s=5, alpha=0.5, label="switch to")
#
#         accepted_jps_id = np.array([i for i in range(accepted_jps.shape[1]) if accepted_jps[1, i] in branches])
#         plt.scatter(accepted_jps[0, accepted_jps_id], accepted_jps[1, accepted_jps_id],
#                     s=1, alpha=0.5, label="jump locations")
#     if njps:
#         plt.plot(-(np.sum(jps, axis=1)), label="total #jumps", alpha=0.5)
#     # plt.scatter(accepted_jps[0], accepted_jps[1], s=1, alpha=0.3, label="jump locations")
#     plt.legend(loc='lower right')
#     plt.title("Proposed jumps / switches")
#     if file is None:
#         plt.show()
#     else:
#         plt.savefig(file, format='pdf')
#     plt.close()

