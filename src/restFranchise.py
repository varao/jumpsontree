"""

PhyloHPYP: restFranchise.py

Created on 4/27/18 10:51 PM

@author: Hanxi Sun

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
from src.rest import Rest
from src.restNode import RestNode
from src.particles import Particles
from src.categorical import Categorical
# from scipy.special import gammaln
from src.utils import get_log, now
from scipy.stats import poisson, chisquare
from tqdm import tqdm


class RestFranchise:
    """
    The Chinese Restaurant Franchise
    """
    def __init__(self, newick=None, newick_format=1, disc=0.5, conc=0,
                 depend=True, base: Categorical = None, **kwargs):
        """
        Initialize the tree
        :param newick: inherited from ete3.Tree
        :param newick_format: inherited from ete3.Tree
        :param disc: the discount parameter
        :param conc: the concentration parameter
        :param depend: whether there is dependency between parent and child
        :param base: a base measure (only supports Categorical for now)
        :param kwargs: arguments for instantiate a Categorical object as the base
        """
        # the tree (root node)
        self._root = RestNode(newick, newick_format)
        # self._root.set_ids()
        self._root.njump = self._root.ROOT_JUMP

        # parameters
        self._disc, self._conc, self._depend = self.check_parameters(disc, conc, depend)
        self._base = None
        if base is not None:
            self.base = base
        elif len(kwargs) > 0:
            self.base = Categorical(**kwargs)

        # total branch length, total number of branches, each branch length, #jumps on each branch
        self._tl, self._nb, self._bls, self._jps = self._root.all_saved_subtree_properties()  # recorded in "preorder"

        # observed nodes
        self._root.init_observed_nodes()
        self._observed = self._root.get_observed()

        # restaurants
        self.init_rests()

        # name internal nodes
        if newick_format != 1:
            idx = 0
            for n in self.root.traverse("preorder"):
                if n.name is None:
                    n.name = "_node_{}".format(idx)
                    idx += 1

        self.set_id()

    #####################
    #    properties     #
    #####################
    def _get_root(self):
        return self._root

    def _get_disc(self):
        return self._disc

    def _set_disc(self, value):
        d, _ = Rest.check_parameters(value, self.conc)
        self._disc = d
        self.init_rests()

    def _get_conc(self):
        return self._conc

    def _set_conc(self, value):
        _, c = Rest.check_parameters(self.disc, value)
        self._conc = c
        self.init_rests()

    def _get_depend(self):
        return self._depend

    def _set_depend(self, value):
        self._depend = self.check_depend(depend=value, conc=self.conc)

    def _get_base(self):
        return self._base

    def _set_base(self, value):
        if value is None:
            self._base = None
        elif type(value) != Categorical:
            raise TypeError("Base measure should be of type Categorical. ")
        else:
            value.normalize()
            self._base = value
            self.init_rests()

    def _get_total_length(self):
        return self._tl

    def _get_num_branches(self):
        return self._nb

    def _get_branch_lengths(self):
        return self._bls

    def _get_jps(self):
        return self._jps

    def _set_jps(self, value):
        self.root.jps = value
        self._jps = self.root.jps
        self.init_rests()

    def _get_observed(self):
        return self._observed

    def _get_leaves(self):
        return self.root.get_leaves()

    def _get_leaf_names(self):
        return self.root.get_leaf_names()

    def _get_nodes(self):  # todo
        return self.root.nodes

    def _get_nodes_by_id(self):
        return self.root.id_nodes

    def __getitem__(self, item):
        if type(item) == str:
            return self.nodes[item]
        else:
            return self.id_nodes[item]

    @property
    def nleaf(self):
        return self.root.nleaf

    root = property(fget=_get_root)  # the tree (root)
    disc = property(fget=_get_disc, fset=_set_disc)  # the discount parameter
    conc = property(fget=_get_conc, fset=_set_conc)  # the concentration parameter
    depend = property(fget=_get_depend, fset=_set_depend)  # whether there is dependency btwn parent & child node
    base = property(fget=_get_base, fset=_set_base)  # the universal base measure
    tl = property(fget=_get_total_length)  # total branch length
    nb = property(fget=_get_num_branches)  # total number of branches
    bls = property(fget=_get_branch_lengths)  # [list] each branch length
    jps = property(fget=_get_jps, fset=_set_jps)  # [list] #jumps on each branch
    observed = property(fget=_get_observed)  # [dict] all observed nodes (name:node)
    leaves = property(fget=_get_leaves)  # all leaves in the tree
    leaf_names = property(fget=_get_leaf_names)  # all leaves in the tree
    nodes = property(fget=_get_nodes)  # [dict] all nodes in the tree
    id_nodes = property(fget=_get_nodes_by_id)  # [dict] all nodes in the tree arranged by its id

    def divs(self, div_name):
        return self.root.divs(div_name)

    #####################
    #     utilities     #
    #####################

    def traverse(self, strategy="preorder"):
        return self.root.traverse(strategy=strategy)

    def set_id(self):
        i = 0
        for n in self.traverse():
            n.set_id(i)
            i += 1

    def write(self, *args, format=1, **kwargs):
        newick = self.root.write(*args, format=format, **kwargs)
        return newick.replace(")1:", "):")

    def get_depth(self, dist=False):
        depth = {}
        for n in self.traverse():
            if n.is_root():
                depth[n.name] = 0
            else:
                depth[n.name] = depth[n.up.name] + (1 if not dist else n.dist)
        return depth

    def get_height(self, dist=False):
        height = {}
        for n in self.traverse(strategy="postorder"):
            # print(n.name)
            if n.is_root():
                break
            if n.is_leaf():
                height[n.name] = (0 if not dist else 0.)

            up_height = height[n.name] + (1 if not dist else n.dist)
            height[n.up.name] = min(up_height, height[n.up.name]) if n.up.name in height else up_height
        return height

    def get_nleaves(self):
        return {n.name: n.nleaf for n in self.traverse()}

    @staticmethod
    def check_depend(depend, conc):
        if type(depend) != bool:
            warnings.warn("depend ({}) is forced to be of type bool. ".format(depend))
            depend = bool(depend)
        if conc > 0 and depend:
            raise ValueError("When conc (={})> 0, depend should be False. ".format(conc))
        return depend

    def check_parameters(self, disc, conc, depend=None):
        disc, conc = Rest.check_parameters(disc, conc)
        depend = self.check_depend(depend, conc)
        return disc, conc, depend

    def set_parameters(self, disc, conc, depend=None):
        if depend is None:
            depend = self.depend
        self._disc, self._conc, self._depend = self.check_parameters(disc, conc, depend)
        self.init_rests()

    def uniform_base(self, labels):
        if type(labels) != list:
            raise TypeError("labels should be a list of labels.")
        self._base = Categorical(labels=labels)

    def is_dirichlet(self):
        return (not self.depend) and self.disc == 0

    def update_observed(self):
        self._observed = self.root.get_observed()

    def copy_properties(self, other):
        """
        reset properties with another RestFranchise
        """
        self._disc, self._conc = other.disc, other.conc
        self._tl = other.tl
        self._nb = other.nb
        self._bls = list(other.bls)
        self._jps = list(other.jps)
        self._observed = self.root.get_observed()

    def deep_copy(self):
        """
        perform a deep copy of current restaurant franchise
        """
        new = self.__class__(disc=self.disc, conc=self.conc, depend=self.depend, base=self.base)
        new._root = self.root.deep_copy()
        new.copy_properties(self)
        return new

    def parameters_by_model(self, model="HPY", var=None):
        # the hierarchical PY model with discount = var
        if model == "HPY":
            if var is None and self.conc > 0:
                var = 1 - 1 / (self.conc + 1)
            else:
                var = self.disc
            self.set_parameters(var, 0, True)

        # the hierarchical Dirichlet model with concentration = var
        elif model == "HDP":
            if var is None and self.disc > 0:
                var = 1 / (1 - self.disc) - 1  # keep the same variance
            else:
                var = self.conc
            self.set_parameters(0, var, False)

        # when type is not acceptable
        else:
            raise ValueError("model can either be \"HPY\" for hierarchical Pitman-Yor processes" +
                             " or \"HDP\" for hierarchical Dirichlet processes, (not {}).".format(model))

    def get_node_by_name(self, node_name):
        node = None
        for n in self.root.traverse():
            if n.name == node_name:
                node = n
                break
        if node is None:
            raise ValueError("Unable to find a node with node_name {}".format(node_name))
        return node

    def is_offspring(self, parent_name, child_name):
        parent = self.get_node_by_name(parent_name)
        return parent.is_offspring(child_name)

    def __str__(self):
        return self.root.__str__()

    def str_print(self, tree=False, pruned=False, pruned_rests=False,
                  rests=False, bases=False, jps=False, observed=False):
        ret = [self.root.str_print(tree=True)] if tree else []
        if pruned:
            jns = [n.name for n in self.root.traverse() if n.njump > 0]
            jp_tree, ref_name = self.jps_prune()
            ref = ""
            for k in set(ref_name.values()):
                ref_k = [k1 for k1, v1 in ref_name.items() if v1 == k]
                ref += ("\n  {0}: ".format(k) + ", ".join(ref_k))
            ret.append("Jump nodes: " + ",".join(jns) + "\n" +
                       "Equivalent Jump Tree:" + jp_tree.__str__() + "\n" +
                       "Reference observation nodes: " + ref + '\n')
        if pruned_rests:
            jp_tree, _ = self.jps_prune()
            node_names = [n.name for n in jp_tree.root.traverse('preorder')]
            pruned_rests = ""
            for n in self.root.traverse('preorder'):
                if n.name in node_names:
                    pruned_rests += ('\n  {0} rest = '.format(n.name) + n.rest.__str__())
            ret.append("Restaurants (pruned): " + pruned_rests + '\n')
        if rests or bases or jps:
            ret.append(self.root.str_print(rests=rests, bases=bases, jps=jps))
        if observed:
            ret.append("Observed Nodes: " + '[' + ', '.join(self.observed.keys()) + ']')
        return '\n'.join(ret)

    ##################
    #   restaurants  #
    ##################

    def init_rests(self, node=None):
        """
        Initiate restaurants for all nodes
        """
        if node is None:
            node = self.root

        node.init_rests(self.disc, self.conc, self.base)

    def seat_new_obs(self, node_name, obs=None):
        """
        Update all restaurants with the table configuration introduced by an observation
        The table configuration is sampled according to (6.2) in progress report of PhyloTreePY @ 03/28/18
        :param node_name: the name of species that the new observation comes from
        :param obs: the new observation (observed category)
        :return: None
        """
        node = self.observed[node_name]
        return node.seat_new_obs(obs, self.depend)

    def rests(self, rn=None):
        """
        A pandas.DataFrame object of nodes & rests
        :return:
        """
        if rn is None:
            _, rn = self.jps_prune(init_rests=False)
        labels = list(self.base.keys())
        names = []
        for i in labels:
            names += ["{}_nt".format(i), "{}_nc".format(i)]
        rests = pd.DataFrame(columns=["node_name"] + names)
        idx = 0
        for n in self.root.traverse(strategy='preorder'):
            if n.njump > 0:
                rests.loc[idx] = [n.name] + [0] * (2 * len(labels))
                for i in labels:
                    tbl = n.rest[i]
                    rests["{}_nt".format(i)][idx] = tbl.nt
                    rests["{}_nc".format(i)][idx] = tbl.nc
                idx += 1
        ref_names = {}
        for n in self.root.traverse(strategy='preorder'):
            if n.njump == 0 and (not n.is_root()):
                ref_names[n.up.name] = n.name
                if n.is_observed():
                    ref_names[n.name] = n.name
            else:
                ref_names[n.name] = n.name
                if not n.is_root():
                    ref_names[n.up.name] = n.up.name
        for k, v in ref_names.items():
            if k != v:
                idx = [k]
                while v != ref_names[v]:
                    idx.append(v)
                    v = ref_names[v]
                while ref_names[k] != v:
                    ref_names[k] = v
        rests.node_name = rests.node_name.map(ref_names)
        rests.node_name = rests.node_name.map(rn)
        return rests

    #####################
    #     jump_rate     #
    #####################

    def sample_prior_jump_rate(self, prior_mean_njumps=1.):
        """
        Sample jump_rate from the prior: exp(1/total_length)
        :return: the sample
        """
        return np.random.exponential(prior_mean_njumps / self.tl)

    def sample_post_jump_rate(self, prior_mean_njumps=1.):
        """
        Sample jump_rate from the posterior given jumps: Gamma(1 + total_jumps, 1/total_length + #branches)
        :return: the sample
        """
        return np.random.gamma(sum(self.jps) + 1, 1 / (self.nb + (prior_mean_njumps / self.tl)))

    def inhomogeneous(self, style: str = "delta time"):
        assert style in ["delta time", "number of leaves"]

        if style == "delta time":
            limits = {self.root.name: 0.}
            for n in self.traverse():
                if n.is_root():
                    continue
                limits[n.name] = limits[n.up.name] + n.dist
            nodes = [n for n in self.traverse()]
            nodes.sort(key=lambda node: limits[node.name])
            nodes = {n: i for i, n in enumerate(nodes)}
            points = np.sort(list(limits.values()))
            mid = ((points[1:] + points[:-1]) / 2).reshape(-1, 1)
            intervals = np.array([[limits[n.up.name], limits[n.name]] for n in self.traverse() if not n.is_root()])
            counts = np.sum((intervals[:, 0] < mid) & (intervals[:, 1] > mid), axis=1)
            lengths = points[1:] - points[:-1]
            for n, i in nodes.items():
                if n.is_root():
                    continue
                i0 = nodes[n.up]
                n.dist = np.sum(lengths[i0:i] / counts[i0:i])

        elif style == "number of leaves":
            for n in self.traverse():
                if n.is_root():
                    continue
                n.dist *= n.nleaf

        # update total branch length
        self._tl = self.root.tl

    #####################
    #  jps management   #
    #####################

    def subtree_has_jump(self, node_name):
        node = self.get_node_by_name(node_name)
        return node.subtree_has_jump()

    def update_jps(self, node=None):
        self._jps = self.root.jps
        self.init_rests(node=node)

    def jps_prune(self, init_rests=True):
        """
        Prune the tree according to its jumps (i.e. remove branches with 0 jump)
        and initialize all the restaurants (with discount (disc), concentration (conc) and base)
        Observed nodes will be preserved and the name_map will be returned
        :return: new: a copy of the original tree with branches pruned,
                 obs_name_ref: a dict that maps observed node names to names after merging {old_name:new_name}
        """
        new = self.deep_copy()

        ref_names = new.root.jps_prune(observed=new.observed)
        new.update_observed()
        if init_rests:
            new.init_rests()

        return new, ref_names

    def partition_by_jps(self):
        ref_names = {n.name: n.name for n in self.traverse()}
        new = self.deep_copy()

        for node in new.traverse():
            if node.njump == 0 and (not node.is_root()):
                parent = node.up
                ref_names[parent.name] = node.name
                parent.name = node.name
                parent.children += node.children
                for child in node.children:
                    child.up = parent
                parent.remove_child(node)
        for k, v, in ref_names.items():
            if k != v:
                idx = [k]
                while v != ref_names[v]:
                    idx.append(v)
                    v = ref_names[v]
                while ref_names[k] != v:
                    ref_names[k] = v

        idx_map = {n.name: i for i, n in enumerate(new.traverse())}
        Z = [idx_map[ref_names[n.name]] for n in self.traverse()]
        del new
        return Z, ref_names

    def full_jps(self, njump):
        """
        Assign the same number of jumps to all branches in the tree, except for the root.
        :param njump: number of jumps on each branch
        :return: None
        """
        self.root.full_jps(njump)
        self.update_jps()

    def poisson_jps(self, jump_rate):
        """
        Sample jumps on each branch from the prior: Poisson(jump_rate * branch_length)
        :param jump_rate: the intensity of Poisson
        :return: None
        """
        self.root.poisson_jps(jump_rate)
        self.update_jps()

    def propose_jps(self, jr, switch, _birthDeath=True):
        """
        Propose a new jps (list of #jumps on branches) based on current jps & jump_rate
        and set the jps in current tree to be the proposed new jps
        Proposing by randomly pick a branch and then randomly sample the number of jumps
        from Poisson(jump_rate)
        :param jr: current jump_rate
        :param switch: whether to randomly select a jump and move it upwards/downwards (to
                        break the coupling effect between nearby branches.
        :param _birthDeath:
        :return: new jps, log(prob_ratio), where prob_ratio is in (6.2) of progress
                report of PhyloTreePY @ 04/20/18
        """
        # self = tree
        nodes = self.nodes
        node_ids = dict(zip(nodes.keys(), range(self.nb + 1)))
        if switch:
            ids, log_ratio, if_same = self.switch_jumps(jr, nodes, node_ids)
        elif _birthDeath:
            ids, log_ratio, if_same = self.new_jump_birthDeath(jr, nodes, node_ids)
        else:
            ids, log_ratio, if_same = self.new_jump_prior(jr, nodes, node_ids)
        if not if_same:
            self.update_jps()  # todo make it only on the affected rests
        return ids, log_ratio, if_same

    def new_jump_birthDeath(self, jr, nodes, node_ids):
        idx = np.random.randint(self.nb) + 1  # pick a branch to change (the root 0 will not be picked)
        jn = list(nodes.values())[idx]
        njump = jn.njump
        if njump == 0:
            jn.njump = 1
            log_ratio = np.log(jr * jn.dist) - np.log(2.)
        else:
            jn.njump = njump + np.random.choice([-1, 1])
            log_ratio = np.log(jr * jn.dist) * (jn.njump - njump) - np.log(max(njump, jn.njump))
            if jn.njump == 0:
                log_ratio += np.log(2.)
        if_same = False
        return [node_ids[jn.name], 0], log_ratio, if_same

    def new_jump_prior(self, jr, nodes, node_ids):
        idx = np.random.randint(self.nb) + 1  # pick a branch to change (the root 0 will not be picked)
        jn = list(nodes.values())[idx]
        njump = jn.njump
        jn.njump = np.random.poisson(jr * jn.dist)
        if_same = (njump == jn.njump)
        log_ratio = 0.
        return [node_ids[jn.name], 0], log_ratio, if_same

    def switch_jumps(self, jr, nodes, node_ids):
        ids, log_ratio, if_same = [0, 0], 0., False

        if sum(self.jps) > self.root.ROOT_JUMP:
            # switch njumps between two adjacent branches
            jump_nodes = [n for n in nodes.values() if n.njump > 0 and not n.is_root()]
            # if len(jump_nodes) > 0:  # when branches other than the root have jumps
            jn = jump_nodes[np.random.randint(len(jump_nodes))]  # pick a jump node
            pjn = jn.up  # get its parent
            # [c for c in pjn.children if c.name!=jn.name]
            move_to = ([] if pjn.is_root() else [pjn]) + jn.children
            n = len(move_to)
            if n > 0:
                jn1 = move_to[np.random.randint(n)]

                # number of branches can be reached from the new jump node
                n1 = len(jn1.children) + (0 if jn1.up.is_root() else 1)  # (len(jn1.up.children) - 1)
                tmp_njump = jn1.njump  # switch the number of jumps
                jn1.njump = jn.njump
                jn.njump = tmp_njump

                log_ratio += ((-np.log(n1)) - (-np.log(n)) +
                              poisson.logpmf(jn.njump, jn.dist*jr) +
                              poisson.logpmf(jn1.njump, jn1.dist*jr) -
                              poisson.logpmf(jn1.njump, jn.dist*jr) -
                              poisson.logpmf(jn.njump, jn1.dist*jr))
                ids = [node_ids[jn.name], node_ids[jn1.name]]

                if tmp_njump == jn1.njump:
                    if_same = True

            else:  # n = 0: jn is a leaf and its parent is the root  # todo: a flag for a non-effective jumps
                ids = [node_ids[jn.name], node_ids[jn.name]]
                log_ratio = 0.
                if_same = True

        return ids, log_ratio, if_same

    def switch_jumps_branches(self, jr, branches, id_nodes=None):
        if id_nodes is None:
            id_nodes = self.id_nodes
        ids, log_ratio, if_same = [0, 0], 0., False

        jn_id = np.where(np.array(self.jps) > 0)[0][1]
        jn = id_nodes[jn_id]

        jn1 = id_nodes[np.random.choice([i for i in branches if i != jn.id])]
        jn1.njump = jn.njump
        jn.njump = 0

        log_ratio += (poisson.logpmf(jn.njump, jn.dist * jr) +
                      poisson.logpmf(jn1.njump, jn1.dist * jr) -
                      poisson.logpmf(jn1.njump, jn.dist * jr) -
                      poisson.logpmf(jn.njump, jn1.dist * jr))
        ids = [jn.id, jn1.id]
        if_same = False
        self.update_jps()
        return ids, log_ratio, if_same

    # def seat_unaffected_data(self, dt, affected_leaves):  # todo
    #     affected_names = [n.name for n in affected_leaves]
    #     a_node_name = []
    #     a_obs = []
    #     a_ndt = 0
    #     self.init_rests()
    #     for idx in range(dt['ndt']):
    #         nn = dt['node_name'][idx]
    #         ob = dt['obs'][idx]
    #         if nn in affected_names:
    #             a_node_name += [nn]
    #             a_obs += [ob]
    #             a_ndt += 1
    #         else:
    #             self.seat_new_obs(node_name=nn, obs=ob)
    #     a_dt = {'node_name': a_node_name,
    #             'obs': a_obs,
    #             'ndt': a_ndt}
    #     return a_dt

    def particleMCMC(self, data, num_particles=5, n_iter=2000, return_particles=False, return_rests=False,
                     if_switch=True, fix_jump_rate=False, prior_mean_njumps=1., init_jr=None, init_log_lik=None,
                     progress_bar=True):  # detailed_info=False, todo
        """
        Generate posterior samples of (jump_rate, jps) with Particle MCMC algorithm,
        where jps denotes jumps on branches.
        :param data: the pandas.DataFrame data
        :param num_particles: number of particles for the particle filtering step
        :param n_iter: number of MCMC iterations
        :param return_particles: whether posterior samples of the particles will
                                 also be returned
        :param return_rests: whether rests of particles will be returned
        :param if_switch: if propose by switching
        :param fix_jump_rate: whether the jump rate is fixed to "init_jr"
        :param init_jr: initial jump rate, will be a sample from the prior if None
        :param init_log_lik: initial log likelihood (in order to connect with previous
                             runs of samples.
        :param prior_mean_njumps: prior mean number of jumps
        :param progress_bar: whether a progress bar will be printed

        :return: info, posterior samples of jump_rate, jps, and particles (None
                 when not return_particles)
        """
        # log = get_log(log_file, out_log, "particleMCMC.txt")
        # if out_log:
        #     log.write("")

        # log of MCMC iterations
        log = {"acc_rate": 0,
               'accepted': [False] * n_iter,
               'log_lik': [0.] * n_iter,
               'log_acc': [0.] * n_iter,
               'proposed': [[0] * (self.nb + 1)] * n_iter,
               "same_proposal_rate": 0.,
               'same_proposal': [False] * n_iter,
               'sampled': [[0, 0]] * n_iter}

        # containers of posterior samples
        post_jrs = np.zeros(n_iter)  # jump rates
        post_jps = np.zeros((n_iter, self.nb + 1), dtype=int)  # nb+1: (it,0) = 1 (# of jumps on base-root branch)
        post_Zs = np.zeros((n_iter, self.nb + 1), dtype=int)  # partition of the tree
        post_particles = [None] * n_iter if return_particles else None
        post_rests = (pd.DataFrame(columns=['iter', 'node_name'] +
                                           [f"{label}_{decor}" for decor in ["nt", "nc"] for label in self.base.keys()])
                      if return_rests else None)

        # initialize jump rate and jps
        jr = prior_mean_njumps / self.tl if init_jr is None else init_jr
        self.poisson_jps(jr)

        # fake records for previous iteration (iteration -1)
        jps_pre, p_pre, Zs_pre = self.jps, None, self.partition_by_jps()[0]
        log_lik_pre = -np.Inf if init_log_lik is None else init_log_lik

        # Start the iteration
        if progress_bar:
            print("Generating posterior samples...", now())

        switch = if_switch  # indicator for whether to switch two jumps or to propose a jump

        p = log_lik = log_acc = None
        for it in tqdm(range(n_iter), disable=(not progress_bar)):
            # new jump rate (if not fixed)
            jr = self.sample_post_jump_rate(prior_mean_njumps) if not fix_jump_rate else jr
            post_jrs[it] = jr

            # propose new jps
            ids, log_ratio, if_same = self.propose_jps(jr, switch=switch)
            if if_switch:
                switch = not switch
            log["sampled"][it] = ids

            # Acceptance
            if if_same:  # if proposed jps is the same with the previous sample
                log['same_proposal'][it] = True
                accepted = True
            else:  # particle filter estimation of data log-likelihood
                ps = Particles(tree=self, num_particles=num_particles, forward=True)
                log_lik = ps.particle_filter(data)  # out_log=out_log, log=log
                p = ps.get_particle() if (return_particles or return_rests) else None
                log_acc = log_lik - log_lik_pre + log_ratio
                accepted = (log_acc > np.log(np.random.rand(1)[0]))

            log['accepted'][it] = accepted
            log["acc_rate"] += accepted
            log['log_lik'][it] = log_lik
            log['log_acc'][it] = log_acc
            log['proposed'][it] = self.jps

            # update and record samples
            if accepted:
                jps_pre = self.jps
                Zs_pre = self.partition_by_jps()[0]
                p_pre = p
                log_lik_pre = log_lik
            else:
                self.jps = jps_pre  # rewind jps
            post_jps[it, :] = jps_pre
            post_Zs[it, :] = Zs_pre
            if return_particles:
                post_particles[it] = p_pre.deep_copy()
            if return_rests:
                if p_pre is None:
                    p_rests = pd.DataFrame(columns=['iter', 'node_name'] + names)
                    p_rests.loc[0] = np.nan
                else:
                    p_rests = p_pre.rests()
                p_rests['iter'] = it
                post_rests = pd.concat([post_rests, p_rests])

        # clean and process info
        log["acc_rate"] /= n_iter
        log['same_proposal_rate'] = np.mean(log['same_proposal'])
        log["last_jr"] = jr  # last jump_rate
        log["last_jps"] = self.jps  # last jps
        log["last_log_lik"] = log_lik_pre  # last log_lik
        self.jps = {'Root': 1}

        if progress_bar:
            print("DONE!", now())

        return {"log": log,
                "jump_rate": post_jrs,
                "jumps": post_jps,
                "partitions": post_Zs,
                "particles": post_particles,
                "restaurants": post_rests}

    ##################
    # synthetic data #
    ##################
    def simulate_jps_byPrior(self, jump_rate=None, njumps=None, min_affected_leaves=0, max_affected_leaves=None,
                             not_on_same_branch=True):
        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        # simulate jumps
        if jump_rate is not None:
            assert njumps is None, "Cannot set jump_rate and njumps simultaneously."
            self.jps = {}
            self.poisson_jps(jump_rate=jump_rate)
            for n in self.traverse():
                if n.nleaf <= min_affected_leaves or n.nleaf >= max_affected_leaves:
                    n.njump = 0
        elif njumps is not None:
            self.jps = {}
            if njumps > 0:
                probs, nodes = [], []
                for n in self.traverse():
                    if not n.is_root() and max_affected_leaves > n.nleaf > min_affected_leaves:
                        probs.append(n.dist)
                        nodes.append(n)
                jump_nodes = np.random.choice(nodes, size=njumps, p=np.array(probs)/sum(probs),
                                              replace=(not not_on_same_branch))
                for n in jump_nodes:
                    n.njump = 1
                self.update_jps()

    def _get_subtree_nodenames(self, node):
        if type(node) != RestNode:
            node = self[node]
        return list(node.nodes.keys())

    def simulate_jps_1equally(self, min_affected_leaves=0, max_affected_leaves=None,
                              not_in_subtree=None, in_subtree=None, CLEAR_OLD_JUMPS=True) -> RestNode:
        invalid = [] if not_in_subtree is None else self._get_subtree_nodenames(not_in_subtree)
        valid = self._get_subtree_nodenames(self.root if in_subtree is None else in_subtree)

        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf

        nodes = []
        for n in self.traverse():
            if not n.is_root() and CLEAR_OLD_JUMPS:
                n.njump = 0
            if (n.name not in invalid) and (n.name in valid) and max_affected_leaves > n.nleaf > min_affected_leaves:
                nodes.append(n)
        assert len(nodes) > 0, "Unable to find appropriate nodes."
        node = np.random.choice(nodes)
        node.njump = 1
        self.update_jps()
        return node

    def simulate_jps_1byNleaf(self, nleaf, tolerance=None, min_affected_leaves=0, max_affected_leaves=None,
                              not_in_subtree=None, in_subtree=None, CLEAR_OLD_JUMPS=True) -> RestNode:
        invalid = [] if not_in_subtree is None else self._get_subtree_nodenames(not_in_subtree)
        valid = self._get_subtree_nodenames(self.root if in_subtree is None else in_subtree)

        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        probs, nodes = [], []
        for n in self.traverse():
            if not n.is_root() and CLEAR_OLD_JUMPS:
                n.njump = 0
            if (n.name not in invalid) and (n.name in valid) and max_affected_leaves > n.nleaf > min_affected_leaves:
                if tolerance is None or abs(n.nleaf - nleaf) < tolerance:
                    nodes.append(n)
                    probs.append(1./((n.nleaf - nleaf + 1) ** 2))

        assert len(nodes) > 0, "Unable to find appropriate nodes."

        node = np.random.choice(nodes, p=np.array(probs)/sum(probs))
        node.njump = 1
        self.update_jps()
        return node

    # =================

    def simulate_data_byPrior(self, each_size: int = 1):
        assert each_size > 0
        new = self.deep_copy()
        assert new.base is not None, "Base measure is not provided."

        # simulate data
        data = pd.DataFrame(columns=["node_name", "obs"])
        for n in new.observed.values():
            for _ in range(each_size):
                obs = n.seat_new_obs(depend=self.depend)
                data = data.append({"node_name": n.name, "obs": obs}, ignore_index=True)
        del new
        return data

    # =================

    def simulate_prior(self, jump_rate=None, njumps=None, min_affected_leaves=0, max_affected_leaves=None,
                       not_on_same_branch=True, each_size: int = 1):
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf

        self.simulate_jps_byPrior(jump_rate, njumps, min_affected_leaves, max_affected_leaves, not_on_same_branch)
        return self.simulate_data_byPrior(each_size)

    def simulate_one_jump(self, min_affected_leaves=0, max_affected_leaves=None, affected_leaves=None, tolerance=None,
                          p0: Categorical = None, p1: Categorical = None,
                          total_variation=0., K: int = None, labels=None, each_size: int = 1):
        assert each_size > 0

        # get distribution before (p0) and after jump (p1)
        if p0 is None or p1 is None:
            assert p0 is None and p1 is None, "Ambiguity in setting p0 and p1"
            if labels is None and K is None:  # use base labels
                labels = self.base.keys()
            elif labels is None:
                labels = [i for i in range(K)]
            K = len(labels)
            dp = total_variation / K
            p0 = Categorical(labels=labels, probs=[1./K + dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])
            p1 = Categorical(labels=labels, probs=[1./K - dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])

        # locate jump
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        if max_affected_leaves is not None and 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if affected_leaves is not None:
            if 0 < affected_leaves < 1:
                affected_leaves *= self.nleaf
            if tolerance is not None and 0 < tolerance < 1:
                tolerance *= self.nleaf
            jn = self.simulate_jps_1byNleaf(
                nleaf=affected_leaves, tolerance=tolerance,
                min_affected_leaves=min_affected_leaves, max_affected_leaves=max_affected_leaves)
        else:
            jn = self.simulate_jps_1equally(
                min_affected_leaves=min_affected_leaves, max_affected_leaves=max_affected_leaves)

        # simulate data
        data = pd.DataFrame(columns=['node_name', 'obs'])
        affected = jn.get_observed().keys()
        for k in self.observed.keys():
            for _ in range(each_size):
                data = data.append({'node_name': k, 'obs':  p1.sample() if k in affected else p0.sample()},
                                   ignore_index=True)
        return data, jn.nleaf
    
    def simulate_two_jumps(self, min_affected_leaves=None, mid_affected_leaves=None, max_affected_leaves=None,
                           single_jump_total_variation=0., each_size: int = 1):
        """
        simulate two jumps, one in the subtree of another, each creates
        single_jump_total_variation amount of difference in distributions before and after it
        The two jumps should be of the same direction
        """
        assert each_size > 0
        assert -0.5 <= single_jump_total_variation <= 0.5

        if min_affected_leaves is None:
            min_affected_leaves = 0
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        if mid_affected_leaves is None:
            mid_affected_leaves = 0.5
        if 0 < mid_affected_leaves < 1:
            mid_affected_leaves *= self.nleaf
        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf

        # get distributions
        labels = self.base.keys()
        K = len(labels)
        dp = 2 * single_jump_total_variation / K
        p0 = Categorical(labels=labels, probs=[1./K + dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])
        p1 = Categorical(labels=labels)  # uniform, probs = 1./K
        p2 = Categorical(labels=labels, probs=[1./K - dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])

        # locate the jumps
        n1 = self.simulate_jps_1byNleaf(nleaf=(mid_affected_leaves + max_affected_leaves)/2,
                                        min_affected_leaves=mid_affected_leaves,
                                        max_affected_leaves=max_affected_leaves)
        n2 = self.simulate_jps_1byNleaf(nleaf=(min_affected_leaves + mid_affected_leaves)/2,
                                        min_affected_leaves=min_affected_leaves,
                                        in_subtree=n1, CLEAR_OLD_JUMPS=False)

        data = pd.DataFrame(columns=['node_name', 'obs'])
        n1_nodes = n1.get_observed().keys()
        n2_nodes = n2.get_observed().keys()
        for k in self.observed.keys():
            for _ in range(each_size):
                data = data.append(
                    {'node_name': k,
                     'obs':  (p2.sample() if k in n2_nodes else p1.sample()) if k in n1_nodes else p0.sample()},
                    ignore_index=True)
        return data, n1.nleaf, n2.nleaf

    def simulate_three_jumps(self, min_affected_leaves=None, max_affected_leaves=None, affected=None,
                             single_jump_total_variation=0., each_size: int = 1):
        """
        simulate two jumps, one in the subtree of another, each creates
        single_jump_total_variation amount of difference in distributions before and after it
        """
        assert each_size > 0
        assert -1/3 <= single_jump_total_variation <= 1/3
        assert affected is None or len(affected) == 3

        if min_affected_leaves is None:
            min_affected_leaves = 0
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if affected is None:
            affected = max_affected_leaves - (np.arange(3) + 1) * (max_affected_leaves - min_affected_leaves) / 4

        # get distributions
        labels = self.base.keys()
        K = len(labels)
        dp = 2 * single_jump_total_variation / K
        p0 = Categorical(labels=labels, probs=[1./K + 1.5 * dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])
        p1 = Categorical(labels=labels, probs=[1./K + 0.5 * dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])
        p2 = Categorical(labels=labels, probs=[1./K - 0.5 * dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])
        p3 = Categorical(labels=labels, probs=[1./K - 1.5 * dp * ((i < (K//2)) - (i >= (K-(K//2)))) for i in range(K)])

        # locate the jumps
        n1 = self.simulate_jps_1byNleaf(nleaf=affected[0],
                                        min_affected_leaves=min_affected_leaves,
                                        max_affected_leaves=max_affected_leaves)
        n2 = self.simulate_jps_1byNleaf(nleaf=affected[1],
                                        min_affected_leaves=min_affected_leaves,
                                        in_subtree=n1, CLEAR_OLD_JUMPS=False)
        n3 = self.simulate_jps_1byNleaf(nleaf=affected[2],
                                        min_affected_leaves=min_affected_leaves,
                                        in_subtree=n2, CLEAR_OLD_JUMPS=False)

        data = pd.DataFrame(columns=['node_name', 'obs'])
        n1_nodes = n1.get_observed().keys()
        n2_nodes = n2.get_observed().keys()
        n3_nodes = n3.get_observed().keys()
        for k in self.observed.keys():
            for _ in range(each_size):
                if k in n3_nodes:
                    obs = p3.sample()
                elif k in n2_nodes:
                    obs = p2.sample()
                elif k in n1_nodes:
                    obs = p1.sample()
                else:
                    obs = p0.sample()
                data = data.append({'node_name': k, 'obs':  obs}, ignore_index=True)
        return data, n1.nleaf, n2.nleaf, n3.nleaf

    # ########################################################################################################
    #
    # def simulation(self, each_size=1, K=2, p0=None):  # simulate data with K categories
    #     new = self.deep_copy()
    #
    #     data = pd.DataFrame(columns=['node_name', 'obs'])
    #
    #     if p0 is None:
    #         p0 = Categorical.uniform(K=K)
    #
    #     for n in new.leaves.values():
    #         sl = p0.sample(each_size)
    #         if each_size == 1:
    #             sl = [sl]
    #         for s in sl:
    #             data = data.append({'node_name': n.name, 'obs': s}, ignore_index=True)
    #
    #     return data
    #
    # def find_jump_and_simulate(self, K=None, labels=None, node_name=None, nleaf=None, delta_p=.0, each_size=1,
    #                            base: Categorical = None, test_subtree=None):
    #     new = self.deep_copy()
    #
    #     data = pd.DataFrame(columns=['node_name', 'obs'])
    #
    #     if base is None:
    #         if K is not None:
    #             labels = [i for i in range(K)]
    #         elif labels is None:
    #             K, labels = 2, [0, 1]
    #         p0 = Categorical.uniform(labels=labels)
    #     else:
    #         labels = list(base.keys())
    #         p0 = base
    #     # p1 = Categorical.uniform(labels=labels)
    #     p1 = Categorical(duplicate_from=p0)
    #     sign = float(np.sign(1 - (p1[labels[0]] + 2 * delta_p)))
    #     p1[labels[0]] += sign * 2. * delta_p
    #     for k in labels[1:]:
    #         p1[k] -= sign * 2. * delta_p / (len(labels) - 1)
    #
    #     if node_name is not None:
    #         node = new.nodes[node_name]
    #     elif nleaf is not None:
    #         if nleaf == 0:
    #             node = new.root
    #         else:
    #             node = new.set_jps_by_nleaf(nleaf=nleaf, not_in_subtree=test_subtree)
    #     else:
    #         node = new.root
    #
    #     tree_observed_name = new.observed.keys()
    #     node_observed_name = node.get_observed().keys()
    #     for k in tree_observed_name:
    #         if k in node_observed_name:
    #             sl = p1.sample(each_size)
    #         else:
    #             sl = p0.sample(each_size)
    #         if each_size == 1:
    #             sl = [sl]
    #         for s in sl:
    #             data = data.append({'node_name': k, 'obs': s}, ignore_index=True)
    #
    #     return data, node.name, new.jps
    #
    # def generator_preprocess(self, disc=None, conc=None, depend=None, jps=None,
    #                          base=None, sizes=None, init_rests=True,
    #                          out_log=False, log_file=None):
    #     """
    #     Pre-process self (new in generator()) and clean parameter for generator
    #     :return: log file and sample size at each node
    #     """
    #     log = get_log(log_file, out_log, "generator_log.txt")
    #
    #     new = self.deep_copy()
    #
    #     if disc is not None:
    #         new.disc = disc
    #     if conc is not None:
    #         new.conc = conc
    #     if depend is not None:
    #         new.depend = depend
    #     if jps is not None:
    #         new.jps = jps
    #
    #     # sample size at each node
    #     node_sizes = {}
    #     node_names = new.observed.keys()
    #     if sizes is None:
    #         if (not type(each_size) == int) or each_size <= 0:
    #             raise ValueError("each_size should be a positive int. ")
    #         else:
    #             for k in node_names:
    #                 node_sizes[k] = each_size
    #     elif type(sizes) == dict:
    #         for n in sizes:
    #             if (not type(sizes[n]) == int) or sizes[n] < 0:
    #                 raise ValueError("Elements of sizes should be non-negative int. ")
    #             if sizes[n] > 0:
    #                 node_sizes[n] = sizes[n]
    #         if sum(node_sizes.values()) == 0:
    #             raise ValueError("There should be at least 1 positive element in sizes. ")
    #     elif type(sizes) == list:
    #         if not len(sizes) == len(node_names):
    #             raise ValueError("Length of sizes should agree with the number of leaves in the tree. ")
    #         else:
    #             for k, num in zip(node_names, sizes):
    #                 if (not type(num) == int) or num < 0:
    #                     raise ValueError("Elements of sizes should be non-negative int. ")
    #                 elif num > 0:
    #                     node_sizes[k] = num
    #             if sum(node_sizes.values()) == 0:
    #                 raise ValueError("There should be at least 1 positive element in sizes. ")
    #     else:
    #         raise TypeError("Type of sizes should be dict or list. ")
    #
    #     # the base measure
    #     if base is not None:
    #         if base == "uniform":
    #             # category labels
    #             if labels is None:
    #                 if num_cat is None:
    #                     raise ValueError("num_cat should not be None when the labels need to be specified.")
    #                 labels = [n for n in range(num_cat)]
    #             labels = list(labels)
    #
    #             new.uniform_base(labels)
    #         else:
    #             if type(base) != Categorical:
    #                 raise TypeError("base should be a valid base measure of type Categorical. ")
    #             for k in base.keys():
    #                 if k not in labels:
    #                     raise ValueError("Invalid base measure: base label {0} not in labels {1}".format(k, labels))
    #         new.base = base
    #     else:
    #         labels = list(new.base.keys())
    #
    #     # prepare restaurants
    #     if init_rests:
    #         new.init_rests()
    #
    #     if out_log:
    #         jump_tree, ref_names = new.jps_prune()
    #
    #         log.write("\n" + ("=" * 80 + "\n") * 3 + "\n" +
    #                   "Full Tree: \n" + new.str_print(tree=True, jps=True, observed=True) + "\n")
    #         log.write("\nPruned tree: " + jump_tree.str_print(tree=True, rests=True) + "\n")
    #
    #     return log, node_sizes, labels, new
    #
    # def generator(self, disc=None, conc=None, depend=None, jps=None,
    #               base=None, sizes=None, init_rests=True,
    #               out_log=False, log_file=None):
    #     """
    #     # todo complete the note
    #     # todo remove all "labels" arguments.
    #     Generate synthetic data given jps
    #     :param disc: discount parameter of Pitman-Yor process
    #     :param conc: ---
    #     :param depend: ---
    #     :param jps: jps, if None then uses the jps specified with the tree
    #     :param base: default: "uniform" ---
    #     :param labels: a list of labels of all available types, if None then use 0, 1, ..., num_cat-1
    #     :param num_cat: number of categories, only active when names is None
    #     :param sizes: a list or dict of number of samples on each node, if None then use each_size
    #     :param each_size: an int to specify number of samples on each node, only active when sizes is None.
    #     :param init_rests: if restaurants will be initialized
    #                        (whether seated customers in the current tree will be removed).
    #     :param out_log: if True ---
    #     :param log_file: ---
    #     :return: generated data (pandas.DataFrame) with node_name and obs
    #     """
    #     log, node_sizes, labels, tree = self.generator_preprocess(disc, conc, depend, jps, base,
    #                                                               sizes, init_rests, out_log, log_file)
    #
    #     # simulate data
    #     data = {}
    #     for node_name in node_sizes.keys():
    #         data[node_name] = dict.fromkeys(labels, 0)
    #
    #         for idx in range(node_sizes[node_name]):
    #             obs = tree.seat_new_obs(node_name)
    #             data[node_name][obs] += 1
    #             if out_log:
    #                 log.write("\n" + node_name + ": sample {0} = {1}".format(idx+1, obs) +
    #                           "\n" + tree.str_print(rests=True) + "\n")
    #
    #     # adjust data format
    #     data = pd.DataFrame(data)
    #     if out_log:
    #         log.write("\n" + "="*80 + "\n\nSynthetic data:\n" + data.__str__())
    #
    #     data = pd.melt(data)
    #     data = pd.concat([data, pd.Series(labels * len(node_sizes))], axis=1)
    #     data.rename(columns={'variable': 'node_name', 'value': 'ct', 0: 'obs'}, inplace=True)
    #     data = data.reindex(data.index.repeat(data.ct))
    #     data.index = [i for i in range(data.shape[0])]
    #     data = data[['node_name', 'obs']]
    #
    #     return data, tree


########################################################################################################################

    # def train_test_split(self, data, test_size=0.2, node_names=None,
    #                      by_subtree=False, subtree_root_name=None, balance=False):
    #     """
    #     :param data:
    #     :param test_size: todo 0<= <=1
    #     :param node_names:
    #     :param by_subtree:
    #     :param subtree_root_name:
    #     :param balance: equal amount samples on each node
    #     :return:
    #     """
    #     n = None
    #     if (node_names is None) and (not by_subtree):
    #         node_names = self.observed.keys()
    #     elif by_subtree:
    #         done = False
    #         for n in self.root.traverse():
    #             if n.name == subtree_root_name:
    #                 done = True
    #                 break
    #         if not done:
    #             raise ValueError("Unable to find the (jump) node with name " + subtree_root_name + ".")
    #         node_names = n.get_observed().keys()
    #
    #     def sample(obj):
    #         return obj.sample(frac=test_size)
    #
    #     if balance:
    #         test = data[data.node_name.isin(node_names)].groupby('node_name').apply(sample)
    #         train = data.drop(test.index.levels[1])
    #     else:
    #         test = sample(data[data.node_name.isin(node_names)])
    #         train = data.drop(test.index)
    #
    #     test.reset_index(drop=True, inplace=True)
    #     train.reset_index(drop=True, inplace=True)
    #
    #     return train, test

    ####################
    # data likelihoods #
    ####################

    # def data_jps_loglik(self, data, model="HPY", particle_filter=None, num_particles=5, forward=True,
    #                     jps=None, var=None, base=None, labels=None):
    #     """
    #     log-likelihood of the data, given the number of jumps on each branch (jps)
    #     :param data:
    #     :param model:
    #     :param particle_filter:
    #     :param num_particles:
    #     :param forward:
    #     :param jps: todo infer from the tree directly?
    #     :param var: todo change name to parameter
    #     :param base:  todo infer from the tree directly?
    #     :param labels:
    #     :return:
    #     """
    #     other = self.deep_copy()
    #     other.parameters_by_model(model, var)
    #     loglik = None
    #     if base == 'uniform':
    #         other.uniform_base(labels)
    #     if jps is not None:
    #         other.jps = jps
    #     if particle_filter is None:
    #         particle_filter = True if model == "HPY" else False
    #
    #     # Calculate the log-likelihood for the hierarchical PY model with discount = param
    #     if model == "HPY":
    #         if not particle_filter:
    #             raise ValueError("Cannot calculate the log likelihood without using particle filter for HPY model.")
    #         ps = Particles(tree=other, num_particles=num_particles, forward=True)
    #         loglik = ps.particle_filter(data)  # out_log=out_log, log=log
    #
    #     # Calculate the log-likelihood for the hierarchical Dirichlet model with concentration = param
    #     elif model == "HDP":
    #         if particle_filter:
    #             ps = Particles(tree=other, num_particles=num_particles, forward=True)
    #             loglik = ps.particle_filter(data)  # out_log=out_log, log=log
    #         else:
    #             jump_tree, ref_names = other.jps_prune()
    #             counts = pd.crosstab(data.obs, data.node_name.map(ref_names)).to_dict()
    #
    #             loglik = 0
    #             for c in counts.values():
    #                 n = np.array([c[k] if k in c.keys() else 0 for k in other.base.keys()])
    #                 alpha = np.array([other.conc * v for v in other.base.values()])
    #
    #                 loglik += (gammaln(np.sum(n)+1) - np.sum(gammaln(n+1)) +
    #                            gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) +
    #                            np.sum(gammaln(n+alpha)) - gammaln(np.sum(n+alpha)))
    #     # del other
    #     return loglik

    # def subtree_data_loglik(self, data, model="HPY", jump_rate=1, node_name=None,
    #                         particle_filter=None, num_particles=5, forward=None):
    #     other = self.deep_copy()
    #
    #     node = self.get_node_by_name(node_name)
    #     node.poisson_jps(jump_rate=jump_rate)  # sample jumps in the subtree
    #     node_base = node.rest.base
    #     node.init_rests(other.disc, other.conc, node_base)
    #     other.update_jps()
    #
    #     return other.data_jps_loglik(data, model=model, particle_filter=particle_filter,
    #                                  num_particles=num_particles, forward=forward)
    #
    # def data_loglik(self, data, model="HPY", jump_rate=1, node_name=None, num_sample=100, particle_filter=None,
    #                 num_particles=5, forward=True, var=None):
    #     # self, data = tree, test
    #     other = self.deep_copy()
    #     other.parameters_by_model(model, var)
    #
    #     logliks = np.zeros(num_sample)
    #     for i in range(num_sample):
    #         if node_name is None:
    #             other.poisson_jps(jump_rate=jump_rate)
    #             logliks[i] = other.data_jps_loglik(data=data, model=model, particle_filter=particle_filter,
    #                                                num_particles=num_particles, forward=forward)
    #         else:
    #             # really need to sample jumps within the subtree? or better,
    #             # restricted the posterior sampling within the part of tree that have data
    #             logliks[i] = other.subtree_data_loglik(data=data, model=model, jump_rate=jump_rate,
    #                                                    node_name=node_name, particle_filter=particle_filter,
    #                                                    num_particles=num_particles, forward=forward)
    #     return np.mean(logliks)
    #
    # def test_post_loglik(self, test, train, model="HPY", node_name=None, num_particles=3, var=None,
    #                      num_sample=700, burn_out=300, progress_bar=False):
    #     if progress_bar:
    #         print("Calculating posterior test likelihood ...")
    #
    #     other = self.deep_copy()
    #     other.parameters_by_model(model, var)
    #
    #     n_iter = burn_out + num_sample
    #     info, post_jump_rate, post_jps, post_particles = other.particleMCMC(train, return_particles=True,
    #                                                                         num_particles=num_particles,
    #                                                                         n_iter=n_iter, progress_bar=progress_bar)
    #
    #     if progress_bar:
    #         print("Estimating likelihood...")
    #     pct10 = num_sample // 10
    #
    #     logliks = np.zeros(num_sample)
    #     for i in range(num_sample):
    #         tree = post_particles[burn_out+i]
    #         jump_rate = post_jump_rate[burn_out+i]
    #
    #         if node_name is None:
    #             logliks[i] = tree.data_jps_loglik(test, model=model)
    #         else:
    #             logliks[i] = tree.data_loglik(test, model=model, jump_rate=jump_rate,
    #                                           node_name=node_name, num_sample=10)
    #         if progress_bar and (i+1) % pct10 == 0:
    #             print("..{}0%".format((i+1)//pct10))
    #
    #     return np.mean(logliks), post_jump_rate, post_jps, post_particles
    #
    #
    # def particleMCMC_branches(self, data, branches,
    #                           num_particles=5, n_iter=2000, return_particles=False, return_rests=False,
    #                           fixed_jump_rate=False, init_jr=None, init_log_lik=-np.Inf,  # div_name=None,
    #                           prior_mean_njumps=1., total_number_of_jumps=1,
    #                           out_log=False, log_file=None, progress_bar=False):  # detailed_info=False, todo
    #     """
    #     Generate posterior samples of (jump_rate, jps) with Particle MCMC algorithm,
    #     where jps denotes jumps on branches.
    #     :param data: the pandas.DataFrame data
    #     :param branches: branches to be considered
    #     :param num_particles: number of particles for the particle filtering step
    #     :param n_iter: number of MCMC iterations
    #     :param return_particles: whether posterior samples of the particles will
    #                              also be returned
    #     :param return_rests: whether rests of particles will be returned
    #     :param div_name: the type of divergence to use
    #     :param fixed_jump_rate: whether the jump rate is fixed to "init_jr"
    #     :param init_jr: initial jump rate, will be a sample from the prior if None
    #     :param init_log_lik: initial log likelihood (in order to connect with previous
    #                          runs of samples.
    #     :param prior_mean_njumps: prior mean number of jumps
    #     :param total_number_of_jumps: total number of jumps allowed
    #     :param branches: a list of branches considered while updating
    #     :param out_log: if True, will output log information for each iteration
    #     :param log_file: the log file name (string)
    #     :param progress_bar: whether a progress bar will be printed
    #
    #     :return: info, posterior samples of jump_rate, jps, and particles (None
    #              when not return_particles)
    #     """
    #     # pct10 = n_iter // 10
    #
    #     log = get_log(log_file, out_log, "particleMCMC.txt")
    #
    #     if out_log:
    #         log.write("")
    #
    #     # initialize the information output
    #     info = {"acc_rate": 0,
    #             'accepted': [False] * n_iter,
    #             'log_lik': [0.] * n_iter,
    #             'log_acc': [0.] * n_iter,
    #             'proposed': [[0] * (self.nb + 1)] * n_iter,
    #             'sampled': [[0, 0]] * n_iter}
    #
    #     # containers of posterior samples
    #     post_jump_rate = np.zeros(n_iter)
    #     post_jps = np.zeros((n_iter, self.nb + 1), dtype=int)  # nb+1: (i,0) = 1 (# of jumps on base-root branch)
    #     post_particles = [None] * n_iter
    #     # post_divs = np.zeros((n_iter, self.nb))
    #     post_partitions = np.empty((n_iter, self.nb+1), dtype=int)  # the partition
    #
    #     names = []
    #     if return_rests:
    #         for lbl in self.base.keys():
    #             names += ["{}_nt".format(lbl), "{}_nc".format(lbl)]
    #     post_rests = pd.DataFrame(columns=['iter', 'node_name'] + names)
    #
    #     if init_jr is None:
    #         if fixed_jump_rate:
    #             raise ValueError("No jump rate (init_jr) set. ")
    #         jr = self.sample_prior_jump_rate(prior_mean_njumps)
    #         log_lik_pre = -np.Inf
    #     else:
    #         jr = init_jr
    #         self.poisson_jps(jr)
    #         log_lik_pre = init_log_lik
    #
    #     self.jps = {}
    #     id_nodes = self.id_nodes
    #     id_nodes[np.random.choice(branches)].njump = total_number_of_jumps
    #     self.update_jps()
    #     # print("init: {}".format(np.where(self.jps)[0]))
    #
    #     # fake records for previous iteration (iteration -1)
    #     # when idx=0, the fake pre will assure acceptance
    #     jps_pre, p_pre, p = self.jps, None, None
    #     # affected_leaves, log_ratio = set(self.leaves.values()), 0
    #     log_ratio = log_lik = log_acc = 0.
    #     if_same = False
    #
    #     # Start the iteration
    #     if progress_bar:
    #         print("Generating posterior samples...", now())
    #
    #     for it in tqdm(range(n_iter), disable=(not progress_bar)):
    #         if if_same:
    #             accepted = True
    #         else:
    #             ps = Particles(tree=self.jps_prune(), num_particles=num_particles, forward=True)
    #             log_lik = ps.particle_filter(data)  # out_log=out_log, log=log
    #             p = ps.get_particle() if (return_particles or return_rests or div_name) else None
    #             log_acc = log_lik - log_lik_pre + log_ratio
    #             accepted = (log_acc > np.log(np.random.rand(1)[0]))
    #
    #         info['accepted'][it] = accepted
    #         info['log_lik'][it] = log_lik
    #         info['log_acc'][it] = log_acc
    #         info['proposed'][it] = self.jps
    #
    #         # update the samples
    #         post_jump_rate[it] = jr
    #
    #         if accepted:  # accept the proposed jps
    #             # print(" -- acc")
    #             post_jps[it, :] = self.jps
    #             jps_pre = self.jps
    #             log_lik_pre = log_lik
    #             p_pre = p.deep_copy()
    #             if return_particles:
    #                 post_particles[it] = p.deep_copy()
    #             if return_rests:
    #                 p_rests = p.rests()
    #                 p_rests['iter'] = it
    #                 post_rests = pd.concat([post_rests, p_rests])
    #             if div_name is not None:
    #                 post_divs[it] = p.divs(div_name)
    #
    #             info["acc_rate"] += 1
    #         else:  # reject the proposed jps
    #             # if out_log:
    #             #     log.write("REJECT!\n")
    #             post_jps[it, :] = jps_pre
    #             self.jps = jps_pre
    #             if return_particles:
    #                 post_particles[it] = p_pre.deep_copy()
    #             if return_rests:
    #                 if p_pre is None:
    #                     p_rests = pd.DataFrame(columns=['iter', 'node_name'] + names)
    #                     p_rests.loc[0] = np.nan
    #                 else:
    #                     p_rests = p_pre.rests()
    #                 p_rests['iter'] = it
    #                 post_rests = pd.concat([post_rests, p_rests])
    #             # if div_name is not None:
    #             #     if p_pre is None:
    #             #         post_divs[it] = np.nan
    #             #     else:
    #             #         post_divs[it] = p_pre.divs(div_name)
    #         # record partition
    #         post_partitions[it] = self.partition_by_jps()[0]
    #
    #         # generate new proposal
    #         jr = self.sample_post_jump_rate(prior_mean_njumps) if not fixed_jump_rate else init_jr
    #         ids, log_ratio, if_same = self.switch_jumps_branches(jr, branches, id_nodes)
    #         # print(ids)
    #         info["sampled"][it] = ids
    #
    #         # print("iter-{}: {}; \t {}".format(it, np.where(self.jps)[0], ids))
    #
    #     # clean and process info
    #     info["acc_rate"] /= n_iter
    #
    #     info["last_jr"] = jr  # last jump_rate
    #     info["last_jps"] = self.jps  # last jps
    #     info["last_ll"] = log_lik_pre  # last log_lik
    #
    #     self.jps = {'Root': 1}  # reset the jps
    #
    #     # print("DONE!", now(), "count_same = {} / {} = {}. ".format(count_same, n_iter, count_same/n_iter))
    #     print("DONE!", now())
    #
    #     return {"info": info,
    #             "jump_rate": post_jump_rate,
    #             "jumps": post_jps,
    #             "partitions": post_partitions,
    #             # "divs": post_divs,
    #             "particles": post_particles,
    #             "restaurants": post_rests}







