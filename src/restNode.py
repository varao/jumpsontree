"""

PhyloHPYP: restNode.py

Created on 5/3/18 5:35 PM

@author: Hanxi Sun

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from ete3 import Tree
# https://github.com/etetoolkit/ete/blob/master/ete3/coretype/tree.py
# http://etetoolkit.org/docs/latest/reference/reference_tree.html
import numpy as np
# from categorical import Categorical
from src.rest import Rest
from src.utils import KL, L1
from scipy.special import gammaln
# from math import factorial


def my_gammaln(n):
    return 0. if n == 0 else gammaln(n)


class RestNode(Tree):
    """
    A (tree) node in the Chinese Restaurant Franchise (corresponds to a
    hierarchical Pitman-Yor process)
    """
    _ROOT_JUMP = 1  # number of jumps at the root
    # _COUNTER = 0  # todo id

    def __init__(self, newick=None, newick_format=1, dist=0, support=None, name=None,
                 quoted_node_names=False):
        """
        Most parameter settings are inherited from ete3.Tree, except
        for "njump", which denotes the number of jumps on the branch
        that connects the current node and its parent.
        """
        super().__init__(newick=newick, format=newick_format, dist=dist, support=support,
                         name=name, quoted_node_names=quoted_node_names)
        # self._id = type(self)._COUNTER
        # type(self)._COUNTER += 1
        self._njump = 0
        self._id = None
        self._rest = None
        self._observed = False  # True if self is a leave in the original tree.
        # self._depth = self._height = 0

    #####################
    #   node property   #
    #####################
    def _get_id(self):
        return self._id

    def set_id(self, value):
        self._id = value

    def _get_njump(self):
        return self._njump

    def _set_njump(self, value):
        if int(value) < 0:
            raise ValueError("The number of jumps should not be negative. ")
        self._njump = int(value)

    def _get_rest(self):
        return self._rest

    def _set_rest(self, value):
        if type(value) == Rest:
            self._rest = value
        else:
            raise TypeError("Incorrect type: restaurant (Rest)")

    # def _get_depth(self):
    #     return self._depth
    #
    # def _get_height(self):
    #     return self._height
    #
    # def set_depth(self, value):
    #     self._depth = int(value)
    #
    # def set_height(self, value):
    #     self._height = int(value)

    njump = property(fget=_get_njump, fset=_set_njump)
    rest = property(fget=_get_rest, fset=_set_rest)
    id = property(fget=_get_id)
    # depth = property(fget=_get_depth)
    # height = property(fget=_get_height)

    #####################
    # sub-tree property #
    #####################
    def _get_total_length(self):
        """
        Calculate the total branch length for the subtree starting at self
        :return: total branch length
        """
        tl = self.dist
        for child in self.children:
            tl += child.tl
        return tl

    def _get_num_branches(self):
        """
        Get the total number of branches in the subtree starting at self
        :return: total number of branches
        """
        nb = len(self.children)
        for child in self.children:
            nb += child.nb
        return nb

    def _get_branch_lengths(self):
        """
        Get the list of each branch length
        :return: the list of branch lengths
        """
        bls = []
        for n in self.traverse(strategy='preorder'):
            bls.append(n.dist)
        return bls

    def _get_jump_list(self):
        """
        Get the list of jumps on each branch
        :return: the list of number of jumps on all branches
        """
        jps = []
        for n in self.traverse(strategy='preorder'):
            jps.append(n.njump)
        return jps

    def _set_jump_list(self, value):
        """
        Set all jumps according to value
        :param value: a list of int+ to represent number of jumps on each branch
        :return: None
        """
        if type(value) == list:
            if self.is_root():
                if value[0] == 0:
                    warnings.warn("Root node njump is set to be the default, {}, not 0. ".format(self.ROOT_JUMP))
                    value[0] = self.ROOT_JUMP
                elif value[0] != self.ROOT_JUMP:
                    warnings.warn("Root node njump is {0}, not the default, {1}".format(value[0], self.ROOT_JUMP))
            for i, n in zip(range(len(value)), self.traverse(strategy='preorder')):
                n.njump = value[i]
        elif type(value) == dict:
            if self.is_root():
                if self.name not in value:
                    # warnings.warn("Root node njump is set to be the default, {}. ".format(self.ROOT_JUMP))
                    value[self.name] = self.ROOT_JUMP
                elif value[self.name] != self.ROOT_JUMP:
                    warnings.warn(
                        "Root node njump is set to be {0}, not the default, {1}".format(value[self.name],
                                                                                        self.ROOT_JUMP))
            for n in self.traverse(strategy='preorder'):
                n.njump = value[n.name] if n.name in value else 0
        else:
            raise TypeError("Incorrect type: all jumps (list of number of jumps, " +
                            "or dict with node names and number of jumps)")

    def _get_nodes(self):
        node_dict = {}
        for n in self.traverse(strategy='preorder'):
            node_dict[n.name] = n
        return node_dict

    def _get_nodes_by_id(self):
        id_nodes = {}
        for n in self.traverse(strategy='preorder'):
            id_nodes[n.id] = n
        return id_nodes

    tl = property(fget=_get_total_length)  # total branch length
    nb = property(fget=_get_num_branches)  # total number of branches
    bls = property(fget=_get_branch_lengths)  # each branch length
    jps = property(fget=_get_jump_list, fset=_set_jump_list)  # #jumps on each branch
    nodes = property(fget=_get_nodes)  # a dict of all nodes with name being the key
    id_nodes = property(fget=_get_nodes_by_id)  # a dict of all nodes with id being the key

    def divs(self, div_name):
        new = self.deep_copy()

        def div(p, c):
            if div_name == "KLcb":
                return KL(p, c, child_base=True)
            elif div_name == "KLpb":
                return KL(p, c, child_base=False)
            elif div_name == "L1":
                return L1(p, c)
            else:
                raise ValueError("Divergence name not recognized. ")

        divs = []
        for n in new.traverse(strategy='preorder'):
            if not n.is_root():
                if n.njump == 0:
                    divs += [0]
                else:
                    n.rest.base = n.up.rest.pm
                    # divs += [div(n.rest.base, n.rest.pm)]
                    m = div(n.rest.base, n.rest.pm)
                    divs += [m]
                    if m == np.inf:
                        print(n.name, n.rest, n.rest.base, n.up.rest, n.up.rest.base)
        return divs

    @property
    def nleaf(self):
        """
        number of leaves that will be affected by jumps at current node.
        :return: nleaf
        """
        if self.is_leaf():
            return 1
        else:
            nleaf = 0
            for child in self.children:
                nleaf += child.nleaf
            return nleaf

    def get_node_by_i(self, node_i):
        i = 0
        n, found = None, False
        for n in self.traverse(strategy="preorder"):
            if i == node_i:
                found = True
                break
            i += 1
        if not found:
            raise ValueError("not enough nodes in the tree to include {}".format(node_i))
        return n

    def get_node_i_by_name(self, node_name):
        i = 0
        found = False
        for n in self.traverse(strategy="preorder"):
            if n.name == node_name:
                found = True
                break
            i += 1
        if not found:
            raise ValueError("no node with name: " + node_name)
        return i

    #####################
    #     utilities     #
    #####################

    def traverse(self, strategy='preorder', is_leaf_fn=None):
        if strategy == "preorder":
            return self._iter_descendants_preorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "levelorder":
            return self._iter_descendants_levelorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "postorder":
            return self._iter_descendants_postorder(is_leaf_fn=is_leaf_fn)

    @property
    def ROOT_JUMP(self):
        return type(self)._ROOT_JUMP

    def is_offspring(self, child_name):
        res = False
        for n in self.traverse('preorder'):
            if n.name == child_name:
                res = True
                break
        return res

    def all_saved_subtree_properties(self):
        """
        get all subtree properties all at once, only need one full travel of the tree
        :return: tl, nb, bls, jps
        """
        tl, nb, bls, jps = self.dist, len(self.children), [self.dist], [self.njump]
        for child in self.children:
            tl1, nb1, bls1, jps1 = child.all_saved_subtree_properties()
            tl += tl1
            nb += nb1
            bls += bls1
            jps += jps1
        return tl, nb, bls, jps

    def deep_copy(self):
        return self.copy(method='deepcopy')

    def __str__(self):
        return self.get_ascii(compact=False, show_internal=True)

    def str_print(self, tree=False, pruned=False, pruned_rests=False,
                  rests=False, bases=False, jps=False, observed=False):
        ret = []
        if tree:
            ret.append(self.__str__())
        if pruned:
            jp_root = self.deep_copy()
            ref_name = jp_root.jps_prune()
            ref = ""
            for k in set(ref_name.values()):
                ref_k = [k1 for k1, v1 in ref_name.items() if v1 == k]
                ref += ("\n  {0}: ".format(k) + ", ".join(ref_k))
            ret.append("Equivalent Jump Tree:" + jp_root.__str__() + "\n" +
                       "Reference observation nodes: " + ref + '\n')
        if pruned_rests:
            jp_root = self.deep_copy()
            _ = jp_root.jps_prune()
            node_names = [n.name for n in jp_root.traverse('preorder')]
            pruned_rests = ""
            for n in self.traverse('preorder'):
                if n.name in node_names:
                    pruned_rests += ('\n  {0} rest = '.format(n.name) + n.rest.__str__())
            ret.append("Restaurants (pruned): " + pruned_rests + '\n')
        if rests:
            rests = ""
            for n in self.traverse('preorder'):
                rests += ('\n  {0} rest = '.format(n.name) + n.rest.__str__())
            ret.append("Restaurants:" + rests + '\n')
        if bases:
            bases = ""
            for n in self.traverse('preorder'):
                bases += ('\n  {0} base = '.format(n.name) + n.rest.base.__str__())
            ret.append("Bases: " + bases + '\n')
        if jps:
            njumps = []
            for n in self.traverse('preorder'):
                njumps.append('{0}: {1}'.format(n.name, n.njump))
            ret.append("Jumps: " + "{" + ", ".join(njumps) + "}\n")
        if observed:
            observed = self.get_observed()
            ret.append("Observed Nodes: " + '[' + ', '.join(observed.keys()) + ']')
        return '\n'.join(ret)

    #####################
    #  observed nodes   #
    #####################
    def init_observed_nodes(self):
        """
        Set leaf nodes in the original tree to be observed
        :return: None
        """
        for n in self.iter_leaves():
            n._observed = True

    def is_observed(self):
        return self._observed

    def iter_observed(self):
        """
        Iterate through all observed nodes
        :return: None
        """
        for n in self.traverse(strategy='preorder'):
            if n.is_observed():
                yield n

    def get_observed(self):
        """
        Get all observed nodes in the tree
        :return: a dictionary of observed nodes (name:node)
        """
        obs_nodes = {}
        for n in self.iter_observed():
            obs_nodes[n.name] = n
        return obs_nodes

    ##################
    #   restaurants  #
    ##################

    # def has_rest(self):
    #     return not self.njump == 0

    def init_rests(self, disc, conc, base):
        """
        Initiate restaurants for all nodes
        :param disc: the discount parameter of a unit branch length in PYP
        :param conc: the concentration parameter in PYP
        :param base: base measure (can be uniform)
        :return: None
        """
        if self.is_root():
            if self.njump == 0:
                self._njump = self.ROOT_JUMP
                warnings.warn("Root node njump is set to be the default, {}, not 0. ".format(self.ROOT_JUMP))
            elif self.njump != self.ROOT_JUMP:
                warnings.warn("Root node njump is {0}, not the default {1}. ".format(self.njump, self.ROOT_JUMP))

        if self.njump == 0:
            self._rest = self.up.rest
        else:
            self._rest = Rest((disc ** self.njump), conc, base)  # branch discount = disc^(#jumps)

        for child in self.children:
            child.init_rests(disc, conc, base)

    def update_child_base(self):
        """
        Update base measures in offspring nodes according to seats of customers & the base measure
        Needs a valid base measure at the restaurant of current node (self.rest)
        Only active when depend = True
        :return: None
        """
        children_base = self.rest.p_key()
        for child in self.children:
            if child.njump > 0:
                child.rest.base = children_base
            child.update_child_base()

    def seat_new_obs(self, obs=None, depend=None):
        """
        Update all restaurants with the table configuration introduced by an
        observation at current node (self). If obs is None, the table configuration
        is sampled according to the prior and the sampled obs is returned. If obs
        is given, the table configuration then will be sampled according to (6.2)
        in progress report of PhyloTreePY @ 03/28/18.
        :param obs: the new observation label, or None.
        :param depend: Boolean, indicating whether the distribution on the child
                       node depends on the parent node.
        :return: sampled label.
        """
        if depend is None:
            raise ValueError("parameter depend not specified (should be True or False). ")

        node = self
        while (not node.is_root()) and node.njump == 0:
            node = node.up

        new_table, k = node.rest.seat_new_customer(obs)  # if obs is None, then generates an obs from the prior

        if depend:
            if new_table and (not node.is_root()):
                parent = node.up
                parent.seat_new_obs(obs=k, depend=depend)
            else:
                node.update_child_base()

        return k

    ##################
    # jps management #
    ##################

    def subtree_has_jump(self):
        return sum(self.jps[1:]) > 0

    def jps_prune(self, observed=None, obs_names=None):
        """
        Prune the tree according to jumps (i.e. remove branches
        with 0 jump) Observed nodes will be preserved and the ref_names will be returned
        :param observed: the list of observed nodes (leaves), If None, then will get it from the tree
        :param obs_names: the list of observed node names. If None, then will get it from the tree
        :return: ref_names, a dict that maps observed node names to names after merging {old_name:new_name}
        """
        if obs_names is None:
            if observed is None:
                observed = self.get_observed()
            obs_names = observed.keys()

        ref_names = {name: name for name in obs_names}

        for node in self.traverse(strategy='preorder'):
            if node.njump == 0 and (not node.is_root()):  # no jump between node & its parent
                parent = node.up

                if node.is_observed():
                    if parent.is_observed():  # merge 2 observed nodes
                        ref_names[parent.name] = node.name
                    parent.name = node.name
                    parent._observed = True

                parent.children += node.children
                for child in node.children:
                    child.up = parent
                parent.remove_child(node)

        # solve recursive references in ref_names
        for k, v in ref_names.items():
            if k != v:
                idx = [k]
                while v != ref_names[v]:
                    idx.append(v)
                    v = ref_names[v]
                while ref_names[k] != v:
                    ref_names[k] = v

        return ref_names

    def full_jps(self, njump=1):
        """
        Assign the same number of jumps to all branches in the tree, except for the root.
        :param njump: number of jumps on each branch
        :return: None
        """
        if self.is_root():  # default jump = 1 at root
            self.njump = self.ROOT_JUMP
        else:
            self.njump = njump
        for child in self.children:
            child.full_jps(njump)

    def poisson_jps(self, jump_rate):
        """
        Sample jumps on each branch from the prior: Poisson(jump_rate * branch_length)
        :param jump_rate: the intensity of Poisson
        :return: None
        """
        if jump_rate < 0:
            raise ValueError("The Poisson intensity (jump rate) should not be negative. ")

        if self.is_root():  # default jump = 1 at root
            self.njump = self.ROOT_JUMP
        else:
            self.njump = np.random.poisson(jump_rate * self.dist)
        for child in self.children:
            child.poisson_jps(jump_rate)


########################################################################################################################

