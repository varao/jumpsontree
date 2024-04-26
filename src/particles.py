"""

phyloHPYP: particles.py

Created on 11/09/18 10:29 PM

@author: Hanxi Sun

"""

import numpy as np
from ete3 import Tree
# from categorical import Categorical


class ParticleNode(Tree):
    def __init__(self, newick=None, newick_format=1):
        super().__init__(newick=newick, format=newick_format)
        self.njump = 0
        self.rests = []  # list of Rest
        self.observed = False  # True if self is a leave in the original tree.

    def update_child_base(self, idx):
        children_base = self.rests[idx].p_key()
        for child in self.children:
            if child.njump > 0:
                child.rests[idx].base = children_base
            child.update_child_base(idx)

    def seat_new_obs(self, obs=None, depend=None, idx=None):
        if depend is None:
            raise ValueError("parameter depend not specified (should be True or False). ")

        # self = self
        # while (not self.is_root()) and self.njump == 0:
        #     self = self.up

        if idx is not None:
            rest = self.rests[idx]
            new_table, k = rest.seat_new_customer(obs)  # if obs is None, then generates an obs from the prior
            if depend:
                if new_table and (not self.is_root()):
                    parent = self.up
                    parent.seat_new_obs(obs=k, depend=depend, idx=idx)
                else:
                    self.update_child_base(idx)
            return k
        else:
            ks = []
            for i in range(len(self.rests)):
                k = self.seat_new_obs(obs=obs, depend=depend, idx=i)
                ks += [k]
            return ks

    def traverse(self, strategy='preorder', is_leaf_fn=None):
        if strategy == "preorder":
            return self._iter_descendants_preorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "levelorder":
            return self._iter_descendants_levelorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "postorder":
            return self._iter_descendants_postorder(is_leaf_fn=is_leaf_fn)


class Particles:
    def __init__(self, tree, num_particles, forward=True):
        """
        Calculate the estimated log-likelihood with particle filtering
        :param tree:
        :param num_particles: number of particles
        :param forward: boolean, if True, then will look forward one step to calculate weight
        :return: estimated log-likelihood p(data|jps)
        """
        self.root = ParticleNode(newick=tree.root.write(format=1))

        self.n = num_particles  # number of particles
        self.forward = forward
        self.depend = tree.depend
        self.base = tree.base
        self.tree = tree

        # get nodes & update info according to tree (name, njump, rests)
        self.nodes = {}
        for n, n0 in zip(self.root.traverse(), tree.root.traverse()):
            n.name = n0.name
            n.njump = n0.njump
            n.rests = [n0.rest.deep_copy() for _ in range(num_particles)] if n.njump > 0 else n.up.rests
            self.nodes[n.name] = n

    def forward_weights(self, new_node=None, new_obs=None):
        if new_node is None:
            return [1.] * self.n
        else:
            return [new_node.rests[i].p_key(new_obs) for i in range(self.n)]

    def get_particle(self, i=0):
        i = i % self.n
        for n0, n in zip(self.tree.root.traverse(), self.root.traverse()):
            n0.rest = n.rests[i]
        return self.tree

    def particle_filter(self, data):  # out_log=False, log=None
        """
        Run whole particle filtering step with given data
        :param data: pandas DataFrame of data
        # :param out_log: todo boolean, if True, will print rests and weights after each update
        # :param log_file: todo str of None, the output log file (default: out_log.txt)
        :return: estimated log-likelihood
        """
        loglik = np.log(self.base[data['obs'][0]]) if self.forward else 0.
        ndt = data.shape[0]

        for idx in range(ndt):  # idx = 0
            node_name, obs = data.iloc[idx]
            node = self.nodes[node_name]

            if self.forward:
                new_node_name, new_obs = data.iloc[idx+1] if (idx + 1) < ndt else (None, None)
                new_node = self.nodes[new_node_name] if new_node_name is not None else None

                node.seat_new_obs(obs, depend=self.depend)
                weights = self.forward_weights(new_node, new_obs)
            else:
                ks = node.seat_new_obs(depend=self.depend)
                weights = [int(k == obs) for k in ks]

            avg_weight = sum(weights) / self.n
            loglik += np.log(avg_weight)

            # resample the particles
            weights = [w / sum(weights) for w in weights]
            indices = np.random.choice(self.n, self.n, replace=True, p=weights)
            for n in self.root.traverse():
                if n.njump == 0:
                    n.rests = n.up.rests
                else:
                    rests = [n.rests[i].deep_copy() for i in indices]
                    n.rests = rests

        return loglik
