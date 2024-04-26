"""

PhyloHPYP: categorical.py

Created on 4/27/18 3:32 PM

@author: Hanxi Sun

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class Categorical(dict):
    """
    Categorical probability distribution (dict of label:prob)
    """

    def __init__(self, labels=None, probs=None, K: int = None, normalize=True):
        super().__init__()
        self._valid = False  # whether it is normalized
        if labels is not None or probs is not None:
            if probs is None:
                for k in labels:
                    self[k] = 1. / len(labels)
            elif labels is None:
                for k, v in enumerate(probs):
                    self[k] = v
            else:
                for k, v in zip(labels, probs):
                    self[k] = v
        elif K is not None:
            assert K > 0
            for k in range(K):
                self[k] = 1. / K

        if normalize:
            self.normalize()
            self._valid = True
        else:
            self._valid = (sum(self.values()) == 1)

    def __getitem__(self, k):
        return super(Categorical, self).__getitem__(k) if k in self else 0  # returns 0 for a non-existing key

    def __setitem__(self, k, v):
        if v < 0:
            raise ValueError("Probs can be un-normalized, but should be non-negative.")
        super().__setitem__(k, v)
        self._valid = False

    def get_total_p(self):
        return sum([v for v in self.values()])

    def is_valid(self):
        """
        Test if the probability measure is valid (normalized).
        :return: True if normalized, False otherwise.
        """
        return self._valid

    def normalize(self):
        """
        normalize the probability measure
        :return: None
        """
        if not self.is_valid():
            total_p = self.get_total_p()
            for k, v in self.items():
                self[k] = v/total_p
            self._valid = True

    def sample(self, n=1):
        """
        Sample n keys from the distribution.
        :param n: number of samples.
        :return: a list of n sampled keys (or only one sample when n=1).
        """
        if not type(n) == int:
            raise TypeError("The number of samples, n ({}), should be an integer. ".format(n))
        if n <= 0:
            raise ValueError("The number of samples, n ({}), should be positive.".format(n))

        if not self.is_valid():
            self.normalize()
        p = list(self.values())
        k = list(self.keys())
        idx = np.random.choice(len(k), size=n, p=p)
        if n == 1:
            s = k[idx[0]]
        else:
            s = [k[i] for i in idx]
        return s
