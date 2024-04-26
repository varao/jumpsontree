"""

PhyloHPYP: rest.py

Created on 4/27/18 10:52 PM

@author: Hanxi Sun

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from src.categorical import Categorical
import numpy as np


class TblCx(object):
    """
    Tables (Tbl) and Customers (Cx) under the same label within a restaurant
    """
    def __init__(self, nt=0, nc=0):
        self._nt = nt  # number of tables
        self._nc = nc  # number of customers

    def add_customer(self, new_table=False):
        self._nc += 1
        if self._nt == 0 or new_table:
            self._nt += 1

    def _get_nt(self):
        return self._nt

    def _get_nc(self):
        return self._nc

    nt = property(fget=_get_nt)
    nc = property(fget=_get_nc)

    def __repr__(self):
        return "({}, {})".format(self.nt, self.nc)

    def __str__(self):
        return "({}, {})".format(self.nt, self.nc)

    def deep_copy(self):
        other = type(self)()
        other._nc = self.nc
        other._nt = self.nt
        return other


class Rest(dict):
    """
    Chinese Restaurant Processes of Pitman-Yor processes.
    The main object is a dictionary of {key:value},
    where key is the label and value is a TblCx object assigned to its key.
    """

    _NEW_TABLE = '__NEW_TABLE__'  # flag for a new table

    def __init__(self, disc=0., conc=0., base: Categorical = None):
        """
        Initialize the restaurant
        :param disc: Pitman-Yor discount parameter
        :param conc: Pitman-Yor concentration parameter
        :param base: a Categorical which represents the base measure
        """
        self._disc, self._conc = self.check_parameters(disc, conc)
        self._nt = 0   # number of tables
        self._nc = 0   # number of customers
        self._base = None
        # self._pm = None  # posterior measure (from the stick breaking)
        if base is not None:
            self.base = base
        super().__init__()

    def __getitem__(self, k):
        return super(Rest, self).__getitem__(k) if k in self else TblCx()

    ######################
    #     properties     #
    ######################

    def _get_disc(self):
        return self._disc

    def _get_conc(self):
        return self._conc

    def _get_nt(self):
        return self._nt

    def _get_nc(self):
        return self._nc

    def _get_base(self):
        return self._base

    def _set_base(self, value: Categorical = None):
        if value is None:
            self._base = None
        elif not type(value) == Categorical:
            raise TypeError("Base measure should be of type Categorical. ")
        else:
            value.normalize()
            self._base = value

    def _get_pm(self, n_atom=50):
        # if self._pm is None:
        #     self._pm = self.post_measure(n_atom=n_atom)
        # return self._pm
        return self.post_measure(n_atom=n_atom)

    disc = property(fget=_get_disc)
    conc = property(fget=_get_conc)
    nt = property(fget=_get_nt)
    nc = property(fget=_get_nc)
    base = property(fget=_get_base, fset=_set_base)
    pm = property(fget=_get_pm)

    #####################
    #     utilities     #
    #####################

    @property
    def NEW_TABLE(self):
        return type(self)._NEW_TABLE

    @staticmethod
    def check_parameters(d, c):
        """
        Check if the discount and concentration parameters are valid
        :param d: the discount parameter
        :param c: the concentration parameters
        :return d, c
        """
        if c == 0 and d == 0:
            warnings.warn("Restaurant with discount=0 and concentration=0 created. ")
            pass
        elif d >= 1 or d < 0:
            raise ValueError("The discount parameter, {}, should be in [0,1). ".format(d))
        elif c + d <= 0:
            raise ValueError("The concentration parameter ({}) and ".format(c) +
                             "the discount parameter ({}) should sum up tp a positive number.".format(d))
        return d, c

    def _p_tbl_numerator(self, k):
        """
        Calculate the numerator for p_tbl(k). Only used in p_tbl()
        :param k: label
        :return: the numerator for p_tbl(k)
        """
        if k == self.NEW_TABLE:
            return self.conc + self.disc * self.nt if self.conc + self.nc > 0 else 1
        else:
            return self[k].nc - self.disc * self[k].nt  # = 0 when self[k] = (0, 0)

    def is_place_holder(self):
        return self.disc == 0 and self.conc == 0

    def is_empty(self):
        return self.nc == 0 and self.nt == 0

    def empty(self):
        self._nc = 0
        self._nt = 0
        self.clear()

    def deep_copy(self):
        other = type(self)(disc=self.disc, conc=self.conc, base=self.base)
        other._nc = self.nc
        other._nt = self.nt
        for k in self.keys():
            other[k] = self[k].deep_copy()
        return other

    def check_base(self):
        if self.base is None:
            raise AttributeError("Base measure undefined for restaurant. ")

    # def assign(self, dic):  # todo: use this to get particles from the stored rests, etc.
    #     for k in dic.keys():
    #         self[k]

    #####################
    #     functions     #
    #####################

    def add_customer(self, k, new_table=False):
        """
        Add new customer in the restaurant, with table label k
        :param k: table label
        :param new_table: if True, a new table will be created for the customer.
                          Default to be False.
        :return: None
        """
        if k not in self:
            self[k] = TblCx()  # empty TblCx for k
            new_table = True
        if new_table:
            self._nt += 1
        self._nc += 1
        self[k].add_customer(new_table)

    def p_tbl(self, k=None):
        """
        Get the probability of sit in an existing table with label k,
        or a new table (when k indicates NEW_TABLE)
        :param k: table label, or NEW_TABLE. If None, then returns a
                  Categorical object of all table probabilities.
        :return: the probability(ies).
        """
        if k is None:
            p_t = Categorical()
            p_t[self.NEW_TABLE] = self._p_tbl_numerator(self.NEW_TABLE)
            for k in self:
                p_t[k] = self._p_tbl_numerator(k)
            p_t.normalize()
            return p_t

        else:
            # denominator
            den = self.nc + self.conc
            den = 1 if den == 0 else den

            return self._p_tbl_numerator(k) / den

    def p_key(self, k=None):
        """
        Get the probability of a new customer being labelled k
        (the new table case with base measure is integrated out).
        Needs a valid BASE measure.
        :param k: the key (label), if None, then returns as a
                  Categorical object of all key probabilities.
        :return: the probability(ies)
        """
        self.check_base()

        if k is None:
            p_t = self.p_tbl()
            p_k = Categorical()

            for k in self.base:
                p_k[k] = p_t[k] + p_t[self.NEW_TABLE] * self.base[k]

            p_k.normalize()
            return p_k

        else:
            if k == self.NEW_TABLE:
                raise ValueError("The key/label (k) cannot be the new table flag, {} ".format(self.NEW_TABLE))
            else:
                return self.p_tbl(k) + self.p_tbl(self.NEW_TABLE) * self.base[k]

    def post_p_key(self, k):
        """
        Get the probability of a new customer sits in a previous
        table labelled k, and the probability of open a new table
        that will be labelled k (the posterior probability of
        seating at table with label k). Needs a valid base measure.
        :param k: the table label
        :return: a Categorical object of {k:p(existing tbl of k), NEW_TABLE:p(new tbl k)}
        """
        self.check_base()

        p_t = Categorical()
        p_t[k] = self.p_tbl(k)
        p_t[self.NEW_TABLE] = self.p_key(k) - p_t[k]
        p_t.normalize()
        return p_t

    def seat_new_customer(self, obs=None):
        """
        Seat a new customer in restaurant with label k when k is given, otherwise
        randomly seat a new customer in the restaurant. Needs a valid base measure.
        :param obs: customer label, if None, then seat a new customer according
                    to the generative process of hierarchical Pitman-Yor process.
        :return: Boolean: if a new table is created
        """
        self.check_base()

        p_t = self.p_tbl() if obs is None else self.post_p_key(obs)

        k = p_t.sample()
        new_table = (k == self.NEW_TABLE)

        if new_table:
            k = self.base.sample() if obs is None else obs

        self.add_customer(k, new_table=new_table)

        return new_table, k

    def post_measure(self, n_atom=50, threshold=.1, MAX=1000):
        # self = ps[0].root.rest.deep_copy()
        if n_atom is None:
            n_atom = 50
        if self.disc > .95:  # put an .1 threshold on it
            n_atom = min(int(- np.log(threshold) / (1 - self.disc)), MAX)
        pk = self.p_key()
        atoms = np.array(pk.sample(n=n_atom))
        # given the existing configuration of the customers, it is still a Pitman-Yor (consider the sequential sampling)
        # i.e. PY(d, c, H) -> PY(d, c1, H1)
        # H1 = weighted average of H & empirical distr. of current configuration of the customers
        # c1 = c + number_of_customers (i.e. nc)
        pi1 = np.random.beta(1 - self.disc, self.nc + (np.arange(n_atom)+1) * self.disc)
        pi = np.zeros(n_atom)
        s = 0
        for i in range(n_atom):
            pi[i] = pi1[i] * (1 - s)
            s += pi[i]
        # print(sum(pi))
        pi /= np.sum(pi)
        post = Categorical()
        for k in pk.keys():
            post[k] = float(np.sum(pi[atoms == k]))
        # print(post)
        return post


########################################################################################################################

# # from rest import Rest
# from categorical import Categorical
# # parent {0: (2, 36), 1: (1, 1)} base {0: 0.5, 1: 0.5}
# rp = Rest(disc=.5, base=Categorical.uniform(2))
# rp.add_customer(0)
# rp.add_customer(1)
# rp.add_customer(0, True)
# for _ in range(36-2):
#     rp.add_customer(0, False)
#
# rp.post_measure()
#

# child {1: (1, 9), 0: (2, 3)} base {0: 1.0, 1: 0.0}

########################################################################
# digamma function

# from scipy.special import digamma
# ds = 1-1e-8
# s = 0.
# i = 0
# while np.exp(s) > .05:
#     i += 1
#     v = i * (digamma(i * ds) - digamma(1 + (i - 1) * ds))
#     # if i < 10:
#     print("{:5d}: ".format(i), v)
#     s += v
# print(i, v)
# n_atom = (1 + i // 50) * 50
# n_atom = int(2.5*1e5)
# r = Rest(disc=ds, base=Categorical.uniform(2))
# pk = r.p_key()
# atoms = np.array(pk.sample(n=n_atom))
# pi1 = np.random.beta(1 - r.disc, r.nc + (np.arange(n_atom)+1) * r.disc)
# pi = np.zeros(n_atom)
# s = 0
# for i in range(n_atom):
#     pi[i] = pi1[i] * (1 - s)
#     s += pi[i]
# print(sum(pi))
# pi /= np.sum(pi)
# post = Categorical()
# for k in pk.keys():
#     post[k] = float(np.sum(pi[atoms == k]))
# print(post)
