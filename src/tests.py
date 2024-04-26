"""

PhyloHPYP: tests.py

Created on 6/8/18 4:04 PM

@author: Hanxi Sun

"""

import numpy as np
from src.categorical import Categorical
from src.rest import TblCx, Rest
from src.restNode import RestNode
from src.restFranchise import RestFranchise


PREFIX = ["", " - ", "   - "]
ACCURACY = 1e-5
ACCURACY_LOW = 0.05


def log(level, name, test=None, message=""):
    print(("\n" if level == 0 else "") + PREFIX[level] + name + ": " +
          ("" if test is None else ("ok." if test else (message + "ERROR!"))))
    return test


def test_Categorical():
    #######################
    N = 1000
    threshold = 0.7
    #######################

    global_test = True

    log(0, "Test categorical.Categorical")

    #
    p = Categorical()
    p[0] = 1
    p[1] = 1
    p['a'] = 1
    p['b'] = 1
    test = (p.__str__() == "{0: 1, 1: 1, 'a': 1, 'b': 1}")
    global_test = global_test and log(1, "create Categorical", test)

    #
    test = (p[0] == 1 and p['a'] == 1 and p['c'] == 0)
    test = test and (p.__str__() == "{0: 1, 1: 1, 'a': 1, 'b': 1}")
    global_test = global_test and log(1, "call elements", test)

    #
    test = (not p.is_valid())
    p.normalize()
    test = test and (p.__str__() == "{0: 0.25, 1: 0.25, 'a': 0.25, 'b': 0.25}")
    test = test and (p.is_valid())
    global_test = global_test and log(1, "normalization", test)

    #
    counts = [0 for _ in [0, 1, 'a', 'b']]
    idx = {0: 0, 1: 1, 'a': 2, 'b': 3}
    samples = p.sample(N)
    for s in samples:
        counts[idx[s]] += 1/(2*N)
        s1 = p.sample()
        counts[idx[s1]] += 1/(2*N)
    diff = 0
    for c in counts:
        diff += np.abs(c - 0.25)
    test = (diff < threshold)
    global_test = global_test and log(1, "sampling", test)

    #
    p = Categorical.uniform(K=5)
    test = (p.__str__() == "{0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}")
    p = Categorical.uniform(K=5, labels=[1, 'a', 3, 4, 'b'])
    test = test and (p.__str__() == "{1: 0.2, 'a': 0.2, 3: 0.2, 4: 0.2, 'b': 0.2}")
    global_test = global_test and log(1, "uniform distribution", test)

    #######################
    log(0, "", global_test)
    return global_test


def test_TblCx():
    global_test = True
    log(0, "Test rest.TblCx")

    #
    tc = TblCx()
    test = (tc.__str__() == "(0, 0)")
    test = test and (tc.nc == 0 and tc.nt == 0)
    global_test = global_test and log(1, "initialization", test)

    #
    for i in range(3):
        tc.add_customer()
    for i in range(3):
        tc.add_customer(new_table=True)
    test = (tc.nc == 6 and tc.nt == 4)
    global_test = global_test and log(1, "add customer", test)

    #######################
    log(0, "", global_test)
    return global_test


def test_Rest():
    #######################
    N = 10000  # number of iterations
    disc = 0.1
    K = 4  # number of categories (for base)
    #######################

    global_test = True
    log(0, "Test rest.Rest")

    #
    test = True
    try:
        Rest(disc="1", conc="1")
        test = test and False
    except TypeError:
        test = test and True
    try:
        Rest(conc=-1)
        test = test and False
    except ValueError:
        test = test and True
    try:
        Rest(disc=-1)
        test = test and False
    except ValueError:
        test = test and True
    global_test = global_test and log(1, "check parameter (disc & conc)", test)

    #
    rest = Rest(disc=disc)
    rest.base = Categorical.uniform(K)
    test = (rest.disc == .1 and rest.conc == 0)
    test = test and (rest.base.__str__() == "{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}")
    global_test = global_test and log(1, "initialization", test)

    #
    rest.add_customer(0)
    rest.add_customer(1)
    rest.add_customer(2)
    rest.add_customer(3)
    rest.add_customer(1)
    rest.add_customer(1, new_table=True)
    rest.add_customer(1, new_table=True)
    test = (rest.__str__() == "{0: (1, 1), 1: (3, 4), 2: (1, 1), 3: (1, 1)}")
    global_test = global_test and log(1, "add customer", test)

    #
    rest1 = Rest(disc=0)
    test = rest1.is_place_holder()
    rest1 = Rest(disc=disc)
    test = (test and rest1.is_empty())
    rest1 = rest.deep_copy()
    rest1.empty()
    test = (test and rest1.is_empty())
    global_test = global_test and log(1, "utils", test)

    #
    rest1 = Rest(disc=disc)
    p_tbl = rest1.p_tbl()
    test = (p_tbl[rest.NEW_TABLE] == 1)
    p_tbl = rest.p_tbl()
    p = rest.nt * rest.disc / rest.nc
    test = test and ((abs(rest.p_tbl(rest.NEW_TABLE) - p) < ACCURACY and
                      abs(p_tbl[rest.NEW_TABLE] - p) < ACCURACY))
    for k in rest.base.keys():
        p = (rest[k].nc - rest[k].nt * rest.disc) / rest.nc if k in rest.keys() else 0
        test = test and (abs(rest.p_tbl(k) - p) < ACCURACY and abs(p_tbl[k] - p) < ACCURACY)
    global_test = global_test and log(1, "p_tbl", test)

    #
    test = True
    p_key = rest.p_key()
    for k in rest.base.keys():
        p = rest.p_tbl(k) + rest.p_tbl(rest.NEW_TABLE) * rest.base[k]
        test = test and (abs(rest.p_key(k) - p) < ACCURACY and abs(p_key[k] - p) < ACCURACY)
    global_test = global_test and log(1, "p_key", test)

    #
    test = True
    for k in rest.base.keys():
        post_p_key = rest.post_p_key(k)
        p_k = rest.p_tbl(k)
        p_new = rest.p_tbl(rest.NEW_TABLE) * rest.base[k]
        p_k, p_new = p_k/(p_k + p_new), p_new/(p_k + p_new)
        test = test and (abs(post_p_key[k] - p_k) < ACCURACY and abs(post_p_key[rest.NEW_TABLE] - p_new) < ACCURACY)
    global_test = global_test and log(1, "post_p_key", test)

    #
    p_tbl1 = [0] * (K+1)  # 0, 1, 2, 3, NEW_TABLE
    p_key1 = [0] * K  # 0, 1, 2, 3
    for i in range(N):
        rest1 = rest.deep_copy()
        new_table, k = rest1.seat_new_customer()
        p_key1[k] += 1/N
        if new_table:
            p_tbl1[K] += 1/N
        else:
            p_tbl1[k] += 1/N
    test = (abs(p_tbl1[K] - p_tbl[rest.NEW_TABLE]) < ACCURACY_LOW)
    for i in range(K):
        test = test and (abs(p_tbl1[i] - p_tbl[i]) < ACCURACY_LOW) and (abs(p_key1[i] - p_key[i]) < ACCURACY_LOW)
    global_test = global_test and log(1, "seat_new_customer", test)

    #######################
    log(0, "", global_test)
    return global_test


def test_RestNode():
    #######################
    jr = 2  # jump rate
    N = 1000  # number of iterations
    newick1 = ("(Bovine:0.69395,(Gibbon:0.36079," +
               "(Orang:0.33636,(Gorilla:0.17147,(Chimp:0.19268," +
               "Human:0.11927)C-H:0.08386)G-CH:0.06124)O-GCH:0.15057)" +
               "G-OGCH:0.54939,Mouse:1.21460)B-M:0.10;")
    bls = [0.10, 0.69395, 0.54939, 0.36079, 0.15057, 0.33636,
           0.06124, 0.17147, 0.08386, 0.19268, 0.11927, 1.21460]
    names = ['B-M', 'Bovine', 'G-OGCH', 'Mouse', 'Gibbon', 'O-GCH',
             'Orang', 'G-CH', 'Gorilla', 'C-H', 'Chimp', 'Human']
    nleaf = [7, 1, 5, 1, 1, 4, 1, 3, 1, 2, 1, 1]

    newick2 = "(A:0.3,(B:0.3,C:0.3)BC:0.1)Root;"
    disc = 0.7
    labels = ['1', '2', '3']
    base = Categorical.uniform(len(labels), labels)
    #######################

    global_test = True

    log(0, "Test RestNode")

    #
    log(1, "Initialization")

    ###
    root = RestNode(newick1)
    test = (root.str_print(tree=True) == ('\n   /-Bovine\n  |\n  |      ' +
                                          '/-Gibbon\n  |-G-OGCH\n  |     |     ' +
                                          '/-Orang\n  |      \\O-GCH\n-B-M         |    ' +
                                          '/-Gorilla\n  |           \\G-CH\n  |              |   ' +
                                          '/-Chimp\n  |               \\C-H\n  |                  ' +
                                          '\\-Human\n  |\n   \\-Mouse'))
    nb = root.nb
    tl = root.tl
    test = test and (nb == 11) and (root.bls == bls) and (abs(tl - sum(bls)) < ACCURACY)
    jps = root.jps
    tl1, nb1, bls1, jps1 = root.all_saved_subtree_properties()
    test = test and (nb1 == nb) and (bls1 == bls) and (tl1 == tl)
    for n, i in zip(root.traverse(), range(nb+1)):
        test = test and ((n.name == names[i]) and (n.rest is None) and
                         (n.njump == 0) and (jps[i] == 0) and (jps1[i] == 0) and
                         n.nleaf == nleaf[i])
    global_test = global_test and log(2, "tree initialization", test)

    ###
    root.init_observed_nodes()
    observed = root.get_observed()
    test = (list(observed.keys()) == ['Bovine', 'Gibbon', 'Orang', 'Gorilla', 'Chimp', 'Human', 'Mouse'])
    global_test = global_test and log(2, "observed nodes initialization", test)

    #
    log(1, "Jumps Management")

    ###
    avg = 0
    root.full_jps()
    test = (root.jps == [root.ROOT_JUMP] + [1] * 11)
    root.full_jps(2)
    test = test & (root.jps == [root.ROOT_JUMP] + [2] * 11)
    for i in range(N):
        root.poisson_jps(jump_rate=jr)
        avg += (sum(root.jps) - root.ROOT_JUMP)/N
    test = test and (abs(avg/(tl - root.dist) - jr) < ACCURACY_LOW)
    global_test = global_test and log(2, "jump sampling", test)

    ###
    jps = [root.ROOT_JUMP, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    root.jps = jps
    jp_root = root.deep_copy()
    ref_name = jp_root.jps_prune()
    test = (ref_name == {'Bovine': 'Gibbon', 'Gibbon': 'Gibbon', 'Orang': 'Gorilla', 'Gorilla': 'Gorilla',
                         'Chimp': 'Human', 'Human': 'Human', 'Mouse': 'Mouse'})
    test = test and (jp_root.str_print(tree=True) == '\n      /-Mouse\n-Gibbon\n      \\Gorilla-Human')
    global_test = global_test and log(2, "jps prune", test)

    ###
    # root.full_jps(0)
    # nodes = root.nodes
    # counts = np.zeros(nb)
    # for i in range(N):
    #     # _, idx = root.mcmc_propose_jps(jump_rate=jr, nb=nb, nodes=nodes)
    #     counts[idx-1] += 1/N
    # test = (np.mean(np.abs(counts - 1/nb)) < ACCURACY_LOW)
    # global_test = global_test and log(2, "mcmc propose jps", test)

    #
    log(1, "restaurants")

    ###
    root = RestNode(newick2)
    root.init_observed_nodes()
    jps = [root.ROOT_JUMP, 1, 0, 2, 0]
    root.jps = jps
    root.init_rests(disc=disc, conc=0, base=base)
    test = True
    for n in root.traverse():
        test = test and n.rest.is_empty()
        test = test and (n.rest.base == base)
    global_test = global_test and log(2, "restaurants initialization", test)

    ###
    root.init_rests(disc=disc, conc=0, base=base)
    nodes = root.nodes
    Root, A, BC, B, C = nodes["Root"], nodes["A"], nodes["BC"], nodes["B"], nodes["C"]
    [Root.rest.seat_new_customer() for _ in range(3)]
    Root.update_child_base()
    # print(root.str_print(tree=True, pruned=True, pruned_rests=True, bases=True))
    test = ((C.rest.p_key() == A.rest.base) and (C.rest.p_key() == B.rest.base))
    global_test = global_test and log(2, "restaurants update children bases", test)

    ###
    root.init_rests(disc=disc, conc=0, base=base)
    [A.seat_new_obs(depend=True) for _ in range(3)]
    [B.seat_new_obs(depend=True) for _ in range(1)]
    [C.seat_new_obs(depend=True) for _ in range(2)]
    C_nc, C_nt = C.rest.nc, C.rest.nt
    ratio = dict.fromkeys(labels + [A.rest.NEW_TABLE], 0)
    for i in range(N):
        tmp_root = root.deep_copy()
        tmp_observed = tmp_root.get_observed()
        tmp_A = tmp_observed["A"]
        tmp_C = tmp_observed["C"]
        obs = tmp_A.seat_new_obs(depend=True)
        ratio[obs] += 1/N
        if tmp_C.rest.nc > C_nc:
            ratio[A.rest.NEW_TABLE] += 1/N
    diff = 0
    for k in labels:
        diff += np.abs(A.rest.p_key(k) - ratio[k])
    diff += np.abs(A.rest.p_tbl(A.rest.NEW_TABLE) - ratio[A.rest.NEW_TABLE])
    test = (diff < ACCURACY_LOW * 2)
    global_test = global_test and log(2, "restaurants seat new observations", test)

    #######################
    log(0, "", global_test)
    return global_test


def test_RestFranchise():
    #######################
    disc, newick = 0.5, "(A:0.3,(B:0.3,C:0.3)I1:0.1)Root;"
    #######################

    global_test = True

    log(0, "Test RestFranchise")
    tree = RestFranchise(newick, disc=disc)
    print(tree)

    #######################
    log(0, "", global_test)
    return global_test


def test_util():
    pass


##################################
##################################
##################################

test_Categorical()
test_TblCx()
test_Rest()
test_RestNode()




















