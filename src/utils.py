"""

PhyloHPYP: utils.py

Created on 6/12/18 5:54 PM

@author: Hanxi Sun

"""

from ete3 import Tree
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os

MAX_DIVS = 200

LOCAL_RUNS = "/Users/hanxi/Documents/Research/local_runs/PhylogenicSP_runs/"
LOCAL_DATA = LOCAL_RUNS + "Data/"
SERVER_RUNS = "/scratch/halstead/s/sun652/phyloHPYP/runs/"
SERVER_DATA = "/home/sun652/phyloHPYP/data/"


def _sync():
    print("rsync -zarv --include='*.py' --include='scripts/' --exclude='*' " +
          os.getcwd() + "/* sun652@halstead.rcac.purdue.edu:~/phyloHPYP/src/")


def now():
    n = datetime.now().strftime('%m%d_%H%M%S.%f').split('.')
    return "%s_%02d" % (n[0], int(n[1]) // 10000)


def today():
    return datetime.now().strftime('%m%d%y')


def setwd(wd):
    if wd[-1] != "/":
        wd += "/"
    if not os.path.exists(wd):
        os.makedirs(wd)
    return wd


def dump(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def newick_internal_nodes(newick, prefix="I"):
    idx, i = 0, 1
    # remove 0.0 for "root"
    if newick[-1] == "\n":
        newick = newick[:-1]
    if newick[-6:] == "):0.0;":
        newick = newick[:-5] + ";"
    while True:
        idx = newick.find("):", idx + 1)
        if idx == -1:
            break
        # Flag = True
        newick = newick[:idx] + ")" + prefix + "{}:".format(i) + newick[idx+2:]
        i += 1
    newick = newick[:-1] + "Root;"
    return newick.replace("'", "")


def newick_rename_nodes(newick):
    tr = Tree(newick, format=1)
    idx = 0
    for n in tr.traverse("preorder"):
        n.name = "n{}".format(idx)
        idx += 1
    return tr.write(format=3)


def newick_from_file(filename, rename=True):
    newick = open(filename, "r").read().replace('\n', '')
    if rename:
        newick = newick_rename_nodes(newick)  # print(tree)
    return newick


def data_to_treeBreaker(data: pd.DataFrame, file: str):
    data.to_csv(file, index=False, sep="\t", header=False)


def tree_to_treeBreaker(newick: str, tree_leaf_names: list):
    idx = idx0 = 0
    pieces = []
    while True:
        idx = newick.find(":", idx+1)
        if idx == -1:
            break
        idx1 = max(newick.rfind("(", 0, idx), newick.rfind(")", 0, idx)) + 1
        # print(newick[:idx1], newick[idx1:idx], newick[idx:], "", pieces, sep="\n")
        if newick[idx1:idx] not in tree_leaf_names:
            pieces.append(newick[idx0:idx1])
            idx0 = idx
    return "".join(pieces) + newick[newick.rfind(":"):]


def get_log(log_file=None, out_log=False, default="log.txt"):
    """
    Get the log stream to output program log.
    :param log_file: string of file name
    :param out_log: bool, if True, then open log_file, otherwise return None.
    :param default: the default log_file
    :return:
    """
    if log_file is None:
        log_file = default
    elif type(log_file) != str:
        raise TypeError("log_file should be of string type. ")
    log = open(log_file, 'a') if out_log else None
    return log


def equal_var_conc(disc):
    return 1 / (1 - disc) - 1


def equal_var_disc(conc):
    return 1 - 1 / (conc + 1)


def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(), return_index=True, return_counts=True)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def pl2nl(tree, tree_size, pl):
    """
    Get nl (number of leaves affected) from pl (proportion of leaves affected)
    """
    all_nl = np.array(sorted(set([n.nleaf for n in tree.nodes.values()])))
    nl = all_nl[np.argmin(np.abs(all_nl - (pl * tree_size)))]
    return nl


# ==================================================================================================================== #

def KL(p, c, child_base=True):
    # calculate KL(p, c) where p (parent), c (child) are Categorical objects
    if child_base:
        c0, c1 = c, p
    else:
        c0, c1 = p, c
    dist = 0
    c0.normalize()
    c1.normalize()
    for i in c0.keys():
        if c0[i] > 0:
            dist += c0[i] * (np.log(c0[i]) - np.log(c1[i]))
    if dist == np.inf:
        print(c0, c1)
    return dist


def L1(c1, c2):
    dist = 0
    c1.normalize()
    c2.normalize()
    for i in set(list(c2.keys()) + list(c1.keys())):
        dist += np.abs(c1[i] - c2[i])
    return dist  # todo make it total variation (i.e. /2)


def div(pa, ch, div_name):
    if div_name == "KLcb":
        return KL(pa, ch, child_base=True)
    elif div_name == "KLpb":
        return KL(pa, ch, child_base=False)
    elif div_name == "L1":
        return L1(pa, ch)
    else:
        raise ValueError("Divergence name not recognized. ")


# ==================================================================================================================== #
# ==================================================================================================================== #

def overlap_area(arr1, arr2, number_bins=50, arr1_weights=None, arr2_weights=None):
    if arr1_weights is None:
        arr1_weights = np.ones(len(arr1)) / len(arr1)
    if arr2_weights is None:
        arr2_weights = np.ones(len(arr2)) / len(arr2)
    # Determine the range over which the integration will occur
    min_value = np.max((arr1.min(), arr2.min()))
    max_value = np.min((arr1.max(), arr2.max()))
    max_value *= 1.001
    # Determine the bin width
    bin_width = (max_value-min_value)/number_bins
    # For each bin, find min frequency
    lower_bound = min_value  # Lower bound of the first bin is the min_value of both arrays
    min_arr = np.empty(number_bins)  # Array that will collect the min frequency in each bin
    for b in range(number_bins):
        higher_bound = lower_bound + bin_width  # Set the higher bound for the bin
        # Determine the share of samples in the interval
        samp_arr1 = ((arr1 >= lower_bound) & (arr1 < higher_bound))
        samp_arr2 = ((arr2 >= lower_bound) & (arr2 < higher_bound))
        # np.sum(samp_arr2)
        freq_arr1 = np.sum(samp_arr1 * arr1_weights)
        freq_arr2 = np.sum(samp_arr2 * arr2_weights)
        # freq_arr1 = np.ma.masked_where((arr1 < lower_bound) | (arr1 >= higher_bound), arr1).count()/len(arr1)
        # freq_arr2 = np.ma.masked_where((arr2 < lower_bound) | (arr2 >= higher_bound), arr2).count()/len(arr2)
        # Conserve the lower frequency
        min_arr[b] = np.min((freq_arr1, freq_arr2))
        # print(freq_arr1, freq_arr2)
        lower_bound = higher_bound  # To move to the next range
    return min_arr.sum()


# ==================================================================================================================== #

def info2np(info):
    # info.keys()
    for k in ['accepted', 'log_lik', 'log_acc', 'proposed', 'sampled']:
        info[k] = np.array(info[k])


# ==================================================================================================================== #
# ==================================================================================================================== #
# ==================================================================================================================== #
# ==================================================================================================================== #

def print_commands(exp="",
                   src_dir="sun652@halstead.rcac.purdue.edu:~/scratch/phyloHPY/src/",
                   script_dir="sun652@halstead.rcac.purdue.edu:~/scratch/phyloHPY/src/scripts",
                   run_dir="sun652@halstead.rcac.purdue.edu:~/scratch/phyloHPY/runs/"):
    # print_utils(EXP)
    print("scp " + (os.getcwd() + "/scripts/* " + script_dir) + '\n' +
          "scp " + (os.getcwd() + "/* " + src_dir) + '\n' +
          "scp " + (run_dir + exp + "* " +
                    os.getcwd() + "/runs/" + exp) + '\n' +
          "ssh sun652@halstead.rcac.purdue.edu" + '\n' +
          "module load anaconda/5.1.0-py36" + '\n' +
          "python ~/scratch/phyloHPY/src/scripts/run.py" + '\n'
          # "cd ~/scratch/phyloHPY/src/scripts/" + '\n' +
          # "cd ~/scratch/phyloHPY/runs/" + '\n' +
          # "qstat -u sun652" + '\n' +
          # "qselect -u sun652 | wc -l" + '\n' +  # number of jobs queuing / running
          # "find . -type f -size 0c -exec mv {} msg/ \;" + "\n")  # mv files with specific size
          )




