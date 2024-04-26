"""

phyloHPYP: fossil_data.py

Created on 12/29/20 4:04 PM

@author: Hanxi Sun

# ======================================================================================================================
Manuscript:
    Pierre-Olivier Antoine, et.al.
    A new rhinoceros clade from the Pleistocene of Asia sheds light on mammal dispersals to the Philippines

"""

import pandas as pd

from src import utils, plots
from src.restFranchise import RestFranchise

DATA = utils.LOCAL_DATA + "fossil/"
# from nexus import NexusReader
# n = NexusReader.from_file(DATA + 'temp_nex.txt')  # remove spacing in taxa names for this to work ...
# n = NexusReader.from_file(DATA + 'matrix philippinensis_v3.nex')

# ======================================================================================================================
# process the consensus tree (get the branch length and linkages table)
table = []  # Goal: [[node, parent, branch_length], ...]
names = []
checkpoints = 0
for line in open(DATA + 'matrix philippinensis_v3_contree.txt', 'r'):
    if checkpoints == 2 and not line.startswith('--'):  # inside the target table region
        while line.find(" " * 3) >= 0:
            line = line.replace(" " * 3, " " * 2)
        line = line.replace("  (", " (")
        line = line.strip()
        table.append(line.replace(" " * 2, ","))
    if checkpoints == 4:  # get full names
        if line[0].isdigit():  # passed the name section
            checkpoints += 1
        else:
            names.append(line[:line.find("  ")])
    elif checkpoints > 4:  # passed the target table region
        break

    if line.startswith('Branch lengths and linkages for tree #1 (unrooted)'):  # the line before the target table
        checkpoints = 1
    if line.startswith('--') and checkpoints > 0:  # passing the headers / endings of tables
        checkpoints += 1

# replace table partial names
for i, row in enumerate(table):  # i, row = 0, table[0]  # i, row = 2, table[2]
    r = row.split(',')[:3]
    if row[0].isdigit():
        table[i] = r
        continue
    partial = r[0].split('(')[0].lower().rstrip()
    for name in names:
        if name.lower().startswith(partial):
            table[i] = [name] + r[1:3]
            break

# address the 3-child root -- remove leaf node "Ronzotherium filholi"
# for i, row in enumerate(table):
#     if row[0] == "Ronzotherium filholi":
#         table.pop(i)
#         break


# ======================================================================================================================
# construct tree with the linkage table
def get_children(curr):
    # _i = 0
    children, branches = [], []  # children, children with branches
    # while _i < len(_table):
    #     if _table[_i][1] == curr:  # curr is the parent
    #         children.append(_table[_i][0])
    #         branches.append(_table[_i][0] + ":" + _table[_i][2])
    #         _table.pop(_i)
    #     else:
    #         _i += 1
    for link in table:
        if link[1] == curr:
            children.append(link[0])
            branches.append(link[0] + ":" + link[2])

    return children, branches


def get_subtree(curr):
    children, branches = get_children(curr)
    if len(children) == 0:
        return ''
    subtree = ",".join([get_subtree(c)+b for c, b in zip(children, branches)])
    return f"({subtree})"


root = '57'
newick = get_subtree(root) + root + ":1;"
open(DATA + "tree1.newick", 'w').write(newick)

tree = RestFranchise(newick=newick)
plots.annotated_tree(tree, file=DATA + "tree1.pdf", scale=1)


# ======================================================================================================================
# get data
# Table SI2, SI3
# data = [["Nesorhinus philippinensis", 1185],  # Table SI3
#         ["Nesorhinus philippinensis", 1140],  # Table SI3
#         ["Nesorhinus philippinensis", 1025],  # Table SI3
#         ["Nesorhinus philippinensis", 1018],  # Table SI3
#         ["Nesorhinus philippinensis", 1148],  # Table SI3
#         ["Nesorhinus philippinensis", 998],  # Table SI3
#         ["Nesorhinus philippinensis", 1140],  # Table SI3
#         ["Nesorhinus hayasakai", 1297],  # Table SI3
#         ["Nesorhinus hayasakai", 1337],  # Table SI3
#         ["Nesorhinus hayasakai", 1377],  # Table SI3
#         ["Nesorhinus hayasakai", 1018],  # Table SI3
#         ["Nesorhinus hayasakai", 1114],  # Table SI3
#         ["Nesorhinus hayasakai", 1215],  # Table SI3
#         ["Nesorhinus hayasakai", 1113],  # Table SI3
#         ["Nesorhinus hayasakai", 1373],  # Table SI3
#         ["Nesorhinus hayasakai", 1670],  # Table SI3
#         ["Nesorhinus hayasakai", 1119],  # Table SI3
#         ["Nesorhinus hayasakai", 1306],  # Table SI3
#         ["Nesorhinus philippinensis", 1025],  # Table SI2
#         ["Nesorhinus philippinensis", 1018],  # Table SI2
#         ["Ceratotherium simum",],
#         ["Diceros bicornis",],
#         ["Dicerorhinus sumatrensis",],
#         ["Rhinoceros sondaicus",],
#         ["Rhinoceros unicornis",],
#         [""]]
# ...
