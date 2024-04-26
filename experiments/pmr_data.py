"""

phyloHPYP: pmr.py

Created on 12/28/20 8:11 AM

@author: Hanxi Sun

========================================================================================================================
cleaning data for PMR (post-marital residence)
Data: https://github.com/J-Moravec/pmr_language_simulation
Paper: https://www.sciencedirect.com/science/article/pii/S1090513817303835

"""

from tqdm import tqdm
import pandas as pd
import dendropy

from src import utils, plots
from src.restFranchise import RestFranchise

DATA = utils.LOCAL_DATA + "pmr/"

# https://matplotlib.org/mpl_examples/color/named_colors.hires.png
COLORS = ["purple", "royalblue", "firebrick", "darkgreen"]

ENCODING = {"ambilocal": 0, "patrilocal": 1, "matrilocal": 2, "neolocal": 3}
LABELS = ["A", "P", "M", "N"]
NAN_CODE = "-"

# clades & data coding
SPECIALS = {"UtoAztecan": (None,
                           {"sep": " ", "header": 0, "names": ["node_name", "obs"]}),
            "IndoEuropean": ({"VU": "ambilocal", "V": "patrilocal", "U": "matrilocal", "N": "neolocal"},
                             {"sep": "\t", "header": 0, "names": ["node_name", "obs", "obs1", "obs2"]}),
            "PamaNyungan": (None,
                            {"sep": " ", "header": 1, "names": ["node_name", "obs"]}),
            "Bantu": ({1: "patrilocal", 3: "matrilocal", 2: "neolocal"},
                      {"sep": "\t", "header": 0, "names": ["node_name", "obs"]}),
            "Austronesian": ({'MP': "ambilocal", 'P': "patrilocal", 'M': "matrilocal", 'N': "neolocal"},
                             {"sep": "\t", "header": 0, "names": ["node_name", "obs", "obs1", "obs2"]})}


for clade, special in tqdm(SPECIALS.items()):  # clade, coding = next(iter(CODINGS.items()))
    wd = DATA + clade + "/"
    coding, pd_kwargs = special

    # tree
    newick = dendropy.Tree.get(path=wd + f"sources/{clade.lower()}.trees",
                               schema="nexus", preserve_underscores=True).as_string(schema='newick').replace("'", "")
    tree = RestFranchise(newick=newick, labels=LABELS)

    # data
    data = pd.read_csv(wd + f"sources/{'original_coding_table' if clade == 'Austronesian' else 'residence'}.txt",
                       **pd_kwargs)
    data = data[["node_name", "obs"]]
    data = data[data.obs != NAN_CODE]
    if clade == "PamaNyungan":
        data.node_name = data.node_name.apply(lambda x: x.capitalize())

    # address inconsistency between data and tree
    data = data[data.node_name.isin([n.name for n in tree.leaves])]
    tree.root.prune(list(data.node_name))

    # encode observation labels
    if coding is not None:
        data.obs.replace(coding, inplace=True)
    data.obs.replace(ENCODING, inplace=True)

    # tree internal node name
    for i, n in enumerate(tree.traverse()):
        if not n.is_root() and not n.is_leaf():
            n.name = f"n{i}"

    # save
    data.to_csv(DATA + clade + ".csv", index=False)
    open(DATA + clade + ".newick", 'w').write(tree.write())

    plots.annotated_tree(tree=tree, data=data, colors=COLORS, circular=tree.nleaf > 50, file=DATA + clade + ".pdf")







