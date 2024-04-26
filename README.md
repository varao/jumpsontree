# phyloHPYP
Hierarchical Pitman-Yor processes on tree-structured data, such as phylogenetic trees.

## Versions
-  v3.0
    - Determine jump location with posterior (median) estimation of clustering (posterior sample that minimizes balanced Binder's loss).
    - Use Bayes factor to determine whether there are jumps or not.
-  v2.4 (stable)
    - modify restFranchise to have multiple rests per node => a better way to represent particles
    - update particle filtering step that only modify part of the tree being affected by the proposed jumps
-  v2.3 (stable)
    - fully functional posterior inference based on divergence between the distributions at parent & child nodes
    - update the way to propose jumps (move jumps locally) => better mixing
    - study the prior distribution of divergence in `scripts/priorDivergence.py`  
- Cached versions (and scripts) are stored in the folder `cache`.
