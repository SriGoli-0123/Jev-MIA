"""First-order (single-fact) sensitivity features derived from the RIFT grids.

D_k = mean over pairs containing F_k and over replacements of [f(original) - f(F_k replaced)],
i.e. f(s11) - f(s01) when F_k is the pair's first fact and f(s11) - f(s10) when it is the second.
These answer "does changing one fact change the decision", the baseline that RIFT must beat.
"""
import numpy as np

FIRST_ORDER_FEATURES = ("D1", "D2", "D3")
# Full per-pair effects (pair-averaged over replacements), saved for analysis.
DELTA_FEATURES = tuple(f"{name}_F{i}F{j}" for i, j in ((1, 2), (1, 3), (2, 3))
                       for name in ("d_j_i1", "d_j_i0", "d_i_j1", "d_i_j0"))


def single_fact_effects(grid_rows):
    """``grid_rows``: dicts with keys i, j, d_i_j1, d_j_i1 (one per pair x replacement)."""
    effects = {1: [], 2: [], 3: []}
    for row in grid_rows:
        effects[row["i"]].append(row["d_i_j1"])
        effects[row["j"]].append(row["d_j_i1"])
    return {f"D{k}": float(np.mean(v)) if v else float("nan") for k, v in effects.items()}
