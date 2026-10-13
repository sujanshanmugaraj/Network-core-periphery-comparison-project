"""Agreement measures between node-score vectors (RQ1)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score

from .robustness import random_tie_order


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def top_set(s, x: float) -> set:
    """Tie-inclusive top-x set: every node scoring at least the ceil(x N)-th largest value.
    Identical rankings therefore always give identical sets, whatever the tie structure.
    The lowest tier is never included (e.g. for a binary core/periphery labelling whose
    core is smaller than x N, the top set is the core)."""
    s = np.asarray(s, float)
    k = max(1, int(np.ceil(x * s.size)))
    thr = np.sort(s)[::-1][k - 1]
    if thr == s.min() and np.any(s > thr):
        return set(np.flatnonzero(s > thr))
    return set(np.flatnonzero(s >= thr))


def topx_jaccard(a, b, x: float, n_draws: int = 50, seed: int = 0) -> float:
    """Jaccard overlap of the tie-inclusive top-x sets (see ``top_set``)."""
    A, B = top_set(a, x), top_set(b, x)
    return len(A & B) / len(A | B)


def topx_jaccard_random_ties(a, b, x: float, n_draws: int = 50, seed: int = 0) -> float:
    """Expected Jaccard overlap of exactly-sized top-x sets with ties broken uniformly at
    random (sensitivity analysis; heavily penalises large tied shells)."""
    n = len(a)
    k = max(1, int(round(x * n)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        A = set(random_tie_order(a, rng)[:k])
        B = set(random_tie_order(b, rng)[:k])
        vals.append(len(A & B) / len(A | B))
    return float(np.mean(vals))


def ami(labels_a, labels_b) -> float:
    return float(adjusted_mutual_info_score(np.asarray(labels_a, int), np.asarray(labels_b, int)))


def natural_core_labels(method: str, scores, be_core_size: int):
    """Binary core/periphery labelling that is natural for each method.

    k-core   : nodes in the main core (coreness == degeneracy)
    cohesion : nodes in the most cohesive k-components (level == max level)
    BE       : the optimised discrete labelling
    Rombach  : top-|BE core| nodes by aggregate core score (size matched to BE)
    degree   : top-|BE core| nodes by degree (reference)"""
    s = np.asarray(scores, float)
    if method in ("kcore", "cohesion"):
        return (s == s.max()).astype(int)
    if method == "be":
        return (s > 0.5).astype(int)
    order = np.argsort(-s, kind="stable")
    lab = np.zeros(s.size, int)
    lab[order[:be_core_size]] = 1
    return lab
