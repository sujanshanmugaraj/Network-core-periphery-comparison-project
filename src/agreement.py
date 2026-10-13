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


def topx_jaccard(a, b, x: float, n_draws: int = 50, seed: int = 0) -> float:
    """Expected Jaccard overlap of the top-x fraction sets, ties broken uniformly at
    random (averaged over ``n_draws`` draws).  Random tie-breaking avoids inflating
    agreement through a shared deterministic tie-breaker such as degree."""
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
