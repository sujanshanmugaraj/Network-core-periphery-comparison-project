"""Paired comparisons across networks: Wilcoxon signed-rank, rank-biserial effect size,
bootstrap confidence intervals, Holm correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def rank_biserial(diff) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014): (W+ - W-)/(W+ + W-)."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_comparison(a, b, seed: int = 0) -> dict:
    """Compare method A against method B over networks (values: robustness R).

    diff = R_A - R_B; negative means A dismantles the networks faster."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d[np.abs(d) < 1e-12] = 0.0
    nz = d[d != 0]  # Wilcoxon's convention: zero differences are discarded
    if nz.size == 0:
        p = 1.0
    else:
        p = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided",
                           method="exact" if nz.size <= 25 else "approx").pvalue)
    lo, hi = bootstrap_ci(d, seed=seed)
    return {
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "ci_low": lo,
        "ci_high": hi,
        "wins_A": int((d < 0).sum()),
        "wins_B": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "wilcoxon_p": p,
        "rank_biserial": rank_biserial(d),
        "n": int(d.size),
        "n_nonzero": int(nz.size),
    }


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
