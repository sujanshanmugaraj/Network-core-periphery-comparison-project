"""Core–periphery scores: k-core coreness, discrete Borgatti–Everett, Rombach et al. (2014).

All functions take a graph whose nodes are 0..N-1 (see datasets.clean_graph) and
return a numpy array ``score[v]`` (higher = more core).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from numba import njit

from .datasets import to_csr


# ============================================================ k-core coreness
def kcore_scores(G: nx.Graph) -> np.ndarray:
    cn = nx.core_number(G)
    return np.array([cn[v] for v in range(G.number_of_nodes())], dtype=float)


# ============================================================ Borgatti–Everett (discrete)
@njit(cache=True)
def _be_corr(M, E, n_p, E_pp):
    """Pearson correlation between the adjacency (upper triangle) and the ideal pattern
    delta_ij = 1 unless both i and j are periphery."""
    P = M - n_p * (n_p - 1) / 2.0  # number of ones in the pattern
    if P <= 0.0 or P >= M:
        return -2.0
    S = E - E_pp  # sum_ij a_ij * delta_ij
    num = M * S - E * P
    den = np.sqrt((M * E - E * E) * (M * P - P * P))
    return num / den


@njit(cache=True)
def _be_greedy(indptr, indices, core0, order_seed, max_passes):
    n = indptr.size - 1
    M = n * (n - 1) / 2.0
    E = indices.size / 2.0
    core = core0.copy()
    # kp[v] = number of periphery neighbours of v
    kp = np.zeros(n, dtype=np.int64)
    n_p = 0
    for v in range(n):
        if not core[v]:
            n_p += 1
        for j in range(indptr[v], indptr[v + 1]):
            if not core[indices[j]]:
                kp[v] += 1
    E_pp = 0.0
    for v in range(n):
        if not core[v]:
            E_pp += kp[v]
    E_pp /= 2.0
    r = _be_corr(M, E, n_p, E_pp)
    np.random.seed(order_seed)
    for _ in range(max_passes):
        changed = False
        perm = np.random.permutation(n)
        for v in perm:
            if core[v]:  # core -> periphery
                n_p2 = n_p + 1
                E_pp2 = E_pp + kp[v]
            else:  # periphery -> core
                n_p2 = n_p - 1
                E_pp2 = E_pp - kp[v]
            r2 = _be_corr(M, E, n_p2, E_pp2)
            if r2 > r + 1e-12:
                delta = 1 if core[v] else -1  # neighbours gain/lose a periphery neighbour
                core[v] = not core[v]
                for j in range(indptr[v], indptr[v + 1]):
                    kp[indices[j]] += delta
                n_p, E_pp, r = n_p2, E_pp2, r2
                changed = True
        if not changed:
            break
    return core, r


def borgatti_everett(G: nx.Graph, n_restarts: int = 20, init_frac: float = 0.10, seed: int = 0):
    """Greedy label switching maximising the BE correlation.

    Initial core = top ``init_frac`` of nodes by degree; each restart visits nodes in a
    different random order; the best of ``n_restarts`` runs is returned.
    Returns (core_labels: bool array, rho: float)."""
    indptr, indices = to_csr(G)
    n = G.number_of_nodes()
    deg = np.diff(indptr)
    n_core = max(1, int(round(init_frac * n)))
    core0 = np.zeros(n, dtype=np.bool_)
    core0[np.lexsort((np.arange(n), -deg))[:n_core]] = True
    best, best_r = None, -np.inf
    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        core, r = _be_greedy(indptr, indices, core0, int(rng.integers(2**31 - 1)), 1000)
        if r > best_r:
            best, best_r = core, r
    return best.astype(bool), float(best_r)


def be_scores(G: nx.Graph, **kw) -> np.ndarray:
    core, _ = borgatti_everett(G, **kw)
    return core.astype(float)


# ============================================================ Rombach et al. (2014)
def rombach_profile(n: int, alpha: float, beta: float) -> np.ndarray:
    """Sharp transition function, Eq. (9) of Rombach et al. (2014), normalised to sum 1.

    Positions i = 1..N; the ⌊βN⌋ lowest positions are periphery, the rest core."""
    i = np.arange(1, n + 1, dtype=float)
    b = int(np.floor(beta * n))
    c = np.empty(n)
    if b > 0:
        c[:b] = i[:b] * (1 - alpha) / (2 * b)
    c[b:] = (i[b:] - b) * (1 - alpha) / (2 * (n - b)) + (1 + alpha) / 2
    return c / c.sum()


@njit(cache=True)
def _has_edge(indptr, indices, u, v):
    lo, hi = indptr[u], indptr[u + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        x = indices[mid]
        if x == v:
            return True
        if x < v:
            lo = mid + 1
        else:
            hi = mid
    return False


@njit(cache=True)
def _rombach_sa(indptr, indices, cstar, init_pos, n_iter, seed):
    """Simulated annealing over assignments C = cstar[pos] maximising R = C^T A C."""
    np.random.seed(seed)
    n = indptr.size - 1
    C = np.empty(n)
    for v in range(n):
        C[v] = cstar[init_pos[v]]
    s = np.zeros(n)  # s = A C
    for v in range(n):
        acc = 0.0
        for j in range(indptr[v], indptr[v + 1]):
            acc += C[indices[j]]
        s[v] = acc
    R = 0.0
    for v in range(n):
        R += C[v] * s[v]
    # temperature scale from typical |dR|
    typ = 0.0
    cnt = 0
    for _ in range(2000):
        u = np.random.randint(n)
        v = np.random.randint(n)
        du = C[v] - C[u]
        if du == 0.0:
            continue
        a = 1.0 if _has_edge(indptr, indices, u, v) else 0.0
        typ += abs(2 * du * (s[u] - s[v]) - 2 * a * du * du)
        cnt += 1
    T0 = typ / max(cnt, 1)
    if T0 <= 0:
        T0 = 1e-12
    T_end = T0 * 1e-6
    cool = (T_end / T0) ** (1.0 / n_iter)
    T = T0
    best_R = R
    best_C = C.copy()
    for _ in range(n_iter):
        u = np.random.randint(n)
        v = np.random.randint(n)
        du = C[v] - C[u]
        if du != 0.0:
            a = 1.0 if _has_edge(indptr, indices, u, v) else 0.0
            dR = 2 * du * (s[u] - s[v]) - 2 * a * du * du
            if dR > 0 or np.random.random() < np.exp(dR / T):
                C[u] += du
                C[v] -= du
                for j in range(indptr[u], indptr[u + 1]):
                    s[indices[j]] += du
                for j in range(indptr[v], indptr[v + 1]):
                    s[indices[j]] -= du
                R += dR
                if R > best_R:
                    best_R = R
                    best_C[:] = C
        T *= cool
    # exact recomputation (removes floating drift)
    R = 0.0
    for v in range(n):
        acc = 0.0
        for j in range(indptr[v], indptr[v + 1]):
            acc += best_C[indices[j]]
        R += best_C[v] * acc
    return best_C, R


ALPHAS = np.round(np.linspace(0.0, 1.0, 6), 2)   # 0, .2, ..., 1
BETAS = np.round(np.linspace(0.1, 0.9, 9), 2)    # .1, .2, ..., .9


def rombach_scores(G: nx.Graph, alphas=ALPHAS, betas=BETAS, iters_per_node: int = 500,
                   min_iters: int = 300_000, seed: int = 0, return_details: bool = False):
    """Aggregate core score CS(i) = Z * sum_{alpha,beta} C_i(alpha,beta) R(alpha,beta),
    normalised so that max CS = 1 (Rombach et al. 2014, Eq. 11).

    Each (alpha, beta) is optimised by simulated annealing started from the
    degree ordering (highest degree -> highest core value)."""
    indptr, indices = to_csr(G)
    n = G.number_of_nodes()
    deg = np.diff(indptr)
    # position (0 = most peripheral) for degree-ordered start
    rank_order = np.lexsort((np.arange(n), deg))  # ascending degree
    init_pos = np.empty(n, dtype=np.int64)
    init_pos[rank_order] = np.arange(n)
    n_iter = max(int(iters_per_node * n), min_iters)
    rng = np.random.default_rng(seed)
    cs = np.zeros(n)
    details = []
    for a in alphas:
        for b in betas:
            cstar = rombach_profile(n, a, b)
            C, R = _rombach_sa(indptr, indices, cstar, init_pos, n_iter, int(rng.integers(2**31 - 1)))
            cs += C * R
            details.append((a, b, R))
    cs /= cs.max()
    return (cs, details) if return_details else cs


# ============================================================ dispatcher
def coreness_scores(G: nx.Graph, method: str, **kw) -> np.ndarray:
    method = method.lower()
    if method in ("kcore", "k-core", "coreness"):
        return kcore_scores(G)
    if method in ("be", "borgatti-everett"):
        return be_scores(G, **kw)
    if method in ("rombach", "rombach2014"):
        return rombach_scores(G, **kw)
    if method == "degree":
        return np.array([G.degree(v) for v in range(G.number_of_nodes())], dtype=float)
    if method == "cohesion":
        from .cohesion import cohesion_levels
        lv = cohesion_levels(G)
        return np.array([lv[v] for v in range(G.number_of_nodes())], dtype=float)
    raise ValueError(method)
