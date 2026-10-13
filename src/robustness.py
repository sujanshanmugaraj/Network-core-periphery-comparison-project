"""Robustness curves under (static) targeted node removal.

S(q): size of the largest connected component (as a fraction of the ORIGINAL N) after
removing the first q nodes of an ordering.  Robustness index R = (1/N) sum_{q=1}^{N} S(q)
(Schneider et al. 2011).  Lower R = the ordering identifies structurally critical nodes
better.  Curves are computed exactly in O(M α(N)) by adding nodes back in reverse order
with a union–find structure (Newman & Ziff 2000).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from numba import njit

from .datasets import to_csr


@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def _curve(indptr, indices, order):
    n = indptr.size - 1
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    present = np.zeros(n, dtype=np.bool_)
    largest_after = np.zeros(n + 1, dtype=np.int64)  # largest_after[q] = LCC of order[q:]
    best = 0
    for q in range(n - 1, -1, -1):
        v = order[q]
        present[v] = True
        if best < 1:
            best = 1
        for j in range(indptr[v], indptr[v + 1]):
            u = indices[j]
            if present[u]:
                ru, rv = _find(parent, u), _find(parent, v)
                if ru != rv:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    size[ru] += size[rv]
                    if size[ru] > best:
                        best = size[ru]
        largest_after[q] = best
    return largest_after


def robustness_curve(G: nx.Graph, order, csr=None):
    """Return (S, R): S[q-1] = LCC fraction after removing q nodes (q = 1..N)."""
    indptr, indices = csr if csr is not None else to_csr(G)
    order = np.asarray(order, dtype=np.int64)
    n = indptr.size - 1
    assert order.size == n and np.unique(order).size == n, "order must be a permutation"
    la = _curve(indptr, indices, order)
    S = la[1:] / n
    return S, float(S.sum() / n)


def robustness_curve_naive(G: nx.Graph, order):
    """Reference implementation (used in tests)."""
    H = G.copy()
    n = G.number_of_nodes()
    S = []
    for v in order:
        H.remove_node(v)
        S.append(max((len(c) for c in nx.connected_components(H)), default=0) / n)
    return np.array(S), sum(S) / n


# ------------------------------------------------------------------ orderings
def order_by(scores, tiebreak, rng=None):
    """Highest score first.  Ties broken by ``tiebreak`` (higher first); if ``rng`` is
    given, remaining ties are broken uniformly at random, otherwise by node id."""
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    last = rng.random(n) if rng is not None else -np.arange(n, dtype=float)
    if tiebreak is None:
        return np.lexsort((-last, -scores))
    return np.lexsort((-last, -np.asarray(tiebreak, dtype=float), -scores))


def random_tie_order(scores, rng):
    """Highest score first; all ties broken uniformly at random."""
    return order_by(scores, None, rng)
