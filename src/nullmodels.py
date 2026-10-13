"""Degree-preserving null model (Maslov–Sneppen double-edge swaps) in numba."""
from __future__ import annotations

import networkx as nx
import numpy as np
from numba import njit


@njit(cache=True)
def _swap(src, dst, keys_sorted_unused, n, nswap, max_tries, seed):
    """Double-edge swaps on an edge list, rejecting self-loops and multi-edges.

    Edge membership is tracked in a hash set implemented with a typed dict."""
    np.random.seed(seed)
    m = src.size
    edges = dict()
    for e in range(m):
        a, b = src[e], dst[e]
        if a > b:
            a, b = b, a
        edges[a * n + b] = 1
    done = 0
    tries = 0
    while done < nswap and tries < max_tries:
        tries += 1
        e1 = np.random.randint(m)
        e2 = np.random.randint(m)
        if e1 == e2:
            continue
        a, b = src[e1], dst[e1]
        c, d = src[e2], dst[e2]
        if np.random.random() < 0.5:
            c, d = d, c
        # new edges (a,d) and (c,b)
        if a == d or c == b or a == c or b == d:
            continue
        x1, y1 = (a, d) if a < d else (d, a)
        x2, y2 = (c, b) if c < b else (b, c)
        k1, k2 = x1 * n + y1, x2 * n + y2
        if k1 in edges or k2 in edges or k1 == k2:
            continue
        o1x, o1y = (a, b) if a < b else (b, a)
        o2x, o2y = (c, d) if c < d else (d, c)
        del edges[o1x * n + o1y]
        del edges[o2x * n + o2y]
        edges[k1] = 1
        edges[k2] = 1
        src[e1], dst[e1] = a, d
        src[e2], dst[e2] = c, b
        done += 1
    return src, dst, done


def degree_preserving_randomization(G: nx.Graph, swaps_per_edge: int = 10, seed: int = 0) -> nx.Graph:
    """Randomised copy of G with the same degree sequence (nswap = swaps_per_edge * M).

    The returned graph is restricted to its largest connected component, as for the
    empirical networks (swaps rarely disconnect more than a handful of nodes)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    E = np.array(list(G.edges()), dtype=np.int64)
    src, dst, done = _swap(E[:, 0].copy(), E[:, 1].copy(), None, n, swaps_per_edge * m,
                           100 * swaps_per_edge * m, seed)
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from(zip(src.tolist(), dst.tolist()))
    lcc = max(nx.connected_components(H), key=len)
    if len(lcc) < n:
        H = nx.convert_node_labels_to_integers(H.subgraph(lcc).copy(), ordering="sorted")
    return H
