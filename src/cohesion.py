"""Exact structural cohesion (Moody & White 2003): k-components and node cohesion levels.

Algorithm (exact, recursive cut-splitting with certificates)
-------------------------------------------------------------
A k-component is a maximal node set inducing a k-connected subgraph (with > k nodes).
For a candidate piece P we

1. prune P to its k-core (a k-connected graph has minimum degree >= k);
2. for every connected component C of the pruned piece, decide whether C is
   k-connected.  If it is, C is a (maximal) k-component and we recurse into C at
   level k+1.  Otherwise we obtain a vertex cut S with |S| < k and recurse on every
   piece D ∪ S, D a component of C − S.

Splitting is exact: a k-component K cannot be disconnected by |S| < k nodes, so K∖S
lies inside a single component D and therefore K ⊆ D ∪ S.  Hence no k-component is
ever destroyed, and a piece that is itself k-connected is necessarily maximal.

The k-connectivity test follows Esfahanian & Hakimi: with v a minimum-degree node,
C is k-connected iff κ(v, w) >= k for every non-neighbour w of v and κ(x, y) >= k for
every non-adjacent pair x, y of neighbours of v.  Most pairs are certified without any
max-flow computation, using two sufficient conditions that are proved in the paper:

* |N(x) ∩ N(y)| >= k  (k internally disjoint paths of length two);
* "bootstrap certification": starting from T = {x} ∪ N(x), repeatedly add any node
  with >= k neighbours in T.  Every node of T is k-connected to x, because a set S with
  |S| < k leaves at least one T-neighbour of every T-node outside S.

Only pairs not certified this way trigger a unit-capacity max-flow (NetworkX,
Edmonds–Karp with cutoff k).  The result has been verified against
``networkx.k_components`` and brute force (see tests/).
"""
from __future__ import annotations

from collections import deque

import networkx as nx
from networkx.algorithms.connectivity import (
    build_auxiliary_node_connectivity,
    local_node_connectivity,
    minimum_st_node_cut,
)
from networkx.algorithms.flow import build_residual_network


class _Certifier:
    """Bootstrap certification of κ(x, ·) >= k inside one piece (adjacency dict of sets)."""

    def __init__(self, adj: dict, x, k: int):
        self.adj, self.k = adj, k
        self.cert = {x}
        self.count: dict = {}
        self._queue = deque()
        for u in adj[x]:
            self.add(u)
        self._propagate()

    def add(self, w):
        if w in self.cert:
            return
        self.cert.add(w)
        self._queue.append(w)

    def _propagate(self):
        adj, k, cert, count, q = self.adj, self.k, self.cert, self.count, self._queue
        while q:
            w = q.popleft()
            for z in adj[w]:
                if z in cert:
                    continue
                c = count.get(z, 0) + 1
                count[z] = c
                if c >= k:
                    cert.add(z)
                    q.append(z)

    def verified(self, w):
        """Mark w as flow-verified and propagate."""
        self.add(w)
        self._propagate()


class _Flow:
    """Lazily built auxiliary digraph for local node connectivity in one piece."""

    def __init__(self, H: nx.Graph):
        self.H = H
        self.aux = None
        self.res = None
        self.n_flows = 0

    def _build(self):
        if self.aux is None:
            self.aux = build_auxiliary_node_connectivity(self.H)
            self.res = build_residual_network(self.aux, "capacity")

    def kappa_at_least(self, s, t, k) -> bool:
        self._build()
        self.n_flows += 1
        return local_node_connectivity(self.H, s, t, auxiliary=self.aux, residual=self.res, cutoff=k) >= k

    def cut(self, s, t) -> set:
        self._build()
        return set(minimum_st_node_cut(self.H, s, t, auxiliary=self.aux, residual=self.res))


def find_small_cut(H: nx.Graph, k: int, stats: dict | None = None):
    """Return a vertex cut of size < k of connected graph H, or None if H is k-connected.

    Assumes min degree of H >= k (so |H| > k)."""
    adj = {u: set(H.adj[u]) for u in H}
    flow = _Flow(H)
    v = min(adj, key=lambda u: len(adj[u]))
    Nv = adj[v]

    def ok_pair(a, b, certifiers):
        if len(adj[a] & adj[b]) >= k:
            return True
        cf = certifiers.get(a)
        if cf is None:
            cf = certifiers[a] = _Certifier(adj, a, k)
        if b in cf.cert:
            return True
        if flow.kappa_at_least(a, b, k):
            cf.verified(b)
            return True
        return False

    certifiers: dict = {}
    # Case 1: v outside the cut -> v must be k-connected to every non-neighbour.
    for w in adj:
        if w == v or w in Nv:
            continue
        if not ok_pair(v, w, certifiers):
            if stats is not None:
                stats["flows"] = stats.get("flows", 0) + flow.n_flows
            return flow.cut(v, w)
    # Case 2: v inside the cut -> two neighbours of v are separated.
    nbrs = sorted(Nv, key=lambda u: len(adj[u]))
    for i, x in enumerate(nbrs):
        for y in nbrs[i + 1:]:
            if y in adj[x]:
                continue
            if not ok_pair(x, y, certifiers):
                if stats is not None:
                    stats["flows"] = stats.get("flows", 0) + flow.n_flows
                return flow.cut(x, y)
    if stats is not None:
        stats["flows"] = stats.get("flows", 0) + flow.n_flows
    return None


def k_components_exact(G: nx.Graph, k_max: int | None = None, stats: dict | None = None) -> dict[int, list[frozenset]]:
    """All k-components of G for k = 1..(k_max or until none remain).

    Returns {k: [frozenset(nodes), ...]} (same format as networkx.k_components)."""
    out: dict[int, list[frozenset]] = {}
    comps = [c for c in nx.connected_components(G) if len(c) > 1]
    if not comps:
        return out
    out[1] = [frozenset(c) for c in comps]
    # level 2 = biconnected components with at least three nodes (linear time)
    stack = []
    for c in nx.biconnected_components(G):
        if len(c) > 2:
            out.setdefault(2, []).append(frozenset(c))
            stack.append((frozenset(c), 3))
    while stack:
        piece, k = stack.pop()
        if k_max is not None and k > k_max:
            continue
        H = nx.k_core(G.subgraph(piece), k)
        for cc in nx.connected_components(H):
            if len(cc) <= k:
                continue
            sub = H.subgraph(cc)
            cut = find_small_cut(sub, k, stats)
            if cut is None:
                out.setdefault(k, []).append(frozenset(cc))
                stack.append((frozenset(cc), k + 1))
            else:
                rest = sub.subgraph(set(cc) - cut)
                for D in nx.connected_components(rest):
                    stack.append((frozenset(D | cut), k))
    for k in out:
        out[k] = sorted(set(out[k]), key=lambda s: (-len(s), min(s)))
    return out


def levels_from_components(G: nx.Graph, kcomps: dict) -> dict:
    lvl = {v: 0 for v in G}
    for k in sorted(kcomps):
        for comp in kcomps[k]:
            for v in comp:
                if k > lvl[v]:
                    lvl[v] = k
    return lvl


def cohesion_levels(G: nx.Graph, return_components: bool = False):
    """Cohesion level of every node: the largest k such that the node lies in a k-component."""
    kc = k_components_exact(G)
    lvl = levels_from_components(G, kc)
    return (lvl, kc) if return_components else lvl
