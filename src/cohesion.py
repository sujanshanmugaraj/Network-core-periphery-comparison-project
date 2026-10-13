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

Only pairs not certified this way trigger a unit-capacity max-flow on the node-split
graph (numba augmenting paths with cutoff k).  The result has been verified against
brute force and ``networkx`` (see tests/).
"""
from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from numba import njit


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


@njit(cache=True)
def _unit_vertex_flow(indptr, to, cap0, rev, src, sink, cutoff):
    """Augmenting-path max-flow on the split graph (all capacities 0/1), stopping at
    ``cutoff``.  Returns (flow value, residual capacities)."""
    cap = cap0.copy()
    n2 = indptr.size - 1
    parent_arc = np.empty(n2, dtype=np.int64)
    queue = np.empty(n2, dtype=np.int64)
    flow = 0
    while flow < cutoff:
        parent_arc[:] = -1
        parent_arc[src] = -2
        head, tail = 0, 0
        queue[tail] = src
        tail += 1
        found = False
        while head < tail and not found:
            x = queue[head]
            head += 1
            for a in range(indptr[x], indptr[x + 1]):
                if cap[a] > 0:
                    y = to[a]
                    if parent_arc[y] == -1:
                        parent_arc[y] = a
                        if y == sink:
                            found = True
                            break
                        queue[tail] = y
                        tail += 1
        if not found:
            break
        y = sink
        while y != src:
            a = parent_arc[y]
            cap[a] -= 1
            cap[rev[a]] += 1
            y = to[rev[a]]
        flow += 1
    return flow, cap


@njit(cache=True)
def _reachable(indptr, to, cap, src):
    n2 = indptr.size - 1
    seen = np.zeros(n2, dtype=np.bool_)
    stack = np.empty(n2, dtype=np.int64)
    top = 0
    stack[top] = src
    top += 1
    seen[src] = True
    while top > 0:
        top -= 1
        x = stack[top]
        for a in range(indptr[x], indptr[x + 1]):
            if cap[a] > 0 and not seen[to[a]]:
                seen[to[a]] = True
                stack[top] = to[a]
                top += 1
    return seen


class _Flow:
    """Vertex-disjoint path counting in one piece via a split graph (v_in = 2i,
    v_out = 2i+1, unit capacities), solved by a numba augmenting-path routine.
    Equivalent to NetworkX's local_node_connectivity / minimum_st_node_cut."""

    def __init__(self, H: nx.Graph):
        self.H = H
        self.built = False
        self.n_flows = 0

    def _build(self):
        if self.built:
            return
        nodes = list(self.H)
        self.idx = {v: i for i, v in enumerate(nodes)}
        self.nodes = nodes
        p = len(nodes)
        e = np.array([(self.idx[u], self.idx[v]) for u, v in self.H.edges()], dtype=np.int64).reshape(-1, 2)
        U = np.concatenate([e[:, 0], e[:, 1]])
        V = np.concatenate([e[:, 1], e[:, 0]])
        vi = np.arange(p, dtype=np.int64)
        ft = np.concatenate([2 * vi, 2 * U + 1])      # forward tails: v_in->v_out, u_out->v_in
        fh = np.concatenate([2 * vi + 1, 2 * V])
        nf = ft.size
        tails = np.concatenate([ft, fh])
        heads = np.concatenate([fh, ft])
        # vertex arcs have capacity 1, edge arcs "infinite" (so minimum cuts use vertices only)
        fcap = np.concatenate([np.ones(p, np.int64), np.full(U.size, p + 1, np.int64)])
        cap = np.concatenate([fcap, np.zeros(nf, np.int64)])
        rev_old = np.concatenate([np.arange(nf, 2 * nf), np.arange(nf)])
        perm = np.argsort(tails, kind="stable")
        pos = np.empty_like(perm)
        pos[perm] = np.arange(perm.size)
        self.to = heads[perm]
        self.cap0 = cap[perm]
        self.rev = pos[rev_old[perm]]
        self.indptr = np.zeros(2 * p + 1, dtype=np.int64)
        np.add.at(self.indptr, tails + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.built = True

    def kappa_at_least(self, s, t, k) -> bool:
        self._build()
        self.n_flows += 1
        f, _ = _unit_vertex_flow(self.indptr, self.to, self.cap0, self.rev,
                                 2 * self.idx[s] + 1, 2 * self.idx[t], k)
        return f >= k

    def cut(self, s, t) -> set:
        """A minimum s-t vertex cut (s, t non-adjacent)."""
        self._build()
        f, cap = _unit_vertex_flow(self.indptr, self.to, self.cap0, self.rev,
                                   2 * self.idx[s] + 1, 2 * self.idx[t], np.iinfo(np.int64).max)
        seen = _reachable(self.indptr, self.to, cap, 2 * self.idx[s] + 1)
        cut = {self.nodes[i] for i in range(len(self.nodes)) if seen[2 * i] and not seen[2 * i + 1]}
        cut.discard(s)
        cut.discard(t)
        assert len(cut) == f, (len(cut), f)
        return cut


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


def broker_scores(n: int, kcomps: dict):
    """Cohesive brokerage B_i = sum_{k>=2} max(0, c_k(i) - 1), where c_k(i) is the number of
    k-components that contain node i.  Overlaps between k-components consist of fewer than k
    nodes and are exactly the vertex cuts separating cohesive groups (bowtie: node X)."""
    import numpy as np

    B = np.zeros(n)
    for k, comps in kcomps.items():
        if k < 2:
            continue
        cnt = np.zeros(n)
        for c in comps:
            cnt[list(c)] += 1
        B += np.maximum(cnt - 1, 0)
    return B


def group_structure(G: nx.Graph, kcomps: dict) -> list[dict]:
    """Group-level comparison of k-cores and k-components for k = 2..max.

    For each k: number of connected components of the k-core (n_cc), number of
    k-components (n_kcomp), coverage (fraction of k-core nodes inside some k-component)
    and size-weighted best-match Jaccard of k-core components to k-components."""
    rows = []
    kmax = max(nx.core_number(G).values())
    for k in range(2, kmax + 1):
        core = nx.k_core(G, k)
        if core.number_of_nodes() == 0:
            break
        ccs = [set(c) for c in nx.connected_components(core)]
        comps = [set(c) for c in kcomps.get(k, [])]
        covered = set().union(*comps) if comps else set()
        tot = sum(len(c) for c in ccs)
        jac = 0.0
        for C in ccs:
            best = max((len(C & K) / len(C | K) for K in comps), default=0.0)
            jac += len(C) * best
        rows.append({"k": k, "core_size": tot, "n_cc": len(ccs), "n_kcomp": len(comps),
                     "coverage": len(covered & set(core)) / tot, "group_jaccard": jac / tot})
    return rows


def group_summary(rows: list[dict]) -> dict:
    """Scalar summaries: fragmentation F = sum n_kcomp / sum n_cc (1 = identical group
    counts) and node-weighted mean best-match Jaccard / coverage over k."""
    if not rows:
        return {"fragmentation": 1.0, "group_jaccard": 1.0, "coverage": 1.0}
    w = sum(r["core_size"] for r in rows)
    return {
        "fragmentation": sum(r["n_kcomp"] for r in rows) / sum(r["n_cc"] for r in rows),
        "group_jaccard": sum(r["group_jaccard"] * r["core_size"] for r in rows) / w,
        "coverage": sum(r["coverage"] * r["core_size"] for r in rows) / w,
    }


def cohesion_levels(G: nx.Graph, return_components: bool = False):
    """Cohesion level of every node: the largest k such that the node lies in a k-component."""
    kc = k_components_exact(G)
    lvl = levels_from_components(G, kc)
    return (lvl, kc) if return_components else lvl
