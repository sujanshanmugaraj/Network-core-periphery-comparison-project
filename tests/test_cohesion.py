"""Correctness checks for the exact k-component algorithm.

Run:  python -m pytest -q tests
"""
import itertools

import networkx as nx
import pytest

from src.cohesion import cohesion_levels, k_components_exact


def bowtie():
    G = nx.Graph([("A", "B"), ("B", "X"), ("A", "X"), ("X", "C"), ("C", "D"), ("D", "X")])
    return G


def brute_force_kcomponents(G):
    """Maximal k-connected node sets (size > k), by exhaustive enumeration."""
    nodes = list(G)
    res = {}
    conn = {}
    for r in range(2, len(nodes) + 1):
        for S in itertools.combinations(nodes, r):
            H = G.subgraph(S)
            if not nx.is_connected(H):
                continue
            conn[frozenset(S)] = nx.node_connectivity(H) if r > 2 else 1
    for S, kap in conn.items():
        for k in range(1, kap + 1):
            if len(S) > k:
                res.setdefault(k, set()).add(S)
    out = {}
    for k, sets in res.items():
        out[k] = {S for S in sets if not any(S < T for T in sets)}
    return out


def test_bowtie():
    G = bowtie()
    lvl, kc = cohesion_levels(G, return_components=True)
    assert nx.core_number(G) == {v: 2 for v in G}
    assert lvl == {v: 2 for v in G}
    assert set(kc[2]) == {frozenset("ABX"), frozenset("XCD")}
    assert 3 not in kc
    assert nx.node_connectivity(G) == 1


@pytest.mark.parametrize("seed", range(25))
def test_against_brute_force(seed):
    G = nx.gnp_random_graph(9, 0.55, seed=seed)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    ours = {k: set(v) for k, v in k_components_exact(G).items()}
    bf = brute_force_kcomponents(G)
    for k in set(ours) | set(bf):
        if k == 1:
            continue
        assert ours.get(k, set()) == bf.get(k, set()), (seed, k)


@pytest.mark.parametrize("name", ["karate", "florentine", "lesmis", "davis"])
def test_against_networkx(name):
    G = {
        "karate": nx.karate_club_graph,
        "florentine": nx.florentine_families_graph,
        "lesmis": nx.les_miserables_graph,
        "davis": nx.davis_southern_women_graph,
    }[name]()
    G = nx.Graph(G.edges())
    ours = {k: set(v) for k, v in k_components_exact(G).items()}
    ref = {k: {frozenset(c) for c in v if len(c) > k} for k, v in nx.k_components(G).items()}
    # networkx's implementation can return non-maximal components (e.g. Les Miserables,
    # k=3: a 32-node set strictly inside a 34-node 3-connected set).  We therefore check
    # that (i) every set we return is k-connected and (ii) every networkx set is contained
    # in one of ours; maximality of ours is covered by the brute-force test.
    assert set(ours) == set(ref)
    for k in ours:
        for S in ours[k]:
            assert len(S) > k and nx.node_connectivity(G.subgraph(S)) >= k, (name, k)
        for T in ref[k]:
            assert any(T <= S for S in ours[k]), (name, k)


@pytest.mark.parametrize("seed", range(6))
def test_planted_cliques_with_bridges(seed):
    # overlapping cliques sharing < k nodes create k-cores that are not k-cohesive
    import random

    rnd = random.Random(seed)
    G = nx.Graph()
    base = 0
    blocks = []
    for _ in range(4):
        size = rnd.randint(5, 8)
        nodes = list(range(base, base + size))
        G.add_edges_from(itertools.combinations(nodes, 2))
        blocks.append(nodes)
        base += size
    for a, b in zip(blocks, blocks[1:]):
        G.add_edge(a[0], b[0])
        G.add_edge(a[1], b[0])
    ref = {k: {frozenset(c) for c in v if len(c) > k} for k, v in nx.k_components(G).items()}
    ours = {k: set(v) for k, v in k_components_exact(G).items()}
    for k in set(ours) | set(ref):
        if k == 1:
            continue
        assert ours.get(k, set()) == ref.get(k, set()), (seed, k)
