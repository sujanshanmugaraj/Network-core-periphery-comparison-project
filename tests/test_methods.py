import networkx as nx
import numpy as np

from src.coreness import borgatti_everett, kcore_scores, rombach_profile, rombach_scores
from src.datasets import clean_graph
from src.nullmodels import degree_preserving_randomization
from src.robustness import order_by, robustness_curve, robustness_curve_naive


def test_kcore_trace_example():
    # triangle A-B-C, D attached to C, E attached to D (worked trace in the plan)
    G = clean_graph([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("D", "E")])
    lab = nx.get_node_attributes(G, "label")
    s = kcore_scores(G)
    assert {lab[v]: s[v] for v in G} == {"A": 2, "B": 2, "C": 2, "D": 1, "E": 1}


def test_robustness_matches_naive():
    rng = np.random.default_rng(1)
    for seed in range(5):
        G = clean_graph(nx.gnm_random_graph(60, 120, seed=seed).edges())
        order = rng.permutation(G.number_of_nodes())
        S1, R1 = robustness_curve(G, order)
        S2, R2 = robustness_curve_naive(G, order)
        assert np.allclose(S1, S2) and abs(R1 - R2) < 1e-12


def test_order_by_tiebreak():
    scores = np.array([1, 2, 2, 0])
    deg = np.array([5, 1, 3, 9])
    assert list(order_by(scores, deg)) == [2, 1, 0, 3]


def test_be_ideal_core_periphery():
    # perfect CP graph: clique of 5 core nodes, 15 periphery nodes each linked to 2 core nodes
    G = nx.complete_graph(5)
    for p in range(5, 20):
        G.add_edge(p, p % 5)
        G.add_edge(p, (p + 1) % 5)
    core, rho = borgatti_everett(G, n_restarts=5)
    assert set(np.where(core)[0]) == set(range(5))
    assert rho > 0.5


def test_rombach_profile():
    c = rombach_profile(10, 1.0, 0.5)
    assert np.allclose(c[:5], 0) and np.allclose(c[5:], 0.2)
    c = rombach_profile(10, 0.3, 0.6)
    assert np.isclose(c.sum(), 1) and np.all(np.diff(c) > 0)


def test_rombach_finds_planted_core():
    G = nx.complete_graph(6)
    for p in range(6, 40):
        G.add_edge(p, p % 6)
    cs = rombach_scores(G, alphas=[0.5, 1.0], betas=[0.5, 0.8])
    assert set(np.argsort(-cs)[:6]) == set(range(6))


def test_null_model_preserves_degrees():
    G = clean_graph(nx.barabasi_albert_graph(300, 3, seed=2).edges())
    H = degree_preserving_randomization(G, seed=3)
    if H.number_of_nodes() == G.number_of_nodes():
        assert sorted(d for _, d in G.degree()) == sorted(d for _, d in H.degree())
    assert nx.number_of_selfloops(H) == 0
