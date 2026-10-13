"""Dataset loading and preprocessing.

Every network is converted to a simple, undirected, unweighted graph; self-loops and
multi-edges are removed and only the largest connected component (LCC) is kept.
Nodes are relabelled to consecutive integers 0..N-1 (the original label is kept in
the node attribute ``label``) so that numba kernels can index arrays directly.
"""
from __future__ import annotations

import csv
import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    name: str
    domain: str
    source: str


DATASETS: dict[str, DatasetInfo] = {
    d.key: d
    for d in [
        DatasetInfo("karate", "Zachary karate club", "Social", "Zachary (1977); NetworkX"),
        DatasetInfo("grqc", "arXiv GR-QC collaboration", "Social", "SNAP ca-GrQc"),
        DatasetInfo("email", "EU email (core)", "Social", "SNAP email-Eu-core"),
        DatasetInfo("facebook", "Facebook ego networks", "Social", "SNAP ego-Facebook"),
        DatasetInfo("yeast", "Yeast PPI (Collins)", "Biological", "Netzschleuder collins_yeast"),
        DatasetInfo("celegans", "C. elegans neural", "Biological", "Netzschleuder celegansneural"),
        DatasetInfo("power", "US power grid", "Infrastructure", "Netzschleuder power"),
        DatasetInfo("airindia", "India domestic air routes", "Infrastructure", "OpenFlights"),
    ]
}
NETWORKS = list(DATASETS)


# --------------------------------------------------------------------------- readers
def _read_snap(fname: str) -> list[tuple[str, str]]:
    edges = []
    with gzip.open(RAW / fname, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split()[:2]
            edges.append((a, b))
    return edges


def _read_netzschleuder(fname: str) -> list[tuple[str, str]]:
    with zipfile.ZipFile(RAW / fname) as z:
        text = z.read("edges.csv").decode("utf-8")
    edges = []
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].startswith("#"):
            continue
        edges.append((row[0].strip(), row[1].strip()))
    return edges


def _read_openflights_country(country: str) -> list[tuple[str, str]]:
    in_country = set()
    with open(RAW / "airports.dat", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) > 3 and row[3] == country:
                in_country.add(row[0])
    edges = []
    with open(RAW / "routes.dat", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            s, t = row[3], row[5]
            if s in in_country and t in in_country:
                edges.append((s, t))
    return edges


def _raw_edges(key: str):
    if key == "karate":
        return list(nx.karate_club_graph().edges())
    if key == "grqc":
        return _read_snap("ca-GrQc.txt.gz")
    if key == "email":
        return _read_snap("email-Eu-core.txt.gz")
    if key == "facebook":
        return _read_snap("facebook_combined.txt.gz")
    if key == "yeast":
        return _read_netzschleuder("collins_yeast.csv.zip")
    if key == "celegans":
        return _read_netzschleuder("celegansneural.csv.zip")
    if key == "power":
        return _read_netzschleuder("power.csv.zip")
    if key == "airindia":
        return _read_openflights_country("India")
    raise KeyError(key)


# --------------------------------------------------------------------------- public API
def clean_graph(edges) -> nx.Graph:
    """Simple undirected unweighted LCC, relabelled to 0..N-1."""
    G = nx.Graph()
    G.add_edges_from((a, b) for a, b in edges if a != b)
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    # deterministic relabelling: sort original labels (numeric where possible)
    def _key(x):
        try:
            return (0, float(x), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(x))
    order = sorted(G.nodes(), key=_key)
    mapping = {old: i for i, old in enumerate(order)}
    H = nx.relabel_nodes(G, mapping, copy=True)
    nx.set_node_attributes(H, {i: str(old) for old, i in mapping.items()}, "label")
    # rebuild with sorted node insertion order so that G.nodes() == range(N)
    out = nx.Graph()
    out.add_nodes_from((i, H.nodes[i]) for i in range(H.number_of_nodes()))
    out.add_edges_from(H.edges())
    return out


_CACHE: dict[str, nx.Graph] = {}


def load_graph(key: str) -> nx.Graph:
    if key not in _CACHE:
        _CACHE[key] = clean_graph(_raw_edges(key))
    return _CACHE[key].copy()


def to_csr(G: nx.Graph):
    """CSR arrays (indptr, indices) with sorted neighbour lists; nodes must be 0..N-1."""
    n = G.number_of_nodes()
    indptr = np.zeros(n + 1, dtype=np.int64)
    nbrs = [sorted(G.adj[v]) for v in range(n)]
    indptr[1:] = np.cumsum([len(x) for x in nbrs])
    indices = np.fromiter((u for x in nbrs for u in x), dtype=np.int64, count=int(indptr[-1]))
    return indptr, indices


def graph_stats(G: nx.Graph) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = np.array([d for _, d in G.degree()])
    core = nx.core_number(G)
    if n <= 6000:
        apl = nx.average_shortest_path_length(G)
        diam = nx.diameter(G)
    else:  # not used for our networks, kept for completeness
        apl, diam = float("nan"), float("nan")
    return {
        "N": n,
        "M": m,
        "density": 2 * m / (n * (n - 1)),
        "avg_degree": degs.mean(),
        "max_degree": int(degs.max()),
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "avg_path_length": apl,
        "diameter": diam,
        "degeneracy": max(core.values()),
    }
