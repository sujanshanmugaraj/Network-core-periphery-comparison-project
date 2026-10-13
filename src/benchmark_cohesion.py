"""Runtime/correctness benchmark: our exact k-components vs networkx.k_components.

    python -m src.benchmark_cohesion [--timeout 1800]

For each network, both algorithms are run in a subprocess with a wall-clock limit.  We
record runtimes and whether the two outputs coincide; where they differ we check which
sets are k-connected and whether one family strictly contains the other.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import time

import networkx as nx
import pandas as pd

from .cohesion import k_components_exact
from .datasets import NETWORKS, load_graph
from .pipeline import TAB


def _run(name, which, q):
    G = load_graph(name)
    if which == "ours":
        # warm-up: exclude Numba's one-off JIT compilation from the measured time
        # (Les Miserables exercises every compiled routine, including max-flow and cut extraction)
        k_components_exact(nx.convert_node_labels_to_integers(nx.Graph(nx.les_miserables_graph().edges())))
    t = time.time()
    if which == "ours":
        kc = k_components_exact(G)
    else:
        kc = nx.k_components(G)
    kc = {k: sorted({frozenset(c) for c in v if len(c) > k}, key=lambda s: (-len(s), min(s)))
          for k, v in kc.items()}
    q.put((time.time() - t, kc))


def timed(name, which, timeout):
    q = mp.Queue()
    p = mp.Process(target=_run, args=(name, which, q))
    p.start()
    # Read the result BEFORE joining: a child that puts a large object on the queue cannot exit
    # until the parent drains the pipe (joining first would deadlock on large outputs).
    try:
        out = q.get(timeout=timeout)
    except queue.Empty:
        out = (None, None)
    if p.is_alive():
        p.terminate()
    p.join()
    return out


def compare(G, ours, ref):
    """Count levels where the families differ, and classify the differences."""
    diff_levels, nonmax, invalid = 0, 0, 0
    for k in sorted(set(ours) | set(ref)):
        if k == 1:
            continue
        A, B = set(ours.get(k, [])), set(ref.get(k, []))
        if A == B:
            continue
        diff_levels += 1
        for T in B - A:
            if any(T < S for S in A):
                nonmax += 1  # reference set strictly inside one of ours
            elif nx.node_connectivity(G.subgraph(T)) < k:
                invalid += 1
    return diff_levels, nonmax, invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--nets", nargs="*", default=NETWORKS)
    a = ap.parse_args()
    rows = []
    for net in a.nets:
        G = load_graph(net)
        t_ours, ours = timed(net, "ours", a.timeout)
        t_nx, ref = timed(net, "networkx", a.timeout)
        row = {"key": net, "N": G.number_of_nodes(), "M": G.number_of_edges(),
               "t_ours_s": t_ours, "t_networkx_s": t_nx}
        if ours is not None and ref is not None:
            d, nm, inv = compare(G, ours, ref)
            row.update({"levels_differ": d, "nx_nonmaximal_sets": nm, "nx_invalid_sets": inv})
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(TAB / "cohesion_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
