"""End-to-end experiment pipeline.

    python -m src.pipeline scores      # Phase 2-4: stats, all node scores (cached)
    python -m src.pipeline agreement   # Phase 5  : RQ1 (node level + group level)
    python -m src.pipeline robustness  # Phase 7  : RQ2
    python -m src.pipeline hybrid      # Phase 8  : RQ3
    python -m src.pipeline null        # Phase 6  : degree-preserving null models
    python -m src.pipeline all

Every stage reads/writes results/cache and writes CSV tables to results/tables.
All randomness is seeded (SEED below).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import agreement as ag
from .cohesion import (broker_scores, group_structure, group_summary, k_components_exact,
                       levels_from_components)
from .coreness import borgatti_everett, kcore_scores, rombach_scores
from .datasets import DATASETS, NETWORKS, ROOT, graph_stats, load_graph, to_csr
from .nullmodels import degree_preserving_randomization
from .robustness import adaptive_degree_order, order_by, robustness_curve
from .stats import holm, paired_comparison

SEED = 20240601
RES = ROOT / "results"
CACHE = RES / "cache"
TAB = RES / "tables"
for d in (CACHE, TAB):
    d.mkdir(parents=True, exist_ok=True)

METHODS = ["degree", "kcore", "be", "rombach", "cohesion"]  # core notions compared in RQ1
LABEL = {
    "random": "Random", "degree": "Degree", "degree_adaptive": "Degree (adaptive)",
    "betweenness": "Betweenness", "kcore": "k-core", "be": "BE (discrete)", "rombach": "Rombach",
    "cohesion": "Cohesion", "broker": "Cohesive brokerage", "hybrid_depth": "Hybrid (core+cohesion)",
    "hybrid_broker": "Hybrid (core+brokerage)",
}
STATIC = ["degree", "betweenness", "kcore", "be", "rombach", "cohesion", "broker"]
HYBRIDS = {"hybrid_depth": ("kcore", "cohesion"), "hybrid_broker": ("kcore", "broker")}
N_RANDOM_ORDER = 30
N_TIE_DRAWS = 50
LAMBDAS = np.round(np.linspace(0, 1, 11), 1)
FOLDS = {  # domain-stratified 2-fold split (2 social + 1 bio + 1 infra per fold)
    "A": ["karate", "email", "yeast", "power"],
    "B": ["grqc", "facebook", "celegans", "airindia"],
}


def hash_str(s) -> int:
    return sum(ord(c) * 131 ** i for i, c in enumerate(str(s))) % 1_000_003


def _seed(*parts) -> int:
    return (SEED + sum((i + 1) * hash_str(p) for i, p in enumerate(parts))) % (2**31 - 1)


def load_cache(net: str) -> dict:
    with open(CACHE / f"{net}.pkl", "rb") as fh:
        r = pickle.load(fh)
    # derived score (cheap): cohesive brokerage from the stored k-component hierarchy
    r["broker"] = broker_scores(r["kcore"].size, r["kcomponents"])
    return r


def norm_rank(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (rankdata(x) - 1) / (x.size - 1)


def hybrid_scores(r: dict, kind: str, lam: float) -> np.ndarray:
    a, b = HYBRIDS[kind]
    return lam * norm_rank(r[a]) + (1 - lam) * norm_rank(r[b])


# ============================================================================ scores
def compute_scores(net: str) -> str:
    out = CACHE / f"{net}.pkl"
    if out.exists():
        return f"{net}: cached"
    G = load_graph(net)
    n = G.number_of_nodes()
    t = {}
    rec = {"net": net}
    t0 = time.time(); rec["stats"] = graph_stats(G); t["stats"] = time.time() - t0
    rec["degree"] = np.array([G.degree(v) for v in range(n)], float)
    t0 = time.time()
    bc = nx.betweenness_centrality(G)
    rec["betweenness"] = np.array([bc[v] for v in range(n)]); t["betweenness"] = time.time() - t0
    t0 = time.time(); rec["kcore"] = kcore_scores(G); t["kcore"] = time.time() - t0
    t0 = time.time()
    core, rho = borgatti_everett(G, n_restarts=20, seed=_seed(net, "be"))
    rec["be"], rec["be_rho"] = core.astype(float), rho; t["be"] = time.time() - t0
    t0 = time.time()
    cs, det = rombach_scores(G, seed=_seed(net, "rombach"), return_details=True)
    rec["rombach"], rec["rombach_grid"] = cs, det; t["rombach"] = time.time() - t0
    t0 = time.time()
    st = {}
    kc = k_components_exact(G, stats=st)
    lv = levels_from_components(G, kc)
    rec["cohesion"] = np.array([lv[v] for v in range(n)], float)
    rec["kcomponents"] = {k: [sorted(c) for c in v] for k, v in kc.items()}
    rec["cohesion_flows"] = st.get("flows", 0); t["cohesion"] = time.time() - t0
    rec["time"] = t
    with open(out, "wb") as fh:
        pickle.dump(rec, fh)
    return f"{net}: done " + json.dumps({k: round(v, 1) for k, v in t.items()})


def stage_scores(nets=NETWORKS, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_scores, n): n for n in nets}
        for f in as_completed(futs):
            print(f.result(), flush=True)
    rows, grows = [], []
    for net in NETWORKS:
        if not (CACHE / f"{net}.pkl").exists():
            continue
        r = load_cache(net)
        G = load_graph(net)
        s = dict(r["stats"])
        kc, coh = r["kcore"], r["cohesion"]
        gs = group_structure(G, r["kcomponents"])
        for g in gs:
            grows.append({"key": net, **g})
        s.update({
            "network": DATASETS[net].name, "key": net, "domain": DATASETS[net].domain,
            "max_cohesion": int(coh.max()),
            "n_bicomponents": len(r["kcomponents"].get(2, [])),
            "BE_rho": r["be_rho"], "BE_core_size": int(r["be"].sum()),
            "frac_gap_nodes": float(np.mean(kc > coh)),
            "cohesion_deficit": float(1 - coh.sum() / kc.sum()),
            "n_brokers": int((r["broker"] > 0).sum()),
            **group_summary(gs),
            "time_cohesion_s": r["time"]["cohesion"],
            "time_rombach_s": r["time"]["rombach"],
            "time_betweenness_s": r["time"]["betweenness"],
        })
        rows.append(s)
    df = pd.DataFrame(rows)
    cols = ["network", "key", "domain", "N", "M", "density", "avg_degree", "max_degree",
            "avg_clustering", "transitivity", "assortativity", "avg_path_length", "diameter",
            "degeneracy", "max_cohesion", "n_bicomponents", "BE_rho", "BE_core_size",
            "frac_gap_nodes", "cohesion_deficit", "n_brokers", "fragmentation", "group_jaccard",
            "coverage", "time_cohesion_s", "time_rombach_s", "time_betweenness_s"]
    df[cols].to_csv(TAB / "dataset_table.csv", index=False)
    pd.DataFrame(grows).to_csv(TAB / "group_structure_by_k.csv", index=False)
    print(df[cols].to_string())


# ============================================================================ RQ1
def stage_agreement():
    rows = []
    for net in NETWORKS:
        r = load_cache(net)
        n_be = int(r["be"].sum())
        labels = {m: ag.natural_core_labels(m, r[m], n_be) for m in METHODS}
        for a, b in combinations(METHODS, 2):
            row = {"key": net, "domain": DATASETS[net].domain, "A": a, "B": b,
                   "spearman": ag.spearman(r[a], r[b]), "ami": ag.ami(labels[a], labels[b])}
            for x in (0.05, 0.10, 0.20):
                row[f"jaccard_top{int(x * 100)}"] = ag.topx_jaccard(r[a], r[b], x, N_TIE_DRAWS,
                                                                    seed=_seed(net, a, b, x))
            rows.append(row)
        print(net, "agreement done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "agreement_long.csv", index=False)
    cp = df[df.B.eq("cohesion") & df.A.isin(["kcore", "be", "rombach"])]
    summ = cp.groupby(["domain", "A"])[["spearman", "jaccard_top10", "ami"]].mean().round(3)
    summ.to_csv(TAB / "agreement_domain_summary.csv")
    print(summ)
    wide = cp.pivot(index="key", columns="A", values=["spearman", "jaccard_top10", "ami"]).loc[NETWORKS]
    wide.columns = [f"{m}_{a}" for m, a in wide.columns]
    wide.to_csv(TAB / "agreement_cp_vs_cohesion_wide.csv")
    print(wide.round(3).to_string())


# ============================================================================ null models
NULL_STATS = ["BE_rho", "degeneracy", "max_cohesion", "cohesion_deficit", "frac_gap_nodes",
              "spearman_kcore_cohesion", "fragmentation", "group_jaccard", "coverage",
              "frac_brokers", "transitivity"]


def _structure_stats(G, kc, coh, kcomps, rho) -> dict:
    B = broker_scores(G.number_of_nodes(), kcomps)
    gsum = group_summary(group_structure(G, kcomps))
    return {"BE_rho": rho, "degeneracy": kc.max(), "max_cohesion": coh.max(),
            "cohesion_deficit": 1 - coh.sum() / kc.sum(), "frac_gap_nodes": float(np.mean(kc > coh)),
            "spearman_kcore_cohesion": ag.spearman(kc, coh), **gsum,
            "frac_brokers": float(np.mean(B > 0)), "transitivity": nx.transitivity(G)}


def _null_one(net: str, i: int, n_be_restarts: int) -> dict:
    G = load_graph(net)
    H = degree_preserving_randomization(G, swaps_per_edge=10, seed=_seed(net, "null", i))
    kc = kcore_scores(H)
    _, rho = borgatti_everett(H, n_restarts=n_be_restarts, seed=_seed(net, "nullbe", i))
    kcomps = k_components_exact(H)
    lv = levels_from_components(H, kcomps)
    coh = np.array([lv[v] for v in range(H.number_of_nodes())], float)
    return {"key": net, "rep": i, "N": H.number_of_nodes(), **_structure_stats(H, kc, coh, kcomps, rho)}


def stage_null(n_rep: int = 100, workers: int = 4, nets=NETWORKS):
    out = CACHE / "null_samples.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["key", "rep"])
    have = set(zip(done.key, done.rep))
    # cheap networks first so that partial results are usable early
    jobs = [(net, i) for net in nets for i in range(n_rep) if (net, i) not in have]
    rows = done.to_dict("records")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_null_one, net, i, 5) for net, i in jobs]
        for j, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (j + 1) % 10 == 0 or j + 1 == len(futs):
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"null {j + 1}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)
    null = pd.DataFrame(rows)
    null.to_csv(out, index=False)
    summarize_null(null)


def summarize_null(null: pd.DataFrame | None = None):
    if null is None:
        null = pd.read_csv(CACHE / "null_samples.csv")
    res = []
    for net in NETWORKS:
        if not (CACHE / f"{net}.pkl").exists():
            continue
        sub = null[null.key == net]
        if sub.empty:
            continue
        r = load_cache(net)
        G = load_graph(net)
        obs = _structure_stats(G, r["kcore"], r["cohesion"], r["kcomponents"], r["be_rho"])
        for stat in NULL_STATS:
            o = obs[stat]
            x = sub[stat].to_numpy(float)
            mu, sd = np.nanmean(x), np.nanstd(x, ddof=1)
            z = (o - mu) / sd if sd > 1e-12 else np.nan
            p_hi = (1 + np.sum(x >= o - 1e-12)) / (1 + x.size)
            p_lo = (1 + np.sum(x <= o + 1e-12)) / (1 + x.size)
            res.append({"key": net, "stat": stat, "observed": o, "null_mean": mu, "null_sd": sd,
                        "null_min": np.nanmin(x), "null_max": np.nanmax(x),
                        "z": z, "p_two_sided": min(1.0, 2 * min(p_hi, p_lo)), "n_null": x.size})
    df = pd.DataFrame(res)
    df.to_csv(TAB / "null_model_zscores.csv", index=False)
    piv = df.pivot(index="key", columns="stat", values="observed")
    piv_null = df.pivot(index="key", columns="stat", values="null_mean")
    print("observed\n", piv.round(3).to_string(), "\nnull mean\n", piv_null.round(3).to_string())


# ============================================================================ RQ2
def _robust_one(net: str) -> dict:
    r = load_cache(net)
    G = load_graph(net)
    csr = to_csr(G)
    n = G.number_of_nodes()
    deg = r["degree"]
    res = {"key": net, "R": {}, "R_randtie": {}, "curves": {}}
    for m in STATIC:
        S, R = robustness_curve(G, order_by(r[m], deg if m != "degree" else None), csr)
        res["R"][m], res["curves"][m] = R, S
        rng = np.random.default_rng(_seed(net, "tie", m))
        vals = [robustness_curve(G, order_by(r[m], None, rng), csr)[1] for _ in range(N_TIE_DRAWS)]
        res["R_randtie"][m] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    S, R = robustness_curve(G, adaptive_degree_order(G, csr), csr)
    res["R"]["degree_adaptive"], res["curves"]["degree_adaptive"] = R, S
    rng = np.random.default_rng(_seed(net, "random"))
    rs, curves = [], []
    for _ in range(N_RANDOM_ORDER):
        S, R = robustness_curve(G, rng.permutation(n), csr)
        rs.append(R); curves.append(S)
    res["R"]["random"] = float(np.mean(rs))
    res["R_random_sd"] = float(np.std(rs, ddof=1))
    res["curves"]["random"] = np.mean(curves, axis=0)
    for kind in HYBRIDS:
        res[kind] = {}
        for lam in LAMBDAS:
            S, R = robustness_curve(G, order_by(hybrid_scores(r, kind, lam), deg), csr)
            res[kind][float(lam)] = (R, S)
    return res


ROB_ORDER = ["random", "degree", "degree_adaptive", "betweenness", "kcore", "be", "rombach",
             "cohesion", "broker"]


def stage_robustness(workers=4):
    allres = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_robust_one, NETWORKS):
            allres[res["key"]] = res
            print(res["key"], "robustness done", flush=True)
    with open(CACHE / "robustness.pkl", "wb") as fh:
        pickle.dump(allres, fh)
    rows = []
    for net in NETWORKS:
        res = allres[net]
        row = {"key": net, "domain": DATASETS[net].domain}
        row.update({m: res["R"][m] for m in ROB_ORDER})
        row.update({f"{m}_randtie": res["R_randtie"][m][0] for m in STATIC})
        row.update({f"{m}_randtie_sd": res["R_randtie"][m][1] for m in STATIC})
        row["random_sd"] = res["R_random_sd"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "robustness_R.csv", index=False)
    print(df[["key"] + ROB_ORDER].round(4).to_string())
    print(df[["key"] + [f"{m}_randtie" for m in STATIC]].round(4).to_string())
    comps = [("cohesion", "kcore"), ("cohesion", "be"), ("cohesion", "rombach"), ("cohesion", "degree"),
             ("broker", "kcore"), ("broker", "cohesion"), ("broker", "degree"),
             ("broker", "degree_adaptive"), ("broker", "betweenness"),
             ("kcore", "degree"), ("rombach", "kcore"), ("be", "kcore")]
    crow = []
    for a, b in comps:
        for variant, suf in (("degree_tiebreak", ""), ("random_tiebreak", "_randtie")):
            ca, cb = a + suf, b + suf
            if ca not in df or cb not in df:  # adaptive degree has no tie variant
                ca = ca if ca in df else a
                cb = cb if cb in df else b
            c = paired_comparison(df[ca], df[cb], seed=_seed(a, b, variant))
            c.update({"A": a, "B": b, "variant": variant})
            crow.append(c)
    cdf = pd.DataFrame(crow)
    for v in cdf.variant.unique():
        m = cdf.variant == v
        cdf.loc[m, "holm_p"] = holm(cdf.loc[m, "wilcoxon_p"])
    cdf.to_csv(TAB / "robustness_comparisons.csv", index=False)
    print(cdf[["A", "B", "variant", "mean_diff", "ci_low", "ci_high", "wins_A", "wins_B",
               "wilcoxon_p", "holm_p", "rank_biserial"]].round(4).to_string())
    st = pd.read_csv(TAB / "dataset_table.csv").set_index("key")
    df = df.set_index("key")
    out = pd.DataFrame({
        "domain": df["domain"],
        "cohesion_minus_kcore": df["cohesion"] - df["kcore"],
        "broker_gain_vs_kcore": (df["kcore"] - df["broker"]) / df["kcore"],
        "broker_minus_degree": df["broker"] - df["degree"],
        "transitivity": st["transitivity"], "avg_clustering": st["avg_clustering"],
        "fragmentation": st["fragmentation"], "frac_gap_nodes": st["frac_gap_nodes"],
    })
    out["broker_gain_vs_degree"] = (df["degree"] - df["broker"]) / df["degree"]
    out.to_csv(TAB / "h2_structure_vs_gain.csv")
    print(out.round(3).to_string())
    stage_correlations()


def stage_correlations():
    """Which network properties go with the node-level gap, the group-level gap and the
    brokerage gain?  Spearman correlations across the eight networks (descriptive)."""
    from scipy.stats import spearmanr

    st = pd.read_csv(TAB / "dataset_table.csv").set_index("key").loc[NETWORKS]
    g = pd.read_csv(TAB / "h2_structure_vs_gain.csv").set_index("key").loc[NETWORKS]
    st["group_disagreement"] = 1 - st["group_jaccard"]
    for c in ["broker_gain_vs_kcore", "broker_gain_vs_degree", "cohesion_minus_kcore"]:
        st[c] = g[c]
    xs = ["N", "density", "avg_degree", "avg_clustering", "transitivity", "assortativity",
          "avg_path_length", "degeneracy", "frac_gap_nodes", "fragmentation", "group_disagreement"]
    ys = ["frac_gap_nodes", "fragmentation", "group_disagreement", "broker_gain_vs_kcore",
          "broker_gain_vs_degree", "cohesion_minus_kcore"]
    corr = []
    for y in ys:
        for x in xs:
            if x == y:
                continue
            rho, p = spearmanr(st[x], st[y])
            corr.append({"y": y, "x": x, "spearman": rho, "p": p, "n": len(st)})
    df = pd.DataFrame(corr)
    df.to_csv(TAB / "h2_correlations.csv", index=False)
    print(df.pivot(index="x", columns="y", values="spearman").round(2).to_string())


def stage_ablation():
    """Brokerage ablation: which levels of the cohesion hierarchy carry the signal?
    Compares articulation points (classical cut vertices) with brokerage restricted to
    k = 2, to k >= 3, and the full hierarchy; all ties broken by degree."""
    rows = []
    for net in NETWORKS:
        r = load_cache(net)
        G = load_graph(net)
        csr = to_csr(G)
        n, deg = G.number_of_nodes(), r["degree"]
        kmax = max(r["kcomponents"])
        sub = lambda ks: broker_scores(n, {k: v for k, v in r["kcomponents"].items() if k in ks})
        art = np.zeros(n)
        art[list(nx.articulation_points(G))] = 1
        scores = {"degree": (deg, None), "articulation": (art, deg),
                  "broker_k2": (sub({2}), deg), "broker_k3plus": (sub(set(range(3, kmax + 1))), deg),
                  "broker": (r["broker"], deg)}
        row = {"key": net, "n_articulation": int(art.sum()),
               "n_broker_k2": int((scores["broker_k2"][0] > 0).sum()),
               "n_broker_k3plus": int((scores["broker_k3plus"][0] > 0).sum()),
               "n_broker": int((r["broker"] > 0).sum())}
        for name, (s, tb) in scores.items():
            row[f"R_{name}"] = robustness_curve(G, order_by(s, tb), csr)[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "broker_ablation.csv", index=False)
    print(df.round(4).to_string())


# ============================================================================ RQ3
def stage_hybrid():
    with open(CACHE / "robustness.pkl", "rb") as fh:
        allres = pickle.load(fh)
    rows, crow = [], []
    for kind, (a, b) in HYBRIDS.items():
        lam_tab = pd.DataFrame({net: {lam: allres[net][kind][lam][0] for lam in allres[net][kind]}
                                for net in NETWORKS})
        lam_tab.index.name = "lambda"
        lam_tab.to_csv(TAB / f"{kind}_lambda_R.csv")
        # relative R (divide by k-core R) so that networks are comparable when averaging
        rel = lam_tab.copy()
        for net in NETWORKS:
            rel[net] = lam_tab[net] / allres[net]["R"]["kcore"]

        def evaluate(train, test, scheme):
            m = rel[train].mean(axis=1)
            lam_star = float(m[m == m.min()].index.max())  # ties -> larger λ (closer to k-core)
            for net in test:
                R = allres[net]["R"]
                rows.append({"hybrid": kind, "scheme": scheme, "train": ",".join(train), "test_net": net,
                             "lambda_star": lam_star, "R_hybrid": allres[net][kind][lam_star][0],
                             "R_a": R[a], "R_b": R[b], "R_degree": R["degree"],
                             "R_degree_adaptive": R["degree_adaptive"], "R_betweenness": R["betweenness"]})

        evaluate(FOLDS["A"], FOLDS["B"], "2-fold")
        evaluate(FOLDS["B"], FOLDS["A"], "2-fold")
        for net in NETWORKS:
            evaluate([x for x in NETWORKS if x != net], [net], "leave-one-out")
    df = pd.DataFrame(rows)
    df["best_single"] = df[["R_a", "R_b"]].min(axis=1)
    df["hybrid_beats_both"] = df["R_hybrid"] < df["best_single"] - 1e-12
    df["hybrid_minus_best"] = df["R_hybrid"] - df["best_single"]
    df.to_csv(TAB / "hybrid_heldout.csv", index=False)
    print(df.round(4).to_string())
    for kind in HYBRIDS:
        for scheme in ["2-fold", "leave-one-out"]:
            sub = df[(df.hybrid == kind) & (df.scheme == scheme)]
            for b in ["R_a", "R_b", "best_single", "R_degree", "R_degree_adaptive", "R_betweenness"]:
                c = paired_comparison(sub["R_hybrid"], sub[b], seed=_seed("hyb", kind, scheme, b))
                c.update({"hybrid": kind, "scheme": scheme, "A": "hybrid", "B": b})
                crow.append(c)
    cdf = pd.DataFrame(crow)
    cdf.to_csv(TAB / "hybrid_comparisons.csv", index=False)
    print(cdf[["hybrid", "scheme", "B", "mean_diff", "ci_low", "ci_high", "wins_A", "wins_B",
               "wilcoxon_p", "rank_biserial"]].round(4).to_string())


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scores", "agreement", "null", "summarize-null", "robustness",
                                      "ablation", "correlations", "hybrid", "all"])
    ap.add_argument("--nets", nargs="*", default=NETWORKS)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.stage in ("scores", "all"):
        stage_scores(a.nets, a.workers)
    if a.stage in ("agreement", "all"):
        stage_agreement()
    if a.stage in ("robustness", "all"):
        stage_robustness(a.workers)
    if a.stage == "correlations":
        stage_correlations()
    if a.stage in ("ablation", "all"):
        stage_ablation()
    if a.stage in ("hybrid", "all"):
        stage_hybrid()
    if a.stage in ("null", "all"):
        stage_null(a.n_null, workers=a.workers, nets=a.nets)
    if a.stage == "summarize-null":
        summarize_null()


if __name__ == "__main__":
    main()
