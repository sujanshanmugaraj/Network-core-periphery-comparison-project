"""All paper figures.  python -m src.figures"""
from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .datasets import DATASETS, NETWORKS, ROOT, load_graph
from .pipeline import CACHE, FOLDS, LABEL, TAB, load_cache

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Fixed entity -> colour mapping (categorical slots in fixed order; never cycled).
COLOR = {
    "kcore": "#2a78d6",           # 1 blue
    "cohesion": "#eb6834",        # 2 orange
    "broker": "#1baf7a",          # 3 aqua
    "degree": "#eda100",          # 4 yellow
    "betweenness": "#e87ba4",     # 5 magenta
    "rombach": "#008300",         # 6 green
    "be": "#4a3aa7",              # 7 violet
    "degree_adaptive": "#e34948", # 8 red
    "random": "#8a8984",          # neutral grey (reference)
    "hybrid": "#0b0b0b",          # ink (only shown with k-core / cohesion / broker)
}
STYLE = {"kcore": "-", "cohesion": "-", "broker": "-", "degree": ":", "betweenness": "-.",
         "rombach": (0, (5, 1)), "be": (0, (3, 1, 1, 1)), "degree_adaptive": (0, (1, 1)),
         "random": "--", "hybrid": "--"}
MARK = {"degree": "v", "degree_adaptive": "<", "betweenness": "^", "kcore": "o", "be": "D",
        "rombach": "P", "cohesion": "s", "broker": "*", "hybrid": "X"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"karate": "Karate", "grqc": "GrQc", "email": "Email-EU", "facebook": "Facebook",
         "yeast": "Yeast PPI", "celegans": "C. elegans", "power": "Power grid", "airindia": "Air India"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 1.4, "axes.titlecolor": INK,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def _domain_dividers(ax):
    ax.axhline(3.5, color=INK2, lw=0.5)
    ax.axhline(5.5, color=INK2, lw=0.5)


def _net_axis(ax):
    y = np.arange(len(NETWORKS))
    ax.set_yticks(y, [SHORT[n] for n in NETWORKS])
    ax.set_ylim(len(NETWORKS) - 0.5, -0.5)
    _domain_dividers(ax)
    return y


# ----------------------------------------------------------------- Fig 1: bowtie + karate
def fig_bowtie_karate():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [0.8, 1, 1]})
    B = nx.Graph([("A", "B"), ("B", "X"), ("A", "X"), ("X", "C"), ("C", "D"), ("D", "X")])
    pos = {"A": (-1, 0.6), "B": (-1, -0.6), "X": (0, 0), "C": (1, 0.6), "D": (1, -0.6)}
    ax = axes[0]
    for comp, col in ((["A", "B", "X"], COLOR["kcore"]), (["X", "C", "D"], COLOR["cohesion"])):
        xy = np.array([pos[v] for v in comp])
        ax.fill(xy[:, 0], xy[:, 1], color=col, alpha=0.15, lw=0)
    nx.draw_networkx_edges(B, pos, ax=ax, edge_color=INK2, width=1.2)
    nx.draw_networkx_nodes(B, pos, ax=ax, nodelist=["A", "B", "C", "D"], node_color="white",
                           edgecolors=INK, node_size=330, linewidths=1.2)
    nx.draw_networkx_nodes(B, pos, ax=ax, nodelist=["X"], node_color="white", edgecolors=COLOR["broker"],
                           node_size=330, linewidths=2.4)
    nx.draw_networkx_labels(B, pos, ax=ax, font_size=8)
    ax.set_title("(a) Bowtie: one 2-core,\ntwo 2-components, broker X")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.1, 1.0); ax.axis("off")

    r = load_cache("karate")
    G = load_graph("karate")
    p = nx.kamada_kawai_layout(G)
    cmap = plt.get_cmap("Blues")
    for ax, key, title in ((axes[1], "kcore", "(b) Karate: k-core coreness"),
                           (axes[2], "cohesion", "(c) Karate: cohesion level")):
        vals = r[key]
        nx.draw_networkx_edges(G, p, ax=ax, edge_color=GRID, width=0.7)
        nodes = nx.draw_networkx_nodes(G, p, ax=ax, node_color=vals, cmap=cmap, vmin=0, vmax=4.5,
                                       node_size=[40 + 8 * G.degree(v) for v in G], edgecolors=INK2,
                                       linewidths=0.5)
        if key == "cohesion":
            br = [v for v in G if r["broker"][v] > 0]
            nx.draw_networkx_nodes(G, p, nodelist=br, ax=ax, node_color="none", edgecolors=COLOR["broker"],
                                   node_size=[160 + 8 * G.degree(v) for v in br], linewidths=1.8)
            for k, comps in r["kcomponents"].items():
                if k == 3:
                    for c in comps:
                        xy = np.array([p[v] for v in c])
                        ax.scatter(xy[:, 0], xy[:, 1], s=0)
        ax.set_title(title); ax.axis("off")
        cb = fig.colorbar(nodes, ax=ax, fraction=0.04, pad=0.01, ticks=range(0, 5))
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    axes[2].legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=COLOR["broker"], mew=1.8,
                                   label="cohesive broker")], loc="lower left", fontsize=6.5)
    save(fig, "fig1_bowtie_karate")


# ----------------------------------------------------------------- Fig 2: node-level agreement heatmaps
def fig_agreement_heatmaps():
    df = pd.read_csv(TAB / "agreement_long.csv")
    methods = ["degree", "kcore", "be", "rombach", "cohesion"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for ax, net in zip(axes.flat, NETWORKS):
        M = np.eye(len(methods))
        sub = df[df.key == net]
        for _, row in sub.iterrows():
            i, j = methods.index(row.A), methods.index(row.B)
            M[i, j] = M[j, i] = row.spearman
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                            color="white" if M[i, j] > 0.6 else INK)
        lab = ["Deg", "k-core", "BE", "Romb.", "Coh."]
        ax.set_xticks(range(5), lab, rotation=45, fontsize=6); ax.set_yticks(range(5), lab, fontsize=6)
        ax.grid(False); ax.set_title(f"{SHORT[net]} ({DATASETS[net].domain[:5]}.)", fontsize=7.5)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
    save(fig, "fig2_agreement_heatmaps")


# ----------------------------------------------------------------- Fig 3: node level vs group level
def fig_node_vs_group():
    df = pd.read_csv(TAB / "agreement_long.csv")
    st = pd.read_csv(TAB / "dataset_table.csv").set_index("key").loc[NETWORKS]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    sel = lambda a, met: [df[(df.key == n) & (df.A == a) & (df.B == "cohesion")][met].values[0] for n in NETWORKS]
    ax = axes[0]
    y = _net_axis(ax)
    for k, (a, off) in enumerate((("kcore", -0.2), ("be", 0.0), ("rombach", 0.2))):
        ax.scatter(sel(a, "spearman"), y + off, s=20, color=COLOR[a], marker=MARK[a], zorder=3,
                   edgecolors="white", linewidths=0.5, label=f"{LABEL[a]}")
    ax.set_xlim(-0.05, 1.05); ax.set_title("(a) Node level: Spearman ρ\nwith cohesion level")
    ax = axes[1]; _net_axis(ax)
    for k, (a, off) in enumerate((("kcore", -0.2), ("be", 0.0), ("rombach", 0.2))):
        ax.scatter(sel(a, "jaccard_top10"), y + off, s=20, color=COLOR[a], marker=MARK[a], zorder=3,
                   edgecolors="white", linewidths=0.5)
    ax.set_xlim(-0.05, 1.05); ax.set_title("(b) Node level: Jaccard\nof top-10% sets")
    ax = axes[2]; _net_axis(ax)
    ax.barh(y, st["group_jaccard"], color=COLOR["kcore"], height=0.5)
    ax.set_xlim(0, 1.05); ax.set_title("(c) Group level: best-match\nJaccard, k-core vs k-comp.")
    ax = axes[3]; _net_axis(ax)
    ax.barh(y, st["fragmentation"], color=COLOR["cohesion"], height=0.5)
    ax.set_xscale("log"); ax.axvline(1, color=INK2, lw=0.8)
    for yi, v in zip(y, st["fragmentation"]):
        ax.text(v * 1.12, yi, f"{v:.1f}", va="center", fontsize=6, color=INK2)
    ax.set_title("(d) Group level: fragmentation F\n(# k-comp. / # k-core groups)")
    ax.set_xlim(0.8, st["fragmentation"].max() * 3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{x} vs cohesion" for x in l], loc="lower center", ncol=3, bbox_to_anchor=(0.32, -0.09),
               fontsize=6.5)
    save(fig, "fig3_node_vs_group")


# ----------------------------------------------------------------- Fig 4: group structure by k
def fig_group_by_k():
    g = pd.read_csv(TAB / "group_structure_by_k.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7))
    for ax, net in zip(axes.flat, NETWORKS):
        s = g[g.key == net]
        ax.plot(s.k, s.n_cc, color=COLOR["kcore"], marker="o", ms=2.5, lw=1.2, label="k-core groups")
        ax.plot(s.k, s.n_kcomp, color=COLOR["cohesion"], marker="s", ms=2.5, lw=1.2, label="k-components")
        ax.set_yscale("log"); ax.set_title(SHORT[net], fontsize=7.5); ax.tick_params(labelsize=6)
        ax.set_ylim(0.8, max(2, s.n_kcomp.max(), s.n_cc.max()) * 1.6)
    for ax in axes[1]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("number of groups")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig4_group_structure_by_k")


# ----------------------------------------------------------------- Fig 5: robustness curves
CURVE_METHODS = ["random", "degree", "degree_adaptive", "betweenness", "kcore", "be", "rombach",
                 "cohesion", "broker"]


def fig_robustness_curves():
    with open(CACHE / "robustness.pkl", "rb") as fh:
        allres = pickle.load(fh)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
    for ax, net in zip(axes.flat, NETWORKS):
        res = allres[net]
        for m in CURVE_METHODS:
            S = res["curves"][m]
            q = np.arange(1, S.size + 1) / S.size
            ax.plot(q, S, color=COLOR[m], ls=STYLE[m], lw=1.6 if m in ("cohesion", "broker", "kcore") else 1.1,
                    label=LABEL[m], zorder=3 if m in ("cohesion", "kcore", "broker") else 2)
        ax.set_title(SHORT[net], fontsize=7.5); ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel("fraction removed q")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(q)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig5_robustness_curves")


def fig_robustness_R():
    df = pd.read_csv(TAB / "robustness_R.csv").set_index("key")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    meths = ["degree", "degree_adaptive", "betweenness", "kcore", "be", "rombach", "cohesion", "broker"]
    for ax, suf, title in ((axes[0], "", "(a) ties broken by degree"),
                           (axes[1], "_randtie", "(b) ties broken at random (mean of 50)")):
        y = _net_axis(ax)
        for m in meths:
            col = m + suf if (m + suf) in df else m
            if suf and col == m and m != "degree_adaptive":
                continue
            vals = df.loc[NETWORKS, col] / df.loc[NETWORKS, "random"]
            ax.scatter(vals, y, color=COLOR[m], marker=MARK[m], s=34 if m == "broker" else 22,
                       label=LABEL[m], edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_xlabel("R / R$_{random}$  (lower = more damaging)")
        ax.set_title(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=6.8)
    save(fig, "fig6_robustness_relative")


# ----------------------------------------------------------------- Fig 7: hybrids
def fig_hybrid():
    with open(CACHE / "robustness.pkl", "rb") as fh:
        allres = pickle.load(fh)
    h = pd.read_csv(TAB / "hybrid_heldout.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, kind, partner, title in ((axes[0], "hybrid_depth", "cohesion", "(a) core + cohesion"),
                                     (axes[1], "hybrid_broker", "broker", "(b) core + brokerage")):
        lam = pd.read_csv(TAB / f"{kind}_lambda_R.csv", index_col=0)
        rel = lam.copy()
        for net in NETWORKS:
            rel[net] = lam[net] / allres[net]["R"]["kcore"]
            ax.plot(rel.index, rel[net], color=INK2, alpha=0.35, lw=0.8)
        for fold, ls in (("A", "-"), ("B", "--")):
            mean = rel[FOLDS[fold]].mean(axis=1)
            ax.plot(rel.index, mean, color=COLOR[partner], lw=1.8, ls=ls, label=f"mean, fold {fold}")
            ax.scatter([mean.idxmin()], [mean.min()], color=COLOR[partner], s=26, zorder=4, edgecolors="white")
        ax.axhline(1, color=COLOR["kcore"], lw=0.8, ls=":")
        ax.set_xlabel(f"λ  (0 = {LABEL[partner].lower()}, 1 = k-core)")
        ax.set_title(title); ax.legend(fontsize=6.3, loc="upper left")
    axes[0].set_ylabel("R(λ) / R$_{k\\text{-core}}$")
    ax = axes[2]
    loo = h[(h.scheme == "leave-one-out") & (h.hybrid == "hybrid_broker")].set_index("test_net").loc[NETWORKS]
    y = _net_axis(ax)
    for col, m in (("R_a", "kcore"), ("R_b", "broker"), ("R_degree", "degree"), ("R_hybrid", "hybrid")):
        ax.scatter(loo[col] / loo["R_a"], y, color=COLOR[m], marker=MARK[m], s=26,
                   label=LABEL[m] if m != "hybrid" else "Hybrid (held out)", edgecolors="white",
                   linewidths=0.5, zorder=3)
    ax.set_yticklabels([f"{SHORT[n]} (λ*={loo.loc[n, 'lambda_star']:.1f})" for n in NETWORKS], fontsize=6.3)
    ax.set_xlabel("R / R$_{k\\text{-core}}$"); ax.set_title("(c) Leave-one-network-out")
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    save(fig, "fig7_hybrid")


# ----------------------------------------------------------------- Fig 8: null models
def fig_null():
    z = pd.read_csv(TAB / "null_model_zscores.csv")
    null = pd.read_csv(CACHE / "null_samples.csv")
    nets = [n for n in NETWORKS if n in set(null.key)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, stat, title, log in ((axes[0], "fragmentation", "(a) fragmentation F", True),
                                 (axes[1], "frac_brokers", "(b) fraction of brokers", False),
                                 (axes[2], "BE_rho", "(c) BE correlation ρ*", False)):
        y = _net_axis(ax)
        for i, net in enumerate(NETWORKS):
            if net not in nets:
                continue
            x = null[null.key == net][stat].to_numpy(float)
            jit = np.random.default_rng(i).uniform(-0.18, 0.18, x.size)
            ax.scatter(x, np.full(x.size, i) + jit, s=3, color=COLOR["random"], alpha=0.5, lw=0)
        o = z[z.stat == stat].set_index("key")
        yy = [NETWORKS.index(n) for n in o.index]
        ax.scatter(o["observed"], yy, color=COLOR["cohesion"], s=26, marker="D", zorder=3,
                   edgecolors="white", linewidths=0.6)
        if log:
            ax.set_xscale("log")
        ax.set_title(title)
    axes[0].legend(handles=[Line2D([], [], marker="D", ls="", color=COLOR["cohesion"], label="observed"),
                            Line2D([], [], marker="o", ls="", ms=3, color=COLOR["random"],
                                   label="100 degree-preserving\nrandomisations")],
                   fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, "fig8_null_models")


def main():
    for f in (fig_bowtie_karate, fig_agreement_heatmaps, fig_node_vs_group, fig_group_by_k,
              fig_robustness_curves, fig_robustness_R, fig_hybrid, fig_null):
        try:
            f()
        except (FileNotFoundError, KeyError) as e:
            print("skip", f.__name__, repr(e))


if __name__ == "__main__":
    main()
