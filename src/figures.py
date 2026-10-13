"""All paper figures.  python -m src.figures"""
from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .datasets import DATASETS, NETWORKS, ROOT, load_graph
from .pipeline import CACHE, FOLDS, LABEL, TAB, load_cache

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Fixed entity -> colour mapping (categorical slots in fixed order; never cycled).
COLOR = {
    "kcore": "#2a78d6",       # 1 blue
    "cohesion": "#eb6834",    # 2 orange
    "hybrid": "#1baf7a",      # 3 aqua
    "degree": "#eda100",      # 4 yellow
    "betweenness": "#e87ba4", # 5 magenta
    "rombach": "#008300",     # 6 green
    "be": "#4a3aa7",          # 7 violet
    "random": "#8a8984",      # neutral grey (reference, not a category)
}
STYLE = {"kcore": "-", "cohesion": "-", "hybrid": "--", "degree": ":", "betweenness": "-.",
         "rombach": (0, (5, 1)), "be": (0, (3, 1, 1, 1)), "random": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"karate": "Karate", "grqc": "GrQc", "email": "Email-EU", "facebook": "Facebook",
         "yeast": "Yeast PPI", "celegans": "C. elegans", "power": "Power grid", "airindia": "Air India"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 1.4,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


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
    nx.draw_networkx_nodes(B, pos, ax=ax, node_color="white", edgecolors=INK, node_size=330, linewidths=1.2)
    nx.draw_networkx_labels(B, pos, ax=ax, font_size=8)
    ax.set_title("(a) Bowtie: 2-core, but $\\kappa$ = 1", color=INK)
    ax.text(0, -1.05, "two 2-components share X", ha="center", fontsize=7, color=INK2)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.25, 1.0); ax.axis("off")

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
        gap = [v for v in G if r["kcore"][v] > r["cohesion"][v]]
        if key == "cohesion" and gap:
            nx.draw_networkx_nodes(G, p, nodelist=gap, ax=ax, node_color="none", edgecolors=COLOR["cohesion"],
                                   node_size=[120 + 8 * G.degree(v) for v in gap], linewidths=1.4)
        ax.set_title(title, color=INK); ax.axis("off")
        cb = fig.colorbar(nodes, ax=ax, fraction=0.04, pad=0.01, ticks=range(0, 5))
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    save(fig, "fig1_bowtie_karate")


# ----------------------------------------------------------------- Fig 2: agreement heatmaps
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


# ----------------------------------------------------------------- Fig 3: CP-vs-cohesion overlap per network
def fig_agreement_summary():
    df = pd.read_csv(TAB / "agreement_long.csv")
    nets = NETWORKS
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    metrics = [("spearman", "Spearman ρ"), ("jaccard_top10", "Jaccard, top 10%"), ("ami", "AMI (core labels)")]
    y = np.arange(len(nets))
    for ax, (met, title) in zip(axes, metrics):
        for k, (a, off) in enumerate((("kcore", -0.22), ("be", 0.0), ("rombach", 0.22))):
            vals = [df[(df.key == n) & (df.A == a) & (df.B == "cohesion")][met].values[0] for n in nets]
            ax.scatter(vals, y + off, s=22, color=COLOR[a], label=f"{LABEL[a]} vs cohesion", zorder=3,
                       edgecolors="white", linewidths=0.6, marker="osD"[k])
        ax.set_title(title); ax.set_xlim(-0.05, 1.05)
        ax.axhline(3.5, color=INK2, lw=0.6); ax.axhline(5.5, color=INK2, lw=0.6)
    axes[0].set_yticks(y, [SHORT[n] for n in nets]); axes[0].invert_yaxis()
    axes[0].text(-0.62, 1.5, "Social", rotation=90, va="center", ha="center", color=INK2, fontsize=7,
                 transform=axes[0].get_yaxis_transform())
    axes[0].text(-0.62, 4.5, "Bio.", rotation=90, va="center", ha="center", color=INK2, fontsize=7,
                 transform=axes[0].get_yaxis_transform())
    axes[0].text(-0.62, 6.5, "Infra.", rotation=90, va="center", ha="center", color=INK2, fontsize=7,
                 transform=axes[0].get_yaxis_transform())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.08))
    save(fig, "fig3_agreement_summary")


# ----------------------------------------------------------------- Fig 4: coreness vs cohesion
def fig_core_vs_cohesion():
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8))
    for ax, net in zip(axes.flat, NETWORKS):
        r = load_cache(net)
        kc, coh = r["kcore"].astype(int), r["cohesion"].astype(int)
        pairs, counts = np.unique(np.stack([kc, coh]), axis=1, return_counts=True)
        mx = max(kc.max(), 1)
        ax.plot([0, mx], [0, mx], color=GRID, lw=1, zorder=1)
        ax.scatter(pairs[0], pairs[1], s=6 + 40 * np.sqrt(counts / counts.max()), color=COLOR["kcore"],
                   alpha=0.75, edgecolors="white", linewidths=0.4, zorder=2)
        gap = np.mean(kc > coh)
        ax.set_title(f"{SHORT[net]}  (gap {100 * gap:.0f}%)", fontsize=7.5)
        ax.set_xlim(-0.03 * mx, 1.05 * mx); ax.set_ylim(-0.03 * mx, 1.05 * mx)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel("k-core coreness")
    for ax in axes[:, 0]:
        ax.set_ylabel("cohesion level")
    save(fig, "fig4_coreness_vs_cohesion")


# ----------------------------------------------------------------- Fig 5: robustness curves
ROB_METHODS = ["random", "degree", "betweenness", "kcore", "be", "rombach", "cohesion"]


def fig_robustness_curves():
    with open(CACHE / "robustness.pkl", "rb") as fh:
        allres = pickle.load(fh)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for ax, net in zip(axes.flat, NETWORKS):
        res = allres[net]
        for m in ROB_METHODS:
            S = res["curves"][m]
            q = np.arange(1, S.size + 1) / S.size
            ax.plot(q, S, color=COLOR[m], ls=STYLE[m], lw=1.2 if m != "cohesion" else 1.6,
                    label=LABEL[m], zorder=3 if m in ("cohesion", "kcore") else 2)
        ax.set_title(SHORT[net], fontsize=7.5); ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel("fraction removed q")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(q)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.05), fontsize=7)
    save(fig, "fig5_robustness_curves")


def fig_robustness_R():
    df = pd.read_csv(TAB / "robustness_R.csv").set_index("key")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    meths = ["degree", "betweenness", "kcore", "be", "rombach", "cohesion"]
    y = np.arange(len(NETWORKS))
    marks = {"degree": "v", "betweenness": "^", "kcore": "o", "be": "D", "rombach": "P", "cohesion": "s"}
    for m in meths:
        vals = df.loc[NETWORKS, m] / df.loc[NETWORKS, "random"]
        ax.scatter(vals, y, color=COLOR[m], marker=marks[m], s=24, label=LABEL[m],
                   edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_yticks(y, [SHORT[n] for n in NETWORKS]); ax.invert_yaxis()
    ax.set_xlabel("R / R$_{random}$  (lower = more damaging)")
    ax.legend(fontsize=6, loc="lower right", handletextpad=0.2)
    save(fig, "fig6_robustness_relative")


# ----------------------------------------------------------------- Fig 7: hybrid lambda
def fig_hybrid():
    lam = pd.read_csv(TAB / "hybrid_lambda_R.csv", index_col=0)
    with open(CACHE / "robustness.pkl", "rb") as fh:
        allres = pickle.load(fh)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    rel = lam.copy()
    for net in NETWORKS:
        rel[net] = lam[net] / allres[net]["R"]["kcore"]
        ax.plot(rel.index, rel[net], color=INK2, alpha=0.45, lw=0.9)
        ax.text(1.01, rel[net].iloc[-1], SHORT[net], fontsize=5.5, color=INK2, va="center")
    for fold, col in (("A", COLOR["hybrid"]), ("B", COLOR["be"])):
        mean = rel[FOLDS[fold]].mean(axis=1)
        ax.plot(rel.index, mean, color=col, lw=2, label=f"mean, fold {fold}")
        ax.scatter([mean.idxmin()], [mean.min()], color=col, s=30, zorder=4, edgecolors="white")
    ax.set_xlabel("λ  (0 = cohesion only, 1 = k-core only)")
    ax.set_ylabel("R(λ) / R$_{k\\text{-core}}$")
    ax.set_title("(a) Robustness vs. mixing weight")
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axes[1]
    h = pd.read_csv(TAB / "hybrid_heldout.csv")
    loo = h[h.scheme == "leave-one-out"].set_index("test_net").loc[NETWORKS]
    y = np.arange(len(NETWORKS))
    for col, m, mk in (("R_kcore", "kcore", "o"), ("R_cohesion", "cohesion", "s"), ("R_hybrid", "hybrid", "D")):
        ax.scatter(loo[col] / loo["R_kcore"], y, color=COLOR[m], marker=mk, s=24, label=LABEL[m],
                   edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_yticks(y, [f"{SHORT[n]} (λ*={loo.loc[n, 'lambda_star']:.1f})" for n in NETWORKS], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel("R / R$_{k\\text{-core}}$ (held out)")
    ax.set_title("(b) Leave-one-network-out evaluation")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig7_hybrid")


# ----------------------------------------------------------------- Fig 8: null model
def fig_null():
    z = pd.read_csv(TAB / "null_model_zscores.csv")
    null = pd.read_csv(CACHE / "null_samples.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax = axes[0]
    y = np.arange(len(NETWORKS))
    obs = z[z.stat == "cohesion_deficit"].set_index("key").loc[NETWORKS]
    for i, net in enumerate(NETWORKS):
        x = null[null.key == net]["cohesion_deficit"]
        ax.scatter(x, np.full(x.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, x.size),
                   s=4, color=COLOR["random"], alpha=0.5, lw=0)
    ax.scatter(obs["observed"], y, color=COLOR["cohesion"], s=30, marker="D", zorder=3, label="observed",
               edgecolors="white")
    ax.scatter([], [], s=6, color=COLOR["random"], label="degree-preserving null")
    ax.set_yticks(y, [SHORT[n] for n in NETWORKS]); ax.invert_yaxis()
    ax.set_xlabel("cohesion deficit Γ"); ax.set_title("(a) Cohesion deficit vs. null")
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    be = z[z.stat == "BE_rho"].set_index("key").loc[NETWORKS]
    ax.barh(y, be["z"], color=COLOR["be"], height=0.55)
    ax.axvline(1.96, color=INK2, ls=":", lw=0.8); ax.axvline(-1.96, color=INK2, ls=":", lw=0.8)
    ax.set_yticks(y, [SHORT[n] for n in NETWORKS]); ax.invert_yaxis()
    ax.set_xlabel("z-score of BE correlation ρ*"); ax.set_title("(b) Core–periphery quality vs. null")
    fig.tight_layout()
    save(fig, "fig8_null_models")


def main():
    fig_bowtie_karate()
    fig_agreement_heatmaps()
    fig_agreement_summary()
    fig_core_vs_cohesion()
    for f in (fig_robustness_curves, fig_robustness_R, fig_hybrid, fig_null):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()
