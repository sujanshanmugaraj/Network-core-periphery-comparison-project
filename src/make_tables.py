"""Generate the LaTeX tables of the paper from results/tables/*.csv.

    python -m src.make_tables          -> paper/tables/*.tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import DATASETS, NETWORKS, ROOT
from .pipeline import TAB

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"karate": "Karate", "grqc": "GrQc", "email": "Email-EU", "facebook": "Facebook",
         "yeast": "Yeast PPI", "celegans": "\\emph{C.~elegans}", "power": "Power grid",
         "airindia": "Air India"}
DOM = {"Social": "Soc.", "Biological": "Bio.", "Infrastructure": "Inf."}


def write(name, body):
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def midrules(i):
    return "\\midrule\n" if NETWORKS[i] in ("yeast", "power") else ""


def tab_datasets():
    d = pd.read_csv(TAB / "dataset_table.csv").set_index("key").loc[NETWORKS]
    s = ["\\setlength{\\tabcolsep}{3pt}\\footnotesize",
         "\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Network & Dom. & $N$ & $M$ & $\\langle d\\rangle$ & $C$ & $r$ & $k_{\\max}$ & $\\kappa_{\\max}$\\\\",
         "\\midrule"]
    for i, (k, r) in enumerate(d.iterrows()):
        s.append(midrules(i) + f"{SHORT[k]} & {DOM[r.domain]} & \\num{{{int(r.N)}}} & \\num{{{int(r.M)}}} & "
                 f"{r.avg_degree:.1f} & {r.avg_clustering:.2f} & {r.assortativity:.2f} & {int(r.degeneracy)} & "
                 f"{int(r.max_cohesion)}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_datasets", "\n".join(s))


def tab_agreement():
    w = pd.read_csv(TAB / "agreement_cp_vs_cohesion_wide.csv").set_index("key").loc[NETWORKS]
    d = pd.read_csv(TAB / "dataset_table.csv").set_index("key").loc[NETWORKS]
    s = ["\\setlength{\\tabcolsep}{2.6pt}\\footnotesize",
         "\\begin{tabular}{l ccc c c | ccc r}", "\\toprule",
         " & \\multicolumn{5}{c|}{Node level} & \\multicolumn{4}{c}{Group level}\\\\",
         " & \\multicolumn{3}{c}{$\\rho_s$ with $\\kappa$} & $J_{10}$ & gap & & & & \\\\",
         "Network & $c$ & BE & CS & $c$,$\\kappa$ & \\% & $F$ & $J_G$ & cov. & brok.\\\\", "\\midrule"]
    for i, k in enumerate(NETWORKS):
        a, r = w.loc[k], d.loc[k]
        s.append(midrules(i) + f"{SHORT[k]} & {fmt(a.spearman_kcore, 2)} & {fmt(a.spearman_be, 2)} & "
                 f"{fmt(a.spearman_rombach, 2)} & {fmt(a.jaccard_top10_kcore, 2)} & {100 * r.frac_gap_nodes:.1f} & "
                 f"{r.fragmentation:.2f} & {r.group_jaccard:.2f} & {r.coverage:.2f} & {int(r.n_brokers)}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_agreement", "\n".join(s))


def tab_null():
    z = pd.read_csv(TAB / "null_model_zscores.csv")
    get = lambda k, st, col: z[(z.key == k) & (z.stat == st)][col].values[0] if not z[(z.key == k) & (z.stat == st)].empty else np.nan
    s = ["\\setlength{\\tabcolsep}{2.5pt}\\footnotesize",
         "\\begin{tabular}{l rr rr rr r}", "\\toprule",
         " & \\multicolumn{2}{c}{gap (\\%)} & \\multicolumn{2}{c}{$F$} & \\multicolumn{2}{c}{brokers (\\%)} & BE $\\rho^*$\\\\",
         "Network & obs. & null & obs. & null & obs. & null & $z$\\\\", "\\midrule"]
    for i, k in enumerate(NETWORKS):
        if np.isnan(get(k, "fragmentation", "observed")):
            continue
        g = lambda st, col: get(k, st, col)
        s.append(midrules(i) + f"{SHORT[k]} & {100 * g('frac_gap_nodes', 'observed'):.1f} & "
                 f"{100 * g('frac_gap_nodes', 'null_mean'):.2f} & {g('fragmentation', 'observed'):.2f} & "
                 f"{g('fragmentation', 'null_mean'):.2f} & {100 * g('frac_brokers', 'observed'):.1f} & "
                 f"{100 * g('frac_brokers', 'null_mean'):.2f} & {fmt(g('BE_rho', 'z'), 1)}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_null", "\n".join(s))


ROB = ["random", "degree", "betweenness", "kcore", "be", "rombach", "cohesion", "broker", "degree_adaptive"]
ROB_HEAD = ["Rand.", "Deg.", "Betw.", "$c$", "BE", "CS", "$\\kappa$", "$b$", "Deg.$^\\dagger$"]


def tab_robustness():
    d = pd.read_csv(TAB / "robustness_R.csv").set_index("key").loc[NETWORKS]
    static = [m for m in ROB if m not in ("random", "degree_adaptive")]
    s = ["\\setlength{\\tabcolsep}{2.4pt}\\footnotesize",
         "\\begin{tabular}{l " + "r" * len(ROB) + "}", "\\toprule",
         "Network & " + " & ".join(ROB_HEAD) + "\\\\", "\\midrule"]
    for i, k in enumerate(NETWORKS):
        r = d.loc[k]
        best = min(r[m] for m in static)
        cells = []
        for m in ROB:
            v = f"{r[m]:.3f}"
            if m in static and abs(r[m] - best) < 1e-12:
                v = f"\\textbf{{{v}}}"
            if m == "degree_adaptive" and r[m] < best - 1e-12:
                v = f"\\underline{{{v}}}"
            cells.append(v)
        s.append(midrules(i) + f"{SHORT[k]} & " + " & ".join(cells) + "\\\\")
    # mean rank among static orderings
    ranks = d[static].rank(axis=1)
    s.append("\\midrule")
    s.append("Mean rank & -- & " + " & ".join(f"{ranks[m].mean():.2f}" for m in static) + " & --\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_robustness", "\n".join(s))


NAMES = {"cohesion": "$\\kappa$", "kcore": "$c$", "be": "BE", "rombach": "CS", "degree": "Deg.",
         "betweenness": "Betw.", "broker": "$b$", "degree_adaptive": "Deg.$^\\dagger$"}


def tab_comparisons():
    c = pd.read_csv(TAB / "robustness_comparisons.csv")
    main = c[c.variant == "degree_tiebreak"].set_index(["A", "B"])
    rt = c[c.variant == "random_tiebreak"].set_index(["A", "B"])
    s = ["\\setlength{\\tabcolsep}{2.5pt}\\footnotesize",
         "\\begin{tabular}{l r c c r r | r r}", "\\toprule",
         " & \\multicolumn{5}{c|}{ties broken by degree} & \\multicolumn{2}{c}{random ties}\\\\",
         "A vs.\\ B & $\\overline{\\Delta R}$ & 95\\% CI & W/L & $p_{\\text{Holm}}$ & $r_{rb}$ & $\\overline{\\Delta R}$ & $r_{rb}$\\\\",
         "\\midrule"]
    for (a, b), r in main.iterrows():
        q = rt.loc[(a, b)]
        s.append(f"{NAMES[a]} vs.\\ {NAMES[b]} & {r.mean_diff:+.3f} & [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & "
                 f"{int(r.wins_A)}/{int(r.wins_B)} & {fmt_p(r.holm_p)} & {r.rank_biserial:+.2f} & "
                 f"{q.mean_diff:+.3f} & {q.rank_biserial:+.2f}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_comparisons", "\n".join(s))


def tab_hybrid():
    h = pd.read_csv(TAB / "hybrid_heldout.csv")
    c = pd.read_csv(TAB / "hybrid_comparisons.csv")
    lbl = {"hybrid_depth": "$c$+$\\kappa$", "hybrid_broker": "$c$+$b$"}
    s = ["\\setlength{\\tabcolsep}{2.6pt}\\footnotesize",
         "\\begin{tabular}{l l c r r r r}", "\\toprule",
         "Hybrid & scheme & $\\lambda^*$ & $\\overline{R}_{H}$ & $\\overline{R}_{c}$ & $\\overline{R}_{x}$ & W/L vs.\\ best\\\\",
         "\\midrule"]
    for kind in ["hybrid_depth", "hybrid_broker"]:
        for scheme in ["2-fold", "leave-one-out"]:
            sub = h[(h.hybrid == kind) & (h.scheme == scheme)]
            lams = sorted(set(sub.lambda_star))
            lam_s = ", ".join(f"{x:.1f}" for x in lams)
            cc = c[(c.hybrid == kind) & (c.scheme == scheme) & (c.B == "best_single")].iloc[0]
            s.append(f"{lbl[kind]} & {scheme} & {lam_s} & {sub.R_hybrid.mean():.3f} & {sub.R_a.mean():.3f} & "
                     f"{sub.R_b.mean():.3f} & {int(cc.wins_A)}/{int(cc.wins_B)}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_hybrid", "\n".join(s))


def tab_ablation():
    a = pd.read_csv(TAB / "broker_ablation.csv").set_index("key").loc[NETWORKS]
    cols = [("R_degree", "Deg."), ("R_articulation", "art."), ("R_broker_k2", "$b_{k=2}$"),
            ("R_broker_k3plus", "$b_{k\\ge3}$"), ("R_broker", "$b$")]
    s = ["\\setlength{\\tabcolsep}{2.8pt}\\footnotesize",
         "\\begin{tabular}{l rrr | rrrrr}", "\\toprule",
         " & \\multicolumn{3}{c|}{\\# nodes} & \\multicolumn{5}{c}{$R$ (ties broken by degree)}\\\\",
         "Network & art. & $b_{k=2}$ & $b_{k\\ge3}$ & " + " & ".join(h for _, h in cols) + "\\\\", "\\midrule"]
    for i, k in enumerate(NETWORKS):
        r = a.loc[k]
        best = min(r[c] for c, _ in cols)
        cells = [f"\\textbf{{{r[c]:.3f}}}" if abs(r[c] - best) < 1e-12 else f"{r[c]:.3f}" for c, _ in cols]
        s.append(midrules(i) + f"{SHORT[k]} & {int(r.n_articulation)} & {int(r.n_broker_k2)} & "
                 f"{int(r.n_broker_k3plus)} & " + " & ".join(cells) + "\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_ablation", "\n".join(s))


def tab_bench():
    b = pd.read_csv(TAB / "cohesion_benchmark.csv").set_index("key")
    s = ["\\setlength{\\tabcolsep}{3pt}\\footnotesize",
         "\\begin{tabular}{l r r r r}", "\\toprule",
         "Network & ours (s) & NetworkX (s) & levels differ & non-max.\\ sets\\\\", "\\midrule"]
    for k in NETWORKS:
        if k not in b.index:
            continue
        r = b.loc[k]
        tn = "$>$\\num{600}" if pd.isna(r.t_networkx_s) else f"{r.t_networkx_s:.1f}"
        dl = "--" if pd.isna(r.get("levels_differ", np.nan)) else f"{int(r.levels_differ)}"
        nm = "--" if pd.isna(r.get("nx_nonmaximal_sets", np.nan)) else f"{int(r.nx_nonmaximal_sets)}"
        s.append(f"{SHORT[k]} & {r.t_ours_s:.2f} & {tn} & {dl} & {nm}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("tab_bench", "\n".join(s))


def main():
    for f in (tab_datasets, tab_agreement, tab_null, tab_robustness, tab_comparisons, tab_hybrid,
              tab_ablation, tab_bench):
        try:
            f()
        except (FileNotFoundError, KeyError, IndexError) as e:
            print("skip", f.__name__, repr(e))


if __name__ == "__main__":
    main()
