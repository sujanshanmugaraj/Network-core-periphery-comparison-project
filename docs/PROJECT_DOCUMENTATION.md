# Project documentation — end to end

**Do Two Notions of "Core" Agree? Core–Periphery Structure versus Structural Cohesion, Validated by
Network Robustness**

This document explains the whole project from start to finish: the research question, the concepts,
the data, every piece of code, every experiment, the results and how to reproduce them, plus the
decisions and corrections made along the way and a set of likely viva questions. It is meant to be read
on its own; the paper (`paper/main.pdf`) is the formal write-up of the same work.

---

## Contents

1. [The research question in one page](#1-the-research-question-in-one-page)
2. [Concepts you must know](#2-concepts-you-must-know)
3. [Datasets](#3-datasets)
4. [Repository structure](#4-repository-structure)
5. [The pipeline, stage by stage](#5-the-pipeline-stage-by-stage)
6. [Algorithms in detail](#6-algorithms-in-detail)
7. [Experimental design](#7-experimental-design)
8. [Results](#8-results)
9. [Figures and tables: where each number comes from](#9-figures-and-tables-where-each-number-comes-from)
10. [Reproducing everything](#10-reproducing-everything)
11. [Testing and validation](#11-testing-and-validation)
12. [Decisions, corrections and lessons learned](#12-decisions-corrections-and-lessons-learned)
13. [Limitations and future work](#13-limitations-and-future-work)
14. [Viva preparation: likely questions and answers](#14-viva-preparation-likely-questions-and-answers)
15. [References](#15-references)

---

## 1. The research question in one page

Network scientists use the word "core" in two different ways.

* **Core–periphery (CP) / k-core view: the core is where ties are dense.** Borgatti & Everett (2000)
  fit an ideal "dense core, sparse periphery" pattern; Rombach et al. (2014) give a continuous core
  score; the k-core decomposition (Seidman 1983) peels away low-degree nodes.
* **Structural-cohesion view: the core is what is hardest to pull apart.** Moody & White (2003) define
  a group's cohesion as the number of nodes you must delete to disconnect it, and organise groups into
  a nested hierarchy of *k-components*.

Every k-component lies inside the k-core, but not every k-core is k-cohesive (two triangles sharing one
node form a 2-core, but deleting the shared node disconnects them). The project asks:

| | Question | Hypothesis (stated in advance) | Outcome |
|---|---|---|---|
| **RQ1** | How much do CP cores and cohesion cores agree? | H1: agreement is lower in spatial networks | **Partly**: true for BE/Rombach, **not** for k-cores (agreement ≈ 1 everywhere at node level) |
| **RQ2** | Which ordering of nodes collapses a network fastest? | H2: cohesion beats coreness on clustered networks | **Not supported**: cohesion ≈ coreness. A new cohesion-derived score (*brokerage*) does help |
| **RQ3** | Does a hybrid of coreness and cohesion beat both on unseen networks? | H3: yes | **Not supported**: no hybrid beats its best component on held-out networks |

**Main finding.** The two notions agree on *who* is central (node rankings are almost identical) but
disagree on *how* the centre is organised: one connected k-core often consists of many overlapping
k-components (up to 7.7× as many groups). This group-level gap is absent from degree-preserving
randomisations, and the nodes that glue the groups together — **cohesive brokers** — are structurally
critical: removing them first beats k-coreness and cohesion in all 8 networks (up to 44 % lower
robustness index).

---

## 2. Concepts you must know

### k-core and coreness
Repeatedly delete every node with fewer than *k* neighbours; what survives is the **k-core**. A node's
**coreness** *c(v)* is the largest *k* for which it survives. Linear time (Batagelj & Zaveršnik 2003).

Worked trace (triangle A–B–C, D attached to C, E attached to D): peeling at k = 2 removes E (degree 1),
which drops D to degree 1, so D goes too; A, B, C remain. Coreness: A = B = C = 2, D = E = 1. Note the
cascade: removing E made D removable.

### Node connectivity, k-components, cohesion level
* **Node connectivity κ(H)**: the minimum number of nodes whose removal disconnects H
  (complete graph: |H| − 1). By **Menger's theorem**, κ ≥ k iff every pair of nodes is joined by k
  node-disjoint paths.
* **k-component**: a *maximal* node set K with |K| > k whose induced subgraph is k-connected.
* **Cohesion level κ(v)**: the largest k such that v lies in some k-component.
* k-components **nest** (a 3-component sits inside a 2-component) and may **overlap**.

### The bowtie (the paper's key example)
Triangles {A, B, X} and {X, C, D} share node X.
* Every node has degree ≥ 2 → the whole graph is the 2-core, coreness = 2 for all.
* Removing X disconnects the graph → κ(G) = 1.
* Each triangle is a 2-component → cohesion level = 2 for all.
* **Node level:** coreness = cohesion everywhere (perfect agreement).
* **Group level:** one 2-core group = two 2-components that overlap in X. X is a **cohesive broker**.

### Three propositions (proved in the paper, Section III)
1. **κ(v) ≤ c(v)** for every node (a k-component has minimum degree ≥ k, so it lies in the k-core).
2. Every k-component lies inside **one connected component of the k-core**, and a k-core component that
   is itself k-connected **is** a k-component.
3. Two distinct k-components share **at most k − 1 nodes** (otherwise their union would be k-connected,
   contradicting maximality). So brokers are always small "glue" sets.

### Cohesive brokerage (new score introduced in this project)
`b(v) = Σ_{k≥2} max(0, m_k(v) − 1)`, where `m_k(v)` = number of k-components containing v.
It counts how many *extra* cohesive groups a node belongs to, over all levels. At k = 2 brokers are
articulation points shared by bicomponents; at k ≥ 3 they are small vertex sets gluing dense groups.

### Core–periphery models
* **Borgatti–Everett (BE), discrete:** label nodes core/periphery to maximise the Pearson correlation
  between the adjacency matrix and the ideal pattern (core–core = 1, core–periphery = 1,
  periphery–periphery = 0).
* **Rombach et al. (2014):** assign each node a value from a piecewise-linear profile controlled by
  α (sharpness) and β (periphery fraction); find the permutation maximising core quality
  R = Σ A_uv C_u C_v; aggregate over an (α, β) grid into a core score CS(v) ∈ [0, 1].
  The formula was checked against the original SIAM paper (Eq. 9 and Eq. 11).

### Robustness index
Remove nodes in some order; S(q) = size of the largest connected component (as a fraction of the
original N) after removing a fraction q. **R = (1/N) Σ_{Q=1}^{N} S(Q/N)** (Schneider et al. 2011),
the area under the curve. **Lower R = the ordering found more critical nodes.**

### Group-level agreement measures (introduced here)
* **Fragmentation F** = Σ_k (#k-components) / Σ_k (#connected k-core groups). F = 1 means every k-core
  group is k-cohesive.
* **Group Jaccard J_G**: size-weighted best-match Jaccard similarity of each k-core group to a
  k-component, averaged over k. 1 = identical group structure. Unlike F, it is not inflated by size.
* **Coverage**: fraction of k-core nodes that belong to some k-component.

---

## 3. Datasets

All networks are converted to **simple, undirected, unweighted** graphs (directions, weights,
self-loops and duplicate edges dropped) and reduced to their **largest connected component**.
Nodes are relabelled 0..N−1 (original labels kept in the node attribute `label`).

| Key | Network | Domain | N | M | Avg. degree | Clustering | Degeneracy | Source |
|---|---|---|---|---|---|---|---|---|
| `karate` | Zachary karate club | Social | 34 | 78 | 4.6 | 0.57 | 4 | NetworkX built-in |
| `grqc` | arXiv GR-QC co-authorship | Social | 4,158 | 13,422 | 6.5 | 0.56 | 43 | SNAP `ca-GrQc` |
| `email` | EU research institution e-mail | Social | 986 | 16,064 | 32.6 | 0.41 | 34 | SNAP `email-Eu-core` |
| `facebook` | Facebook ego networks (combined) | Social | 4,039 | 88,234 | 43.7 | 0.61 | 115 | SNAP `ego-Facebook` |
| `yeast` | Yeast protein interactions (Collins 2007) | Biological | 1,004 | 8,319 | 16.6 | 0.65 | 40 | Netzschleuder `collins_yeast` |
| `celegans` | *C. elegans* neural network | Biological | 297 | 2,148 | 14.5 | 0.29 | 10 | Netzschleuder `celegansneural` |
| `power` | Western US power grid | Infrastructure | 4,941 | 6,594 | 2.7 | 0.08 | 5 | Netzschleuder `power` |
| `airindia` | Indian domestic air routes | Infrastructure | 68 | 179 | 5.3 | 0.61 | 6 | OpenFlights (both airports in India) |

Notes:
* **Facebook** (McAuley & Leskovec, NIPS 2012) merges the friend lists of 10 anonymised survey
  volunteers; nodes are people (numbers only, no personal data), edges are friendships.
* **Yeast**: the Collins et al. (2007) high-confidence interactome; 1,622 proteins before the LCC step.
* **Air India**: built from OpenFlights `airports.dat` + `routes.dat`, keeping routes whose origin and
  destination airports both have country = "India".
* Full statistics (density, assortativity, diameter, path length, …): `results/tables/dataset_table.csv`.

Raw files are **not** committed (licensing/size); `python -m src.download_data` fetches them
(≈ 4 MB total) from SNAP, Netzschleuder and the OpenFlights GitHub repository.

---

## 4. Repository structure

```
├── README.md                      quick start and summary
├── requirements.txt               pinned Python dependencies
├── docs/PROJECT_DOCUMENTATION.md  this file
├── data/raw/                      downloaded raw data (git-ignored)
├── src/
│   ├── download_data.py           fetch the 8 datasets
│   ├── datasets.py                load_graph(), cleaning, CSR conversion, statistics
│   ├── coreness.py                k-core, Borgatti–Everett (numba), Rombach (numba SA)
│   ├── cohesion.py                exact k-components, cohesion levels, brokerage, group metrics
│   ├── robustness.py              exact robustness curves (union–find), orderings, adaptive degree
│   ├── agreement.py               Spearman, tie-inclusive top-x% Jaccard, AMI, core labels
│   ├── nullmodels.py              degree-preserving double-edge swaps (numba)
│   ├── stats.py                   Wilcoxon, rank-biserial, bootstrap CI, Holm
│   ├── pipeline.py                all experiment stages (cached, seeded, parallel)
│   ├── benchmark_cohesion.py      runtime/correctness benchmark vs networkx
│   ├── figures.py                 all paper figures
│   └── make_tables.py             LaTeX tables from the result CSVs
├── tests/                         59 unit tests (pytest)
├── results/
│   ├── tables/                    every result as CSV
│   ├── figures/                   every figure as PDF (paper) and PNG (slides)
│   └── cache/                     per-network scores (git-ignored; null_samples.csv is kept)
├── paper/
│   ├── main.tex, main.pdf         IEEE-format paper
│   ├── references.bib             bibliography (46 entries)
│   ├── tables/                    generated LaTeX tables
│   └── literature_notes.md        Phase-1 reading notes
└── slides/                        presentation deck source (one HTML file per slide)
```

---

## 5. The pipeline, stage by stage

Every stage is a sub-command of `python -m src.pipeline <stage>`. Stages read and write
`results/cache/` so they can be re-run independently; all randomness is derived from `SEED = 20240601`.

| Stage | What it does | Main outputs |
|---|---|---|
| `scores` | For each network: statistics, degree, exact betweenness, k-core, BE (20 restarts), Rombach (54 (α,β) pairs), exact k-components → cohesion levels | `cache/<net>.pkl`, `tables/dataset_table.csv`, `tables/group_structure_by_k.csv` |
| `agreement` | RQ1 node level: Spearman, top-5/10/20 % Jaccard, AMI for every pair of {degree, k-core, BE, Rombach, cohesion} | `tables/agreement_long.csv`, `agreement_cp_vs_cohesion_wide.csv`, `agreement_domain_summary.csv` |
| `robustness` | RQ2: robustness curves and R for every ordering (degree, betweenness, k-core, BE, Rombach, cohesion, brokerage, adaptive degree, random×30), random-tie sensitivity (50 draws), paired statistics, structure-vs-gain correlations | `cache/robustness.pkl`, `tables/robustness_R.csv`, `robustness_comparisons.csv`, `h2_structure_vs_gain.csv`, `h2_correlations.csv` |
| `ablation` | Brokerage restricted to k = 2, to k ≥ 3, all levels, vs all articulation points | `tables/broker_ablation.csv` |
| `hybrid` | RQ3: λ-sweep for core+cohesion and core+brokerage; 2-fold (domain-stratified) and leave-one-out held-out evaluation | `tables/hybrid_*_lambda_R.csv`, `hybrid_heldout.csv`, `hybrid_comparisons.csv` |
| `null` | 100 degree-preserving randomisations per network; recompute BE ρ*, gap, fragmentation, J_G, coverage, brokers; z-scores and empirical p | `cache/null_samples.csv`, `tables/null_model_zscores.csv` |
| `correlations` | Recompute only the structure-vs-gain correlations | `tables/h2_correlations.csv` |

Separate scripts: `src.benchmark_cohesion` (Table VIII), `src.figures` (Figs. 1–8),
`src.make_tables` (LaTeX tables in `paper/tables/`).

---

## 6. Algorithms in detail

### 6.1 Borgatti–Everett (discrete) — `coreness.py`
The correlation has a closed form in four counts: pairs Π = N(N−1)/2, edges M, periphery size n_p,
periphery–periphery edges E_pp:

```
P = Π − n_p(n_p−1)/2,  Σ = M − E_pp,
ρ = (Π·Σ − M·P) / sqrt((Π·M − M²)(Π·P − P²))
```

Flipping node v changes n_p by ±1 and E_pp by its number of periphery neighbours, so each candidate flip
is scored in O(1) and applied in O(degree). Greedy label switching starts from the top 10 % of nodes by
degree as core, visits nodes in random order, flips when ρ improves, stops after a pass with no change;
best of 20 visiting orders. Compiled with Numba.

### 6.2 Rombach et al. — `coreness.py`
Profile (Rombach 2014, Eq. 9), positions i = 1..N, b = ⌊βN⌋:
`C*_i = i(1−α)/(2b)` for i ≤ b, `C*_i = (i−b)(1−α)/(2(N−b)) + (1+α)/2` for i > b; normalised to sum 1.
Simulated annealing over swaps of two nodes' values. With s = A·C and Δ = C_v − C_u, a swap changes the
quality by `ΔR = 2Δ(s_u − s_v) − 2A_uvΔ²` (O(log degree) to evaluate, O(degree) to apply). Start from
the degree ordering; max(500N, 300 000) proposals; geometric cooling from T₀ (mean |ΔR| of 2000 random
proposals) to 10⁻⁶T₀. Grid α ∈ {0, 0.2, …, 1}, β ∈ {0.1, …, 0.9} (54 pairs; the original uses
100 × 100, infeasible here). `CS(v) = Z Σ C_v(α,β) R(α,β)`, normalised so max CS = 1.

### 6.3 Exact k-components — `cohesion.py` (methodological contribution)
**Why not networkx?** `networkx.k_components` can return **non-maximal** components: on the
Les Misérables graph at k = 3 it reports a 32-node set strictly inside a 3-connected 34-node set (also
on Air India). Its heuristics are not guaranteed exact.

**Algorithm (certified cut splitting):**
1. Level 2 = biconnected components with ≥ 3 nodes (Hopcroft–Tarjan, linear time).
2. For a piece P at level k: prune to its k-core; for each connected component C with |C| > k ask
   `SmallCut(C, k)`:
   * no cut of size < k → C is a k-component; recurse into C at level k + 1;
   * cut S found → recurse on D ∪ S for every component D of C − S (same k).
3. `SmallCut` uses the Even/Esfahanian test: with v of minimum degree, C is k-connected iff κ(v, w) ≥ k
   for every non-neighbour w of v and κ(x, y) ≥ k for every non-adjacent pair of v's neighbours.

**Why splitting is exact:** a k-component K survives the removal of any |S| < k nodes, so K∖S lies in
one component D and K ⊆ D ∪ S — nothing is ever lost, and a piece that is k-connected is necessarily
maximal.

**Certificates (Lemma 1)** — most pairs need no max-flow:
* |N(x) ∩ N(y)| ≥ k ⇒ κ(x, y) ≥ k (k disjoint paths of length 2);
* "bootstrap closure": start with T = {x} ∪ N(x); repeatedly add any node with ≥ k neighbours in T (or
  verified by flow). Every node of T is k-connected to x (proof by induction: a cut S with |S| < k
  leaves at least one earlier T-neighbour outside S).

Remaining pairs: unit-capacity augmenting-path max-flow on the node-split graph (v_in → v_out capacity
1; edge arcs "infinite"), with a cutoff at k, compiled with Numba; the minimum vertex cut is read from
the residual graph.

**Validation:** identical to exhaustive brute force on 25 random graphs; every set k-connected and every
networkx set contained in one of ours on 4 classic graphs; identical output to the earlier pure-networkx
flow implementation on 4 real networks. Facebook's full hierarchy (k up to 115): ≈ 83 s.

### 6.4 Robustness curves — `robustness.py`
Naive recomputation of the largest component after every removal is O(N·M). We instead **add nodes back
in reverse order** with a union–find structure (Newman–Ziff), which gives the exact curve in
O(M α(N)). Verified against the naive implementation in the tests. Adaptive degree attack: repeatedly
remove the current highest-degree node and update neighbours' degrees.

**Tie-breaking** (important: coreness, cohesion and brokerage have huge ties): main analysis = score,
then degree, then node index. Sensitivity analysis = all ties broken uniformly at random, R averaged over
50 draws.

### 6.5 Null model — `nullmodels.py`
Maslov–Sneppen double-edge swaps: pick edges (a,b), (c,d), rewire to (a,d), (c,b) unless that creates a
self-loop or multi-edge; 10 × M accepted swaps; keep the LCC. Implemented in Numba with a hash set of
edges. Degree sequence preserved exactly (tested).

### 6.6 Statistics — `stats.py`
* Paired over networks: mean ΔR with a 95 % bootstrap CI (10⁴ resamples of networks).
* Exact two-sided Wilcoxon signed-rank test **after discarding zero differences** (exact ties between
  methods are common).
* Effect size: matched-pairs rank-biserial r_rb = (W⁺ − W⁻)/(W⁺ + W⁻).
* Holm correction within each family of comparisons.
* With n = 8 the smallest possible exact two-sided p is 2/2⁸ ≈ 0.008; after Holm over 12 comparisons
  the smallest adjusted p is ≈ 0.094. **We therefore report effect sizes and directions, not
  significance stars.**

---

## 7. Experimental design

* **RQ1 agreement.** Node level: Spearman; Jaccard of tie-inclusive top-x % sets (x = 5, 10, 20);
  AMI between "natural" binary core labels (main k-core; deepest k-components; BE labels; top-|BE core|
  by Rombach CS). Group level: F, J_G, coverage, number of brokers.
* **RQ1 null model.** 100 randomisations per network → z-scores and empirical p
  (p = min{1, 2·min(p≥, p≤)}, p≥ = (1 + #{null ≥ observed})/101; smallest attainable 0.0198).
* **RQ2 robustness.** Static orderings: degree, betweenness (exact), k-core, BE, Rombach, cohesion,
  brokerage; references: random (mean of 30), adaptive degree.
* **RQ3 hybrid.** `H(λ) = λ·rank(coreness) + (1−λ)·rank(x)`, x ∈ {cohesion, brokerage}, ranks
  normalised to [0, 1], λ ∈ {0, 0.1, …, 1}. λ chosen on training networks by minimising mean
  R(λ)/R_kcore, evaluated **only** on held-out networks:
  * 2-fold, domain-stratified, both directions — A = {Karate, Email-EU, Yeast, Power grid},
    B = {GrQc, Facebook, C. elegans, Air India};
  * leave-one-network-out.
* **Ablation.** Brokerage at k = 2 only, k ≥ 3 only, all levels, vs all articulation points.
* **Drivers.** Spearman correlations (across the 8 networks) between the gaps/gains and network
  properties (size, density, clustering, assortativity, …).

---

## 8. Results

### RQ1 — node level: the two cores are the same ranking
| Network | ρ(k-core, κ) | ρ(BE, κ) | ρ(Rombach, κ) | Top-10 % Jaccard (k-core, κ) | Gap nodes |
|---|---|---|---|---|---|
| Karate | 1.00 | 0.54 | 0.78 | 1.00 | 0 % |
| GrQc | 0.99 | 0.37 | 0.92 | 1.00 | 4.6 % |
| Email-EU | 1.00 | 0.40 | 0.99 | 1.00 | 0.4 % |
| Facebook | 1.00 | 0.39 | 0.98 | 0.96 | 15.7 % |
| Yeast | 1.00 | 0.36 | 0.95 | 1.00 | 8.3 % |
| C. elegans | 1.00 | 0.24 | 0.95 | 1.00 | 0 % |
| Power grid | 0.99 | 0.27 | 0.73 | 1.00 | 1.2 % |
| Air India | 1.00 | 0.36 | 0.98 | 1.00 | 0 % |

* 87 % of gap nodes lose exactly one level; the cohesion deficit (1 − Σκ/Σc) is ≤ 1 % everywhere.
* Main k-core = deepest k-components in all 8 networks (AMI = 1).
* CP models agree least with cohesion on infrastructure networks (Rombach mean ρ: social 0.92,
  biological 0.95, infrastructure 0.85) → H1 holds for CP models but **not** for k-cores.

### RQ1 — group level: the organisation differs
| Network | Fragmentation F | Group Jaccard J_G | Brokers |
|---|---|---|---|
| Karate | 1.67 | 0.86 | 1 |
| GrQc | 5.72 | 0.71 | 558 |
| Email-EU | 1.00 | 1.00 | 0 |
| Facebook | 4.66 | 0.53 | 101 |
| Yeast | 4.25 | 0.63 | 80 |
| C. elegans | 1.00 | 1.00 | 0 |
| Power grid | 7.67 | 0.88 | 71 |
| Air India | 1.60 | 0.92 | 4 |

* GrQc: one connected 2-core contains **260** 2-components.
* Drivers: 1 − J_G rises with transitivity (ρ = 0.67) and assortativity (0.64), not with size (0.40);
  F itself grows with size (ρ = 0.78), so it is treated as descriptive only.

### RQ1 — null model: the gap is real structure
In 100 degree-preserving randomisations of every network the gap vanishes (F ≈ 1, no brokers, no gap
nodes), as Łuczak (1991) predicts for random graphs. For **GrQc, Facebook, Yeast and the power grid**
the observed F, broker fraction and gap lie **above all 100 randomisations** (p = 0.02, the minimum).
Side finding: the BE correlation is *lower* than random in 5 of 8 networks (Facebook z = −83,
GrQc −54, Yeast −48): clustered networks look *less* core–periphery-like than their randomisations.

### RQ2 — robustness (R, lower is better; ties broken by degree)
| Network | Random | Degree | Betweenness | k-core | BE | Rombach | Cohesion | **Brokerage** | Adaptive degree |
|---|---|---|---|---|---|---|---|---|---|
| Karate | 0.404 | 0.142 | 0.145 | 0.155 | 0.142 | 0.237 | 0.155 | **0.137** | 0.136 |
| GrQc | 0.368 | 0.131 | **0.086** | 0.198 | 0.127 | 0.190 | 0.197 | 0.128 | 0.107 |
| Email-EU | 0.474 | 0.315 | **0.293** | 0.327 | 0.315 | 0.328 | 0.327 | 0.315 | 0.292 |
| Facebook | 0.456 | 0.295 | **0.104** | 0.336 | 0.295 | 0.342 | 0.343 | 0.224 | 0.266 |
| Yeast | 0.382 | 0.274 | **0.129** | 0.327 | 0.274 | 0.321 | 0.336 | 0.185 | 0.218 |
| C. elegans | 0.477 | **0.331** | 0.348 | 0.344 | **0.331** | 0.383 | 0.344 | **0.331** | 0.286 |
| Power grid | 0.213 | 0.063 | 0.066 | 0.084 | **0.059** | 0.165 | 0.081 | 0.065 | 0.052 |
| Air India | 0.408 | **0.070** | 0.079 | 0.080 | **0.070** | 0.092 | 0.080 | 0.074 | 0.064 |

Key comparisons (mean ΔR over 8 networks, 95 % bootstrap CI, wins/losses):
* Cohesion vs k-core: +0.002 [−0.001, +0.005], 2/3 → **no difference (H2 not supported)**.
* k-core vs degree: +0.029 [+0.015, +0.044], 0/8 → k-cores are worse than plain degree.
* **Brokerage vs k-core: −0.049 [−0.086, −0.017], 8/0, r_rb = −1.**
* **Brokerage vs cohesion: −0.051 [−0.090, −0.018], 8/0, r_rb = −1.**
* Brokerage vs degree: −0.020 [−0.047, +0.000], 4/2 (2 ties: networks without brokers).
* Brokerage vs betweenness: 4/4; vs adaptive degree: 2/6 — brokerage is *not* a universal winner.
* Gain of brokerage over k-core tracks group disagreement 1 − J_G (ρ = 0.88, p = 0.004);
  gain over degree tracks it too (ρ = 0.75, p = 0.03) and is unrelated to size (ρ = 0.08).
* No comparison survives Holm correction at α = 0.05 (minimum adjusted p ≈ 0.094 with n = 8).

### RQ2 — ablation: brokerage is more than articulation points
| Network | Degree | Articulation pts | Brokers k=2 | Brokers k≥3 | All levels |
|---|---|---|---|---|---|
| Facebook | 0.295 | 0.283 | 0.283 | **0.224** | **0.224** |
| Yeast | 0.274 | 0.198 | 0.230 | 0.240 | **0.185** |
| GrQc | 0.131 | 0.135 | **0.126** | 0.128 | 0.128 |
| Power grid | 0.063 | 0.113 | 0.065 | **0.063** | 0.065 |

Articulation points hurt the power grid; on Facebook the whole gain is from k ≥ 3 brokers; on Yeast
only the full hierarchy reaches the best R.

### RQ3 — hybrids
* Core + cohesion: never beats the better single score on held-out networks; λ* flips between 0 and 1.
* Core + brokerage: training always selects λ* = 0 (pure brokerage).
* → **H3 not supported**; the held-out design prevented an overfitted "hybrid wins" claim.

### Tie-breaking sensitivity
With uniformly random tie-breaking every coarse score gets worse and brokerage collapses (86–100 % of
nodes have b = 0). Brokerage is a **prioritisation layer** on top of a base ranking, not a full ranking.

---

## 9. Figures and tables: where each number comes from

| Paper item | Content | Generated by | Data file |
|---|---|---|---|
| Fig. 1 | Bowtie + karate (coreness vs cohesion, broker) | `figures.fig_bowtie_karate` | `cache/karate.pkl` |
| Fig. 2 | Node-level Spearman heatmaps | `figures.fig_agreement_heatmaps` | `tables/agreement_long.csv` |
| Fig. 3 | Node level vs group level | `figures.fig_node_vs_group` | `agreement_long.csv`, `dataset_table.csv` |
| Fig. 4 | #k-core groups vs #k-components per k | `figures.fig_group_by_k` | `group_structure_by_k.csv` |
| Fig. 5 | Null-model comparison | `figures.fig_null` | `cache/null_samples.csv`, `null_model_zscores.csv` |
| Fig. 6 | Robustness curves | `figures.fig_robustness_curves` | `cache/robustness.pkl` |
| Fig. 7 | R / R_random, degree vs random ties | `figures.fig_robustness_R` | `robustness_R.csv` |
| Fig. 8 | Hybrid λ curves and leave-one-out | `figures.fig_hybrid` | `hybrid_*_lambda_R.csv`, `hybrid_heldout.csv` |
| Table I | Dataset statistics | `make_tables.tab_datasets` | `dataset_table.csv` |
| Table II | Agreement (node + group) | `make_tables.tab_agreement` | `agreement_cp_vs_cohesion_wide.csv`, `dataset_table.csv` |
| Table III | Null model | `make_tables.tab_null` | `null_model_zscores.csv` |
| Table IV | Robustness R | `make_tables.tab_robustness` | `robustness_R.csv` |
| Table V | Paired comparisons | `make_tables.tab_comparisons` | `robustness_comparisons.csv` |
| Table VI | Brokerage ablation | `make_tables.tab_ablation` | `broker_ablation.csv` |
| Table VII | Hybrids held out | `make_tables.tab_hybrid` | `hybrid_heldout.csv`, `hybrid_comparisons.csv` |
| Table VIII | Runtime vs networkx | `make_tables.tab_bench` | `cohesion_benchmark.csv` |

No number in the paper's tables is typed by hand: they are generated from the CSVs.

---

## 10. Reproducing everything

```bash
pip install -r requirements.txt
python -m src.download_data
python -m pytest -q tests
python -m src.pipeline scores --workers 3
python -m src.pipeline agreement
python -m src.pipeline robustness --workers 3
python -m src.pipeline ablation
python -m src.pipeline hybrid
python -m src.pipeline null --n-null 100 --workers 4
python -m src.benchmark_cohesion --timeout 600
python -m src.figures
python -m src.make_tables
cd paper && tectonic main.tex
```

Runtime on a 12-thread, 8 GB Windows laptop: scores ≈ 6 min; agreement/robustness/ablation/hybrid
≈ 10 min; null models ≈ 1.5 h (Facebook ≈ 105 s per randomisation when run alone); benchmark ≈ 1 h
(networkx hits its 10-min limit on the large networks). **Memory:** each worker uses ≈ 600 MB — keep
`--workers` ≤ 4 on 8 GB machines, and do not run the benchmark at the same time as the null models.

Everything is deterministic given the seed (`SEED` in `src/pipeline.py`; per-task seeds are derived
from it by name).

**Compiling the paper:** `tectonic main.tex` (used here) or pdfLaTeX + BibTeX. On Overleaf, upload
`paper/` and put `results/figures/*.pdf` in a `figures/` folder (already on the `\graphicspath`).

---

## 11. Testing and validation

`python -m pytest -q tests` → 59 tests:

* `test_cohesion.py` — bowtie (coreness, cohesion, components, broker, group metrics); exact equality with
  **brute-force enumeration** on 25 random graphs; validity/maximality vs networkx on 4 classic graphs;
  planted cliques; numba max-flow vs networkx local connectivity and cuts; random-graph k-cores are
  cohesive (Łuczak).
* `test_methods.py` — k-core worked trace; union–find robustness curve = naive recomputation; tie-break
  order; BE recovers a planted core; Rombach profile shape and planted core; null model preserves
  degrees.
* `test_stats.py` — exact Wilcoxon p, zero-difference handling, rank-biserial, Holm, tie-inclusive top
  sets.

---

## 12. Decisions, corrections and lessons learned

These are recorded because examiners often ask "what went wrong and how did you handle it?".

1. **Hypotheses mostly not supported → the paper was re-centred, not the data.** Node-level agreement
   turned out ≈ 1, which made H2/H3 fail. Instead of hiding this, we looked one level up (groups) and
   found the real difference, then tested it with a null model and a functional yardstick.
2. **networkx `k_components` returned non-maximal components** (Les Misérables, Air India). We wrote an
   exact algorithm with a proof and brute-force tests.
3. **Speed.** The first exact version used networkx max-flow and did not finish Facebook in 30 min. A
   numba unit-capacity flow plus the certificates produced *identical* output and finished Facebook in
   ≈ 83 s (GrQc: 495 s → 38 s).
4. **Bug caught: edge capacities in the split graph.** Edges initially had capacity 1, so minimum cuts
   could use edges instead of vertices; a test assertion caught it and edges were set to "infinite".
5. **Top-x % Jaccard with ties.** With random tie-breaking, two *identical* rankings scored only
   0.14–0.21 on Karate/C. elegans. We switched to tie-inclusive top sets (identical rankings → 1).
6. **Wilcoxon with zero differences.** SciPy silently fell back to a normal approximation when methods
   tied exactly; we now drop zeros and use the exact test.
7. **Confounding by size.** Fragmentation F correlates with N (ρ = 0.78). We therefore rely on the
   size-normalised J_G and on the size-matched null model for structural claims.
8. **Benchmark fairness.** The first benchmark included Numba's one-off JIT compilation in our timing
   (karate 1.14 s instead of 0.002 s); a warm-up run fixes this.
9. **Resource limits.** 8 GB RAM forced ≤ 4 parallel workers; running the benchmark alongside the null
   models caused swapping and was stopped and re-run alone.
10. **Honest statistics.** With 8 networks, no comparison is significant after Holm correction; the
    paper says so and reports effect sizes and bootstrap CIs.

---

## 13. Limitations and future work

* **Sample size:** 8 networks → claims are "in our sample", not "in general".
* **Graph type:** undirected, unweighted, LCC only; weighted/directed cohesion may differ.
* **Attacks:** static orderings and R only; adaptive/optimal-percolation attacks are stronger, and
  adaptive degree beats brokerage on 6/8 networks. Next step: *adaptive brokerage* (recompute the
  hierarchy after removals).
* **Approximations:** Rombach on a 6 × 9 grid (not 100 × 100); BE greedy with 20 restarts.
* **Null model:** preserves degrees, not clustering. A clustering-preserving null model would test
  whether the group-level gap exceeds what triangles alone imply.
* **Scale:** exact cohesion takes seconds to minutes at a few thousand nodes; very large graphs need
  approximation or parallelism.

---

## 14. Viva preparation: likely questions and answers

**Q: Why is cohesion level ≤ coreness?** A k-component is k-connected with more than k nodes, so every
member has ≥ k neighbours inside it (otherwise its < k neighbours would separate it). Hence it lies in
the k-core (Proposition 1).

**Q: If they rank nodes the same, why does cohesion matter?** Because rankings hide organisation. A
single k-core can be several overlapping cohesive groups (F up to 7.7), and the overlaps — brokers — are
critical for connectivity.

**Q: Isn't brokerage just articulation points?** Only at k = 2. The ablation shows articulation points
*hurt* on the power grid, and on Facebook all the gain comes from k ≥ 3 brokers.

**Q: Why not just use betweenness?** Betweenness is still the strongest static ordering on 4 networks;
we report that. Brokerage is comparable on average rank and is derived from the cohesion hierarchy
without shortest paths; it is an interpretable structural signal, not a claim to beat every baseline.

**Q: Your results are not statistically significant — so what?** With n = 8 networks the minimum
Holm-adjusted p is ≈ 0.09, so significance is unattainable by design. We report effect sizes
(r_rb = −1 for brokerage vs k-core: it wins on every network) and bootstrap CIs that exclude 0.

**Q: How do you know the gap is not just because big networks have more groups?** The null model keeps
each network's size and degrees; the gap vanishes there (F ≈ 1). And J_G, which is size-normalised,
tracks clustering and assortativity, not size.

**Q: Why should we trust your k-components when networkx disagrees?** Exactness proof (splitting is
safe; Esfahanian test is complete; Lemma 1 for certificates), brute-force agreement on 25 random graphs,
and every networkx set is contained in one of ours.

**Q: Why did the hybrid fail?** Core + cohesion mixes two almost identical rankings, so there is nothing
to combine. Core + brokerage always chose λ = 0 on training data. The held-out design is exactly what
stopped us from over-claiming.

**Q: Why is BE's correlation lower than random?** Rewiring destroys local clustering and creates ties
among hubs, which is closer to BE's ideal pattern. It shows clustered networks are less CP-like in the
BE sense than their degree sequence alone would suggest.

**Q: How were ties handled in robustness?** Score, then degree, then node id; plus a sensitivity
analysis with 50 random tie-breaks (reported in Table V and Fig. 7b).

---

## 15. References

The full bibliography (46 entries) is in `paper/references.bib`; reading notes for the core papers are
in `paper/literature_notes.md`. Key sources: Borgatti & Everett (2000); Moody & White (2003);
Rombach et al. (2014, 2017); Gallagher et al. (2021); Seidman (1983); Batagelj & Zaveršnik (2003);
Łuczak (1991); Even (1975); Esfahanian (2013); Schneider et al. (2011); Albert et al. (2000);
Newman & Ziff (2000); Maslov & Sneppen (2002); Liu et al. (2015); Kitsak et al. (2010).
