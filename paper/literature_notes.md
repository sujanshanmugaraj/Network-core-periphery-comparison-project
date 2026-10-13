# Literature notes (Phase 1)

Each entry: what it defines · the algorithm · what it leaves open (→ how our study uses it).

## Core papers (read in this order)

**Borgatti & Everett (2000), "Models of core/periphery structures", *Social Networks* 21(4):375–395.**
- Defines the *ideal* core–periphery (CP) pattern: core–core ties present, periphery–periphery ties absent,
  core–periphery ties present (or treated as missing data in a variant). Introduces a discrete model (each node
  is core or periphery) and a continuous model (each node gets a coreness c_i, ideal tie strength c_i c_j).
- Fit = Pearson correlation between the observed adjacency matrix and the ideal pattern; maximised by
  combinatorial search (they used a genetic algorithm in UCINET).
- Leaves open: how the core relates to *connectivity*. The correlation rewards density, not resistance to
  disconnection, and a single core is assumed. → Our "BE (discrete)" method; greedy label switching, 20 restarts.

**Moody & White (2003), "Structural cohesion and embeddedness: a hierarchical concept of social groups", *American Sociological Review* 68(1):103–127.**
- Defines a group's structural cohesion as its node connectivity κ (Menger: κ ≥ k ⇔ every pair is joined by
  k node-independent paths). k-components = maximal k-connected subgraphs; they nest hierarchically, and
  embeddedness = the depth of a node in this hierarchy.
- Algorithm: recursively find minimum vertex cuts, split the graph along them, and recurse; the k-core is
  used as a pre-filter.
- Leaves open: computational cost for large graphs, and how the hierarchy relates to CP models (the two
  literatures barely cite each other). → Our cohesion level; we give an exact certificate-accelerated algorithm.

**Rombach, Porter, Fowler & Mucha (2014), "Core–periphery structure in networks", *SIAM J. Appl. Math.* 74(1):167–190.**
- Defines core quality R_{α,β} = Σ_ij A_ij C_i C_j, where C is a permutation of a profile C* given by a
  piecewise-linear "sharp" transition function (Eq. 9): β sets the core size (⌊βN⌋ periphery positions) and
  α the size of the jump between periphery and core. Aggregate core score CS(i) = Z Σ_{α,β} C_i(α,β) R_{α,β}
  (Eq. 11), normalised so max CS = 1.
- Algorithm: simulated annealing over permutations for each (α,β); the paper uses a 100×100 grid.
- Leaves open: speed (they say so explicitly), choice of the (α,β) grid, and whether CS identifies
  functionally important nodes. → Our "Rombach" method (6×9 grid, SA from the degree ordering).

**Gallagher, Young & Foucault Welles (2021), "A clarified typology of core–periphery structure in networks", *Science Advances* 7(12):eabc9800.**
- Distinguishes *hub-and-spoke* CP (core members each attach to many periphery nodes) from *layered* CP
  (nested shells like the k-core), and shows that different CP methods detect different types. Fits
  stochastic-block-model versions of both by Bayesian inference.
- Leaves open: a functional criterion for choosing between notions of "core". → Motivates our use of
  robustness as an external, function-based yardstick and our comparison across domains.

## Supporting literature

- **Seidman (1983)**, *Social Networks* 5:269–287 — introduces k-cores; observes that a k-core need not be
  k-connected (cohesion is a stricter notion). **Batagelj & Zaveršnik (2003)**, arXiv:cs/0310049 — O(M) core
  decomposition (used by `networkx.core_number`).
- **White & Harary (2001)**, *Sociological Methodology* 31:305–359 — node connectivity as the measure of
  cohesion; contrasts it with density-based measures.
- **Even (1975)**, *SIAM J. Comput.* 4(3):393–396, and **Esfahanian (2013)** — testing k-connectivity with
  O(N + δ²) local connectivity queries; basis of our k-connectivity test.
- **Torrents & Ferraro (2015)**, *J. Social Structure* 16 — heuristics for fast k-component computation;
  NetworkX's `k_components` implementation follows this line. We found that it can return non-maximal
  components (Les Misérables, k = 3), which motivated our exact implementation and brute-force tests.
- **Holme (2005)**, *Phys. Rev. E* 72:046111 — closeness-based core–periphery coefficient; CP measured
  relative to a degree-preserving null model (our Phase 6 follows this logic).
- **Maslov & Sneppen (2002)**, *Science* 296:910–913 — degree-preserving double-edge-swap null model.
- **Csermely, London, Wu & Uzzi (2013)**, *J. Complex Networks* 1(2):93–123 — review of CP structure and its
  dynamical roles (robustness, spreading).
- **Kojaku & Masuda (2017)**, *Phys. Rev. E* 96:052313 — multiple CP pairs; `cpnet` library.
- **Zhang, Martin & Newman (2015)**, *Phys. Rev. E* 91:032803 — SBM-based CP detection.
- **Albert, Jeong & Barabási (2000)**, *Nature* 406:378–382 — error and attack tolerance (degree-targeted
  attacks).
- **Schneider, Moreira, Andrade, Havlin & Herrmann (2011)**, *PNAS* 108(10):3838–3841 — robustness index
  R = (1/N) Σ_Q S(Q).
- **Newman & Ziff (2000)**, *Phys. Rev. Lett.* 85:4104 — union–find percolation (our exact O(M α(N)) curves).
- **Kitsak et al. (2010)**, *Nature Physics* 6:888–893 — k-shell identifies influential spreaders better than
  degree; **Liu, Tang, Zhou & Do (2015)**, *Sci. Rep.* 5:9602 — "core-like groups" (dense but weakly linked to
  the rest) make k-shell rankings misleading. This is the same phenomenon as our k-core / k-component gap.
- **Morone & Makse (2015)**, *Nature* 524:65–68 — optimal percolation / collective influence (stronger
  adaptive dismantling baseline; outside our scope, noted as a limitation).
- **Burleson-Lesser, Morone, Tomassone & Makse (2020)**, *Sci. Rep.* 10:3357 — k-core robustness in ecological
  and financial networks.
- **Vinh, Epps & Bailey (2010)**, *JMLR* 11:2837–2854 — adjusted mutual information.
- **Kerby (2014)**, *Comprehensive Psychology* 3 — rank-biserial effect size for the Wilcoxon test.

## The gap our paper fills

CP models (Borgatti–Everett, Rombach) and k-core decompositions define "core" by **density / degree**;
structural cohesion defines it by **resistance to disconnection**. The two literatures are rarely compared
head-to-head, never (to our knowledge) across domains with a common functional yardstick, and the
k-core/k-component gap has not been quantified against a degree-preserving null model.
