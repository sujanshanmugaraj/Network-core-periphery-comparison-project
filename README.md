# Do Two Notions of "Core" Agree?
### Core–periphery structure vs. structural cohesion, validated by network robustness

Code, data pipeline and paper for a comparative study of **core–periphery** (k-core, Borgatti–Everett,
Rombach et al.) and **structural-cohesion** (Moody–White k-components) notions of a network "core"
across 8 networks from social, biological and infrastructure domains.

* **RQ1** – How much do CP cores and cohesion cores agree? (Spearman, top-x% Jaccard, AMI)
* **RQ2** – Which node ordering dismantles a network fastest? (robustness index R)
* **RQ3** – Does a rank-hybrid of coreness and cohesion beat both on held-out networks?

## Layout

```
data/raw/            downloaded datasets (not versioned; run the download step)
src/
  datasets.py        loading, cleaning (undirected, simple, LCC), statistics
  coreness.py        k-core, Borgatti–Everett (numba greedy), Rombach 2014 (numba SA)
  cohesion.py        exact k-components / cohesion levels (certificate-accelerated)
  robustness.py      exact robustness curves via union–find; ordering helpers
  agreement.py       Spearman, tie-aware top-x% Jaccard, AMI
  nullmodels.py      degree-preserving double-edge swaps (numba)
  stats.py           Wilcoxon, rank-biserial, bootstrap CI, Holm
  pipeline.py        all experiments (cached, seeded, parallel)
  figures.py         all paper figures
tests/               unit tests (incl. brute-force validation of k-components)
results/tables/      CSV tables used in the paper
results/figures/     PDF + PNG figures
paper/               LaTeX source (IEEEtran), bibliography, literature notes
slides/              presentation deck
```

## Reproduce everything

Python 3.12. Exact versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

```bash
python -m src.download_data
```

```bash
python -m pytest -q tests
```

```bash
python -m src.pipeline scores --workers 8
```

```bash
python -m src.pipeline agreement
```

```bash
python -m src.pipeline robustness
```

```bash
python -m src.pipeline hybrid
```

```bash
python -m src.pipeline null --n-null 100 --workers 11
```

```bash
python -m src.figures
```

All randomness is seeded (`SEED` in `src/pipeline.py`). Each step caches its results in `results/cache/`.
Delete that folder to recompute from scratch. Approximate runtimes on a 12-thread laptop: scores
about 25 min (dominated by Facebook betweenness and cohesion), null models about 1–2 h, everything else
under 2 min.

## Compile the paper

Upload `paper/` (main.tex, references.bib) and `results/figures/*.pdf` to Overleaf, or run locally with
a TeX distribution:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.tex` loads figures from `../results/figures/`. On Overleaf, copy that folder to `figures/` and
change `\graphicspath` at the top of `main.tex`.
