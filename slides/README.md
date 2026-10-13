# Presentation deck

12-slide viva/presentation deck for the project, with speaker notes on every slide.

* `deck.json` — slide order, sections and fonts (Source Serif 4 + IBM Plex Sans).
* `slides/<id>.html` — one 1920×1080 slide per file; speaker notes are in each file's `<aside>`.

The deck was built as a Claude Slides artifact, which renders these files and exports them to
**PPTX or PDF** (Share › Export). Images are referenced as uploaded assets (`/_blob/...`); they are the
PNG versions of the paper figures in `../results/figures/`:

| Slide | Figure |
|---|---|
| bowtie | `fig1_bowtie_karate.png` |
| grouplevel | `fig4_group_structure_by_k.png` |
| nullmodel | `fig8_null_models.png` |
| robustness | `fig5_robustness_curves.png` |

Slide order: cover · question · bowtie · design · algorithm · nodelevel · grouplevel · nullmodel ·
robustness · ablation · hybrid · takeaways.
