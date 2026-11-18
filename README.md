# Geometry-MMKG: Depth-Aware Multi-Modal Knowledge Graph Completion

Thesis code. **Research question:** is depth needed for multi-modal knowledge graphs, and if so, how should it be fused?
The hypothesis is that depth helps scene-graph KGs (MMSG, where entities share one viewing space)
more than concept-level KGs (MMKG), and that an adaptive depth-gated fusion works on both.

The proposed model, **GeometryMMKG**, fuses Text, RGB and Depth features of each entity
(gated RGB–Depth fusion followed by cross-attention with text) and scores triples with TransE.
The baselines are NativE, SNAG, MoMoK, MKGformer and CMR (RESEARCH_PLAN.md 3.2).

**Start here:** [CONTRIBUTING.md](CONTRIBUTING.md) (onboarding and rules) and [RESEARCH_PLAN.md](RESEARCH_PLAN.md) (design, protocol, tasks and status).

## Datasets (v0)

| Dataset | HF repo (tag `v0`) | Train / Valid / Test | Setting |
|---|---|---|---|
| FB15k-237-IMG | `callmezuy/fb15k237-img` | 272,115 / 17,535 / 20,466 | MMKG, transductive (main) |
| MKG-Y | `callmezuy/mkg-y` | 21,310 / 2,665 / 2,663 | MMKG, transductive |
| FB15K-237-ind | `callmezuy/fb15k237-ind` | 215,082 / 42,164 / 52,870 | MMKG, inductive |
| Spatial-VRD | `callmezuy/spatial-vrd` | 77,227 / 8,614 / 21,563 | MMSG, inductive (split by image) |
| Spatial-VRD-Trans | `callmezuy/spatial-vrd-trans` | 77,371 / 8,578 / 21,455 | MMSG, transductive control |
| Spatial-VG | `callmezuy/spatial-vg` | 319,069 / 26,372 / 137,984 | MMSG, inductive (official image split) |

Every entity has 768-d `ent_{rgb,depth,text}_emb.pt` features (DINOv2 for RGB and ZipDepth depth, RoBERTa for text).
Details, file formats and rebuild commands are in [data/README.md](data/README.md).

## Layout

```
core/            dataset, fusion, GeometryMMKG model, unified evaluation, seeding
notebooks/       Marimo notebooks run on Molab (feature extraction, NativE, ...)
scripts/         data building, feature extraction, train.py, compute_metrics.py, baseline data preparation
baselines/       NativE, SNAG (original code + allowed changes, see baselines/PROVENANCE.md), MKGformer (submodule)
tests/           brute-force checks of the evaluation code
ZipDepth/        depth estimator used to extract depth features
```

## Quick start

```bash
git clone --recurse-submodules https://github.com/callmezuy/Geometry-MMKG.git
pip install -r requirements.txt
python tests/test_evaluation.py && python tests/test_valid_selector.py
python scripts/train.py                     # GeometryMMKG on data/Spatial-VRD (local)
python scripts/compute_metrics.py --data data/Spatial-VRD --ranks geometry=<ranks.csv>
```

Experiments are run with the notebooks in `notebooks/` on [Molab](https://molab.marimo.io); they download data from HF and code from this repo's `main` branch.

## Evaluation protocol (all models)

- Filtered link prediction over **all** entities, **head and tail** ranked, no type constraint, strict tie rule.
- Checkpoint chosen on **validation MRR**; test is evaluated once.
- Every model writes per-triple ranks; all reported numbers come from `scripts/compute_metrics.py`
  (overall, per relation category, per relation, and paired-bootstrap ΔMRR).
