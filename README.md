# Geometry-MMKG: Depth-Aware Multi-Modal Knowledge Graph Completion

Thesis code. **Research question:** is depth needed for multi-modal knowledge graphs, and if so, how should it be fused?
The hypothesis is that depth helps scene-graph KGs (MMSG, where entities share one viewing space)
more than concept-level KGs (MMKG), and that an adaptive depth-gated fusion works on both.

The proposed model, **GeometryMMKG**, fuses Text, RGB and Depth features of each entity
(gated RGB–Depth fusion followed by cross-attention with text) and scores triples with TransE.
The baselines are NativE, SNAG and MKGformer.

**Start here:** [CONTRIBUTING.md](CONTRIBUTING.md) (onboarding and rules) and [RESEARCH_PLAN.md](RESEARCH_PLAN.md) (design, protocol, tasks and status).

## Datasets (v2)

| Dataset | HF repo | Train / Valid / Test | Setting |
|---|---|---|---|
| Merged-MMKG | `callmezuy/unified-mmkg` | 48,798 / 1,395 / 6,022 | transductive |
| Spatial-VRD | `callmezuy/spatial-vrd` | 77,227 / 8,614 / 21,563 | inductive (split by image) |
| Spatial-VG | `callmezuy/spatial-vg` | 319,069 / 26,372 / 137,984 | inductive (official image split) |
| Spatial-VRD-Trans | `callmezuy/spatial-vrd-trans` | 77,331 / 8,592 / 21,481 | transductive control |

Every entity has pre-extracted 768-d `ent_{text,rgb,depth}_emb.pt` features. Details, file formats and rebuild commands are in [data/README.md](data/README.md).

## Layout

```
core/            dataset, encoders, fusion, GeometryMMKG model, unified evaluation, seeding
notebooks/       Marimo notebooks run on Molab (GeometryMMKG, NativE, MKGformer)
scripts/         data building/cleaning, train.py, compute_metrics.py, patch_mkgformer.py
baselines/       NativE, SNAG (vendored, modified), MKGformer (git submodule)
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
