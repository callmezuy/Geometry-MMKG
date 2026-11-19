---
license: other
task_categories:
- graph-ml
tags:
- knowledge-graph
- link-prediction
- inductive
- scene-graph
- multimodal
- depth
pretty_name: Spatial-VG (Geometry-MMKG v0)
---

# Spatial-VG (Geometry-MMKG, data v0)

Inductive multimodal KG completion over Visual Genome scene graphs: every **entity is one object box**, triples are
the annotated relations between boxes, and the split is by image, so valid / test entities never occur in train.
Built for the Geometry-MMKG thesis (does monocular depth help multimodal KG completion?).
Code: [github.com/callmezuy/Geometry-MMKG](https://github.com/callmezuy/Geometry-MMKG)
(`scripts/clean_splits.py`, `scripts/extract_features.py`, `notebooks/extract-vg-features.py`).

| Split | Triples | Entities in triples |
|---|---:|---:|
| train | 319,069 | 389,413 |
| valid | 26,372 | 33,049 |
| test | 137,984 | 176,343 |

1,441,984 entities in `entity2id.txt` (every box of `VG-SGG.h5`; entity id = box index, so most boxes are in no
triple), 50 relations, 108,073 images (105,814 of them have at least one box).

## Sources and cleaning

- `VG-SGG.h5`, `VG-SGG-dicts.json`: **VG150-curated, variant `no_part_whole`** (Neau et al., ICCVW 2023,
  `Maelic/VG_curated`), not the standard VG150 of Xu et al. (2017). Images: Visual Genome 1.4 (Krishna et al., 2017);
  not redistributed here.
- Split = the official image split of the H5 file; duplicates within a split removed (train 103,092, valid 8,723,
  test 40,935); no valid/test triple occurs in train. `clean_stats.json` has every count.
- `relation_category.tsv`: GEOMETRIC / POSSESSIVE / SEMANTIC / MISC per relation (our classification following the
  definitions of Zellers et al., CVPR 2018).

## Features

`*.pt`, float32, row i = entity id i of `entity2id.txt`. Extracted on Molab with `notebooks/extract-vg-features.py`
(`scripts/extract_features.py vg`, 105,814 images that have at least one box), same recipe as `callmezuy/spatial-vrd`.

| File | Shape | Content |
|---|---|---|
| `ent_rgb_emb.pt` | [1441984, 768] | DINOv2-base pooler output of the box crop |
| `ent_depth_emb.pt` | [1441984, 768] | ZipDepth (base, 384 px, per-image min-max) depth map of the whole image, box region cropped (roi_align) -> DINOv2-base pooler |
| `ent_text_emb.pt` | [1441984, 768] | RoBERTa-base masked mean of `entity2text.txt` (the class name: only 150 distinct strings) |
| `ent_img_mask.pt` | [1441984] bool | True = entity has a usable box |
| `ent_fill.pt` | dict | vectors written into the rows of entities without a usable box |
| `ent_depth_stats.pt` | [1441984, 9] | means of a 3x3 grid of the box on the same ZipDepth depth map (per-image min-max, 1 = nearest point of the image); objects of one image stay comparable (`scripts/depth_stats.py`, `depth_stats.json`) |

6 boxes (ids 77901, 77906, 77908, 77914, 77915, 77916, all of image index 5763) lie entirely outside their image
(annotation error); they are treated as missing and filled with the mean rgb / depth feature (and depth statistics)
of the 389,410 train entities that have a box. This repo has **no** mean / concat tables (compute `(rgb + depth) / 2` or `[rgb, depth]`
from the two tables if needed).

Quality (`feature_quality.json`; linear probe of the 150 object classes on 30,000 entities, majority class 8.2%):
rgb 40.1%, depth 13.6%, text 100% (the text is the class name). Mean pairwise cosine: rgb 0.066, depth 0.348,
text 0.986. The 10 nearest neighbours of depth and of rgb overlap by 1.5% (chance 0.005%).
`feature_stats.json`, `extract.log` and `quality.log` record the run (code commit, ZipDepth commit, timings).

## File formats

- `train/valid/test.txt`: first line = count, then `head \t relation \t tail` (entity names are box indices);
  `*2id.txt`: OpenKE order `head tail relation`.
- `entity2id.txt`, `relation2id.txt`: first line = count, then `name \t id`.
- `entity2text.txt`: first line = count, then `name \t class name`.

## Versions

`v0` (18/11/2026) is the first version of this repo (earlier repos with this name were deleted). Pin it with `snapshot_download("callmezuy/spatial-vg", repo_type="dataset", revision="v0")`. On 19/11/2026 `ent_depth_stats.pt`, `depth_stats.json` and `depth_stats.log` were added and the tag moved; no other file changed.

## License

Derived from Visual Genome (CC BY 4.0) and VG150-curated; use under the terms of those sources.
