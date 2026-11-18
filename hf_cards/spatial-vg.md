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
triple), 50 relations, 108,073 images.

## Sources and cleaning

- `VG-SGG.h5`, `VG-SGG-dicts.json`: **VG150-curated, variant `no_part_whole`** (Neau et al., ICCVW 2023,
  `Maelic/VG_curated`), not the standard VG150 of Xu et al. (2017). Images: Visual Genome 1.4 (Krishna et al., 2017);
  not redistributed here.
- Split = the official image split of the H5 file; duplicates within a split removed (train 103,092, valid 8,723,
  test 40,935); no valid/test triple occurs in train. `clean_stats.json` has every count.
- `relation_category.tsv`: GEOMETRIC / POSSESSIVE / SEMANTIC / MISC per relation (our classification following the
  definitions of Zellers et al., CVPR 2018).

## Features

**Not in this commit yet.** The v0 features (`ent_rgb_emb.pt`, `ent_depth_emb.pt`, `ent_text_emb.pt`,
`ent_img_mask.pt`, `ent_fill.pt`, `feature_stats.json`, `feature_quality.json`) are being extracted with
`notebooks/extract-vg-features.py` and will be added before the tag `v0` is created. Same recipe as
`callmezuy/spatial-vrd`: DINOv2-base on the box crop (rgb), ZipDepth depth map of the whole image cropped to the box
and encoded by DINOv2-base (depth), RoBERTa-base on the class name (text). This repo has no mean / concat tables.

## File formats

- `train/valid/test.txt`: first line = count, then `head \t relation \t tail` (entity names are box indices);
  `*2id.txt`: OpenKE order `head tail relation`.
- `entity2id.txt`, `relation2id.txt`: first line = count, then `name \t id`.
- `entity2text.txt`: first line = count, then `name \t class name`.

## Versions

`v0` is the first version of this repo (earlier repos with this name were deleted); the tag is created once the
features are in. Pin it with `snapshot_download("callmezuy/spatial-vg", repo_type="dataset", revision="v0")`.

## License

Derived from Visual Genome (CC BY 4.0) and VG150-curated; use under the terms of those sources.
