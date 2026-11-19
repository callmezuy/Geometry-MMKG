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
pretty_name: Spatial-VRD (Geometry-MMKG v0)
---

# Spatial-VRD (Geometry-MMKG, data v0)

Inductive multimodal KG completion over scene graphs: every **entity is one object box** of a VRD image, triples are
the annotated relations between boxes, and the split is by image, so valid / test entities never occur in train.
Built for the Geometry-MMKG thesis (does monocular depth help multimodal KG completion?).
Transductive control with the same triples: [callmezuy/spatial-vrd-trans](https://huggingface.co/datasets/callmezuy/spatial-vrd-trans).
Code: [github.com/callmezuy/Geometry-MMKG](https://github.com/callmezuy/Geometry-MMKG)
(`scripts/build_vrd_relation_review.py`, `scripts/clean_splits.py`, `scripts/extract_features.py`).

| Split | Triples | Entities |
|---|---:|---:|
| train | 77,227 | 59,950 |
| valid | 8,614 | 6,694 |
| test | 21,563 | 16,908 |

93,809 entities (object boxes of 4,999 images), 52 relations. Entity names are `<photo_id>_<k>` (box k of the image).

## Sources and cleaning

- Annotations: the free-form `sg_dataset` annotations of VRD (Lu et al., ECCV 2016): `sg_train_annotations.json`,
  `sg_test_annotations.json` (1,310 raw predicates, object names and attributes). Images are **not** redistributed.
- Relations: raw predicates mapped to the curated predicate vocabulary of
  [callmezuy/spatial-vg](https://huggingface.co/datasets/callmezuy/spatial-vg) plus a few VRD-only labels
  (`left of`, `right of`, `touching`, ...); 1.0% of triples dropped, some inverted. Mapping: `relation_review.tsv`.
- Relations with < 10 triples dropped; duplicates within a split removed; no valid/test triple occurs in train.
- Test = original VRD test images; valid = 10% of the train images (grouped by image).
- `relation_category.tsv`: GEOMETRIC / POSSESSIVE / SEMANTIC / MISC per relation (our classification following the
  definitions of Zellers et al., CVPR 2018).
- `clean_stats.json`: before / after counts of every step.

## Features (`*.pt`, float32, row i = entity id i of `entity2id.txt`)

| File | Shape | Content |
|---|---|---|
| `ent_rgb_emb.pt` | [93809, 768] | DINOv2-base pooler output of the box crop |
| `ent_depth_emb.pt` | [93809, 768] | ZipDepth (base, 384 px, per-image min-max) depth map of the whole image, box region cropped (roi_align) -> DINOv2-base pooler |
| `ent_text_emb.pt` | [93809, 768] | RoBERTa-base masked mean of `entity2text.txt` (attributes + object name) |
| `ent_rgb_depth_mean_emb.pt` | [93809, 768] | (rgb + depth) / 2 |
| `ent_rgb_depth_concat_emb.pt` | [93809, 1536] | [rgb, depth] |
| `ent_img_mask.pt` | [93809] bool | True = entity has a usable box |
| `ent_fill.pt` | dict | vectors written into the rows of entities without a usable box |
| `ent_depth_stats.pt` | [93809, 9] | means of a 3x3 grid of the box on the same ZipDepth depth map (per-image min-max, 1 = nearest point of the image); objects of one image stay comparable (`scripts/depth_stats.py`, `depth_stats.json`) |

2 boxes (ids 61233, 61284) lie entirely outside their image (annotation error); they are treated as missing and
filled with the mean rgb / depth feature (and depth statistics) of the train entities. Note that the text names the object class, so text
alone predicts the class almost perfectly (`feature_quality.json`).

## File formats

- `train/valid/test.txt`: first line = count, then `head \t relation \t tail`; `*2id.txt`: OpenKE order `head tail relation`.
- `entity2id.txt`, `relation2id.txt`: first line = count, then `name \t id`.
- `entity2text.txt`: `name \t text`, no count line (23 extra lines name boxes that are not in `entity2id.txt`; ignore them).

## Versions

`v0` (18/11/2026) is the first version of this repo (earlier repos with this name were deleted). Pin it with
`snapshot_download("callmezuy/spatial-vrd", repo_type="dataset", revision="v0")`. On 19/11/2026 `ent_depth_stats.pt`
and `depth_stats.json` were added and the tag moved; no other file changed.

## License

Derived from the VRD dataset (Lu et al., 2016); use under its terms.
