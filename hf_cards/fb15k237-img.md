---
license: other
task_categories:
- graph-ml
tags:
- knowledge-graph
- link-prediction
- multimodal
- depth
pretty_name: FB15k-237-IMG (Geometry-MMKG v0)
---

# FB15k-237-IMG (Geometry-MMKG, data v0)

Transductive multimodal knowledge graph completion on FB15k-237, with entity images, text and
**depth features**. Built for the Geometry-MMKG thesis (does monocular depth help multimodal KG completion?).
Code and rebuild script: [github.com/callmezuy/Geometry-MMKG](https://github.com/callmezuy/Geometry-MMKG)
(`scripts/build_fb15k237_img.py`, `scripts/extract_features.py`). Rebuilding gives byte-identical split files.

| Split | Triples | Entities |
|---|---:|---:|
| train | 272,115 | 14,505 |
| valid | 17,535 | 9,809 |
| test | 20,466 | 10,348 |

14,541 entities, 237 relations. The split is the standard FB15k-237 split (as distributed with KG-BERT),
unchanged: 8 valid and 29 test entities do not occur in train.

## Sources

- Split and text: FB15k-237 (Toutanova & Chen, 2015) as distributed by KG-BERT (Yao et al., 2019):
  `entity2text.txt` (names), `entity2textlong.txt` (descriptions; 26 entities without a description use their name).
- Images: the MKGformer (Chen et al., SIGIR 2022) FB15k image store (Bing images, up to ~10 per entity).
  Images are **not** redistributed here, only features and the list of images used (`image_index.tsv`).

## Images and placeholders

Of 146,019 images of FB15k-237 entities, 1,541 are placeholders (blank, "image unavailable", logos of the
image host, ...) and are excluded (`image_index.tsv`, column `status`):
- `placeholder_hash`: md5 in the reviewed list `image_placeholders.tsv` (colour entities such as *White* are exempt for blank images);
- `placeholder_uniform`: max RGB std < 3 (not for colour entities);
- `placeholder_near_dup`: dHash within 2 bits of a reviewed placeholder.

144,478 images are used; **14,292 entities have at least one image, 249 have none**.

## Features (`*.pt`, float32, row i = entity id i of `entity2id.txt`)

| File | Shape | Content |
|---|---|---|
| `ent_rgb_emb.pt` | [14541, 768] | DINOv2-base pooler output, mean over the entity's images |
| `ent_depth_emb.pt` | [14541, 768] | ZipDepth (base, 384 px, per-image min-max) depth map -> DINOv2-base pooler, mean over images |
| `ent_text_emb.pt` | [14541, 768] | RoBERTa-base masked mean of `entity2textlong.txt` (max 512 tokens) |
| `ent_rgb_depth_mean_emb.pt` | [14541, 768] | (rgb + depth) / 2, for baselines that take one visual vector |
| `ent_rgb_depth_concat_emb.pt` | [14541, 1536] | [rgb, depth] |
| `ent_img_mask.pt` | [14541] bool | True = entity has at least one image |
| `ent_fill.pt` | dict | vectors written into the rows of entities without an image |

Rows of entities without an image hold a constant: the mean rgb / depth feature of the **train** entities that have
an image (`ent_fill.pt`). Models that handle missing images themselves should use `ent_img_mask.pt` instead.
`feature_stats.json` and `feature_quality.json` describe the extraction and simple quality checks (mean cosine,
linear probe of the Freebase type, depth/RGB nearest-neighbour overlap).

## File formats

- `train/valid/test.txt`: first line = number of triples, then `head \t relation \t tail` (Freebase mids).
- `train2id/valid2id/test2id.txt`: first line = count, then `head_id \t tail_id \t relation_id` (OpenKE order).
- `entity2id.txt`, `relation2id.txt`: first line = count, then `name \t id`.
- `entity2text.txt`, `entity2textlong.txt`, `relation2text.txt`: `name \t text`, no count line.
- `clean_stats.json`: every check and count of the build.

## Versions

`v0` (18/11/2026) is the first version of this repo. Pin it with
`snapshot_download("callmezuy/fb15k237-img", repo_type="dataset", revision="v0")`.

## License

The files are derived from FB15k-237, KG-BERT and the MKGformer image store; use them under the terms of those sources.
