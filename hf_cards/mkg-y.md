---
license: other
task_categories:
- graph-ml
tags:
- knowledge-graph
- link-prediction
- multimodal
- depth
pretty_name: MKG-Y (Geometry-MMKG v0)
---

# MKG-Y (Geometry-MMKG, data v0)

Transductive multimodal KG completion on MKG-Y (YAGO subset of MMRNS, Xu et al., ACM MM 2022), with the official
MMRNS images and descriptions and **depth features**.
Code and rebuild script: [github.com/callmezuy/Geometry-MMKG](https://github.com/callmezuy/Geometry-MMKG)
(`scripts/build_mkgy.py`, `scripts/extract_features_v4.py`). Rebuilding gives byte-identical split files.

| Split | Triples | Entities |
|---|---:|---:|
| train | 21,310 | 14,388 |
| valid | 2,665 | 4,265 |
| test | 2,663 | 4,332 |

15,000 entities, 28 relations. The split is the MMRNS split, unchanged (371 valid and 352 test entities do not
occur in train, as in the original). Entity / relation ids follow the NativE benchmark files of MKG-Y.

## Sources

- Split, images (`MKG-Y_img.zip`, up to 3 images per entity) and descriptions: the MMRNS Google Drive release.
  Images are **not** redistributed here, only features and the list of images used (`image_index.tsv`).
- Image folders are named after entities with Windows-unsafe characters replaced; `image_folder_match.tsv` maps
  every entity to its folder (`exact` 14,244, `windows` 222, `alnum` 15; all 14,481 folders matched).

## Images

43,353 image files: 3 unreadable, 79 duplicates within an entity, 7 uniform placeholders (max RGB std < 3); no
placeholder by hash was found on review (`image_placeholders.tsv` is empty). 43,264 images are used;
**14,481 entities have at least one image, 519 have none**.

## Features (`*.pt`, float32, row i = entity id i of `entity2id.txt`)

| File | Shape | Content |
|---|---|---|
| `ent_rgb_emb.pt` | [15000, 768] | DINOv2-base pooler output, mean over the entity's images |
| `ent_depth_emb.pt` | [15000, 768] | ZipDepth (base, 384 px, per-image min-max) depth map -> DINOv2-base pooler, mean over images |
| `ent_text_emb.pt` | [15000, 768] | RoBERTa-base masked mean of `entity2textlong.txt` (max 512 tokens) |
| `ent_rgb_depth_mean_emb.pt` | [15000, 768] | (rgb + depth) / 2 |
| `ent_rgb_depth_concat_emb.pt` | [15000, 1536] | [rgb, depth] |
| `ent_img_mask.pt` | [15000] bool | True = entity has at least one image |
| `ent_fill.pt` | dict | vectors written into the rows of entities without an image |

Rows of entities without an image hold the mean rgb / depth feature of the **train** entities that have an image.

## File formats

- `train/valid/test.txt`: first line = count, then `head \t relation \t tail`; `*2id.txt`: OpenKE order `head tail relation`.
- `entity2id.txt`, `relation2id.txt`: first line = count, then `name \t id`.
- `entity2text.txt`: `name \t readable name`; `entity2textlong.txt`: `name \t description` (MMRNS descriptions with
  quotes, `@en` and the trailing ` .` removed; 12,316 entities have one, the other 2,684 use their name).
- `clean_stats.json`: every check and count of the build.

## Versions

`v0` (18/11/2026) is the first version of this repo. Pin it with
`snapshot_download("callmezuy/mkg-y", repo_type="dataset", revision="v0")`.

## License

Derived from the MMRNS release of MKG-Y (YAGO); use under the terms of those sources.
