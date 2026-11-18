---
license: other
task_categories:
- graph-ml
tags:
- knowledge-graph
- link-prediction
- inductive
- multimodal
- depth
pretty_name: FB15K-237-ind (Geometry-MMKG v0)
---

# FB15K-237-ind (Geometry-MMKG, data v0)

Inductive (dynamic) multimodal KG completion: the entity-disjoint FB15k-237 split of BLP (Daza et al., WWW 2021),
with the same entity text, images and **depth features** as
[callmezuy/fb15k237-img](https://huggingface.co/datasets/callmezuy/fb15k237-img).
Code and rebuild script: [github.com/callmezuy/Geometry-MMKG](https://github.com/callmezuy/Geometry-MMKG)
(`scripts/build_fb15k237_ind.py`). Rebuilding gives byte-identical files.

| Split | Triples | Entities of the split (`*_entities.txt`) |
|---|---:|---:|
| train | 215,082 | 11,633 |
| valid | 42,164 | 1,454 |
| test | 52,870 | 1,454 |

The three entity sets are disjoint. Train triples use only train entities; every valid / test triple contains at
least one entity of its own split (the other end may be a train entity, and for test also a valid entity).
237 relations, all seen in train. The union of the three splits equals the standard FB15k-237 entity and triple set.
Checked against Table 1 of the BLP paper (`clean_stats.json`, `checks`).

## Sources

- Split: BLP, Zenodo record 4501273, `FB15k-237.tar.gz` (md5 `c7cf03efea81b5fbef958f6ce3bfc198`).
- Text, images, features: identical to `callmezuy/fb15k237-img` (same `entity2id.txt`, 14,541 entities); see its card
  for the image store, placeholder filtering and feature extraction. 14,292 entities have an image, 249 do not.

## Files

Same layout and formats as `fb15k237-img` (split files with a count line, OpenKE `*2id.txt`, `ent_*_emb.pt`,
`ent_img_mask.pt`, `ent_fill.pt`, `image_index.tsv`, `image_placeholders.tsv`), plus
`train_entities.txt`, `valid_entities.txt`, `test_entities.txt` (one entity per line, no count line).

`ent_fill.pt` differs from `fb15k237-img`: the constant for entities without an image is the mean feature of the
**train entities of this split** that have an image.

## Versions

`v0` (18/11/2026) is the first version of this repo. Pin it with
`snapshot_download("callmezuy/fb15k237-ind", repo_type="dataset", revision="v0")`.

## License

Derived from FB15k-237, BLP, KG-BERT and the MKGformer image store; use under the terms of those sources.
