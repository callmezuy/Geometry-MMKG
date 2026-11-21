---
license: other
task_categories:
- graph-ml
tags:
- knowledge-graph
- link-prediction
- scene-graph
- multimodal
- depth
pretty_name: Spatial-VRD-Trans (Geometry-MMKG v0)
---

# Spatial-VRD-Trans (Geometry-MMKG, data v0)

Transductive control for [callmezuy/spatial-vrd](https://huggingface.co/datasets/callmezuy/spatial-vrd): the same
107,016 triples, entities and features, re-split so that **every valid / test entity occurs in train**. Comparing a
model on both repos separates the effect of the inductive setting from the effect of the data.
Code: [github.com/callmezuy/Geometry-MMKG](https://github.com/callmezuy/Geometry-MMKG)
(`scripts/make_transductive_split.py --src data/Spatial-VRD --dst data/Spatial-VRD-Trans --stratify relation`).

| Split | Triples | Entities |
|---|---:|---:|
| train | 77,083 | 83,485 |
| valid | 8,550 | 14,221 |
| test | 21,383 | 27,848 |

93,809 entities in `entity2id.txt` (identical to `spatial-vrd`), 83,485 of them in triples; 52 relations, all in
every split. The hold-out is stratified by relation (valid 8% / test 20%), and a triple is held out only if both of
its entities keep a triple in train. `split_stats.json` has the details.

## Files

Same formats as `spatial-vrd` (see its card): split files with a count line, OpenKE `*2id.txt`, `entity2text.txt`,
`relation_category.tsv`, and the same features (`ent_rgb_emb.pt`, `ent_depth_emb.pt`, `ent_text_emb.pt`,
`ent_rgb_depth_mean_emb.pt`, `ent_rgb_depth_concat_emb.pt`, `ent_img_mask.pt`, `ent_fill.pt`, `ent_depth_stats.pt`
[93809, 9] with `depth_stats.json`). Only the fill differs: the 2 entities without a usable box get the mean feature
(and depth statistics) of the train entities of **this** split.

## Versions

`v0` (18/11/2026) is the first version of this repo (earlier repos with this name were deleted). Pin it with
`snapshot_download("callmezuy/spatial-vrd-trans", repo_type="dataset", revision="v0")`. On 19/11/2026
`ent_depth_stats.pt` and `depth_stats.json` were added and the tag moved; no other file changed. On 21/11/2026
`spatial-vrd` dropped its self-loop triples (subject box = object box); this repo was re-split from it with the
same command and seed, the fill vectors were recomputed, and the tag moved.

## License

Derived from the VRD dataset (Lu et al., 2016); use under its terms.
