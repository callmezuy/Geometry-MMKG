# Data layout (v0, 18/11/2026)

Each `data/<name>/` folder holds the current version of a dataset and is published as its own public HF dataset
repo, tagged `v0` (RESEARCH_PLAN.md P0.8). The dataset card of each repo is `hf_cards/<repo>.md`;
`python scripts/publish_hf.py` uploads a folder plus its card. Pin data with
`snapshot_download(..., revision="v0")`.

| Folder | HF repo | Train / Valid / Test | Ent. | Rel. | Setting | Built from |
|---|---|---|---:|---:|---|---|
| `FB15k-237-IMG/` | `callmezuy/fb15k237-img` | 272,115 / 17,535 / 20,466 | 14,541 | 237 | transductive | KG-BERT split + text, MKGformer images |
| `FB15K-237-ind/` | `callmezuy/fb15k237-ind` | 215,082 / 42,164 / 52,870 | 14,541 | 237 | inductive (entity-disjoint) | BLP split; text, images, entity2id of FB15k-237-IMG |
| `MKG-Y/` | `callmezuy/mkg-y` | 21,310 / 2,665 / 2,663 | 15,000 | 28 | transductive | MMRNS split, images, descriptions; NativE ids |
| `Spatial-VRD/` | `callmezuy/spatial-vrd` | 77,227 / 8,614 / 21,563 | 93,809 | 52 | inductive (split by image) | `sg_*_annotations.json` + `relation_review.tsv` |
| `Spatial-VRD-Trans/` | `callmezuy/spatial-vrd-trans` | 77,371 / 8,578 / 21,455 | 93,809 | 52 | transductive control | `Spatial-VRD/`, stratified by relation |
| `Spatial-VG/` | `callmezuy/spatial-vg` | 319,069 / 26,372 / 137,984 | 1,441,984 | 50 | inductive (official image split) | `VG-SGG.h5` + `VG-SGG-dicts.json` (VG150-curated, `no_part_whole`) |

`spatial-vg` features were extracted on Molab (`notebooks/extract-vg-features.py`) and are on HF only (13 GB; not in
`data/Spatial-VG/` locally). VG images for the notebook come from the private copy `callmezuy/vg-images-raw`. Statistics of every dataset: `experiment_results/dataset_stats.md`
(`python scripts/dataset_stats.py --out experiment_results/dataset_stats.md`).

## File conventions

- `train/valid/test.txt`: first line = row count, then `head \t relation \t tail`.
- `train2id/valid2id/test2id.txt`: first line = row count, then `head_id \t tail_id \t relation_id` (OpenKE).
- `entity2id.txt`, `relation2id.txt`: first line = count, then `name \t id`. Row i of every `ent_*_emb.pt` = entity id i.
- `entity2text.txt` (names / short text) and `entity2textlong.txt` (descriptions, MMKG only): `name \t text`; only
  Spatial-VG has a count line.

## Features (all datasets, `scripts/extract_features.py`)

| File | Content |
|---|---|
| `ent_rgb_emb.pt` [n, 768] | DINOv2-base pooler of the image (MMKG: mean over the entity's images) or of the box crop (VRD, VG) |
| `ent_depth_emb.pt` [n, 768] | ZipDepth (base, 384 px, per-image min-max) depth map -> box crop (roi_align) -> DINOv2-base pooler |
| `ent_text_emb.pt` [n, 768] | RoBERTa-base masked mean of `entity2textlong.txt` if present, else `entity2text.txt` |
| `ent_rgb_depth_mean_emb.pt` [n, 768] | (rgb + depth) / 2, for baselines with one visual input (not for Spatial-VG) |
| `ent_rgb_depth_concat_emb.pt` [n, 1536] | [rgb, depth] (not for Spatial-VG) |
| `ent_img_mask.pt` [n] bool | entity has at least one usable image / box |
| `ent_fill.pt` | the constant written into the rgb / depth rows of entities without an image |

Missing images (RESEARCH_PLAN.md 3.2): rows with mask False hold the mean feature of the **train** entities that have
an image, so baselines can read the tables as they are; GeometryMMKG replaces those rows by its learned missing
vector. `feature_stats.json` logs the extraction, `feature_quality.json` the quality checks (`scripts/feature_quality.py`).

## Extra files

- `clean_stats.json`: every check and count of the build (MMKG: images, placeholders, text, coverage).
- `image_index.tsv` (MMKG): `entity \t image member \t status`; only `ok` images are encoded.
- `image_placeholders.tsv` (MMKG): reviewed placeholder md5 list (FB15k-237: 59; MKG-Y: none).
- `MKG-Y/image_folder_match.tsv`: entity -> image folder, with the match tier (`exact` / `windows` / `alnum`).
- `FB15K-237-ind/{train,valid,test}_entities.txt`: the entity sets of the inductive split.
- `relation_category.tsv` (scene graphs): GEOMETRIC / POSSESSIVE / SEMANTIC / MISC per relation.
- `Spatial-VRD/relation_review.tsv`: raw predicate -> label (+ head/tail inversion). Edit it, then rebuild.
- `Spatial-VRD-Trans/split_stats.json`: statistics of the transductive re-split.

## Rebuild (from `Code/`)

Raw sources go to `data/raw/` (see RESEARCH_PLAN.md P0.1–P0.6 for where each comes from).

```bash
# MMKG (each rebuild is byte-identical, checked 18/11/2026)
python scripts/build_fb15k237_img.py --raw data/raw/fb15k237 --dst data/FB15k-237-IMG
python scripts/build_fb15k237_ind.py --raw data/raw/fb15k237_ind --img data/FB15k-237-IMG --dst data/FB15K-237-ind
python scripts/build_mkgy.py --raw data/raw/mmrns_official --bench baselines/NativE/benchmarks/MKG-Y --dst data/MKG-Y

# Scene graphs (byte-identical, checked 11/11/2026)
python scripts/build_vrd_relation_review.py        # only if the label rules change (overwrites relation_review.tsv)
python scripts/clean_splits.py --src data/Spatial-VRD --dst data/_build/Spatial-VRD \
    --vrd-review data/Spatial-VRD/relation_review.tsv --valid-group image-prefix --valid-ratio 0.1
python scripts/clean_splits.py --src data/Spatial-VG --dst data/_build/Spatial-VG \
    --vg-h5 data/Spatial-VG/VG-SGG.h5 --vg-dicts data/Spatial-VG/VG-SGG-dicts.json
# then move the files from data/_build/<name>/ into data/<name>/, and:
python scripts/make_transductive_split.py --src data/Spatial-VRD --dst data/Spatial-VRD-Trans --stratify relation

# Features (write to data/_feat/<name>/, then move into data/<name>/)
python scripts/extract_features.py kg --data data/FB15k-237-IMG --out data/_feat/FB15k-237-IMG \
    --zip data/raw/fb15k237/fb15k_wn18_images.zip --mean-concat
python scripts/extract_features.py fill --feat data/_feat/FB15k-237-IMG --data data/FB15K-237-ind \
    --out data/_feat/FB15K-237-ind --mean-concat
python scripts/extract_features.py kg --data data/MKG-Y --out data/_feat/MKG-Y \
    --zip data/raw/mmrns_official/MKG-Y_img.zip --mean-concat
python scripts/extract_features.py vrd --data data/Spatial-VRD --out data/_feat/Spatial-VRD \
    --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip --mean-concat
python scripts/extract_features.py fill --feat data/_feat/Spatial-VRD --data data/Spatial-VRD-Trans \
    --out data/_feat/Spatial-VRD-Trans --mean-concat
# Spatial-VG: notebooks/extract-vg-features.py on Molab (no --mean-concat)
python scripts/feature_quality.py --data data/FB15k-237-IMG --labels fb15k237   # also: vrd, vg, none (MKG-Y)
```
