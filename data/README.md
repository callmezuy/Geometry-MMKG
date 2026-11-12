# Data layout (v2, 11/11/2026)

Each `data/<name>/` folder holds the **current (v2)** version of a dataset and has the same
name as the folder the notebooks download into. File conventions are unchanged:

- `train/valid/test.txt`: first line = row count, then `head \t relation \t tail`
- `train2id/valid2id/test2id.txt`: first line = row count, then `head_id \t tail_id \t relation_id` (OpenKE)
- `entity2id.txt` is identical to v1, so every `ent_*_emb.pt` row still matches its entity.

| Folder | HF repo | Train / Valid / Test | Rel. | Setting | Built from |
|---|---|---|---:|---|---|
| `Merged-MMKG/` | `callmezuy/unified-mmkg` | 48,798 / 1,395 / 6,022 | 429 | transductive | v1 splits, leakage removed |
| `Spatial-VRD/` | `callmezuy/spatial-vrd` | 77,227 / 8,614 / 21,563 | 52 | inductive (split by image) | `sg_*_annotations.json` + `relation_review.tsv` |
| `Spatial-VG/` | `callmezuy/spatial-vg` | 319,069 / 26,372 / 137,984 | 50 | inductive (official image split) | `VG-SGG.h5` + `VG-SGG-dicts.json` (VG150-curated, `no_part_whole`) |
| `Spatial-VRD-Trans/` | `callmezuy/spatial-vrd-trans` | 77,331 / 8,592 / 21,481 | 52 | transductive control | `Spatial-VRD/` v2 |

Uploaded to HF on 12/11/2026: `main` = v2. Each repo has tags `v1` (original) and `v2`; pin with `snapshot_download(..., revision="v2")` for reproducibility.

## Extra files

- `clean_stats.json`: what each cleaning step removed, before/after statistics.
- `relation_category.tsv` (MMSG only): GEOMETRIC / POSSESSIVE / SEMANTIC / MISC per relation.
- `Spatial-VRD/relation_review.tsv`: raw predicate → label (+ head/tail inversion). Edit it, then rebuild.
- `Spatial-VRD/relation_audit.tsv`: what the old heuristic (`scripts/legacy/`) did to each raw predicate.
- `Spatial-VRD-Trans/split_stats.json`: statistics of the transductive control split.
- `_archive_v1/<name>/`: the v1 split files (+ `entity2id.txt`). Also available on HF at tag `v1`, so safe to delete.

## Rebuild (from `Code/`)

```bash
python scripts/build_vrd_relation_review.py        # only if the label rules change (overwrites relation_review.tsv)
python scripts/clean_splits.py --src data/Spatial-VRD --dst data/_build/Spatial-VRD \
    --vrd-review data/Spatial-VRD/relation_review.tsv --valid-group image-prefix --valid-ratio 0.1
python scripts/clean_splits.py --src data/Spatial-VG --dst data/_build/Spatial-VG \
    --vg-h5 data/Spatial-VG/VG-SGG.h5 --vg-dicts data/Spatial-VG/VG-SGG-dicts.json
python scripts/clean_splits.py --src data/_archive_v1/Merged-MMKG --dst data/_build/Merged-MMKG
# then move the files from data/_build/<name>/ into data/<name>/, and:
python scripts/make_transductive_split.py --src data/Spatial-VRD --dst data/Spatial-VRD-Trans
```

These commands reproduce the current split files byte for byte (checked 11/11/2026).
