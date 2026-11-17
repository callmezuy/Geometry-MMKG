# Data layout (v3, 18/11/2026)

Each `data/<name>/` folder holds the **current (v3)** version of a dataset and has the same
name as the folder the notebooks download into. File conventions are unchanged:

- `train/valid/test.txt`: first line = row count, then `head \t relation \t tail`
- `train2id/valid2id/test2id.txt`: first line = row count, then `head_id \t tail_id \t relation_id` (OpenKE)
- `entity2id.txt` is identical to v1, so every `ent_*_emb.pt` row still matches its entity.

| Folder | HF repo | Train / Valid / Test | Rel. | Setting | Built from |
|---|---|---|---:|---|---|
| `Merged-MMKG/` | `callmezuy/unified-mmkg` | 45,192 / 5,001 / 6,022 | 429 | transductive | v2 test kept; valid re-drawn from train+valid, stratified by source (v3) |
| `Spatial-VRD/` | `callmezuy/spatial-vrd` | 77,227 / 8,614 / 21,563 | 52 | inductive (split by image) | `sg_*_annotations.json` + `relation_review.tsv` |
| `Spatial-VG/` | `callmezuy/spatial-vg` | 319,069 / 26,372 / 137,984 | 50 | inductive (official image split) | `VG-SGG.h5` + `VG-SGG-dicts.json` (VG150-curated, `no_part_whole`) |
| `Spatial-VRD-Trans/` | `callmezuy/spatial-vrd-trans` | 77,371 / 8,578 / 21,455 | 52 | transductive control | `Spatial-VRD/` v2, stratified by relation (v3) |

Uploaded to HF on 12/11/2026 (v2) and 18/11/2026 (v3): `main` = v3. Each repo has tags `v1` (original), `v2` and `v3` (`spatial-vrd-trans`: `v2`, `v3`); pin with `snapshot_download(..., revision=...)` for reproducibility.

**v3 (18/11/2026)** changes only `Merged-MMKG` and `Spatial-VRD-Trans` (`Spatial-VRD`, `Spatial-VG` are unchanged; their v3 tag = v2 content):
- `Merged-MMKG`: in v2 leakage removal left only 1,395 valid triples, **40 of them DB15K** although DB15K is 77% of test, so checkpoint selection barely saw DB15K. v3 keeps test byte for byte and re-draws 5,001 valid triples from train + v2 valid with the source mix of test (DB15K / MKG-W / MKG-Y = 76.9 / 19.7 / 3.4%). Every valid entity occurs in train (v2: 9.5% did not); test entities unseen in train drop from 267 to 242.
- `Spatial-VRD-Trans`: the v2 greedy hold-out favoured GEOMETRIC triples (test 79.5% GEOMETRIC vs train 66.9%), because POSSESSIVE triples more often hit degree-1 entities. v3 caps the hold-out per relation, so train / test are 70.3 / 70.5% GEOMETRIC and 25.1 / 24.9% POSSESSIVE.
- Statistics of every dataset: `experiment_results/dataset_stats.md` (`python scripts/dataset_stats.py`).
- Results on v2 for these two datasets are not comparable with v3 (train changed).

## Extra files

- `clean_stats.json`: what each cleaning step removed, before/after statistics.
- `relation_category.tsv` (MMSG only): GEOMETRIC / POSSESSIVE / SEMANTIC / MISC per relation.
- `Spatial-VRD/relation_review.tsv`: raw predicate → label (+ head/tail inversion). Edit it, then rebuild.
- `Spatial-VRD/relation_audit.tsv`: what the old heuristic (`scripts/legacy/`) did to each raw predicate.
- `split_stats.json` (`Spatial-VRD-Trans/`, `Merged-MMKG/`): statistics of the v3 re-split, incl. stratum shares.
- v1 split files are no longer kept locally; download them from HF at tag `v1` if needed (e.g. for `scripts/audit_vrd_relations.py --stored`).

## Rebuild (from `Code/`)

```bash
python scripts/build_vrd_relation_review.py        # only if the label rules change (overwrites relation_review.tsv)
python scripts/clean_splits.py --src data/Spatial-VRD --dst data/_build/Spatial-VRD \
    --vrd-review data/Spatial-VRD/relation_review.tsv --valid-group image-prefix --valid-ratio 0.1
python scripts/clean_splits.py --src data/Spatial-VG --dst data/_build/Spatial-VG \
    --vg-h5 data/Spatial-VG/VG-SGG.h5 --vg-dicts data/Spatial-VG/VG-SGG-dicts.json
python scripts/clean_splits.py --src <v1 Merged-MMKG from HF tag v1> --dst data/_build/Merged-MMKG
# then move the files from data/_build/<name>/ into data/<name>/, and (v3):
python scripts/make_transductive_split.py --src data/Merged-MMKG --dst data/_build/Merged-MMKG-v3 \n    --keep-test --stratify source --valid-size 5000   # from the v2 Merged-MMKG, then move into data/Merged-MMKG/
python scripts/make_transductive_split.py --src data/Spatial-VRD --dst data/Spatial-VRD-Trans --stratify relation
```

These commands reproduce the current split files byte for byte (checked 11/11/2026; v3 steps 18/11/2026). `--stratify none` reproduces v2 `Spatial-VRD-Trans` byte for byte.
