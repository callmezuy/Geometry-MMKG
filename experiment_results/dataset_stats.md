# Thống kê dữ liệu

Sinh bởi `scripts/dataset_stats.py` từ `data/`.

## Merged-MMKG

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 45,192 | 13,170 | 423 | 0.0 | 0.0 |
| valid | 5,001 | 5,064 | 253 | 0.0 | 0.0 |
| test | 6,022 | 5,745 | 262 | 4.2 | 4.1 |

- Thực thể trong entity2id: **15,458** (xuất hiện trong ít nhất một split: 13,412); quan hệ: **429**; tổng bộ ba: **56,215**.
- Bậc trong train: trung bình 6.86, trung vị 4, max 3,959; thực thể bậc 1: 20.0%.
- Mật độ train (bộ ba / thực thể): 3.43.

- Quan hệ: top-1 chiếm 11.8% train, top-5 29.1%, top-10 42.0%; quan hệ có < 10 bộ ba train: 207/423; quan hệ không có trong train: 6.
- Quan hệ có trong test nhưng không có trong train: 6 (6 bộ ba test).
- Top-5 quan hệ train: `DB15K::<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>` 5,332, `DB15K::<http://dbpedia.org/ontology/birthPlace>` 2,944, `DB15K::<http://dbpedia.org/ontology/genre>` 1,940, `DB15K::<http://www.w3.org/2000/01/rdf-schema#seeAlso>` 1,605, `DB15K::<http://dbpedia.org/ontology/country>` 1,342.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 216 | 8.8 |
| 1-N | 40 | 1.5 |
| N-1 | 141 | 58.7 |
| N-N | 26 | 30.9 |
| unseen | 0 | 0.1 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 7 (`DB15K::<http://dbpedia.org/ontology/spouse>`, `DB15K::<http://dbpedia.org/ontology/partner>`, `DB15K::<http://dbpedia.org/ontology/sisterCollege>`, `MKG-W::http://www.wikidata.org/entity/P190`, `MKG-W::http://www.wikidata.org/entity/P47`, `MKG-W::http://www.wikidata.org/entity/P26`, `MKG-Y::isMarriedTo`).

| source | relations | train | valid | test |
|---|---:|---:|---:|---:|
| DB15K | 263 | 33,748 | 3,848 | 4,634 |
| MKG-W | 142 | 9,790 | 984 | 1,185 |
| MKG-Y | 24 | 1,654 | 169 | 203 |

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 15,458 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_emb.pt` | 15,458 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 15,458 × 768 | float32 | 0 | 0 | ✓ |

## Spatial-VRD

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 77,227 | 59,950 | 52 | 0.0 | 0.0 |
| valid | 8,614 | 6,694 | 50 | 100.0 | 100.0 |
| test | 21,563 | 16,908 | 52 | 100.0 | 100.0 |

- Thực thể trong entity2id: **93,809** (xuất hiện trong ít nhất một split: 83,552); quan hệ: **52**; tổng bộ ba: **107,404**.
- Bậc trong train: trung bình 2.58, trung vị 2, max 22; thực thể bậc 1: 44.7%.
- Mật độ train (bộ ba / thực thể): 1.29.

- Quan hệ: top-1 chiếm 22.2% train, top-5 65.2%, top-10 90.2%; quan hệ có < 10 bộ ba train: 2/52; quan hệ không có trong train: 0.
- Quan hệ có trong test nhưng không có trong train: 0 (0 bộ ba test).
- Top-5 quan hệ train: `on` 17,107, `near` 10,848, `has` 10,054, `above` 6,309, `under` 6,055.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 50 | 70.2 |
| 1-N | 1 | 7.0 |
| N-1 | 1 | 22.8 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 0.

| category | relations | train (%) | test (%) |
|---|---:|---:|---:|
| GEOMETRIC | 21 | 70.4 | 70.3 |
| POSSESSIVE | 5 | 25.0 | 25.3 |
| SEMANTIC | 24 | 4.6 | 4.5 |
| MISC | 2 | 0.0 | 0.0 |

| split | images | entities / image | triples / image |
|---|---:|---:|---:|
| train | 3,600 | 16.65 | 21.45 |
| valid | 400 | 16.73 | 21.54 |
| test | 999 | 16.92 | 21.58 |

- Ảnh test cũng có trong train: 0/999 (0.0%).

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 93,809 × 768 | float32 | 10 | 0 | ✓ |
| `ent_rgb_depth_concat_emb.pt` | 93,809 × 1536 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_mean_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |

## Spatial-VRD-Trans

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 77,371 | 83,552 | 52 | 0.0 | 0.0 |
| valid | 8,578 | 14,210 | 52 | 0.0 | 0.0 |
| test | 21,455 | 27,937 | 52 | 0.0 | 0.0 |

- Thực thể trong entity2id: **93,809** (xuất hiện trong ít nhất một split: 83,552); quan hệ: **52**; tổng bộ ba: **107,404**.
- Bậc trong train: trung bình 1.85, trung vị 1, max 18; thực thể bậc 1: 63.1%.
- Mật độ train (bộ ba / thực thể): 0.93.

- Quan hệ: top-1 chiếm 22.3% train, top-5 65.2%, top-10 90.3%; quan hệ có < 10 bộ ba train: 2/52; quan hệ không có trong train: 0.
- Quan hệ có trong test nhưng không có trong train: 0 (0 bộ ba test).
- Top-5 quan hệ train: `on` 17,286, `near` 10,772, `has` 10,144, `above` 6,247, `under` 6,031.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 51 | 92.9 |
| 1-N | 1 | 7.1 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 0.

| category | relations | train (%) | test (%) |
|---|---:|---:|---:|
| GEOMETRIC | 21 | 70.3 | 70.5 |
| POSSESSIVE | 5 | 25.1 | 24.9 |
| SEMANTIC | 24 | 4.5 | 4.6 |
| MISC | 2 | 0.0 | 0.0 |

| split | images | entities / image | triples / image |
|---|---:|---:|---:|
| train | 4,999 | 16.71 | 15.48 |
| valid | 4,080 | 3.48 | 2.10 |
| test | 4,888 | 5.72 | 4.39 |

- Ảnh test cũng có trong train: 4,888/4,888 (100.0%).

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 93,809 × 768 | float32 | 10 | 0 | ✓ |
| `ent_rgb_depth_concat_emb.pt` | 93,809 × 1536 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_mean_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |

## Spatial-VG

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 319,069 | 389,413 | 50 | 0.0 | 0.0 |
| valid | 26,372 | 33,049 | 50 | 100.0 | 100.0 |
| test | 137,984 | 176,343 | 50 | 100.0 | 100.0 |

- Thực thể trong entity2id: **1,441,984** (xuất hiện trong ít nhất một split: 598,805); quan hệ: **50**; tổng bộ ba: **483,425**.
- Bậc trong train: trung bình 1.64, trung vị 1, max 26; thực thể bậc 1: 67.1%.
- Mật độ train (bộ ba / thực thể): 0.82.

- Quan hệ: top-1 chiếm 29.4% train, top-5 68.8%, top-10 85.3%; quan hệ có < 10 bộ ba train: 0/50; quan hệ không có trong train: 0.
- Quan hệ có trong test nhưng không có trong train: 0 (0 bộ ba test).
- Top-5 quan hệ train: `on` 93,760, `in` 37,173, `wearing` 33,545, `has` 30,345, `near` 24,581.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 50 | 100.0 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 0.

| category | relations | train (%) | test (%) |
|---|---:|---:|---:|
| GEOMETRIC | 15 | 50.1 | 49.5 |
| POSSESSIVE | 6 | 38.4 | 38.5 |
| SEMANTIC | 25 | 10.8 | 11.5 |
| MISC | 4 | 0.7 | 0.5 |

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 1,441,984 × 768 | float32 | 380 | 0 | ✓ |
| `ent_rgb_emb.pt` | 1,441,984 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 1,441,984 × 768 | float32 | 0 | 0 | ✓ |

