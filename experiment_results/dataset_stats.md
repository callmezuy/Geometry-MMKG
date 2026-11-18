# Thống kê dữ liệu

Sinh bởi `scripts/dataset_stats.py` từ `data/`.

## FB15k-237-IMG

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 272,115 | 14,505 | 237 | 0.0 | 0.0 |
| valid | 17,535 | 9,809 | 223 | 0.1 | 0.1 |
| test | 20,466 | 10,348 | 224 | 0.3 | 0.1 |

- Thực thể trong entity2id: **14,541** (xuất hiện trong ít nhất một split: 14,541); quan hệ: **237**; tổng bộ ba: **310,116**.
- Bậc trong train: trung bình 37.52, trung vị 22, max 7,614; thực thể bậc 1: 2.5%.
- Mật độ train (bộ ba / thực thể): 18.76.

- Quan hệ: top-1 chiếm 5.9% train, top-5 22.6%, top-10 36.3%; quan hệ có < 10 bộ ba train: 0/237; quan hệ không có trong train: 0.
- Quan hệ có trong test nhưng không có trong train: 0 (0 bộ ba test).
- Top-5 quan hệ train: `/award/award_nominee/award_nominations./award/award_nomination/award_nominee` 15,989, `/film/film/release_date_s./film/film_regional_release_date/film_release_region` 12,893, `/award/award_nominee/award_nominations./award/award_nomination/award` 12,157, `/people/person/profession` 10,945, `/film/actor/film./film/performance/film` 9,494.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 17 | 0.9 |
| 1-N | 26 | 6.3 |
| N-1 | 86 | 22.0 |
| N-N | 108 | 70.7 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 24 (`/education/field_of_study/students_majoring./education/education/major_field_of_study`, `/government/legislative_session/members./government/government_position_held/legislative_sessions`, `/people/person/sibling_s./people/sibling_relationship/sibling`, `/base/popstra/celebrity/dated./base/popstra/dated/participant`, `/award/award_winner/awards_won./award/award_honor/award_winner`, `/people/person/spouse_s./people/marriage/spouse`, `/award/award_nominee/award_nominations./award/award_nomination/award_nominee`, `/military/military_combatant/military_conflicts./military/military_combatant_group/combatants`, `/award/award_nominated_work/award_nominations./award/award_nomination/nominated_for`, `/base/localfood/seasonal_month/produce_available./base/localfood/produce_availability/seasonal_months`).

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_concat_emb.pt` | 14,541 × 1536 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_mean_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |

## FB15K-237-ind

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 215,082 | 11,633 | 237 | 0.0 | 0.0 |
| valid | 42,164 | 8,965 | 225 | 16.2 | 100.0 |
| test | 52,870 | 10,645 | 232 | 22.7 | 100.0 |

- Thực thể trong entity2id: **14,541** (xuất hiện trong ít nhất một split: 14,541); quan hệ: **237**; tổng bộ ba: **310,116**.
- Bậc trong train: trung bình 36.98, trung vị 22, max 6,851; thực thể bậc 1: 2.0%.
- Mật độ train (bộ ba / thực thể): 18.49.

- Quan hệ: top-1 chiếm 5.3% train, top-5 22.7%, top-10 35.3%; quan hệ có < 10 bộ ba train: 0/237; quan hệ không có trong train: 0.
- Quan hệ có trong test nhưng không có trong train: 0 (0 bộ ba test).
- Top-5 quan hệ train: `/film/film/release_date_s./film/film_regional_release_date/film_release_region` 11,371, `/award/award_nominee/award_nominations./award/award_nomination/award_nominee` 10,437, `/people/person/profession` 9,869, `/award/award_nominee/award_nominations./award/award_nomination/award` 9,572, `/award/award_category/nominees./award/award_nomination/nominated_for` 7,489.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 17 | 1.4 |
| 1-N | 26 | 5.4 |
| N-1 | 85 | 16.7 |
| N-N | 109 | 76.6 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 24 (`/education/field_of_study/students_majoring./education/education/major_field_of_study`, `/government/legislative_session/members./government/government_position_held/legislative_sessions`, `/people/person/sibling_s./people/sibling_relationship/sibling`, `/base/popstra/celebrity/dated./base/popstra/dated/participant`, `/award/award_winner/awards_won./award/award_honor/award_winner`, `/people/person/spouse_s./people/marriage/spouse`, `/award/award_nominee/award_nominations./award/award_nomination/award_nominee`, `/military/military_combatant/military_conflicts./military/military_combatant_group/combatants`, `/award/award_nominated_work/award_nominations./award/award_nomination/nominated_for`, `/base/localfood/seasonal_month/produce_available./base/localfood/produce_availability/seasonal_months`).

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_concat_emb.pt` | 14,541 × 1536 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_mean_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 14,541 × 768 | float32 | 0 | 0 | ✓ |

## MKG-Y

| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |
|---|---:|---:|---:|---:|---:|
| train | 21,310 | 14,388 | 28 | 0.0 | 0.0 |
| valid | 2,665 | 4,265 | 27 | 8.7 | 15.0 |
| test | 2,663 | 4,332 | 25 | 8.1 | 14.6 |

- Thực thể trong entity2id: **15,000** (xuất hiện trong ít nhất một split: 15,000); quan hệ: **28**; tổng bộ ba: **26,638**.
- Bậc trong train: trung bình 2.96, trung vị 2, max 78; thực thể bậc 1: 24.6%.
- Mật độ train (bộ ba / thực thể): 1.48.

- Quan hệ: top-1 chiếm 28.8% train, top-5 79.2%, top-10 95.6%; quan hệ có < 10 bộ ba train: 5/28; quan hệ không có trong train: 0.
- Quan hệ có trong test nhưng không có trong train: 0 (0 bộ ba test).
- Top-5 quan hệ train: `actedIn` 6,131, `created` 3,230, `isAffiliatedTo` 3,128, `playsFor` 2,836, `directed` 1,559.

| mapping | relations | test triples (%) |
|---|---:|---:|
| 1-1 | 11 | 3.3 |
| 1-N | 6 | 33.1 |
| N-1 | 8 | 9.2 |
| N-N | 3 | 54.3 |

- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): 1 (`isMarriedTo`).

| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |
|---|---|---|---:|---:|---|
| `ent_depth_emb.pt` | 15,000 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_concat_emb.pt` | 15,000 × 1536 | float32 | 0 | 0 | ✓ |
| `ent_rgb_depth_mean_emb.pt` | 15,000 × 768 | float32 | 0 | 0 | ✓ |
| `ent_rgb_emb.pt` | 15,000 × 768 | float32 | 0 | 0 | ✓ |
| `ent_text_emb.pt` | 15,000 × 768 | float32 | 0 | 0 | ✓ |

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
| `ent_depth_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |
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
| `ent_depth_emb.pt` | 93,809 × 768 | float32 | 0 | 0 | ✓ |
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

