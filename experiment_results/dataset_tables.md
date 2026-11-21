# Thống kê dữ liệu (bảng kiểu paper)

Sinh bởi `scripts/dataset_tables.py` từ `data/` (v0). Thống kê kiểm tra chi tiết từng bộ: `dataset_stats.md`.

### T1. Tổng quan các bộ dữ liệu

| Dataset | Graph | Setting | #Ent | #Rel | #Train | #Valid | #Test | #Images |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FB15k-237-IMG | MMKG | Trans. | 14,541 | 237 | 272,115 | 17,535 | 20,466 | 144,478 |
| FB15K-237-ind | MMKG | Ind. | 14,541 | 237 | 215,082 | 42,164 | 52,870 | 144,478 |
| MKG-Y | MMKG | Trans. | 15,000 | 28 | 21,310 | 2,665 | 2,663 | 43,264 |
| Spatial-VRD | MMSG | Ind. | 93,809 | 52 | 76,959 | 8,571 | 21,486 | 4,999 |
| Spatial-VRD-Trans | MMSG | Trans. | 93,809 | 52 | 77,083 | 8,550 | 21,383 | 4,999 |
| Spatial-VG | MMSG | Ind. | 1,441,984 | 50 | 319,069 | 26,372 | 137,984 | 105,814 |

#Ent = toàn bộ `entity2id` (= không gian ứng viên khi xếp hạng, mục 4.2 của RESEARCH_PLAN). #Images: MMKG = số ảnh dùng được (trạng thái `ok` trong `image_index.tsv`); MMSG = số ảnh cảnh có ít nhất một box. Trans. = transductive, Ind. = inductive.

### T2. Độ phủ và đặc điểm các modality

| Dataset | Ent. w/ image (%) | Img/ent. mean / med. / max | Images removed | Ent. w/ description (%) | Text words mean / med. / max | Truncated @512 tok. | Test: both / one / no image |
|---|---:|---:|---:|---:|---:|---:|---:|
| FB15k-237-IMG | 14,292 (98.29) | 10.11 / 10 / 48 | 1,541 (1.06%) | 99.82 | 138.7 / 132 / 697 | 24 | 19,205 / 1,247 / 14 |
| FB15K-237-ind | 14,292 (98.29) | 10.11 / 10 / 48 | 1,541 (1.06%) | 99.82 | 138.7 / 132 / 697 | 24 | 49,338 / 3,494 / 38 |
| MKG-Y | 14,481 (96.54) | 2.99 / 3 / 3 | 86 (0.20%) | 82.11 | 67.7 / 47 / 834 | 154 | 2,496 / 165 / 2 |
| Spatial-VRD | 93,807 (100.00) | 1 / 1 / 1 | 2 box ngoài ảnh | – | 2.3 / 2 / 21 | 0 | 21,486 / 0 / 0 |
| Spatial-VRD-Trans | 93,807 (100.00) | 1 / 1 / 1 | 2 box ngoài ảnh | – | 2.3 / 2 / 21 | 0 | 21,383 / 0 / 0 |
| Spatial-VG | 1,441,978 (100.00) | 1 / 1 / 1 | 6 box ngoài ảnh | – | 1.0 / 1 / 1 | 0 | 137,984 / 0 / 0 |

MMKG: một thực thể có nhiều ảnh (đặc trưng = trung bình); MMSG: mỗi thực thể là một box (1 crop). Images removed = ảnh đọc được nhưng bị loại khỏi tập `ok`: ảnh giữ chỗ, gần một màu, ảnh trùng trong cùng thực thể (P0.1, P0.3; không tính ảnh không đọc được: FB15k-237 0, MKG-Y 3). MMSG: box nằm hẳn ngoài ảnh (lỗi chú thích) được tính là thiếu ảnh. Văn bản: `entity2textlong.txt` nếu có, ngược lại `entity2text.txt` (VRD: thuộc tính + tên vật thể; VG: tên lớp). "Ent. w/ description" = có mô tả dài khác tên. Cột cuối: số bộ ba test có cả hai / một / không đầu nào có ảnh (mục 4.10).

### T3. Cấu trúc đồ thị (tính trên train)

| Dataset | Deg. mean / med. / max | Deg.-1 ent. (%) | Isolated ent. | Triples / ent. | 1-1 / 1-N / N-1 / N-N (#rel) | Test in N-N (%) | Top-5 rel. (% train) | Rel. < 10 train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FB15k-237-IMG | 37.52 / 22 / 7,614 | 2.5 | 0 | 18.76 | 17 / 26 / 86 / 108 | 70.7 | 22.6 | 0 |
| FB15K-237-ind | 36.98 / 22 / 6,851 | 2.0 | 0 | 18.49 | 17 / 26 / 85 / 109 | 76.6 | 22.7 | 0 |
| MKG-Y | 2.96 / 2 / 78 | 24.6 | 0 | 1.48 | 11 / 6 / 8 / 3 | 54.3 | 79.2 | 5 |
| Spatial-VRD | 2.57 / 2 / 22 | 44.8 | 10,324 | 1.28 | 50 / 1 / 1 / 0 | 0.0 | 65.2 | 2 |
| Spatial-VRD-Trans | 1.85 / 1 / 17 | 63.2 | 10,324 | 0.92 | 51 / 1 / 0 / 0 | 0.0 | 65.2 | 2 |
| Spatial-VG | 1.64 / 1 / 26 | 67.1 | 843,179 | 0.82 | 50 / 0 / 0 / 0 | 0.0 | 68.8 | 0 |

Bậc = số bộ ba train chứa thực thể (đầu hoặc đuôi), chỉ tính thực thể có trong train. Isolated = thực thể có trong `entity2id` nhưng không thuộc bộ ba nào (MMSG: box không có quan hệ; vẫn nằm trong không gian ứng viên). Loại ánh xạ theo Bordes et al. (2013), ngưỡng 1,5 trên train. Ở MMSG mỗi thực thể là một box riêng nên gần như mọi quan hệ thành 1-1: đây là tính chất của đồ thị cảnh (thực thể không lặp lại giữa các ảnh), không phải lỗi.

### T4. Thực thể theo split và mức độ inductive

| Dataset | Ent. train / valid / test | Unseen ent. valid (%) | Unseen ent. test (%) | Test: both seen | Test: only head new | Test: only tail new | Test: both new |
|---|---:|---:|---:|---:|---:|---:|---:|
| FB15k-237-IMG | 14,505 / 9,809 / 10,348 | 0.1 | 0.3 | 20,438 (99.9%) | 23 (0.1%) | 3 (0.0%) | 2 (0.0%) |
| FB15K-237-ind | 11,633 / 8,965 / 10,645 | 16.2 | 22.7 | 0 (0.0%) | 25,949 (49.1%) | 20,431 (38.6%) | 6,490 (12.3%) |
| MKG-Y | 14,388 / 4,265 / 4,332 | 8.7 | 8.1 | 2,275 (85.4%) | 128 (4.8%) | 224 (8.4%) | 36 (1.4%) |
| Spatial-VRD | 59,900 / 6,685 / 16,900 | 100.0 | 100.0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 21,486 (100.0%) |
| Spatial-VRD-Trans | 83,485 / 14,221 / 27,848 | 0.0 | 0.0 | 21,383 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Spatial-VG | 389,413 / 33,049 / 176,343 | 100.0 | 100.0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 137,984 (100.0%) |

Thực thể theo split = thực thể xuất hiện trong bộ ba của split đó. "New" = không xuất hiện trong bộ ba train. Cách chia test theo đầu nào là thực thể mới giống Daza et al. (BLP, 2021). FB15K-237-ind: test có cả thực thể của valid (cũng chưa thấy khi train). Nếu chỉ coi thực thể của test (không có trong train ∪ valid) là mới, như `test_entities.txt`: chỉ đầu mới 27,801, chỉ đuôi mới 22,536, cả hai mới 2,533.

### T5. Thống kê đồ thị cảnh (MMSG)

| Dataset | Images train / valid / test | #Obj. classes (≥ 10 boxes) | #Pred. | #Boxes (in ≥1 triple) | Boxes / img. | Triples / img. | Pred. head / body / tail | Box area / img. (med.) | Subj.-obj. boxes overlap (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial-VRD | 3,600 / 400 / 999 | 6,745 (855) | 52 | 93,809 (83,485) | 18.77 | 21.41 | 2 / 13 / 37 | 2.3% | 79.1 |
| Spatial-VRD-Trans | 4,999 / 4,080 / 4,895 | 6,745 (855) | 52 | 93,809 (83,485) | 18.77 | 21.41 | 3 / 12 / 37 | 2.3% | 79.1 |
| Spatial-VG | 59,949 / 4,875 / 26,980 | 150 (150) | 50 | 1,441,984 (598,805) | 13.63 | 5.27 | 8 / 29 / 13 | 4.0% | 89.0 |

Ảnh theo split = ảnh có ít nhất một bộ ba trong split (Spatial-VRD-Trans chia theo bộ ba nên một ảnh nằm ở nhiều split). #Obj. classes: VRD là tên vật thể tự do (`names[0]`, chữ thường), VG là 150 lớp. Boxes / img. tính mọi box của ảnh; Triples / img. tính trên ảnh có bộ ba. Head / body / tail theo ngưỡng của BGNN (Li et al., 2021): > 10.000 / 500–10.000 / < 500 bộ ba train. Diện tích box = phần box nằm trong ảnh chia diện tích ảnh. Overlap = tỉ lệ bộ ba (mọi split) có box chủ thể và khách thể giao nhau.

### T6. Nhóm quan hệ (phân loại của chúng tôi theo định nghĩa của Zellers et al., 2018)

| Dataset | GEOMETRIC #rel / train % / test % | POSSESSIVE #rel / train % / test % | SEMANTIC #rel / train % / test % | MISC #rel / train % / test % |
|---|---:|---:|---:|---:|
| Spatial-VRD | 21 / 70.5 / 70.3 | 5 / 24.9 / 25.2 | 24 / 4.6 / 4.5 | 2 / 0.0 / 0.0 |
| Spatial-VRD-Trans | 21 / 70.4 / 70.5 | 5 / 25.0 / 24.9 | 24 / 4.5 / 4.6 | 2 / 0.0 / 0.0 |
| Spatial-VG | 15 / 50.1 / 49.5 | 6 / 38.4 / 38.5 | 25 / 10.8 / 11.5 | 4 / 0.7 / 0.5 |

Bảng gán từng quan hệ vào nhóm: `relation_category.tsv` của mỗi bộ MMSG (phụ lục).

### T7. Đối chiếu với số công bố của nguồn gốc

| Dataset | Source | Published | Ours (v0) | Match |
|---|---|---|---|---|
| FB15k-237-IMG | Toutanova & Chen (2015); split KG-BERT / MKGformer | 14,541 / 237 / 272,115 / 17,535 / 20,466 | 14,541 / 237 / 272,115 / 17,535 / 20,466 | ✓ |
| FB15K-237-ind | Daza et al. (2021); CMR Table 1 | ent. 11,633 / 1,454 / 1,454; triples 215,082 / 42,164 / 52,870 | ent. 11,633 / 1,454 / 1,454; triples 215,082 / 42,164 / 52,870 | ✓ |
| MKG-Y | Xu et al. (2022, MMRNS); NativE | 15,000 / 28 / 21,310 / 2,665 / 2,663 | 15,000 / 28 / 21,310 / 2,665 / 2,663 | ✓ |
| Spatial-VRD | Lu et al. (2016), `sg_dataset` | 5,000 images (4,000 train / 1,000 test) | 4,999 images (999 test) | ≈ (thiếu 1 ảnh test GIF) |
| Spatial-VG | Xu et al. (2017); Neau et al. (2023) | 108,073 images / 150 classes / 50 predicates | 108,073 images / 150 classes / 50 predicates | ✓ |

Chỉ liệt kê số mà nguồn gốc công bố. Ảnh VRD thiếu là `4392556686_44d71ff5a0_o.gif` (ảnh test duy nhất dạng GIF; 23 object, 23 relationship trong JSON): không có thực thể nào của ảnh này trong `entity2id`. Spatial-VRD dùng annotation tự do `sg_dataset` (không phải bản 100 lớp / 70 predicate của Lu et al., 2016) và gán lại predicate (P0.4), nên chỉ so được số ảnh. Spatial-VG dùng VG150-curated của Neau et al. (2023), cùng tập 108.073 ảnh và cùng từ vựng 150 lớp / 50 predicate với VG150 (Xu et al., 2017).
