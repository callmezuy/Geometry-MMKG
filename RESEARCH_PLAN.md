# Kế Hoạch Nghiên Cứu & Danh Sách Công Việc (v2)

**Cập nhật:** 10/11/2026
**Thay thế:** Implementation Plan cũ (4 giai đoạn: Nạp dữ liệu → Benchmark → Phân tích → Viết bài)
**Lý do sửa:** Phát hiện VRD/VG là thiết lập *inductive* (100% thực thể test không có trong train), trong khi Unified-MMKG là *transductive*. Nếu so sánh trực tiếp, hiệu ứng của Depth bị trộn lẫn với hiệu ứng của chế độ đánh giá.

---

## 1. Câu hỏi nghiên cứu

| RQ | Câu hỏi | Bằng chứng cần có |
|---|---|---|
| **RQ1** | Depth có cần thiết cho MMKG không? Khi nào thì cần? | Δdepth trên ma trận 2×2 (mục 2); Δdepth theo nhóm SPATIAL / SEMANTIC |
| **RQ2** | Nếu cần thì fuse Depth như thế nào? | So sánh các cách fusion trên **cùng một backbone**; phân tích hành vi gate |
| **RQ3** | Một mô hình có dùng được cho cả MMKG và MMSG không? | Hiệu năng trên cả hai loại đồ thị; giải thích vì sao baseline transductive không chuyển được sang MMSG |

**Giả thuyết chính:** Depth có ích khi các thực thể có chung không gian nhìn (MMSG), và gần như trung tính với MMKG khái niệm. Hiệu ứng này phụ thuộc vào **loại đồ thị**, không phụ thuộc vào **chế độ inductive/transductive**.

---

## 2. Thiết kế thí nghiệm

### 2.1. Đại lượng đo chính

> **Δdepth = Metric(có Depth) − Metric(không Depth)**, tính trong **cùng mô hình, cùng dataset, cùng chế độ đánh giá**.

Mọi kết luận về Depth phải dựa trên Δdepth. Không dùng kết quả "GeometryMMKG thắng baseline X" để kết luận về Depth.

### 2.2. Ma trận 2×2 tách hai yếu tố

| | Transductive | Inductive |
|---|---|---|
| **MMKG** (thực thể độc lập) | `Merged-MMKG` (chính) | `Merged-MMKG-Ind` (tùy chọn) |
| **MMSG** (chung không gian nhìn) | `Spatial-VRD-Trans` (đối chứng) | `Spatial-VRD`, `Spatial-VG` (chính) |

Kết quả kỳ vọng: Δdepth ≈ 0 ở hàng MMKG và Δdepth > 0 ở hàng MMSG, **ở cả hai cột**.

### 2.3. Vai trò từng dataset

| Dataset | Chế độ | Vai trò |
|---|---|---|
| Merged-MMKG | Transductive | Benchmark chính cho MMKG; mọi baseline chạy nguyên bản |
| Spatial-VRD | Inductive | Benchmark chính cho MMSG |
| Spatial-VG | Inductive | Chứng minh khả năng mở rộng (1.44M thực thể) |
| Spatial-VRD-Trans | Transductive | **Đối chứng**: tách hiệu ứng inductive khỏi hiệu ứng Depth; baseline chạy nguyên bản |
| Merged-MMKG-Ind | Inductive | Tùy chọn: lấp ô còn lại của ma trận |

---

## 3. Mô hình và baseline

### 3.1. GeometryMMKG và các ablation (bắt buộc, chạy trên mọi dataset)

| Biến thể | Mục đích |
|---|---|
| Full (RGB + Depth + Text, gating) | Mô hình đề xuất |
| w/o Depth (RGB + Text) | Δdepth của chính mô hình |
| Fusion = concat | RQ2 |
| Fusion = mean(RGB, Depth) | RQ2 |
| Fusion = trọng số cố định (scalar học được) | RQ2: gating thích nghi có tốt hơn trọng số tĩnh không |
| + BBox geometry (tùy chọn) | Đối chứng: lợi ích đến từ Depth hay chỉ từ hình học 2D |

### 3.2. Baseline

| Baseline | Merged-MMKG | VRD-Trans | VRD / VG (inductive) |
|---|---|---|---|
| NativE (3 kịch bản) | Nguyên bản ✅ đã có | Nguyên bản | Bản *feature-only* (ghi rõ "adapted") |
| SNAG | Nguyên bản | Nguyên bản | Bản *feature-only* |
| MKGformer | Nguyên bản | Nguyên bản | Không chạy; giải thích: entity-as-token, không inductive, không mở rộng tới 1.44M |
| Feature-TransE (RGB+Text / RGB+Depth+Text, MLP) | Có | Có | Có: baseline đơn giản cùng đặc trưng, cô lập đóng góp của Router |
| Baseline thêm sau | Nguyên bản | Nguyên bản | Feature-only nếu là transductive |

**Feature-only adaptation:** thay embedding theo ID bằng `e = MLP(đặc trưng modal)`, giữ nguyên hàm score và cách fusion gốc. Báo thêm một dòng bản nguyên gốc (ID embedding ngẫu nhiên cho thực thể chưa thấy) để tham khảo.

---

## 4. Giao thức đánh giá thống nhất

1. **Filtered link prediction**, xếp hạng **cả head và tail**, lấy trung bình.
2. Không gian ứng viên = toàn bộ `entity2id` (N), không dùng type constraint.
3. Filter set = train ∪ valid ∪ test (toàn bộ triplet đúng).
4. Xử lý trường hợp bằng điểm: thống nhất một cách duy nhất cho mọi mô hình (đề xuất: rank = 1 + số ứng viên có điểm *cao hơn hẳn*, giống OpenKE).
5. Metric: MRR, MR, Hits@1/3/10.
6. **Mọi mô hình dump rank theo từng triplet** ra `ranks_<model>_<dataset>.csv` (`h, r, t, rank_head, rank_tail`). **Một script chung duy nhất** tính metric và phân tích theo nhóm quan hệ.
7. Chọn checkpoint theo MRR trên valid, đánh giá trên test một lần.
8. Seed cố định (42), chạy **3 seed** cho các cấu hình chính và báo mean ± std.
9. Ghi rõ trong bài: VG validation dùng tập con 5.000 mẫu; NativE trên VG pool 768 → 256.

---

## 5. Danh sách công việc

### P0. Sửa dữ liệu (làm trước mọi thí nghiệm)

P0.1–P0.3b làm bằng `scripts/clean_splits.py` (11/11/2026). **Bản v2 hiện là dữ liệu chính tại `data/<tên>/`** (cùng tên với repo HF); các file split v1 được lưu ở `data/_archive_v1/`. Chi tiết trong `clean_stats.json` của mỗi thư mục; cấu trúc và lệnh tái tạo ở `data/README.md`.

- [x] **P0.1** Gán lại nhãn quan hệ cho Spatial-VRD **từ JSON gốc** (`scripts/build_vrd_relation_review.py`): 1.310 predicate gốc → **52 quan hệ**, loại 1.0% triplet, đảo chiều 115 triplet bị động. Đã sửa 91 triplet bị script cũ `clean_mmsg_aliases.py` gán sai (so khớp chuỗi con, mất thông tin hướng); chi tiết ở `data/Spatial-VRD/relation_audit.tsv`. Các quyết định cần duyệt đang theo đề xuất mặc định (cột `needs_review`).
- [x] **P0.2** Loại triplet trùng trong từng split (VRD: 3.175 / 318 / 925; VG: 92.520 / 10.572 / 40.935).
- [x] **P0.3** Loại rò rỉ: Merged-MMKG bỏ **3.586 / 4.981 triplet valid** (nằm trong train hoặc test), valid còn **1.395**. VRD/VG hết rò rỉ sau P0.3b.
  - [ ] Kiểm tra NativE có dùng valid để chọn checkpoint không; nếu có thì chạy lại NativE trên Merged-MMKG v2.
  - [ ] Cân nhắc: valid 1.395 có đủ để chọn checkpoint không, hay tách thêm từ train.
  - [ ] Merged-MMKG có 6 quan hệ trong test không xuất hiện trong train (không thể dự đoán đúng). Quyết định giữ hay bỏ.
- [x] **P0.3b** Chia lại valid theo ảnh: VRD theo tiền tố `image_id` (10% số ảnh train), VG dùng **split chính thức của H5**. Thực thể valid chưa thấy trong train: VRD 20.6% → **100%**, VG 31.1% → **100%**.

| Dataset (v2) | Train | Valid | Test | Quan hệ | Cách tạo |
|---|---:|---:|---:|---:|---|
| Spatial-VRD | 77.227 | 8.614 | 21.563 | 52 | JSON gốc + `relation_review.tsv`; valid chia theo ảnh (10%) |
| Spatial-VG | 319.069 | 26.372 | 137.984 | 50 | `VG-SGG.h5` curated; **split chính thức theo ảnh** (0/1/2) |
| Merged-MMKG | 48.798 | 1.395 | 6.022 | 429 | loại trùng + rò rỉ |
| Spatial-VRD-Trans | 77.331 | 8.592 | 21.481 | 52 | đối chứng transductive từ Spatial-VRD v2 |

**Cập nhật 11/11/2026, nguồn gốc dữ liệu MMSG (quan trọng cho phần viết bài):**
- **Spatial-VG** được tạo từ **VG150-curated, biến thể `no_part_whole`** (Neau et al., ICCVW 2023, repo `Maelic/VG_curated`), **không phải VG150 chuẩn** (Xu et al., 2017). Đã xác minh: dựng lại triplet từ `VG-SGG.h5` khớp chính xác tập test (178.919) và train+valid cũ. File H5 và dicts nằm ở `data/Spatial-VG/`.
- **Spatial-VRD** dùng annotation dạng tự do trong `sg_dataset` (1.310 predicate gốc), không phải bản 70 predicate của Lu et al. 2016. Các predicate được gán lại vào **cùng bộ từ vựng với Spatial-VG** (50 predicate curated), cộng các nhãn riêng của VRD (`left of`, `right of`, `touching`, `around`, `taller than`, `larger than`, `facing`, `pulling`, `cutting`). Bảng ánh xạ: `data/Spatial-VRD/relation_review.tsv`.
- Mỗi thư mục MMSG có `relation_category.tsv` (GEOMETRIC / POSSESSIVE / SEMANTIC / MISC), dùng cho P5.1. **Đã đối chiếu với Neural Motifs (Table 1)**: paper chỉ cho ví dụ, số lớp và tỉ lệ của từng nhóm, không công bố danh sách đầy đủ. Bảng của mình khớp mọi ví dụ trong paper, nhưng phải trình bày là *phân loại của chúng tôi theo định nghĩa của Zellers et al.*, kèm bảng đầy đủ ở phụ lục. Chi tiết sai lệch ghi trong `scripts/build_vrd_relation_review.py`.
- Valid của VG cũ là 10% triplet cắt ngẫu nhiên từ split train của H5; split valid chính thức (5.404 ảnh) chưa từng được dùng. Valid của VRD cũ cũng cắt ngẫu nhiên theo triplet.

- [x] **P0.4** (12/11/2026) Đã đẩy v2 lên HF: `main` = v2, mỗi repo có tag `v1` và `v2`, thêm repo mới `callmezuy/spatial-vrd-trans`. Kế hoạch ban đầu: Đề xuất: tạo tag `v1` trên commit hiện tại của mỗi repo HF, rồi upload thư mục `data/<tên>/` (file `.pt` không đổi, nên LFS không tải lại), sau đó tag `v2`. Spatial-VRD-Trans cần repo HF mới. Notebook vẫn dùng repo ID cũ và tải nhánh `main`, nên sẽ tự nhận v2 sau khi đẩy. (Đã cập nhật mô tả trong notebook và `CONVERSATION_SUMMARY.md`.)
- [x] **P0.5** Tạo split đối chứng từ VRD đã sạch:
  `python scripts/make_transductive_split.py --src data/Spatial-VRD --dst data/Spatial-VRD-Trans`
  0 thực thể valid/test chưa thấy trong train; test phủ 72/73 quan hệ.
- [ ] **P0.6** *(Tùy chọn)* Tạo `Merged-MMKG-Ind`: giữ lại một tập thực thể, đưa mọi triplet chứa chúng vào test.

### P1. Hạ tầng đánh giá

- [ ] **P1.1** `core/models/evaluation.py`: thêm head prediction (`t − r` so với mọi thực thể), trả MR, trả mảng rank theo từng triplet.
- [ ] **P1.2** Thêm seed (`random`, `numpy`, `torch`, `cuda`, `DataLoader generator`) vào 3 notebook `geometry-*.py`.
- [ ] **P1.3** Viết `scripts/compute_metrics.py`: đọc các file rank, tính metric tổng và theo nhóm quan hệ, xuất bảng markdown/LaTeX.
- [ ] **P1.4** NativE: tính filtered rank trong Python từ mảng score (trước khi gọi C++), dump ra file rank. Đối chiếu metric với tester C++ để xác nhận khớp.
- [ ] **P1.5** MKGformer, SNAG: thêm hook dump rank tương tự.
- [ ] **P1.6** Eval trên VG: để `all_h_entity` trên GPU (hoặc fp16) nếu VRAM đủ, tránh copy 4.4 GB từ CPU sang GPU ở mỗi batch.

### P2. Mô hình đề xuất

- [ ] **P2.1** Thêm cờ cấu hình cho các ablation ở mục 3.1 (`use_depth`, `fusion_mode`).
- [ ] **P2.2** Gate có điều kiện theo quan hệ: `G = σ(W[h_rgb ; h_depth ; r])`, khớp giả thuyết "quan hệ không gian cần Depth". So sánh với gate hiện tại.
- [ ] **P2.3** Sửa hoặc mô tả đúng bước cross-attention: với độ dài chuỗi 1, softmax luôn bằng 1, nên text không điều khiển attention. Bước này hiện tương đương `Linear(h_vision)` cộng residual.
- [ ] **P2.4** Log trọng số gate (phần dành cho Depth) theo từng triplet test để phân tích ở P5.
- [ ] **P2.5** *(Tùy chọn)* Modality BBox geometry (x, y, w, h chuẩn hóa, diện tích).

### P3. Baseline

- [ ] **P3.1** Feature-TransE (2-modal và 3-modal).
- [ ] **P3.2** NativE feature-only cho dữ liệu inductive.
- [ ] **P3.3** SNAG: chạy nguyên bản (MMKG, VRD-Trans) và feature-only (VRD/VG).
- [ ] **P3.4** MKGformer trên Merged-MMKG và VRD-Trans.

### P4. Chạy thí nghiệm (Molab)

Ưu tiên theo thứ tự:

- [ ] **P4.1** Merged-MMKG: GeometryMMKG + ablation; SNAG; MKGformer; Feature-TransE; NativE (chạy lại nếu P0.3 yêu cầu).
- [ ] **P4.2** VRD-Trans: GeometryMMKG (full, w/o Depth); NativE nguyên bản 3 kịch bản; Feature-TransE.
- [ ] **P4.3** VRD inductive: GeometryMMKG + ablation; NativE/SNAG feature-only; Feature-TransE.
- [ ] **P4.4** VG inductive: GeometryMMKG (full, w/o Depth), NativE feature-only. Theo dõi RAM/VRAM ở lần eval đầu tiên.
- [ ] **P4.5** Chạy 3 seed cho các cấu hình chính.
- [ ] **P4.6** Đẩy checkpoint `.pth` và file rank lên HF sau mỗi lần chạy (storage của Molab có thể không bền).

### P5. Phân tích

- [ ] **P5.1** Phân loại quan hệ thành `SPATIAL_RELS` và `SEMANTIC_RELS` (sau P0.1). Ghi tiêu chí phân loại; xử lý các quan hệ lưng chừng (near, with, has); báo số triplet test mỗi nhóm.
- [ ] **P5.2** Bảng Δdepth trên ma trận 2×2 cho từng mô hình.
- [ ] **P5.3** Δdepth theo nhóm SPATIAL / SEMANTIC.
- [ ] **P5.4** Phân bố trọng số gate theo dataset và nhóm quan hệ.
- [ ] **P5.5** So sánh baseline nguyên bản trên VRD-Trans với bản feature-only trên VRD inductive, để định lượng mức sụt do inductive.

### P6. Viết bài

- [ ] **P6.1** Experimental Setup: hai chế độ (transductive / inductive), giao thức ở mục 4, thống kê split (tỉ lệ thực thể chưa thấy, bậc thực thể, số thực thể trong `entity2id` không xuất hiện trong triplet).
- [ ] **P6.2** Implementation Details: `type_constrain=False` là **thiết lập gốc của tác giả NativE**; chunked `cdist` và offload để trong phụ lục.
- [ ] **P6.3** Discussion: Concept (MMKG) vs Instance (MMSG); vai trò của biểu diễn dựa trên đặc trưng cho KG lai.
- [ ] **P6.4** Đoạn định vị so với Scene Graph Generation (MOTIFS, VCTree...).

---

## 6. Các tuyên bố được và không được phép

| ✅ Được nói (nếu dữ liệu ủng hộ) | ❌ Không được nói |
|---|---|
| Depth cải thiện MMSG (Δdepth > 0) ở cả transductive và inductive | Mô hình MMKG truyền thống thất bại trên MMSG vì thiếu Depth |
| Depth gần như trung tính với MMKG; gating thích nghi không làm giảm hiệu năng | Depth làm hại baseline cũ (chênh 0.003 MRR, 1 seed, không có ý nghĩa thống kê) |
| Baseline transductive sụt trên MMSG chủ yếu do chế độ inductive (bằng chứng từ VRD-Trans) | Chunking/offload là đóng góp khoa học |
| Biểu diễn dựa trên đặc trưng + fusion Depth thích nghi phù hợp cho KG lai | Đánh giá "hoàn toàn công bằng" khi chưa thống nhất head/tail prediction |

---

## 7. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Δdepth trên VRD-Trans ≈ 0 (Depth chỉ giúp khi inductive) | Vẫn là kết quả có giá trị: Depth đóng vai trò đặc trưng định danh cho thực thể mới. Điều chỉnh lại luận điểm RQ1 |
| Gating không tốt hơn trọng số cố định | Chuyển trọng tâm RQ2 sang gate theo quan hệ (P2.2) |
| BBox tốt bằng Depth | Báo cáo trung thực; nhấn mạnh tính bổ sung nếu có (Depth + BBox) |
| Không đủ GPU cho 3 seed | Ưu tiên 3 seed cho Merged-MMKG và VRD; VG chạy 1 seed |
| Nhóm SEMANTIC trong VRD quá nhỏ | Gộp thêm VG; báo khoảng tin cậy bootstrap |
