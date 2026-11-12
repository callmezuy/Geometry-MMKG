# Tổng Hợp Ngữ Cảnh Dự Án Luận Văn (Session Handoff Context)

**Cập nhật lần cuối:** 11/11/2026  
**Kế hoạch hiện hành:** `RESEARCH_PLAN.md` (v2) · **Mục lục dữ liệu:** `data/README.md`  
**Đề tài:** Multi-Modal Knowledge Graph Completion with Geometric and Spatial Priors (`Geometry-MMKG`)  
**GitHub Repository:** `https://github.com/callmezuy/Geometry-MMKG.git` (Branch `main`, Commit mới nhất: `ed293a1`)  
**Workspace Local:** `d:\University\Thesis\Code`

---

## 1. Kiến Trúc Dữ Liệu & Hugging Face Remote Storage

Dữ liệu local đã được làm sạch thành **v2** (11/11/2026, xem `data/README.md`). Đã đẩy lên HF ngày 12/11/2026: nhánh `main` = v2, bản cũ giữ ở tag `v1`. Bảng dưới đây là v2; số liệu v1 để trong ngoặc.

| Dataset (Local & HF) | Hugging Face Repo ID | Số Thực Thể (`entity2id.txt`) | Số Quan Hệ (`relation2id.txt`) | Train / Valid / Test Triplets | Đặc Trưng Trích Xuất Sẵn (`.pt`) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Unified-MMKG** *(Local: `Merged-MMKG`)* | `callmezuy/unified-mmkg` | `15,458` | `429` | `48,798` / `1,395` (v1 `4,981`, rò rỉ) / `6,022` | `ent_rgb_emb.pt`, `ent_depth_emb.pt`, `ent_text_emb.pt` (`[15458, 768]`) |
| **Spatial-VRD** | `callmezuy/spatial-vrd` | `93,809` | `52` (v1 thực tế là 782) | `77,227` / `8,614` / `21,563` (v1 `81,046` / `9,006` / `22,632`) | `ent_rgb_emb.pt`, `ent_depth_emb.pt`, `ent_text_emb.pt` (`[93809, 768]`) |
| **Spatial-VG** *(VG150-curated `no_part_whole`)* | `callmezuy/spatial-vg` | `1,441,984` | `50` | `319,069` / `26,372` / `137,984` (v1 `379,944` / `42,217` / `178,919`) | `ent_rgb_emb.pt`, `ent_depth_emb.pt`, `ent_text_emb.pt` (`[1441984, 768]`, ~4.43 GB/file) |

Thêm bộ đối chứng **Spatial-VRD-Trans** (transductive, `77,331` / `8,592` / `21,481`), repo HF `callmezuy/spatial-vrd-trans`.

**Những thay đổi chính của v2:**
- VRD/VG là thiết lập **inductive**: 100% thực thể valid/test không có trong train. Valid được chia theo ảnh; VG dùng split chính thức của H5.
- Đã loại triplet trùng lặp và triplet rò rỉ giữa các split.
- VRD được gán lại nhãn từ JSON gốc (`relation_review.tsv`). VG dùng tên predicate thật thay cho ID số.
- **Spatial-VG được tạo từ VG150-curated (Neau et al., ICCVW 2023), không phải VG150 chuẩn.**

### Quy ước định dạng file trong cả 3 bộ dữ liệu:
1. **Nhóm file văn bản (`train.txt`, `valid.txt`, `test.txt`):**
   - Dòng đầu tiên là tổng số dòng (`count header`).
   - Các dòng tiếp theo tuân thủ nghiêm ngặt chuẩn **`Head \t Relation \t Tail` (`s \t p \t o`)**.
   - Được sử dụng trực tiếp bởi `FastMMSGDataset` (`core/data_engineering/fast_dataset.py`) và `MMSGDataset` (`core/data_engineering/dataset.py`) của mô hình đề xuất **GeometryMMKG**.
2. **Nhóm file số nguyên OpenKE (`train2id.txt`, `valid2id.txt`, `test2id.txt`):**
   - Dòng đầu tiên là tổng số dòng (`count header`).
   - Các dòng tiếp theo tuân thủ chuẩn C++ OpenKE **`head_id \t tail_id \t relation_id` (`h \t t \t r`)**.
   - Được sử dụng trực tiếp bởi **NativE** (`baselines/NativE/mmkgc/base/Reader.h`) và script chuyển đổi sang **MKGformer**.

---

## 2. Hệ Thống Thực Nghiệm Trên Molab (`molab.marimo.io`)

Đã chuyển đổi luồng thực nghiệm từ Kaggle `.ipynb` sang **7 Marimo Reactive Notebooks (`.py`)** tại `Code/notebooks/`, kết nối trực tiếp tới **Hugging Face Remote Storage** qua `hf_fs = HfFileSystem()` và `snapshot_download`:

### A. Mô hình đề xuất (`GeometryMMKG`)
1. `notebooks/geometry-mmkg.py` — Huấn luyện & đánh giá trên `callmezuy/unified-mmkg`.
2. `notebooks/geometry-vrd.py` — Huấn luyện & đánh giá trên `callmezuy/spatial-vrd`.
3. `notebooks/geometry-vg.py` — Huấn luyện & đánh giá trên `callmezuy/spatial-vg` (nạp `.pt` với `mmap=True`, đánh giá validation mỗi 5 epochs bằng chunked `torch.cdist` + CPU RAM offloading để không bao giờ bị GPU OOM với 1.44M thực thể).

### B. Baseline `NativE` (Chạy đủ 3 kịch bản: Control 2-Modal, Ablation 1 Early Fusion Mean, Ablation 2 Late Fusion 3-Modal)
4. `notebooks/native-mmkg-reproduce.py` — Trên `callmezuy/unified-mmkg`.
5. `notebooks/native-vrd-reproduce.py` — Trên `callmezuy/spatial-vrd`.
6. `notebooks/native-vg-reproduce.py` — Trên `callmezuy/spatial-vg` (tự động pool chiều `768 -> 256` theo từng chunk trên CPU với `mmap=True` trước khi lưu `.pth` để giảm dung lượng mỗi bảng từ `4.43 GB` xuống `1.47 GB`, chống tràn VRAM).

### C. Baseline `MKGformer` (Phase 1 Entity Modeling `pretrain=1` + Phase 2 Link Prediction `pretrain=0`)
7. `notebooks/mkgformer-mmkg-reproduce.py` — Tự động chuyển đổi `callmezuy/unified-mmkg` sang format TSV (`train.tsv`, `dev.tsv`, `test.tsv`, `entities.txt`, `relations.txt`, `entity2text.txt`, `relation2text.txt`), chạy Phase 1 và tự động tìm `.ckpt` tốt nhất trong `output/` để chạy tiếp Phase 2.

---

## 3. Các Tối Ưu Hóa Kỹ Thuật & Đảm Bảo Công Bằng Đánh Giá (Commit `ed293a1`)

1. **`core/models/evaluation.py` (`evaluate_link_prediction`):**
   - Thay vòng lặp từng mẫu bằng phép tính khoảng cách vector hóa theo từng khối `torch.cdist(..., p=1)` (`entity_chunk_size=32768`).
   - Tự động offload bảng thực thể đã hợp nhất (`all_entity_embs`) sang CPU RAM khi `num_entities > 250,000` (giúp đánh giá `Spatial-VG` 1.44M thực thể chỉ tốn ~120 MB VRAM cho mỗi chunk).
2. **`core/models/end_to_end.py` (`GeometryMMKGModel`):**
   - Thêm tham số `use_encoders=False` khi huấn luyện từ đặc trưng `.pt` trích xuất sẵn để bỏ qua việc tải/khởi tạo `roberta-base`, `ResNet50` và `MiDaS`.
3. **`baselines/NativE/`:**
   - `AdvRelRotatEDB15K.py`: Tự động nhận diện `self.img_dim = img_emb.shape[1]`, `self.text_dim = text_emb.shape[1]`, `self.numeric_dim = numeric_emb.shape[1]` thay vì fix cứng `4096 -> 256` của VGG16 cũ.
   - `run_adv_wgan_gp.py` & `run_adv_wgan_gp_3modal.py` & `BaseModule.py`: Truyền `type_constrain=False` vào `TestDataLoader(..., "link", False)` (khớp với `tester.run_link_prediction(type_constrain=False)` gốc của tác giả NativE), bật lại `kge_score.load_checkpoint(args.save)` trước khi test ở bản 3-modal, và thêm `weights_only=False` cho `torch.load` trên PyTorch 2.6+.
4. **Đảm bảo tính công bằng (Fair Comparison):**
   - Cả 3 mô hình (`GeometryMMKG`, `NativE`, `MKGformer`) đều đánh giá **Filtered Link Prediction** trên toàn bộ tập thực thể $N$ mà **không dùng bất kỳ bộ lọc Type Constraint nào** (`type_constrain=False`).

---

## 4. Trạng Thái Kết Quả Thực Nghiệm & Công Việc Tiếp Theo

### Kết quả đã có (chạy trên dữ liệu v1):
* **NativE trên `Unified-MMKG` (`Merged-MMKG`, 1000 Epochs)** — lưu tại `experiment_results/native_mmkg_results.md` (chạy ngày `05/10/2026` trên bộ `*2id.txt` và `.pt` giữ nguyên từ `04/10/2026`):
  * **Control (RGB + Text):** `MRR = 0.384820` | `MR = 913.64` | `Hits@1 = 0.303388` | `Hits@3 = 0.429508` | `Hits@10 = 0.532962`
  * **Ablation 1 (Mean RGB+Depth + Text):** `MRR = 0.381977` | `MR = 936.28` | `Hits@1 = 0.300232` | `Hits@3 = 0.427018` | `Hits@10 = 0.530306`
  * **Ablation 2 (3-Modal RGB+Depth+Text):** `MRR = 0.382994` | `MR = 926.93` | `Hits@1 = 0.301063` | `Hits@3 = 0.427350` | `Hits@10 = 0.533128`

> ⚠️ Kết quả NativE ở trên được chạy với valid v1, trong đó 72% triplet bị rò rỉ sang train/test. Cần kiểm tra NativE có dùng valid để chọn checkpoint không; nếu có thì phải chạy lại trên v2 (P0.3 trong `RESEARCH_PLAN.md`).

### Các thực nghiệm cần chạy mới (xem thứ tự ưu tiên P1–P4 trong `RESEARCH_PLAN.md`):
1. **GeometryMMKG** trên cả 3 bộ (`geometry-mmkg.py`, `geometry-vrd.py`, `geometry-vg.py`) theo quy trình chọn best model trên `valid.txt` rồi đánh giá trên `test.txt`.
2. **NativE** trên 2 bộ không gian mới (`native-vrd-reproduce.py`, `native-vg-reproduce.py`).
3. **MKGformer** trên `Unified-MMKG` (`mkgformer-mmkg-reproduce.py`).

---

## 5. Quy Tắc Làm Việc (User Preferences)
* Sử dụng đường dẫn tương đối (relative paths) trong tất cả các script của dự án.
* Không ghi đè dữ liệu gốc: script làm sạch luôn ghi ra `--dst` khác `--src`.
* **Luôn liệt kê và báo cáo chi tiết mọi thay đổi trong commit cho người dùng kiểm tra trước khi chạy lệnh `git push`**.
