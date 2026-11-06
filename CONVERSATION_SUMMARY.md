# Tổng Hợp & Tóm Tắt Toàn Diện Tiến Độ Luận Văn (Conversation Summary)

**Thời gian cập nhật:** 27/10/2026  
**Đề tài:** Multi-Modal Knowledge Graph Completion with Geometric and Spatial Priors (Geometry-MMKG)  
**Kho mã nguồn:** [Geometry-MMKG (GitHub)](https://github.com/callmezuy/Geometry-MMKG.git) (Branch `main`, commit `fae6fde`)

---

## 1. Tối Ưu Hóa & Giải Quyết Các Vấn Đề Hạ Tầng Kaggle

### A. Vấn đề tràn ổ đĩa Kaggle Working (Limit 19.5 GB vs Data 20 GB)
* **Thực trạng:** Gói dữ liệu `mkgformer-mmkg` nặng ~20.3 GB (ảnh RGB chiếm 18 GB, trọng số HuggingFace `hf_weights` chiếm 1.5 GB). Nếu dùng lệnh `cp -r` sao chép vào `/kaggle/working`, dung lượng sẽ vượt mức cho phép (19.5 GB) ngay lập tức khiến tiến trình train bị hủy (`No space left on device`).
* **Giải pháp:** Áp dụng **Linux Symbolic Link (`ln -s`)**:
  * Chỉ sao chép mã nguồn `MKGformer` (~5 MB) vào `/kaggle/working` để phục vụ chỉnh sửa script Python.
  * Các thư mục lớn gồm `images`, `hf_weights`, và dữ liệu TSV được tạo liên kết mềm (`ln -sfn`) trỏ trực tiếp từ `/kaggle/input/` sang cấu trúc thư mục mong đợi trong `/kaggle/working`.
  * **Hiệu quả:** `/kaggle/working` chỉ chiếm ~10 MB, bảo toàn trọn vẹn ~19.5 GB ổ đĩa trống để lưu trữ checkpoints và nhật ký huấn luyện.

### B. Cấu hình chạy Offline trên Kaggle
* **Vấn đề `pip install`:** Lệnh `pip install -r requirements.txt` trong mã nguồn gốc MKGformer sẽ bị lỗi ngắt kết nối (`ConnectionError`) khi tắt mạng (`Internet: Off`), đồng thời chứa các gói phiên bản cũ (2021) có nguy cơ xung đột với driver CUDA của Kaggle.
* **Xử lý:** Môi trường Kaggle GPU đã có sẵn đầy đủ `torch`, `torchvision`, `transformers`, `pytorch-lightning`, `scikit-learn`... nên đã vô hiệu hóa dòng lệnh `pip install` trong notebook.
* **Bảo mật trọng số mô hình:** Toàn bộ trọng số của `bert-base-uncased` và `clip-vit-base-patch32` đã được chuyển đổi sang định dạng `safetensors` trong thư mục `hf_weights` nhằm vượt qua kiểm tra lỗ hổng bảo mật `torch.load` (CVE-2025-32434).

---

## 2. Quy Hoạch & Chuẩn Hóa Notebooks

Toàn bộ các file Jupyter Notebook trong thư mục `notebooks/` đã được đổi tên theo cú pháp thống nhất: `[tên model]_[tên dataset]_[Reproduce (nếu là baseline)]`:

| Tên File Notebook | Mô hình | Tập Dữ Liệu | Mục Đích |
| :--- | :--- | :--- | :--- |
| [mkgformer-mmkg-reproduce.ipynb](file:///d:/University/Thesis/Code/notebooks/mkgformer-mmkg-reproduce.ipynb) | MKGformer | Merged-MMKG | Tái hiện baseline MKGformer (Offline + Symlink) |
| [native-mmkg-reproduce.ipynb](file:///d:/University/Thesis/Code/notebooks/native-mmkg-reproduce.ipynb) | NativE | Merged-MMKG | Tái hiện baseline NativE trên Merged-MMKG |
| [native-vrd-reproduce.ipynb](file:///d:/University/Thesis/Code/notebooks/native-vrd-reproduce.ipynb) | NativE | Spatial-VRD | Tái hiện baseline NativE trên tập VRD |
| [geometry-mmkg.ipynb](file:///d:/University/Thesis/Code/notebooks/geometry-mmkg.ipynb) | GeometryMMKG | Merged-MMKG | Huấn luyện mô hình GeometryMMKG đề xuất |
| [geometry-vrd.ipynb](file:///d:/University/Thesis/Code/notebooks/geometry-vrd.ipynb) | GeometryMMKG | Spatial-VRD | Huấn luyện mô hình GeometryMMKG trên Spatial-VRD |
| [vg150-curated-extraction.ipynb](file:///d:/University/Thesis/Code/notebooks/vg150-curated-extraction.ipynb) | Feature Extractor | VG150 Curated | Trích xuất đặc trưng ViT, RoBERTa và ZipDepth |

---

## 3. Quản Lý Dữ Liệu & Giải Phóng Dung Lượng Ổ Đĩa Local

### A. Tình trạng các thư mục dữ liệu tại `Code/data/Merged-MMKG/` (~51 GB)
* **`images_depth/` (~32 GB):** Đã hoàn thành nhiệm vụ trích xuất vector đặc trưng `ent_depth_emb.pt` (47 MB). Cả 3 mô hình (`GeometryMMKG`, `NativE`, `MKGformer`) khi huấn luyện đều chỉ đọc file vector `.pt`, hoàn toàn không sử dụng lại thư mục này $\rightarrow$ **Có thể xóa để giải phóng 32 GB**.
* **`images/` (~18 GB):** `GeometryMMKG` và `NativE` chạy trực tiếp bằng `ent_rgb_emb.pt`. Chỉ có `MKGformer` cần ảnh RGB, tuy nhiên toàn bộ 15.458 ảnh đã được đóng gói sẵn vào `mkgformer-mmkg.zip` (20.3 GB) $\rightarrow$ **Có thể xóa thư mục thô này để giải phóng thêm 18 GB**.
* **Dung lượng sau khi dọn:** Thư mục `Merged-MMKG` chỉ còn ~140 MB (gồm các file `.pt` và `.txt`), siêu nhẹ và đủ điều kiện chạy thực nghiệm.

### B. Thư mục `d:\University\Thesis\Dataset` (~22 GB)
* Chứa dữ liệu gốc ban đầu (`VG`: 16 GB, `VRD`: ~6 GB, `VG_curated`: 130 MB).
* Codebase hiện tại đã chuyển toàn bộ dữ liệu đã tiền xử lý vào `Code/data/` (`Spatial-VG`, `Spatial-VRD`, `Merged-MMKG`). Các pipeline trên Kaggle đều gắn trực tiếp với Public Datasets của Kaggle, do đó thư mục local này hoàn toàn có thể xóa hoặc đưa lên Drive lưu trữ.

### C. Ý nghĩa của script `scripts/construct_spatial_mmsg.py`
* Là script cốt lõi chuyển đổi bài toán Scene Graph từ tập thô VRD sang Knowledge Graph đa phương thức không gian.
* Đọc các nhãn quan hệ, cắt 93.806 bounding boxes của từng vật thể ở cả kênh **RGB** và **Depth map** tương ứng, sinh ra các file định danh: `entity2id.txt`, `relation2id.txt`, `train.txt`, `test.txt` (hiện đặt trong `Code/data/Spatial-VRD/`).

---

## 4. Định Hướng Kiến Trúc: GeometryMMKG V1 vs. V2

* **Bản chất nâng cấp V2 (KG Reasoning & Multi-Modal Fusion):**
  * Nâng cấp hàm chấm điểm hình học: Chuyển sang **RotatE** trong không gian số phức ($h \circ r \approx t$).
  * Nâng cấp hàm mất mát: Sử dụng **Self-Adversarial Negative Sampling Loss** để tự động gán trọng số phạt các mẫu âm khó.
  * Tinh chỉnh cơ chế Attention Gating giữa các phương thức (Text, RGB, Depth).
* **Việc tái sử dụng Embedding:**
  * Toàn bộ vector nhúng `ent_rgb_emb.pt`, `ent_depth_emb.pt`, `ent_text_emb.pt` hiện có **được tái sử dụng 100% cho V2**, không cần xử lý lại ảnh raw.
  * Chỉ khi thay đổi mô hình nền tảng thị giác (ví dụ sang DINOv2 hoặc Depth-Anything-V2) thì mới cần forward lại ảnh raw, và công việc này sẽ được phân bổ thực thi trên Kaggle GPU.

---

## 5. Trạng Thái Đồng Bộ Git

* Đã cấu hình [.gitignore](file:///d:/University/Thesis/Code/.gitignore) chặn các tệp nén lớn (`*.zip`, `mkgformer-mmkg/`, `ZipDepth/`, `checkpoints/`, `hf_weights/`).
* Đã hoàn tất commit và push lên nhánh `main` của repo từ xa:
  * Commit ID: `fae6fde`
  * Trạng thái Working Tree: Hoàn toàn sạch (`clean`).
