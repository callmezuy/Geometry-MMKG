# Kế Hoạch Nghiên Cứu & Danh Sách Công Việc

**Cập nhật:** 19/11/2026. **Dữ liệu:** v0 (tag `v0` trên HF, mục P0). Mọi thí nghiệm chạy trên dữ liệu v0 và theo giao thức ở mục 4.

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
| **MMKG** (thực thể độc lập) | `FB15k-237-IMG` (chính); `MKG-Y` (phụ) | `FB15K-237-ind` (split của BLP, dùng trong CMR) |
| **MMSG** (chung không gian nhìn) | `Spatial-VRD-Trans` (đối chứng) | `Spatial-VRD`, `Spatial-VG` (chính) |

Kết quả kỳ vọng: Δdepth ≈ 0 ở hàng MMKG và Δdepth > 0 ở hàng MMSG, **ở cả hai cột**.

### 2.3. Vai trò từng dataset

| Dataset | Chế độ | Vai trò |
|---|---|---|
| FB15k-237-IMG | Transductive | Benchmark chính cho MMKG; benchmark chuẩn, so được với số trong paper (MKGformer và các bài sau); mọi baseline chạy nguyên bản |
| MKG-Y | Transductive | Bộ phụ cho MMKG. Ảnh chính thức của MMRNS phủ 96,5% thực thể; thuộc họ benchmark báo MRR nên so được với số công bố của NativE, SNAG, AdaMF-MAT, MyGO, MoMoK. Split giữ nguyên của MMRNS |
| FB15K-237-ind | Inductive | Ô MMKG × inductive. Split của BLP (Daza et al., 2021), được CMR (Zhao et al., SIGIR 2024) thêm ảnh và mô tả; 11.633 / 1.454 / 1.454 thực thể train / valid / test, thực thể valid và test chưa thấy khi train. Giao thức của CMR trùng giao thức mục 4 (xếp hạng đầy đủ, filtered, head + tail), nên so được với số công bố của CMR, MoSE-MI, BLP, StATIK |
| Spatial-VRD | Inductive | Benchmark chính cho MMSG |
| Spatial-VG | Inductive | Chứng minh khả năng mở rộng (1,44M thực thể) |
| Spatial-VRD-Trans | Transductive | **Đối chứng**: tách hiệu ứng inductive khỏi hiệu ứng Depth; baseline chạy nguyên bản |
| DB15K | — | Chỉ dùng cho P3.0b (tái lập NativE bằng đặc trưng gốc của tác giả). Không dùng cho thí nghiệm depth: không nguồn nào phát hành ảnh gốc |

**Không dùng** (lý do ghi trong bài): MKG-W (ảnh chính thức phủ tối đa khoảng 62%), WN18-IMG (19%), VTKG/VISTA (không phát hành ảnh gốc), các bộ gộp nhiều nguồn tự dựng (quan hệ không ghép được giữa nguồn, không so được với bài báo nào).

---

## 3. Mô hình và baseline

### 3.1. GeometryMMKG và các ablation (bắt buộc, chạy trên mọi dataset)

| Biến thể | Mục đích |
|---|---|
| Full (RGB + Depth + Text, gating) | Mô hình đề xuất (thiết kế ở đầu mục P2) |
| w/o Depth (RGB + Text) | Δdepth của chính mô hình |
| Fusion = concat | RQ2 |
| Fusion = mean(RGB, Depth) | RQ2 |
| Fusion = trọng số cố định (scalar học được) | RQ2: gating thích nghi có tốt hơn trọng số tĩnh không |
| + BBox geometry (tùy chọn) | Đối chứng: lợi ích đến từ Depth hay chỉ từ hình học 2D |
| Thiếu ảnh = vector hằng, không mặt nạ (dataset có thực thể thiếu ảnh) | Công bằng với baseline: tách lợi ích của cơ chế xử lý thiếu ảnh khỏi phần còn lại của mô hình |

### 3.2. Baseline

**Mức 1 (làm trước; 3 seed ở các ô chính):**

| Baseline | FB15k-237-IMG, MKG-Y | VRD-Trans | FB15K-237-ind, VRD / VG (inductive) | Δdepth? |
|---|---|---|---|---|
| NativE (Control / +D mean / +D concat) | Nguyên bản | Nguyên bản | Nguyên bản trên FB15K-237-ind và Spatial-VRD, 1 seed; không chạy VG | Có, tầng dữ liệu |
| SNAG (COLING 2025) | Nguyên bản | Nguyên bản | Như NativE | Có, tầng dữ liệu |
| MoMoK (ICLR 2025) | Nguyên bản | Nguyên bản | Như NativE | Có, tầng dữ liệu (không thêm expert) |
| MKGformer (SIGIR 2022) | Nguyên bản (FB15k-237-IMG là dataset gốc; so với số arXiv v5 đã sửa lỗi: H@1 0,256 · H@3 0,367 · H@10 0,504 · MR 221, paper không báo MRR) | Nguyên bản | Không chạy; giải thích: entity-as-token, không inductive, không mở rộng tới 1,44M | Không: đọc ảnh thô, làm mốc tham chiếu |
| CMR (SIGIR 2024) | Không bắt buộc | Không bắt buộc | Nguyên bản trên FB15K-237-ind (số paper: MRR 0,371 · H@1 0,269 · H@10 0,567) và Spatial-VRD; VG không chạy (bi-encoder PLM + ViT trên 1,44M thực thể quá nặng) | Không: đọc ảnh thô, làm mốc tham chiếu. Code CMR bắt buộc mỗi thực thể có đúng một ảnh (tác giả tự crawl bù ảnh còn thiếu, P0.2); cách cấp ảnh cho thực thể thiếu ảnh quyết định ở P3.7 |
| Feature-TransE (RGB+Text / RGB+Depth+Text, MLP) | Có | Có | Có: baseline đơn giản cùng đặc trưng, cô lập đóng góp của Router | Có |

Δdepth giữa các mô hình được kiểm chứng trên NativE, SNAG, MoMoK, Feature-TransE và GeometryMMKG. MKGformer và CMR là mốc so sánh với mặt bằng chung của lĩnh vực, không có bản thêm depth.

**Mức 2 (chỉ làm nếu còn GPU; 1 seed):** AdaMF-MAT (COLING 2024; sinh vector đối kháng cho modality thiếu), IKRL và TBKGC (đã có sẵn trong `baselines/NativE/mmkgc/module/model/`), BLP-TransE (mốc chỉ dùng text cho FB15K-237-ind). MyGO lên mức 2 nếu code tách token đã được phát hành.

**Chỉ trích số từ paper, không chạy lại** (ghi lý do một dòng trong bài): RSME, TransAE, VBKGC, OTKGE, IMF, QEB, VISTA, LAFA, SGMPT, MyGO (nếu không lên mức 2), MCKGC (không có code hoặc không đưa depth vào được); KICGPT, MKGL, K-ON, GLTW, ELMM (dùng LLM, ngoài ngân sách tính toán); GraIL, NBFNet, ULTRA (thuần cấu trúc, không dùng modality); StATIK, StAR, kNN-KGE (chỉ dùng text; số trên FB15K-237-ind lấy từ CMR).

**Lưu ý so số với paper:** hai họ benchmark tách biệt. Họ FB15k-237-IMG (MKGformer, RSME, SGMPT, LAFA, ELMM) thường chỉ báo Hits@k và MR. Họ DB15K / MKG-W / MKG-Y của MMRNS (NativE, SNAG, AdaMF-MAT, MyGO, MoMoK) báo MRR. Khi đối chiếu số paper chỉ so cùng dataset và cùng metric. Trên MKG-Y, baseline gốc dùng đặc trưng của tác giả còn mình dùng đặc trưng DINOv2 (tiêu chí 4 của P3.0), nên số của mình có thể lệch số paper; phần tái lập với đặc trưng gốc nằm ở P3.0 tiêu chí 1.

**Thiếu ảnh (xử lý ở tầng dữ liệu, không sửa kiến trúc baseline).** Depth suy ra từ RGB nên thực thể thiếu ảnh thì thiếu cả RGB lẫn depth. Không bỏ thực thể, không suy đoán bù.
1. **Baseline dùng bảng đặc trưng (NativE, SNAG, Feature-TransE, ...):** dùng chung một bảng đặc trưng; mọi thực thể thiếu ảnh nhận **cùng một vector hằng** = trung bình đặc trưng của các thực thể có ảnh (tính trên thực thể train, riêng cho RGB và depth). Nhiều ảnh cho một thực thể: lấy trung bình đặc trưng DINOv2. Không sửa code mô hình: lớp chiếu học được biến vector hằng `c` thành `Wc + b`, gần tương đương một vector "thiếu" học được.
2. **Không dùng vector ngẫu nhiên riêng cho từng thực thể** (cách hay gặp trong các bộ MMKG): ở chế độ transductive nó đóng vai trò embedding theo ID miễn phí, và bản có depth nhận thêm một bảng ID như vậy, làm Δdepth dương giả. Ngoại lệ duy nhất là cơ chế có sẵn trong code gốc của baseline (SNAG đọc `missing.json` trên bộ MKG, P3.0d), vì quy tắc P3 giữ nguyên code gốc.
3. **Baseline đọc ảnh thô (MKGformer, ...):** giữ nguyên cách gốc (MKGformer: ảnh toàn 0 cho thực thể thiếu ảnh, chọn ngẫu nhiên tối đa 7 ảnh chia 1 chính / 3 phụ / 3 RCNN).
4. **GeometryMMKG:** mặt nạ có/không + một vector "thiếu" học được thay cho phần thị giác (RGB và depth) của thực thể thiếu ảnh (P2.6). Đây là **thiết kế của mô hình đề xuất**, không phải quy tắc chung. Kèm ablation dùng cùng vector hằng như baseline, không mặt nạ (mục 3.1), để phần chênh lệch với baseline không bị quy cho cách xử lý thiếu ảnh.

Δdepth (RQ1) không bị ảnh hưởng bởi lựa chọn này vì luôn so trong cùng một mô hình: bản có và không có depth thiếu ở cùng thực thể và được xử lý như nhau.

**Baseline ở ô inductive:** NativE, SNAG, MoMoK chạy nguyên bản: thực thể chưa thấy giữ embedding ID ngẫu nhiên, chỉ phần đặc trưng mang thông tin. Không viết bản "feature-only" (thay embedding ID bằng MLP của đặc trưng là sửa kiến trúc). Mọi baseline giữ kiến trúc gốc; depth chỉ vào ở tầng dữ liệu (quy tắc ở P3).

---

## 4. Giao thức đánh giá thống nhất

1. **Filtered link prediction**, xếp hạng **cả head và tail**, lấy trung bình.
2. Không gian ứng viên = toàn bộ `entity2id` (N), không dùng type constraint. FB15K-237-ind cũng vậy: ứng viên gồm cả thực thể đã thấy và chưa thấy, đúng như CMR.
3. Filter set = train ∪ valid ∪ test (toàn bộ triplet đúng).
4. Bằng điểm: rank = 1 + số ứng viên có điểm *tốt hơn hẳn* (giống OpenKE), cho mọi mô hình.
5. Metric: MRR, MR, Hits@1/3/10.
6. **Mọi mô hình dump rank theo từng triplet** ra `ranks_<model>_<dataset>.csv` (`h, r, t, rank_head, rank_tail`). **Một script chung duy nhất** (`scripts/compute_metrics.py`) tính metric và phân tích theo nhóm quan hệ.
7. Chọn checkpoint theo MRR trên valid, đánh giá trên test một lần.
8. Seed cố định (42), chạy **3 seed (42, 43, 44)** cho các cấu hình chính và báo mean ± std.
   - **Cấu hình chính** = các cặp có Depth / không Depth của mỗi mô hình trên FB15k-237-IMG, FB15K-237-ind, Spatial-VRD và VRD-Trans (các ô dùng để kết luận Δdepth).
   - **Chỉ 1 seed (42):** kiểm tra tái lập paper (P3.0b), ablation fusion phụ (RQ2) nếu thiếu GPU, MKG-Y, Spatial-VG, và baseline ở ô inductive.
   - Lý do: khoảng tin cậy bootstrap chỉ phản ánh dao động do mẫu test, không phản ánh dao động do huấn luyện.
9. Ghi rõ trong bài: VG validation dùng tập con 5.000 mẫu (GeometryMMKG).
10. Với dataset có thực thể thiếu ảnh: báo trên test đầy đủ, kèm bảng chia nhóm theo số đầu bộ ba có ảnh (cả hai / một / không). Nhóm **"không"** là đối chứng âm: Δdepth ở đó phải xấp xỉ 0. Chỉ kết luận từ nhóm "không" khi nhóm đủ lớn: FB15k-237-IMG chỉ có 14 triplet test ở nhóm này, MKG-Y chỉ có 2 (P0.1, P0.3), nên nhóm so sánh chính là nhóm "một" (FB15k-237-IMG: 1.247 triplet). **Cần quyết định** (hoãn): tạo nhóm "không" lúc test bằng cách che ảnh (đặt mask = 0, dùng vector thay thế) của cả hai đầu trên toàn bộ test, rồi đo Δdepth. Việc này chỉ đổi cách đánh giá, không đụng dữ liệu, đặc trưng hay huấn luyện. Hạn quyết: trước khi chạy đánh giá các cấu hình chính. FB15k-237-IMG báo thêm Δdepth trên thực thể vật lý so với thực thể trừu tượng (ảnh Bing của khái niệm trừu tượng ít mang thông tin hình học).

---

## 5. Danh sách công việc

### P0. Dữ liệu v0

Sáu bộ, mỗi bộ một repo HF public, tag `v0`; cấu trúc file, lệnh dựng lại và đặc trưng ở `data/README.md`, thẻ dataset ở `hf_cards/`. Mọi lệnh dựng lại cho kết quả giống hệt từng byte.

| Dataset | Train / Valid / Test | Thực thể | Quan hệ | Nguồn |
|---|---|---:|---:|---|
| FB15k-237-IMG | 272.115 / 17.535 / 20.466 | 14.541 | 237 | split + văn bản KG-BERT, ảnh MKGformer |
| FB15K-237-ind | 215.082 / 42.164 / 52.870 | 14.541 | 237 | split BLP; văn bản, ảnh, `entity2id` của FB15k-237-IMG |
| MKG-Y | 21.310 / 2.665 / 2.663 | 15.000 | 28 | split, ảnh, mô tả chính thức của MMRNS; id của NativE |
| Spatial-VRD | 77.227 / 8.614 / 21.563 | 93.809 | 52 | `sg_dataset` của VRD + `relation_review.tsv` |
| Spatial-VRD-Trans | 77.371 / 8.578 / 21.455 | 93.809 | 52 | chia lại Spatial-VRD, phân tầng theo quan hệ |
| Spatial-VG | 319.069 / 26.372 / 137.984 | 1.441.984 | 50 | VG150-curated `no_part_whole` (`VG-SGG.h5`) |

- [x] **P0.1 FB15k-237-IMG** (`scripts/build_fb15k237_img.py`).
  - Split KG-BERT giữ nguyên (`dev.tsv` thành `valid.txt`), giống hệt bản trong `baselines/MKGformer/MKG/dataset/FB15k-237/`; không trùng, không rò rỉ sang train; valid có 8 và test có 29 thực thể không xuất hiện trong train (đặc điểm của split chuẩn, giữ nguyên). `entity2id` / `relation2id` theo thứ tự `entities.txt` / `relations.txt`. Văn bản: `entity2text.txt` (tên ngắn), `entity2textlong.txt` (26 thực thể không có mô tả dài thì dùng tên ngắn), `relation2text.txt`.
  - Ảnh: kho ảnh MKGformer (Google Drive id `197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5-`, `data/raw/fb15k237/fb15k_wn18_images.zip`, 3,03 GB), `image_index.tsv` (`entity \t zip member \t status`), 146.019 ảnh, đọc được hết.
  - **Ảnh giữ chỗ** (không thể hiện gì về thực thể: icon ảnh hỏng dùng chung cho 510 thực thể, "image no longer available", ảnh trắng/đen hoặc 1×1, thumbnail mặc định của MTV, Vimeo, timeanddate.com) vẫn nằm trong index, chỉ được đánh dấu. Ba luật: (1) md5 thuộc danh sách đã duyệt bằng mắt `scripts/fb15k237_image_placeholders.tsv` (59 hash; đã duyệt mọi md5 dùng chung cho ≥3 thực thể, 200 ảnh gần trùng nhất, mọi ảnh có std RGB 1–8); (2) dHash cách một mẫu trong danh sách ≤2 bit (chỉ dùng mẫu có std ≥10); (3) gần một màu (std RGB <3). 21 thực thể màu (đuôi của quan hệ `*/colors`, như Maroon) được miễn luật 1 và luật 3. Kết quả: 1.541 ảnh giữ chỗ (1,06%).
  - Độ phủ: **14.292 / 14.541 thực thể (98,29%)** có ít nhất một ảnh thật; 1–48 ảnh mỗi thực thể (trung vị 10). Test chia theo số đầu bộ ba có ảnh: cả hai 19.205 / một 1.247 / không 14.
- [x] **P0.2 FB15K-237-ind** (`scripts/build_fb15k237_ind.py`).
  - Nguồn: bản lưu trên Zenodo (record 4501273, `FB15k-237.tar.gz`, md5 khớp; link surfdrive trong README của BLP đã chết). Split `ind-*.tsv` giống hệt từng byte với `data/FB15k237_ind/` trong repo CMR.
  - Khớp Table 1 của CMR (11.633 / 215.082; 1.454 / 42.164; 1.454 / 52.870; 237 quan hệ). Thực thể train / valid / test không giao nhau; triplet train chỉ chứa thực thể train; quan hệ ở valid và test đều có trong train. Gộp ba split được đúng 310.116 triplet của FB15k-237.
  - Dùng chung `entity2id`, `relation2id`, văn bản và `image_index.tsv` với FB15k-237-IMG, nên đặc trưng dùng chung được (điền lại vector hằng theo train của bộ này, P0.7). Kèm `{train,valid,test}_entities.txt`.
  - Cấu trúc test: 27.801 triplet chỉ có đầu là thực thể mới, 22.536 chỉ có đuôi, 2.533 có cả hai; 962 thực thể valid xuất hiện trong test. BLP và CMR xếp hạng test trên toàn bộ 14.541 thực thể.
  - Ảnh: 98,14% thực thể test có ảnh. Test chia theo số đầu có ảnh: cả hai 49.338 / một 3.494 / không 38.
  - **Khác CMR về ảnh:** CMR dùng một ảnh "tốt nhất" mỗi thực thể (`fb15k_best_img.pickle`, không có trong repo) và tự crawl bù ảnh cho thực thể thiếu (bộ ảnh để trên BaiduNetDisk). Mình dùng kho MKGformer và không bù ảnh (mục 3.2); ghi rõ khi so với số paper.
- [x] **P0.3 MKG-Y** (`scripts/build_mkgy.py`).
  - Split, ảnh (`MKG-Y_img.zip`, 6,3 GB) và mô tả chính thức trong Drive của MMRNS (folder `1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj`, về `data/raw/mmrns_official/`). Ánh xạ qua `entity2id` / `relation2id` của NativE thì trùng `baselines/NativE/benchmarks/MKG-Y/` cả thứ tự; giữ id của NativE để đặc trưng gốc của tác giả khớp dòng. Không trùng, không rò rỉ sang train; valid có 371, test có 352 thực thể không xuất hiện trong train (split gốc, giữ nguyên).
  - Ghép thư mục ảnh với thực thể (tên thư mục bị Windows sửa): trùng y nguyên 14.244, trùng sau chuẩn hoá Windows 222, trùng sau khi chỉ giữ chữ và số 15; mỗi tầng chỉ nhận khi ra đúng một thực thể chưa ghép (`image_folder_match.tsv`).
  - Ảnh: 43.353 file, 3 file không đọc được, 79 bản trùng trong cùng thực thể, 7 khung gần một màu; đã duyệt bằng mắt mọi md5 dùng chung cho ≥2 thực thể và mọi ảnh có std RGB <12, không có ảnh giữ chỗ nào khác (`scripts/mkgy_image_placeholders.tsv` rỗng).
  - **Độ phủ 14.481 / 15.000 thực thể (96,54%)**, 1–3 ảnh mỗi thực thể. Test chia theo số đầu có ảnh: cả hai 2.496 / một 165 / **không 2**.
  - Văn bản: `entity2textlong.txt` là mô tả MMRNS (12.316 thực thể), 2.684 thực thể còn lại dùng tên; `entity2text.txt` là tên.
- [x] **P0.4 Spatial-VRD** (`scripts/build_vrd_relation_review.py`, `scripts/clean_splits.py`, đọc JSON gốc qua `scripts/vrd_raw.py`).
  - Annotation dạng tự do trong `sg_dataset` (1.310 predicate gốc), không phải bản 70 predicate của Lu et al. 2016. Predicate được gán vào **cùng bộ từ vựng với Spatial-VG** (50 predicate curated) cộng các nhãn riêng của VRD (`left of`, `right of`, `touching`, `around`, `taller than`, `larger than`, `facing`, `pulling`, `cutting`): **52 quan hệ**, loại 1,0% triplet, đảo chiều 115 triplet bị động. Luật làm việc trên từ (không so chuỗi con) để giữ thông tin hướng. Bảng ánh xạ: `data/Spatial-VRD/relation_review.tsv`; các dòng cần duyệt đang theo đề xuất mặc định (cột `needs_review`).
  - Loại triplet trùng trong từng split; không có triplet valid/test nằm trong train. Test = ảnh test gốc của VRD; valid = 10% số ảnh train (chia theo tiền tố `image_id`), nên 100% thực thể valid/test chưa thấy trong train.
  - 4.999 / 5.000 ảnh: ảnh test `4392556686_44d71ff5a0_o.gif` (ảnh duy nhất dạng GIF; 23 object, 23 relationship) không có thực thể nào trong `entity2id`, nên test có 999 ảnh.
- [x] **P0.5 Spatial-VG** (`scripts/clean_splits.py --vg-h5`).
  - Từ **VG150-curated, biến thể `no_part_whole`** (Neau et al., ICCVW 2023, repo `Maelic/VG_curated`), **không phải VG150 chuẩn** (Xu et al., 2017). File H5 và dicts ở `data/Spatial-VG/`. Dùng **split chính thức theo ảnh của H5**; loại triplet trùng trong từng split.
- [x] **P0.6 Spatial-VRD-Trans** (`scripts/make_transductive_split.py --stratify relation`).
  - Chia lại Spatial-VRD thành transductive: 0 thực thể valid/test chưa thấy trong train. Phân tầng theo quan hệ để phân bố nhóm quan hệ khớp Spatial-VRD (GEOMETRIC train/test 70,3/70,5%, POSSESSIVE 25,1/24,9%; Spatial-VRD 70,4/70,3%); không phân tầng thì triplet POSSESSIVE (hay dính thực thể bậc 1) ít được tách ra và test lệch về GEOMETRIC.
- [x] **Nhóm quan hệ** (`relation_category.tsv` ở mỗi bộ MMSG): GEOMETRIC / POSSESSIVE / SEMANTIC / MISC, dùng cho P5. Đã đối chiếu với Neural Motifs (Table 1): paper chỉ cho ví dụ, số lớp và tỉ lệ của từng nhóm, không công bố danh sách đầy đủ. Bảng của mình khớp mọi ví dụ trong paper, nhưng phải trình bày là *phân loại của chúng tôi theo định nghĩa của Zellers et al.*, kèm bảng đầy đủ ở phụ lục. Chi tiết sai lệch ghi trong `scripts/build_vrd_relation_review.py`.
- [x] **P0.7 Đặc trưng** (`scripts/extract_features.py`): xong cả 6 bộ; Spatial-VG chạy trên Molab (`notebooks/extract-vg-features.py`, 18/11/2026).
  - RGB: **DINOv2 ViT-B/14** (`facebook/dinov2-base`, `pooler_output`, 768 chiều, đóng băng, không có lớp chiếu). Depth: ZipDepth (384 px) → bản đồ float, min-max theo từng ảnh → cắt theo box bằng `roi_align` → cùng DINOv2. Text: RoBERTa-base, trung bình token theo mask, `entity2textlong.txt` nếu có, tối đa 512 token (bị cắt: FB15k-237 24, MKG-Y 154, VRD 0). Bản đồ depth giữ dạng float từ đầu đến cuối và mọi bộ mã hoá đều là mạng pretrained, không có lớp khởi tạo ngẫu nhiên.
  - MMKG (chế độ `kg`): đọc `image_index.tsv`, chỉ ảnh `ok`; đặc trưng thực thể = trung bình các ảnh, RGB và depth tính riêng. Scene graph: một box mỗi thực thể; box nằm hẳn ngoài ảnh (lỗi chú thích, VRD 2 box) tính là thiếu ảnh.
  - Thực thể thiếu ảnh nhận **một vector hằng** = trung bình đặc trưng các thực thể train có ảnh (riêng RGB, depth; `ent_fill.pt`), kèm `ent_img_mask.pt`. Không lưu bảng thô riêng: GeometryMMKG thay đúng các dòng có mặt nạ 0 bằng vector học được. Lệnh `fill` điền lại theo train của bộ khác cùng `entity2id` (FB15K-237-ind từ FB15k-237-IMG, VRD-Trans từ VRD).
  - Bảng cho baseline (+D ở tầng dữ liệu): `ent_rgb_depth_mean_emb.pt` (768) và `ent_rgb_depth_concat_emb.pt` (1536); không xuất cho Spatial-VG (baseline không chạy VG; bảng concat khoảng 8,9 GB).
  - VG chạy trên Molab (RTX PRO 6000 96 GB, batch DINOv2 1024 / ZipDepth 64, 44 phút cho 105.814 ảnh); ZipDepth là bản sao trong repo (upstream commit `91f3fd21`, md5 checkpoint khớp); kết quả đẩy lên HF dưới dạng pull request, duyệt rồi merge. Ảnh VG lấy từ bản sao private `callmezuy/vg-images-raw` (chép nguyên 3 file zip từ máy, đúng kích thước gốc), vì Stanford chỉ cho tải khoảng 1 MB/s. 6 box (ảnh số 5763) nằm hẳn ngoài ảnh, tính là thiếu ảnh.
  - Hiệu năng RTX 3050 4 GB: batch DINOv2 64 (175 ảnh/s); RoBERTa với mô tả 512 token giữ khoảng 2 GB bộ nhớ đệm, phải `empty_cache` trước bước ảnh. Thời gian: VRD 20 phút, MKG-Y 22 phút, FB15k-237 43 phút.
  - Chất lượng (`scripts/feature_quality.py`, chỉ trên thực thể có ảnh; probe hồi quy logistic, 80/20):

    | Dataset | Thiếu ảnh (điền) | Cosine TB RGB / depth / text | Probe RGB / depth / text (đa số) | 10-NN depth ∩ RGB (ngẫu nhiên) |
    |---|---|---|---|---|
    | FB15k-237-IMG | 249 | 0,18 / 0,61 / 0,95 | 63,0 / 57,4 / 82,7% (31,7%; 30 kiểu top-50 của BLP) | 15,4% (0,07%) |
    | MKG-Y | 519 | 0,17 / 0,48 / 0,94 | không có nhãn | 9,1% (0,07%) |
    | Spatial-VRD | 2 | 0,07 / 0,38 / 0,96 | 40,3 / 11,7 / 98,8% (4,5%; 543 lớp vật thể) | 3,7% (0,01%) |
    | Spatial-VG | 6 | 0,07 / 0,35 / 0,99 | 40,1 / 13,6 / 100% (8,2%; 150 lớp vật thể) | 1,5% (0,005%) |

    Text VRD gần 100% vì mô tả chứa tên lớp; text VG **chính là** tên lớp (150 chuỗi khác nhau), nên probe text = 100% và cần nói rõ trong bài. **Lưu ý cho RQ1:** ở FB15k-237, depth của ảnh web đoán kiểu thực thể gần bằng RGB (57% so với 63%): depth toàn ảnh mã hoá cả "thể loại ảnh" (chân dung, bản đồ, logo), không chỉ hình học, nên Δdepth > 0 ở MMKG không tự động là bằng chứng về thông tin hình học; cần đối chứng (ví dụ depth xáo trộn giữa các thực thể cùng kiểu) khi phân tích.
- [x] **P0.8 Phát hành v0 trên HF** (`scripts/publish_hf.py`, thẻ ở `hf_cards/`).
  - Tag `v0` (public): `callmezuy/fb15k237-img`, `callmezuy/fb15k237-ind`, `callmezuy/mkg-y`, `callmezuy/spatial-vrd`, `callmezuy/spatial-vrd-trans`, `callmezuy/spatial-vg` (đặc trưng merge từ PR #1 sau khi kiểm tra kích thước, mặt nạ, vector điền, log; tag 18/11/2026).
  - Private: `callmezuy/vg-images-raw` (ảnh VG gốc, chỉ để notebook tải nhanh; không gắn tag).
  - Thống kê: `experiment_results/dataset_stats.md` (`scripts/dataset_stats.py`, kiểm tra chi tiết từng bộ); bảng kiểu paper T1–T7 (tổng quan, modality, cấu trúc đồ thị, mức inductive, scene graph, nhóm quan hệ, đối chiếu số công bố) ở `experiment_results/dataset_tables.{md,tex}` (`scripts/dataset_tables.py`, 19/11/2026).

### P1. Hạ tầng đánh giá

- [x] **P1.1** `core/models/evaluation.py`: xếp hạng cả head và tail, trả về dict (MRR/MR/Hits gộp hai chiều + từng chiều), `return_ranks`, `save_ranks()`. Kiểm tra bằng `tests/test_evaluation.py` (so với brute-force). **Sửa 19/11/2026:** điểm đáp án đúng (cộng trực tiếp) và điểm ứng viên (`torch.cdist`) lệch nhau ở chữ số cuối, nên đôi khi đáp án đúng bị đếm là "tốt hơn chính nó" (rank +1); thực thể bị lọc cũng có thể bị đếm lệch. Nay đáp án đúng và các thực thể bị lọc bị loại ngay trong phép so sánh. Ảnh hưởng (Spatial-VRD, lần chạy thử): khoảng 16% bộ ba test có rank +1, MRR bị báo thấp khoảng 0,012–0,015. Thêm test mô hình phụ thuộc quan hệ, test này bắt được lỗi. Baseline không bị: `Tester` / `ValidSelector` của NativE, SNAG lấy điểm đáp án đúng từ chính mảng điểm ứng viên.
- [x] **P1.2** `core/reproducibility.py` (`set_seed`, `seed_worker`); `scripts/train.py` dùng `SEED = 42` và lưu `ranks_geometry_<ds>_seed<SEED>_test.csv`.
- [x] **P1.3** `scripts/compute_metrics.py`: metric tổng, theo nhóm (`relation_category.tsv`), theo từng quan hệ; `--reference` cho ΔMRR với khoảng tin cậy bootstrap ghép cặp. (Chưa xuất LaTeX.)
- [x] **P1.4** NativE `Tester.run_link_prediction(rank_file=...)` + tham số `-rank_file`; in metric Python cạnh metric C++ để đối chiếu. Đã kiểm tra với lib giả theo đúng luật OpenKE; **cần xác nhận hai bộ số khớp nhau ở lần chạy thật đầu tiên**.
- [x] **P1.5** SNAG: giống NativE. MKGformer (submodule upstream): `scripts/patch_mkgformer.py`, ghi rank khi đặt `MKG_RANK_FILE`, rank theo luật đếm chặt (MKGformer gốc dùng argsort nên khi bằng điểm thì xếp tùy ý).
- [x] **P1.6** Bảng thực thể đã fuse được giữ trên GPU nếu chiếm < 50% VRAM trống (`keep_on_gpu=None`), ngược lại để trên CPU.
- [ ] **P1.7** Đánh giá cả head và tail trên VG (138K triplet × 1,44M thực thể × 2 chiều) rất nặng; đo thời gian trên Molab trước khi chạy P4.4.

### P2. Mô hình đề xuất

**Thiết kế (quyết định 18/11/2026).** Thực thể = hàm của đặc trưng đóng băng (text t, RGB x, depth z), không có embedding theo ID, nên cùng một mô hình chạy ở cả bốn ô 2×2. (1) Gộp RGB–depth thành vision v theo `fusion_mode` (P2.1). (2) Thực thể thiếu ảnh: v ← một vector học được `v_miss` (P2.6). (3) Cổng text–vision: u = W_v v, g = σ(W_tg[t; u]), h = LN(t + Dropout(g ⊙ u)) / √d (P2.3). LN không có gain/bias học được, nên mọi thực thể có chuẩn L2 bằng 1 như TransE gốc (quyết định 19/11/2026): với gain học được, mô hình tự co/giãn cả không gian, làm margin và nhiệt độ đổi ý nghĩa trong lúc train (đo trên Spatial-VRD: khoảng cách tới đáp án đúng 435 lúc khởi tạo, 88 sau 1 epoch; với chuẩn cố định: 35,9 → 32,3 sau 3 epoch). (4) Điểm TransE L1 với embedding quan hệ. Train: âm mẫu **hai phía trong batch** (mọi thực thể khác nhau của batch thay head và thay tail, như BLP); bỏ âm mẫu là đáp án đúng **theo train** (không dùng valid/test khi train); loss margin có trọng số self-adversarial (detach), trung bình hai phía. **Bỏ GCN:** ở chế độ inductive, cạnh của thực thể test chỉ nằm trong chính các bộ ba test (dùng để truyền tin là rò rỉ), còn chỉ dùng đồ thị train thì thực thể test không có hàng xóm, biểu diễn train và test khác bản chất. **Bỏ loss căn chỉnh** text/RGB/depth: nó được tính trên đặc trưng đóng băng nên không có gradient. Code: `core/fusion/geometry_guided_fusion.py`, `core/models/{end_to_end,kg_reasoning}.py`, `core/data_engineering/fast_dataset.py`.

- [x] **P2.1** Cờ ablation (mục 3.1): `use_depth`, `fusion_mode` ∈ {`gate`: G = σ(W_g[x; z]), v = G⊙x + (1−G)⊙z; `concat`: Linear(2d→d); `mean`; `fixed`: một scalar học được α, v = σ(α)x + (1−σ(α))z}, `missing_mode` ∈ {`learned_mask`, `constant`}. Biến thể có tên (`VARIANTS`): `full`, `no_depth` (v = x), `concat`, `mean`, `fixed`, `constant`.
  - `scripts/train.py --data --variant --epochs --seed` cùng các siêu tham số (`--lr --margin --adv-temp --batch-size --dropout --weight-decay --valid-every --valid-max`). Đánh giá valid mỗi `--valid-every` epoch và sau epoch cuối, giữ checkpoint có MRR valid cao nhất, test một lần. Spatial-VG: valid trên tập con 5.000 bộ ba cố định (seed 0, chung cho mọi biến thể và seed; mục 4.9). Ghi `ranks_geometry_<ds>_<variant>_seed<s>_test.csv`, `metrics_…json` (test, valid tốt nhất, epoch tốt nhất, lịch sử từng epoch, tham số, commit) và checkpoint.
  - Nhiệt độ self-adversarial mặc định 1 (như RotatE): chạy thử 3 epoch trên Spatial-VRD, số âm mẫu hiệu dụng (exp entropy của trọng số, ghi trong log mỗi epoch) giảm từ khoảng 1.300 xuống 250 trên khoảng 1.980 ứng viên; nhiệt độ 2 chỉ còn khoảng 9, gần với chỉ lấy âm mẫu khó nhất.
  - Khoảng cách L1 trong loss dùng backward tự viết, chia khúc (`L1CDist`): backward của `torch.cdist(p=1)` dựng tensor B×C×d (khoảng 6 GB với batch 1024).
  - Kiểm tra: `tests/test_negatives.py` (mặt nạ âm mẫu, thực thể lặp trong batch, chỉ lọc theo train, gradcheck của L1). Chạy 1 epoch đủ 6 biến thể trên Spatial-VRD bằng RTX 3050 4 GB: khoảng 1 phút/epoch train.
  - **Tuning** (quyết định (d), đổi 19/11/2026): **tune riêng từng biến thể trên cùng một lưới** (cùng số cấu hình cho mỗi biến thể và cho baseline, tiêu chí 5). Lý do: lần chạy thử cho thấy các biến thể cần cấu hình khác nhau (trên Spatial-VRD `full` học thuộc sau epoch 20, `no_depth` thì không), nên dùng chung cấu hình thắng của `no_depth` có thể làm Δdepth âm chỉ vì tham số. Lưới và ngân sách epoch chốt **sau** khi đóng băng thiết kế (P2.8).
- [ ] **P2.2** Gate có điều kiện theo quan hệ: `G = σ(W_g[x; z] + b_r)`, b_r là bias theo quan hệ (tương đương W[x; z; one-hot(r)]), khởi tạo 0. Cài dưới cờ `--rel-gate` (P2.8, thử đầu tiên). Biểu diễn thực thể khi đó phụ thuộc quan hệ (như phép chiếu theo quan hệ của TransR): train fuse ứng viên một lần cho mỗi quan hệ trong batch, đánh giá fuse toàn bộ thực thể một lần cho mỗi quan hệ.
- [x] **P2.3** Cross-attention với chuỗi dài 1 (softmax luôn bằng 1, tương đương `Linear(h_vision)` cộng residual) được thay bằng **cổng text–vision**: u = W_v v, g = σ(W_tg[t; u]), h = LN(t + Dropout(g ⊙ u)); text quyết định từng chiều nhận bao nhiêu thông tin thị giác, g log được như gate RGB–depth (P2.4).
- [ ] **P2.4** Log trọng số gate (phần dành cho Depth) theo từng triplet test để phân tích ở P5.
- [ ] **P2.5** *(Tùy chọn)* Modality BBox geometry (x, y, w, h chuẩn hóa, diện tích).
- [x] **P2.6** Thiếu ảnh: đọc `ent_img_mask.pt`; với `learned_mask`, vision v của thực thể có mặt nạ 0 được thay bằng **một** vector học được `v_miss`, sau bước gộp RGB–depth. RGB và depth thiếu cùng lúc (cùng nguồn ảnh) nên một vector ở tầng vision là đủ, và thực thể thiếu ảnh đi qua đúng cùng một phép tính ở mọi biến thể; đặt hai vector riêng trước gate thì bản có và không có depth sẽ cho đầu ra khác nhau. Biến thể `constant` bỏ qua mặt nạ, đọc bảng đã điền vector hằng như baseline. Kiểm tra `tests/test_missing_modality.py`: cùng tham số chung thì thực thể thiếu ảnh cho đầu ra giống hệt giữa `no_depth` và mọi `fusion_mode`; đầu ra không phụ thuộc giá trị RGB/depth của dòng thiếu; `constant` thì có; batch toàn thực thể thiếu ảnh không tạo gradient cho phần gộp RGB–depth. (Loss căn chỉnh đã bỏ, xem đầu mục P2.)
- [x] **P2.7** `notebooks/geometry.py` (Molab, theo mẫu `native.py`): chọn 1 trong 6 dataset, tải `v0` từ HF (chỉ file `*2id`, bảng đặc trưng, mặt nạ, `relation_category.tsv`), kiểm tra bảng đặc trưng; ô HF token dạng mật khẩu; chọn biến thể, seed, tham số; mỗi biến thể chạy `scripts/train.py`, xong là đẩy log, rank, metric, checkpoint lên `geometry-mmkg-results/geometry/<dataset>/<variant>/`; biến thể đã có metric của seed đó thì bỏ qua. Cell cuối chạy `compute_metrics.py` so với `no_depth` (Δdepth). `marimo check` sạch. **Chưa chạy trên Molab.**
- [ ] **P2.8** Hoàn thiện thiết kế trước khi tune (thứ tự: thiết kế → đóng băng → tune → chạy chính thức; quyết định 19/11/2026).
  - **Lần chạy thử** (seed 42, cấu hình mặc định, 100 epoch, code `0455091`, còn lỗi đánh giá ở P1.1):

    | Dataset | `full`: epoch tốt nhất / MRR valid | `no_depth`: epoch tốt nhất / MRR valid | Ghi chú |
    |---|---|---|---|
    | Spatial-VRD | 20 / 0,178 | 90 / 0,217 | `full` học thuộc: valid tụt còn 0,099 ở epoch 100 trong khi loss train vẫn giảm |
    | MKG-Y | 100 / 0,164 | 100 / 0,089 | chưa hội tụ (21 batch/epoch); valid còn tăng mạnh |
    | FB15k-237-IMG | 90 / 0,277 | 100 / 0,280 | gần hội tụ, không học thuộc |

    Các lần chạy này đã lỡ tính và xem số test (chế độ dev chưa có); từ nay giai đoạn phát triển chỉ nhìn valid.
    Phân tích checkpoint VRD: phần depth của gate gần như hằng (trung bình 0,34, sd giữa thực thể 0,03), tức gate chưa thích nghi
    theo thực thể; vật thể ở ảnh khác chiếm gần hết các ứng viên xếp trên đáp án đúng (khoảng 1.300–1.600 so với 4 vật thể cùng ảnh).
  - **Chế độ dev** (`train.py --dev`, ô "Chế độ dev" trong notebook): chỉ đánh giá valid, không bao giờ chạy test, không ghi rank;
    kết quả ở `geometry-dev/<dataset>/<cờ>/<biến thể>/`. Mỗi lần đánh giá valid ghi phần depth của gate (trung bình, độ lệch giữa
    thực thể, giữa quan hệ). Notebook có bảng tổng hợp valid của mọi lần chạy dev.
  - **Các cờ thử, theo thứ tự, cộng dồn** (mặc định tắt; cùng cờ cho mọi biến thể, biến thể không dùng được cờ thì bỏ cờ đó):
    1. `--rel-gate`: gate theo quan hệ (P2.2). Nhắm vào: gate hằng; giả thuyết "quan hệ không gian cần depth".
    2. `--input-norm`: LayerNorm riêng (có tham số) cho RGB và depth trước khi gộp. Nhắm vào: hai bảng DINOv2 khác thang.
    3. `--depth-dropout P` (P = 0,3): lúc train, mỗi thực thể bỏ depth với xác suất P (đi đường không depth). Nhắm vào: `full`
       học thuộc trên VRD.
    4. `--text-proj`: phép chiếu học được cho text (khởi tạo đơn vị) trước residual và cổng. Nhắm vào: text không được biến đổi.
    Các cờ 1 và 4 khởi tạo sao cho mô hình lúc đầu giống hệt bản không có cờ. Kiểm tra: `tests/test_model_options.py`
    (gồm loss nhóm theo quan hệ khớp với cách tính từng câu hỏi, thực thể thiếu ảnh vẫn cho cùng đầu ra với mọi cờ).
  - **Giao thức:** Spatial-VRD và FB15k-237-IMG, seed 42, cấu hình mặc định, 100 epoch, `full` và `no_depth` (thêm biến thể
    khác nếu cờ ảnh hưởng tới nó). **Luật giữ một cờ:** MRR valid của các biến thể chịu ảnh hưởng tăng trung bình trên hai bộ và
    không tụt quá 0,005 ở bộ nào. Ghi mọi lần thử vào đây, kể cả lần không cải thiện. Mục tiêu là mô hình tốt hơn nói chung,
    không phải làm Δdepth dương.
  - **Nhật ký các vòng** (MRR valid tốt nhất / epoch; seed 42, 100 epoch, cấu hình mặc định, Molab RTX PRO 6000):

    | Vòng | Cờ | VRD `full` | VRD `no_depth` | FB15k `full` | FB15k `no_depth` | Quyết định |
    |---|---|---|---|---|---|---|
    | 0 | không (`base`, code `1c81cfd`) | 0,1887 / 20 | 0,2315 / 100 | 0,2977 / 90 | 0,3013 / 100 | mốc so sánh |
    | 1 | `rel_gate` | 0,1853 / 20 | (= vòng 0) | 0,2963 / 70 | (= vòng 0) | **bỏ** (−0,0034 / −0,0014) |
    | 2 | `input_norm` | 0,1928 / 20 | 0,2384 / 100 | 0,2970 / 100 | 0,3011 / 100 | **giữ** (`full` +0,0041 / −0,0007; `no_depth` +0,0069 / −0,0002) |
    | 3 | + `depth_dropout 0.3` | 0,2059 / 30 | (= vòng 2) | 0,3028 / 100 | (= vòng 2) | **giữ** (+0,0131 / +0,0058) |
    | 4 | + `text_proj` | 0,2364 / 30 | 0,2512 / 70 | 0,3225 / 100 | 0,3253 / 100 | **giữ** (`full` +0,0305 / +0,0197; `no_depth` +0,0128 / +0,0242) |
    | 5 (P2.9) | + `depth_stats` DAV2 `grid9` | 0,2492 / 30 | (= vòng 4) | (không chạy) | (không áp dụng) | **giữ** (+0,0128, chỉ VRD) |
    | 5b (P2.9) | + `depth_stats` ZipDepth `grid9` | 0,2466 / 30 | (= vòng 4) | (không chạy) | (không áp dụng) | +0,0102; DAV2 − ZipDepth = +0,0026 (cỡ nhiễu) |

    Vòng 0 (19/11/2026): lặp lại lần chạy thử với code đã sửa lỗi đánh giá; MRR valid cao hơn lần chạy thử 0,011–0,021, epoch
    tốt nhất và hình dạng đường valid giống. VRD `full` vẫn học thuộc: loss train 1,11 ở epoch 100 (so với 2,37 của `no_depth`),
    valid tụt từ 0,189 (epoch 20) xuống 0,106. Phần depth của gate: VRD 0,34 ở epoch tốt nhất (sd giữa thực thể 0,03), tăng dần lên
    0,44; FB15k ở yên quanh 0,50 (giá trị lúc khởi tạo), sd 0,03, tức gate gần như không học. VRD `no_depth` và FB15k cả hai biến
    thể còn tăng nhẹ ở epoch 100.

    Vòng 1 (19/11/2026): `rel_gate` làm MRR valid của `full` giảm nhẹ ở cả hai bộ (trung bình −0,0024), nên bỏ theo luật giữ cờ.
    Bias theo quan hệ gần như không rời 0: độ lệch phần depth giữa các quan hệ chỉ 0,0006 (VRD) và 0,0014 (FB15k), nhỏ hơn nhiều so
    với độ lệch giữa thực thể (0,02–0,03); đường valid và loss train gần trùng vòng 0, chênh lệch cỡ nhiễu. Kết luận chỉ là: với
    cấu hình mặc định, gate theo quan hệ không học được gì thêm; chưa kiểm định được giả thuyết "quan hệ không gian cần depth".
    Chi phí: 10,2 s/epoch so với khoảng 4,7 trên VRD, 108 s/epoch so với khoảng 13 trên FB15k (182 phút cho 100 epoch).
    Vì bias theo quan hệ gần như không học, vòng 1 xấp xỉ một lần chạy lại vòng 0 với dòng ngẫu nhiên khác, nên chênh lệch
    0,001–0,003 của nó là ước lượng thô cho mức nhiễu giữa hai lần chạy cùng seed.

    Vòng 2 (19/11/2026): `input_norm` đạt luật giữ cờ ở cả hai biến thể (trung bình `full` +0,0017, `no_depth` +0,0034; không bộ
    nào tụt quá 0,005), nên **giữ** và bật ở các vòng sau. Mức tăng nằm ở VRD (+0,004 và +0,007); FB15k gần như không đổi. Mức tăng
    của `full` chỉ hơi lớn hơn mức nhiễu ước lượng ở vòng 1, nên chưa chắc chắn; các lần chạy 3 seed sau khi đóng băng sẽ cho biết.
    Trước khi chạy đã lường trước tác dụng nhỏ: `pooler_output` của DINOv2 đã đi qua LayerNorm, nên độ lớn từng vector RGB và depth
    vốn gần nhau. Trên VRD `full` vẫn học thuộc (đỉnh ở epoch 20, valid còn 0,115 ở epoch 100); khoảng cách với `no_depth` tăng lên
    −0,046. Phần depth của gate ở FB15k nhích từ 0,50 lên 0,53.
    Ghi chú dữ liệu (`feature_quality.json`): vector depth giống nhau hơn hẳn vector RGB (cosine trung bình giữa thực thể: VRD 0,38
    so với 0,07; FB15k 0,61 so với 0,18), tức các mảnh depth dồn về một hướng chung trong không gian DINOv2. LayerNorm theo từng vector
    không sửa được điều này.

    Vòng 3 (19/11/2026): `depth_dropout 0.3` (cộng `input_norm`) tăng MRR valid của `full` ở cả hai bộ (+0,0131 VRD, +0,0058 FB15k,
    trung bình +0,0095, lớn hơn rõ mức nhiễu) → **giữ**. Học thuộc trên VRD giảm nhưng chưa hết: đỉnh dời từ epoch 20 sang 30,
    loss train ở epoch 100 là 2,08 (vòng 2: 1,16), valid ở epoch 100 là 0,174 (vòng 2: 0,115). Phần depth của gate giảm (VRD 0,32 →
    0,25, FB15k 0,53 → 0,36) và khác nhau giữa thực thể nhiều hơn ở FB15k (sd 0,03 → 0,06). So với `no_depth` (vòng 2): VRD −0,0325,
    FB15k +0,0017 (cỡ nhiễu). Chỉ thử P = 0,3; P có đưa vào lưới tune hay không quyết định lúc chốt lưới (lưu ý tiêu chí 5: cùng số
    cấu hình cho mọi biến thể).

    Vòng 4 (19/11/2026): `text_proj` (cộng hai cờ đã giữ) tăng mạnh ở cả hai biến thể và cả hai bộ (`full` +0,0305 VRD, +0,0197
    FB15k; `no_depth` +0,0128, +0,0242) → **giữ**. Đây là mức tăng lớn nhất trong 4 vòng: text là xương sống của biểu diễn, và
    vector RoBERTa đi thẳng vào mà không được biến đổi là nút thắt chính. Thêm 590 nghìn tham số (VRD `full` 3,59 triệu). Trên VRD
    `full` vẫn học thuộc (đỉnh epoch 30, còn 0,2075 ở epoch 100); `no_depth` đạt đỉnh ở epoch 70 rồi đi ngang. FB15k cả hai biến thể
    còn tăng nhẹ ở epoch 100 (valid H@10 khoảng 0,49, MR khoảng 155). So với `no_depth`: VRD −0,0148, FB15k −0,0028 (cỡ nhiễu).
  - **Thiết kế chốt sau P2.8:** `input_norm` + `depth_dropout 0.3` + `text_proj`; bỏ `rel_gate`. So với vòng 0, MRR valid `full`
    tăng +0,048 (VRD) và +0,025 (FB15k), `no_depth` +0,020 và +0,024. Giới hạn của giao thức: thử tham lam theo thứ tự cố định, một
    seed, nên `rel_gate` chỉ được thử trên cấu hình cũ (tham số của nó gần như không học, nên không thử lại); chọn cờ trên valid làm
    valid hơi lạc quan, test chưa bị đụng tới.
- [ ] **P2.9** Có làm tiếp thiết kế sau P2.8 không (quyết định 19/11/2026: kiểm tra trước, tối đa 3 vòng cờ nữa với cùng luật,
  rồi đóng băng).
  - [x] **A. Probe depth** (`scripts/depth_probe.py`, kết quả `experiment_results/depth_probe.{md,_mlp.json,_linear.json}`):
    đặc trưng depth có chứa thông tin về quan hệ không gian mà RGB, text và khung 2D không có không? Cặp vật cùng ảnh trong
    Spatial-VRD, bộ phân loại nhỏ (MLP 512 ẩn và tuyến tính) trên đặc trưng cố định, train trên ảnh train (giữ 10% ảnh để dừng
    sớm), báo trên valid, không đọc test, 3 seed. Tác vụ: 52 quan hệ; "in front of" / "behind" (thứ tự sâu, nơi depth phải giúp);
    trên / dưới và trái / phải (đối chứng 2D). `box` = khung chuẩn hoá của hai vật và hình học cặp (lấy từ annotation).
    Kết quả (MLP, độ chính xác top-1; tuyến tính cho cùng kết luận):

    | Đặc trưng | 52 quan hệ | trước / sau | trên / dưới | trái / phải |
    |---|---|---|---|---|
    | đoán lớp đông nhất | 0,243 | 0,629 | 0,563 | 0,503 |
    | rgb | 0,487 | 0,748 | 0,847 | 0,509 |
    | depth | 0,452 | 0,723 | 0,833 | 0,504 |
    | rgb + depth | 0,500 | 0,757 | 0,875 | 0,572 |
    | text + rgb | 0,593 | 0,779 | 0,899 | 0,534 |
    | text + rgb + depth | 0,587 | 0,789 | 0,904 | 0,497 |
    | box | 0,565 | **0,804** | **0,956** | **0,986** |
    | text + rgb + box | **0,621** | 0,808 | 0,920 | 0,644 |

    (sd giữa seed 0,001–0,02.) Kết luận: (1) vector depth có tín hiệu (hơn hẳn đoán bừa) nhưng yếu hơn RGB, kể cả ở tác vụ
    trước / sau là nơi depth lẽ ra mạnh nhất; khi đã có text + RGB, thêm depth chỉ đổi −0,006 đến +0,010, tức thông tin gần như
    trùng lặp. Khớp với kết quả KG (Δdepth ≤ 0). (2) Khung 2D một mình đoán trước / sau (0,804) tốt hơn mọi tổ hợp đặc trưng ảnh,
    và là thứ duy nhất giải được trái / phải (mảnh cắt không biết vị trí của vật trong ảnh). Mô hình hiện tại không có thông tin
    này. Giới hạn: ghép 24 chiều của box với 1.536+ chiều đặc trưng ảnh làm bộ phân loại kém hơn box một mình (ví dụ trái / phải
    0,644 so với 0,986, chỉ 1.717 mẫu train), nên probe không trả lời sạch được câu "depth có thêm gì ngoài box"; và probe phân loại
    quan hệ cho một cặp đã biết, khác với bài toán xếp hạng của KG.
  - **Nguyên tắc (19/11/2026): không dùng tọa độ box làm đặc trưng của mô hình.** Box chỉ có ở 3/6 dataset (scene graph), các quan
    hệ như left of / above được gán nhãn từ chính vị trí hai khung (probe: box một mình giải trái / phải 0,986), và baseline KGC không
    nhận box. Box vẫn được dùng như hiện tại để xác định vùng của thực thể (cắt RGB / depth), tương đương "ảnh của thực thể" ở MMKG.
    Kết quả với box trong probe chỉ dùng để phân tích.
  - [x] **A2. Số độ sâu theo vùng** (`scripts/vrd_depth_stats.py` → `results/probe/vrd_depth_stats.pt`; probe
    `--depth-stats`, kết quả `experiment_results/depth_probe_a2_{mlp,linear,mlp_pca64}.json`). Giả thuyết: cắt mảnh depth rồi đưa qua
    DINOv2 làm mất thang so sánh chung giữa các vật cùng ảnh (DINOv2 gần như bỏ qua mức sáng tuyệt đối, LayerNorm ở đầu ra, mảnh nào
    cũng co về 224×224). `dstat` = 5 con số trên depth map đã chuẩn hoá theo ảnh, trong vùng của vật: trung bình, trung vị, phân vị
    10 / 90, độ lệch chuẩn (cùng ZipDepth, cùng vùng như lúc trích đặc trưng; 1,5 phút trên RTX 3050). Kiểm tra nhanh: với
    "in front of", vật đầu gần hơn (trung vị lớn hơn) ở 71% cặp, "behind" 29%, "near" và "wearing" 50%.
    Thay đổi so với `text + rgb` khi thêm từng loại depth (3 seed):

    | Probe | trước / sau: + vector depth | trước / sau: + `dstat` | 52 quan hệ: + vector depth | 52 quan hệ: + `dstat` |
    |---|---|---|---|---|
    | MLP | +0,010 | **+0,022** | −0,006 | +0,006 |
    | tuyến tính | +0,001 | +0,008 | +0,002 | +0,003 |
    | MLP, PCA 64 chiều | −0,005 | **+0,017** | −0,008 | +0,002 |

    Kết luận: giả thuyết được ủng hộ. 10 con số (5 cho mỗi vật) mang thông tin thứ tự sâu nhiều hơn vector 1.536 chiều: `dstat` một
    mình đoán trước / sau ngang vector depth (0,715 so với 0,723), và thêm vào text + RGB thì tăng đều ở cả ba kiểu probe, còn vector
    depth thì không. Thêm cả hai không hơn `dstat` một mình. Nhưng mức tăng nhỏ (trước / sau +0,02; 52 quan hệ +0,006), và "in front
    of" / "behind" chỉ chiếm khoảng 12% bộ ba của VRD, nên lợi ích trên KG nếu có cũng khiêm tốn. Ngoài khung 2D thì `dstat` không thêm
    gì (box + `dstat` so với box: −0,013 đến +0,004), tức phần lớn thông tin gần / xa trùng với vị trí trong ảnh. Ở FB15k / MKG-Y mỗi
    thực thể là một ảnh riêng, không có thang chung giữa các thực thể, nên `dstat` không có nghĩa "gần / xa hơn vật kia".
  - [x] **A3. Mô hình depth và cách tính số độ sâu** (`scripts/vrd_depth_stats.py --model {zipdepth,dav2}`, kết quả
    `experiment_results/depth_probe_a3_{mlp,linear}.json`). Tài liệu: Sharifzadeh et al. (bản ICLR 2020) cho thấy đưa depth map
    qua mạng pretrained trên RGB thì không tăng (predicate R@50 45,8–48,9 so với 90,5 khi train từ đầu trên depth), khớp chẩn đoán
    vector DINOv2 của depth yếu; ADVRD (Gan & He, KBS 2022) và DM-KG (arXiv 2607.12319) đều tách một giá trị độ sâu cho mỗi vật (trung
    bình theo vùng có trọng số / trung vị trong khoảng P10–P90) rồi lấy hiệu giữa hai vật; mức tăng do depth trong các bài này nhỏ
    (Sharifzadeh: VRD predicate R@50 90,34 → 90,47, zero-shot 70,06 → 72,28). So hai mô hình: ZipDepth (bản đang dùng, mô hình
    nhẹ) và Depth Anything V2 base (`depth-anything/Depth-Anything-V2-Base-hf`, 518 px; 5 phút cho 5.000 ảnh trên RTX 3050); ba cách
    tính: `stats5` (5 số như A2), `robust` (trung vị trong khoảng P10–P90, 1 số), `grid9` (trung bình lưới 3×3). Tỉ lệ vật đầu gần hơn
    (trung vị): "in front of" 0,706 (ZipDepth) / 0,784 (DAV2), "behind" 0,286 / 0,147, "above" 0,144 / 0,216, "under" 0,831 / 0,698;
    tương quan giữa hai mô hình chỉ 0,47. ZipDepth dựa nhiều vào vị trí dọc (thấp hơn = gần hơn), DAV2 tách thứ tự sâu rõ hơn.
    Probe MLP (3 seed; tuyến tính cùng xu hướng):

    | Số độ sâu | trước / sau, một mình | trước / sau, + text + rgb | trên / dưới, + text + rgb | 52 quan hệ, + text + rgb |
    |---|---|---|---|---|
    | (không, text + rgb) | | 0,779 | 0,899 | 0,593 |
    | ZipDepth `stats5` | 0,715 | 0,801 | 0,909 | 0,599 |
    | ZipDepth `grid9` | 0,729 | 0,799 | 0,910 | 0,598 |
    | DAV2 `stats5` | **0,816** | 0,810 | 0,903 | 0,594 |
    | DAV2 `robust` | 0,815 | 0,793 | 0,897 | 0,594 |
    | DAV2 `grid9` | 0,813 | **0,821** | 0,903 | 0,599 |

    Kết luận: mô hình depth quan trọng hơn cách tính. 10 số của DAV2 một mình đoán trước / sau 0,816, hơn text + RGB (0,779) và ngang
    khung 2D (0,804); thêm vào text + RGB tăng tới +0,042 (`grid9`; tuyến tính +0,042), so với +0,020 của ZipDepth. Trung vị đơn
    lẻ (`robust`) yếu nhất khi ghép. Ở 52 quan hệ mức tăng vẫn nhỏ (tối đa +0,006), vì trước / sau chỉ khoảng 12% bộ ba.
  - [ ] **Vòng 5 (quyết định 19/11/2026: thử trên KG trước, chưa đổi dữ liệu v0).** Cờ `--depth-stats PATH[:KEY]` (train.py,
    notebook ô "Số độ sâu"): h ← h + W_s s sau bước chuẩn hoá, s = `grid9` của DAV2 (9 số, chuẩn hoá bằng trung bình / độ lệch của
    thực thể train có ảnh), W_s tuyến tính không bias, khởi tạo 0 (lúc đầu mô hình giống hệt bản không cờ). Vì cộng sau chuẩn hoá,
    h + r − t chứa đúng W_s (s_h − s_t), tức hiệu độ sâu đầu − đuôi như ADVRD. Số hạng bằng 0 với thực thể thiếu ảnh và với thực thể
    bị depth_dropout bỏ depth; `no_depth` không dùng cờ này (thông tin depth). Log thêm độ lớn trung bình của W_s s (biểu diễn không
    có nó có chuẩn 1). Kiểm tra: `tests/test_model_options.py` (khởi tạo 0, bỏ qua thực thể thiếu ảnh, tuyến tính theo hiệu độ sâu,
    p = 1 bằng `no_depth`, loss và đánh giá truyền đúng s, chuẩn hoá chỉ dùng thực thể train). File `probe/vrd_depth_stats_dav2.pt`
    nằm trong repo kết quả private (chỉ có cho Spatial-VRD). Giao thức: chỉ Spatial-VRD (số độ sâu không có nghĩa so sánh ở MMKG),
    `full` với thiết kế chốt (+ `input_norm`, `depth_dropout 0.3`, `text_proj`), seed 42, 100 epoch; mốc là vòng 4 (0,2364). **Luật
    giữ:** MRR valid tăng > 0,005 (hơn mức nhiễu). Đạt thì mới tính trích lại depth bằng DAV2 cho cả 6 dataset (phiên bản dữ liệu
    mới, chạy lại các run baseline có depth).

    **Kết quả (19/11/2026, code `3a66fa6`):** MRR valid tốt nhất **0,2492** (epoch 30; H@1 0,167, H@10 0,409, MR 1162) so với 0,2364
    của vòng 4 → **+0,0128, đạt luật giữ.** Cao hơn vòng 4 ở mọi lần đánh giá (+0,010 đến +0,017 từ epoch 10 đến 100), nên không phải
    nhiễu của một điểm. |W_s s| trung bình 0,16–0,19 (biểu diễn không có nó có chuẩn 1), tức mô hình dùng số độ sâu thật. So với
    `no_depth` vòng 4 (0,2512): −0,002, cỡ nhiễu (vòng 4: −0,0148). Học thuộc vẫn còn (đỉnh epoch 30, 0,2166 ở epoch 100). Chưa tách
    được phần tăng do cách biểu diễn (số độ sâu thay vì vector DINOv2) và do mô hình depth (DAV2 thay vì ZipDepth): cần thêm lần
    chạy cùng cờ với `grid9` của ZipDepth.

    **Vòng 5b (19/11/2026, cùng cờ, `grid9` của ZipDepth):** MRR valid tốt nhất **0,2466** (epoch 30; H@1 0,165, H@10 0,405, MR
    1216), +0,0102 so với vòng 4. DAV2 hơn ZipDepth 0,0026 (0,002–0,004 ở mọi lần đánh giá, một seed, cỡ nhiễu). Vậy khoảng 80% mức
    tăng đến từ **cách biểu diễn** (số độ sâu giữ thang chung của ảnh, cộng tuyến tính để TransE thấy hiệu độ sâu), phần còn lại
    nhỏ từ mô hình depth. Probe cho DAV2 hơn hẳn ở trước / sau (0,816 so với 0,715) nhưng ZipDepth hơn ở trên / dưới (0,865 so với
    0,800), và trước / sau chỉ khoảng 12% bộ ba, nên trên KG hai mô hình gần nhau.
  - **Quyết định (19/11/2026, người dùng duyệt):**
    - **Thiết kế đóng băng:** `input_norm` + `depth_dropout 0.3` + `text_proj` + `depth_stats` (ZipDepth `grid9`, bảng của dataset,
      `--depth-stats data`); bỏ `rel_gate`. `depth_stats` **chỉ dùng cho 3 bộ scene graph** (Spatial-VRD, Spatial-VRD-Trans,
      Spatial-VG): ở MMKG mỗi thực thể là một ảnh riêng, không có cảnh chung để so gần / xa. Ghi rõ quy tắc này trong bài.
    - **Mô hình depth: ZipDepth** (giữ nhất quán với `ent_depth_emb.pt` của v0; DAV2 chỉ hơn 0,0026, cỡ nhiễu).
    - **Dữ liệu: cách (a)**: thêm `ent_depth_stats.pt` (n × 9) + `depth_stats.json` vào repo HF của 3 bộ scene graph rồi dời tag
      `v0`; không file nào khác đổi. Script `scripts/depth_stats.py` (cùng depth map, cùng box, cùng cắt box như
      `extract_features.py`; dòng thiếu ảnh = trung bình thực thể train; kiểm tra tập có số độ sâu đúng bằng `ent_img_mask.pt`).
      VRD-Trans dùng lại giá trị của VRD, chỉ điền lại dòng thiếu ảnh theo train của nó (`fill`). VG chạy trên máy
      (RTX 3050; notebook `notebooks/vg-depth-stats.py` để dự phòng trên Molab). Bảng VRD lệch bản probe trung bình 0,0004
      (batch fp16 khác).
    - **Baseline có nhận số độ sâu không: quyết sau** (trước khi chạy chính thức).

### P3. Baseline

**Quy tắc bắt buộc: baseline phải chạy đúng kiến trúc gốc. Không sửa code can thiệp vào kiến trúc.**
- *Kiến trúc* = mọi thứ quyết định cách mô hình tính điểm một bộ ba: các module và lớp (projection, pooling, fusion, attention, expert, generator), cách gộp modality, hàm score, cách khởi tạo embedding. Giữ nguyên code gốc của tác giả, kể cả các số chiều gắn cứng.
- *Được phép sửa* (ngoài kiến trúc, mỗi thay đổi ghi vào phụ lục kèm diff so với mã gốc): đường dẫn và nạp dữ liệu; giao thức đánh giá (ghi rank, filtered head + tail, luật bằng điểm chặt); chọn checkpoint theo valid (`ValidSelector`); seed; sửa lỗi tương thích phiên bản thư viện (ví dụ `weights_only=False`); cờ có sẵn trong code gốc (ví dụ `type_constrain=False`, `-use_pool 0`).
- *Đưa depth vào baseline:* chỉ ở **tầng dữ liệu**. Ghép đặc trưng RGB và depth (trung bình hoặc nối) rồi đưa vào ô ảnh có sẵn, khi mô hình gốc tự lấy số chiều từ dữ liệu; hoặc dùng một đường N-modality mà tác giả đã phát hành sẵn. Không thêm nhánh, lớp hay expert mới.
- Nếu một baseline không nhận được dữ liệu của mình mà không sửa kiến trúc (ví dụ gắn cứng số chiều đặc trưng), thì **không chạy baseline đó ở ô ấy** và giải thích trong bài. Không viết bản chỉnh sửa.
- Baseline transductive (embedding theo ID) ở ô inductive: **chạy nguyên bản**, 1 seed, trên FB15K-237-ind và Spatial-VRD; không chạy trên Spatial-VG (1,44M thực thể, tốn GPU mà không thêm thông tin). `entity2id` chứa mọi thực thể nên code gốc chạy được; số đo được là mức mô hình gốc tổng quát hoá sang thực thể mới, báo đúng như vậy. Δdepth ở hàng inductive dựa chủ yếu trên GeometryMMKG và Feature-TransE; CMR là mốc tham chiếu inductive nguyên bản.
- Phụ lục: `baselines/PROVENANCE.md` (nguồn mã gốc, bảng thay đổi theo loại được phép) và `baselines/patches/<baseline>.diff` (diff đầy đủ với mã gốc).

Bảng dưới là các tiêu chí phải đạt trước khi đưa số baseline vào bài:

| # | Tiêu chí | Trạng thái |
|---|---|---|
| 0 | **Kiến trúc baseline giữ nguyên code gốc**; mọi thay đổi ngoài kiến trúc có diff trong phụ lục | [x] NativE, SNAG (P3.0d); [ ] MoMoK, CMR khi thêm vào |
| 1 | Chạy lại được số của paper gốc trên dataset gốc bằng script gốc (sai lệch khoảng ±1 MRR), với `-select_on original`. Mục đích: chứng minh code đã sửa (rank_file, ValidSelector) vẫn giống code gốc khi tắt các phần thêm | [ ] NativE trên DB15K (đang chạy), SNAG trên MKG-Y |
| 2 | **Mọi model chọn checkpoint theo valid MRR, chỉ đụng tới test một lần** | [x] code (P3.0a) |
| 3 | Cùng giao thức đánh giá (filtered, head+tail, không ràng buộc type, luật bằng điểm chặt) | [x] P1 |
| 4 | Cùng feature (`ent_*_emb.pt`, 768 chiều) cho mọi model; ghi rõ là không dùng feature gốc của baseline | [x] dữ liệu; [ ] ghi trong bài |
| 5 | Tuning siêu tham số với **cùng ngân sách** trên valid cho mọi model, kể cả GeometryMMKG; xuất phát từ config gốc của tác giả | [ ] |
| 6 | 3 seed cho các ô chính, báo mean ± std; Δ kèm khoảng tin cậy bootstrap | [ ] |
| 7 | Đặt tên phân biệt: `NativE` (gốc) / `NativE+D(mean)`, `NativE+D(concat)` (depth đưa vào ô ảnh ở tầng dữ liệu); tương tự cho SNAG, MoMoK | [ ] |

- [x] **P3.0a Chọn checkpoint theo valid.** SNAG gốc (`Trainer.run`) cứ mỗi 100 epoch chạy Tester trên **test**, giữ weights có MRR test cao nhất và early-stop theo MRR test (rò rỉ test); NativE `WCGTrainerKuai16KGP` cũng in metric test trong lúc train.
  - `mmkgc/config/ValidSelector.py` (giống hệt nhau trong NativE và SNAG): đánh giá valid bằng Python (filtered, head+tail, luật bằng điểm chặt), giữ weights tốt nhất trong RAM (chỉ tham số học được), tùy chọn early-stop và tập con có seed.
  - Tham số: `-select_on {valid,original}` (mặc định `valid`; `original` = hành vi gốc), `-valid_steps`, `-patience` (0 = tắt, giữ nguyên ngân sách epoch), `-valid_max` (0 = toàn bộ).
  - Đã gắn vào `WCGTrainerGP`, `WCGTrainerDB15KGP`, `WCGTrainerKuai16KGP`, SNAG `Trainer` và các run script. Kiểm tra: `tests/test_valid_selector.py`. **Chưa chạy end-to-end** với lib C++ thật (chỉ có trên Linux/Molab).
  - Ở chế độ `valid`, SNAG đặt tên checkpoint `-MRR<x>` theo **MRR valid**; nếu file cùng tên đã tồn tại, `run_mmkgc.py` nạp file cũ (hành vi gốc), nên xoá `checkpoint/` giữa các lần chạy.
- [ ] **P3.0b Kiểm tra tái lập** (tiêu chí 1).
  - NativE/DB15K: `notebooks/native-db15k-sanity.py` tải embedding gốc từ Google Drive của tác giả (visual `[12842, 4096]`, textual/numeric `[12842, 768]`) và chạy đúng `scripts/run_db15k.sh` với `-select_on=original`, model `AdvRelRotatEDB15K` gốc. Số mục tiêu (arXiv 2406.17605, Table 2/3, NativE "All"): **MRR 37.16 · H@1 28.01 · H@3 42.25 · H@10 54.13**. Đang chạy trên Molab (18/11/2026), kết quả về `geometry-mmkg-results/native/DB15K-sanity/`.
  - [ ] SNAG trên **MKG-Y** (đổi từ MKG-W, 18/11/2026): split và id MKG-Y của tác giả SNAG giống hệt `data/MKG-Y` (và của NativE) theo nội dung, nên lần tái lập này vừa kiểm tra code SNAG vừa kiểm tra dữ liệu v0; sau này SNAG trên MKG-Y v0 chỉ khác đặc trưng (của tác giả so với của mình). Script gốc: `run_base.sh 0 MKG-Y 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 Y001`, thêm `-select_on=original -rank_file`.
    - Số mục tiêu (arXiv 2403.06832, Table 1 "MKGC performance on DB15K, MKG-W and MKG-Y", dòng "SnAg (Ours)", cột MKG-Y; đọc trực tiếp bản v1 11/03/2024 và v4 15/01/2025, hai bản giống nhau): **MRR .395 · H@1 .354 · H@3 .411 · H@10 .471**. Bản không có GMNM ("- w/o GMNM"): .387 / .345 / .407 / .457. Paper ghi "hidden dimension d = 256": khớp script (`EMB_DIM=128`, RotatE dùng `dim_e = 2·dim = 256`). Paper không nói cách chọn checkpoint cho MKGC; code gốc chọn theo MRR test (P3.0a), phải ghi rõ trong bài.
    - `notebooks/snag-mkgy-sanity.py`: tải zip của tác giả từ Drive, kiểm tra sha256 của zip và của `MKG-Y-{visual,textual}.pth` (`[15000, 383]`, `[15000, 384]`), `missing.json` (756 thiếu ảnh, 2.684 thiếu mô tả), chỉ giải nén ba file đó. Lệnh = `run_base.sh` (đã kiểm tra trên máy: mở rộng script bằng một `python` giả, ra đúng 21 tham số giống notebook, `CUDA_VISIBLE_DEVICES=0`), cộng `-select_on=original -rank_file`. Nút thứ hai (tuỳ chọn): `-select_on=valid -valid_steps=100 -patience=10` (luật gốc, chỉ thay test bằng valid) để đo mức lạc quan do chọn theo test. Kết quả về `geometry-mmkg-results/snag/MKG-Y-sanity/`. **Chưa chạy trên Molab.**
    - Quyết định (19/11/2026): coi là tái lập được khi **MRR lệch ≤ 1 điểm** so với paper (H@k chỉ báo kèm); chạy cả bản `select_on=valid` ngay sau bản gốc. Nếu lệch nhiều: báo nguyên nhân khả dĩ trước, không sửa kiến trúc.
- [x] **P3.0d Mã gốc cho NativE và SNAG** (18/11/2026).
  - NativE: `AdvRelRotatEDB15K.py` giống từng byte với `zjukg/NativE` commit `c72291c` (giữ pooling gắn cứng VGG 4096, chỉ dùng cho P3.0b). Dữ liệu của mình dùng `AdvRelRotatE` (`run_adv_wgan_gp.py`, script gốc `run_mkgy.sh`), vốn tự lấy số chiều từ dữ liệu. Không dùng đường 4-modality (`AdvRelRotatEKuai16K` bắt buộc đủ ảnh, text, audio, video).
  - SNAG: repo GitHub `zjukg/SNAG` không chứa code MKGC; mã gốc là file zip "Dataset and Code" trên Google Drive (`1jIYo7JWXNNi7LZqBIsl103FL8tHcoW-q`, sha256 trong PROVENANCE). `AdvMixRotatE.py` giống từng byte với bản đó. Cờ gốc `use_pool`: khi `use_pool=1` và tên dataset không chứa "MKG", nó pooling cứng theo VGG 4096 chiều, nên dùng `-use_pool 0` cho FB15k-237-IMG và VRD.
  - `notebooks/native.py`: một notebook cho cả 5 dataset (tải `v0` từ HF), ba kịch bản `control` / `dmean` / `dconcat`. Mỗi kịch bản là một `-dataset` riêng, symlink tới cùng benchmark và tới bảng đặc trưng tương ứng. Cell setup kiểm tra bảng đặc trưng (đúng số thực thể, hữu hạn, dòng thiếu ảnh đúng là vector hằng). Tham số mặc định = `run_mkgy.sh` (dim 250, margin 4, batch 1024, neg 128, lr 1e-4, 1000 epoch), đánh giá valid mỗi 50 epoch. Kịch bản đã có rank CSV trên repo kết quả thì bỏ qua. **Chưa chạy trên Molab.**
  - `scripts/prepare_snag_data.py`: dữ liệu cho SNAG dưới tên dataset riêng (ví dụ `MKG-Y-v0`), không dùng tên của tác giả; ghi `benchmarks/<tên>/` (kèm `type_constrain.txt` theo logic `n-n.py` của tác giả, thử trên split MKG-Y của tác giả ra đúng file của họ), embedding `<tên>-{visual,textual}.pth`, và gộp khoá vào `missing.json` (giữ khoá của tác giả). Code gốc đọc `missing.json` khi tên dataset chứa "MKG" và điền `mean + std·randn` vào các dòng có trong danh sách. Trên MKG-Y: **visual** = 519 thực thể không có ảnh (đều nằm trong 756 thực thể thiếu ảnh của tác giả); **textual** = 2.684 thực thể không có mô tả, **trùng khớp danh sách của tác giả** (`entity2id` giống hệt). Như vậy SNAG thay text-tên của 2.684 thực thể này bằng nhiễu, đúng như code gốc làm với dữ liệu của tác giả, trong khi các model khác vẫn thấy text-tên.
- [ ] **P3.1** Feature-TransE (2-modal và 3-modal).
- [ ] **P3.3** SNAG: notebook trên dữ liệu v0 (dùng `prepare_snag_data.py`), chạy nguyên bản trên mọi ô trừ Spatial-VG.
- [ ] **P3.4** MKGformer trên FB15k-237-IMG (dataset gốc của paper, so trực tiếp được với số công bố) và VRD-Trans. Notebook phải dùng ảnh và mô tả thật (`dataset/FB15k-images`, mô tả KG-BERT), không được để thư mục ảnh rỗng hay text giả.
- [ ] **P3.5** Kiểm tra file embedding gốc của tác giả NativE (DB15K) xem họ điền gì cho thực thể thiếu ảnh, ghi vào phụ lục; P3.0b vẫn dùng embedding gốc đó.
    - **Kết quả `original` (19/11/2026, Molab RTX PRO 6000, torch 2.11, commit `aea72de` + cài `unidecode`, seed 42, 1 lần chạy): MRR 39.02 · H@1 34.53 · H@3 41.08 · H@10 46.40** (MR 1293), lệch so với paper −0.48 · −0.87 · −0.02 · −0.70 điểm → **đạt ngưỡng (|ΔMRR| ≤ 1)**. Rank C++ và Python trùng nhau (2.663 bộ ba test, head + tail). MRR test đạt đỉnh ở epoch 899, dừng sớm ở epoch 1899 (10 lần đánh giá test liên tiếp không vượt), tổng 19,5 phút. Như vậy code (lib C++, patch ghi rank), split MKG-Y và embedding gốc cho ra đúng số của tác giả; checkpoint được chọn theo test như code gốc (P3.0a).
- [ ] **P3.6** MoMoK (mức 1): clone repo `zjukg/MoMoK` làm submodule; gắn `ValidSelector` (`-select_on valid`) và ghi rank theo P1; depth vào ở tầng dữ liệu (+D mean / +D concat vào ô ảnh), không thêm expert, không sửa kiến trúc; ở ô inductive chạy nguyên bản như NativE. Kiểm tra trước: code MoMoK có tự lấy số chiều đặc trưng từ dữ liệu không; nếu gắn cứng thì không chạy ở ô đó. Kiểm tra tái lập (tiêu chí 1) trên MKG-Y bằng đặc trưng gốc của tác giả.
- [ ] **P3.7** CMR (mức 1): clone repo `OreOZhao/CMR`; chạy nguyên bản trên FB15K-237-ind, đối chiếu số paper (MRR 0,371), đây là kiểm tra tái lập cho ô inductive. Thêm ghi rank theo P1. Chạy trên Spatial-VRD (text = tên lớp vật thể, ảnh = vùng cắt theo box). Code CMR đọc đúng một ảnh mỗi thực thể và không chạy được nếu thực thể thiếu ảnh (P0.2). Cần quyết định cách cấp ảnh cho 249 thực thể thiếu ảnh ở tầng dữ liệu, không sửa code: ảnh đen (giống cách MKGformer điền 0) hoặc dùng bộ ảnh BaiduNetDisk của tác giả (chỉ để kiểm tra tái lập). Ảnh dùng cho CMR: ảnh `ok` đầu tiên trong `image_index.tsv`, hoặc ảnh CMR chọn nếu lấy được `fb15k_best_img.pickle`.
- [ ] **P3.8** *(Mức 2, chỉ làm nếu còn GPU)* AdaMF-MAT, IKRL, TBKGC, BLP-TransE; MyGO nếu code tách token đã phát hành.

### P4. Chạy thí nghiệm (Molab)

Ưu tiên theo thứ tự:

- [ ] **P4.1** FB15k-237-IMG: GeometryMMKG + ablation; NativE; SNAG; MoMoK; MKGformer; Feature-TransE. MKG-Y: GeometryMMKG (full, w/o Depth); NativE; SNAG; MoMoK; Feature-TransE.
- [ ] **P4.1b** FB15K-237-ind: GeometryMMKG (full, w/o Depth); CMR nguyên bản; NativE/SNAG/MoMoK nguyên bản (1 seed); Feature-TransE.
- [ ] **P4.2** VRD-Trans: GeometryMMKG (full, w/o Depth); NativE 3 kịch bản (Control / +D mean / +D concat); SNAG; MoMoK; MKGformer; Feature-TransE.
- [ ] **P4.3** VRD inductive: GeometryMMKG + ablation; NativE/SNAG/MoMoK nguyên bản (1 seed); CMR nguyên bản; Feature-TransE.
- [ ] **P4.4** VG inductive: GeometryMMKG (full, w/o Depth); Feature-TransE. Baseline transductive không chạy. Theo dõi RAM/VRAM ở lần eval đầu tiên.
- [ ] **P4.5** Chạy 3 seed cho các cấu hình chính.
- [ ] **P4.6** Đẩy checkpoint và file rank lên HF sau mỗi lần chạy (storage của Molab có thể không bền), vào dataset repo private `callmezuy/geometry-mmkg-results` (`<model>/<dataset>/`). Molab không có ô secret: notebook nhận token qua ô nhập dạng mật khẩu. Đã có trong `native.py` và `native-db15k-sanity.py`.

### P5. Phân tích

- [ ] **P5.1** Phân loại quan hệ thành `SPATIAL_RELS` và `SEMANTIC_RELS` (từ `relation_category.tsv`). Ghi tiêu chí phân loại; xử lý các quan hệ lưng chừng (near, with, has); báo số triplet test mỗi nhóm.
- [ ] **P5.2** Bảng Δdepth trên ma trận 2×2 cho từng mô hình.
- [ ] **P5.3** Δdepth theo nhóm SPATIAL / SEMANTIC.
- [ ] **P5.4** Phân bố trọng số gate theo dataset và nhóm quan hệ.
- [ ] **P5.5** So sánh baseline nguyên bản trên VRD-Trans với chính nó (nguyên bản) trên VRD inductive: cùng mô hình, cùng code, chỉ khác chế độ split, để định lượng mức sụt do inductive.

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
| Depth gần như trung tính với MMKG; gating thích nghi không làm giảm hiệu năng | Kết luận từ chênh lệch nhỏ của 1 seed, không có khoảng tin cậy |
| Baseline transductive sụt trên MMSG chủ yếu do chế độ inductive (bằng chứng từ VRD-Trans) | Chunking/offload là đóng góp khoa học |
| Biểu diễn dựa trên đặc trưng + fusion Depth thích nghi phù hợp cho KG lai | Đánh giá "hoàn toàn công bằng" khi các mô hình chưa theo cùng giao thức mục 4 |

---

## 7. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Δdepth trên VRD-Trans ≈ 0 (Depth chỉ giúp khi inductive) | Vẫn là kết quả có giá trị: Depth đóng vai trò đặc trưng định danh cho thực thể mới. Điều chỉnh lại luận điểm RQ1 |
| Gating không tốt hơn trọng số cố định | Chuyển trọng tâm RQ2 sang gate theo quan hệ (P2.2) |
| BBox tốt bằng Depth | Báo cáo trung thực; nhấn mạnh tính bổ sung nếu có (Depth + BBox) |
| Không đủ GPU cho 3 seed | Ưu tiên 3 seed cho FB15k-237-IMG và VRD; FB15K-237-ind nếu còn GPU; MKG-Y và VG chạy 1 seed; bỏ hẳn baseline mức 2 |
| Ảnh Bing của FB15k-237 nhiễu, Δdepth ≈ 0 vì ảnh xấu chứ không vì loại đồ thị | Phân tích vật lý / trừu tượng (mục 4, điểm 10); MKG-Y làm bộ thứ hai; kiểm tra chất lượng đặc trưng bằng `scripts/feature_quality.py` |
| Không tái lập được số CMR trên FB15K-237-ind (MRR 0,371) | Báo số tự chạy cạnh số paper và ghi rõ sai lệch; Δdepth vẫn so trong cùng giao thức nên không bị ảnh hưởng |
| MoMoK chưa có sẵn trong máy, tích hợp tốn thời gian hơn dự kiến | Ưu tiên NativE và SNAG (đã có hạ tầng); MoMoK làm sau P4.1 lượt đầu |
| Nhóm SEMANTIC trong VRD quá nhỏ | Gộp thêm VG; báo khoảng tin cậy bootstrap |
