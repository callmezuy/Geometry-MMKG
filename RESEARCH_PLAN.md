# Kế Hoạch Nghiên Cứu & Danh Sách Công Việc

**Cập nhật:** 18/11/2026. **Dữ liệu:** v0 (tag `v0` trên HF, mục P0). Mọi thí nghiệm chạy trên dữ liệu v0 và theo giao thức ở mục 4.

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
| Full (RGB + Depth + Text, gating) | Mô hình đề xuất |
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
4. **GeometryMMKG:** vector "thiếu" học được + mặt nạ có/không, fusion bỏ qua modality thiếu (P2.6). Đây là **thiết kế của mô hình đề xuất**, không phải quy tắc chung. Kèm ablation dùng cùng vector hằng như baseline, không mặt nạ (mục 3.1), để phần chênh lệch với baseline không bị quy cho cách xử lý thiếu ảnh.

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
  - Thống kê: `experiment_results/dataset_stats.md` (`scripts/dataset_stats.py`).

### P1. Hạ tầng đánh giá

- [x] **P1.1** `core/models/evaluation.py`: xếp hạng cả head và tail, trả về dict (MRR/MR/Hits gộp hai chiều + từng chiều), `return_ranks`, `save_ranks()`. Kiểm tra bằng `tests/test_evaluation.py` (so với brute-force).
- [x] **P1.2** `core/reproducibility.py` (`set_seed`, `seed_worker`); `scripts/train.py` dùng `SEED = 42` và lưu `ranks_geometry_<ds>_seed<SEED>_test.csv`.
- [x] **P1.3** `scripts/compute_metrics.py`: metric tổng, theo nhóm (`relation_category.tsv`), theo từng quan hệ; `--reference` cho ΔMRR với khoảng tin cậy bootstrap ghép cặp. (Chưa xuất LaTeX.)
- [x] **P1.4** NativE `Tester.run_link_prediction(rank_file=...)` + tham số `-rank_file`; in metric Python cạnh metric C++ để đối chiếu. Đã kiểm tra với lib giả theo đúng luật OpenKE; **cần xác nhận hai bộ số khớp nhau ở lần chạy thật đầu tiên**.
- [x] **P1.5** SNAG: giống NativE. MKGformer (submodule upstream): `scripts/patch_mkgformer.py`, ghi rank khi đặt `MKG_RANK_FILE`, rank theo luật đếm chặt (MKGformer gốc dùng argsort nên khi bằng điểm thì xếp tùy ý).
- [x] **P1.6** Bảng thực thể đã fuse được giữ trên GPU nếu chiếm < 50% VRAM trống (`keep_on_gpu=None`), ngược lại để trên CPU.
- [ ] **P1.7** Đánh giá cả head và tail trên VG (138K triplet × 1,44M thực thể × 2 chiều) rất nặng; đo thời gian trên Molab trước khi chạy P4.4.

### P2. Mô hình đề xuất

Vấn đề đã biết của code hiện tại (`core/`): GeometryMMKG chỉ lấy mẫu âm phía tail (in-batch, hardest negative) trong khi đánh giá cả head; module GCN được khởi tạo nhưng không dùng.

- [ ] **P2.1** Thêm cờ cấu hình cho các ablation ở mục 3.1 (`use_depth`, `fusion_mode`).
- [ ] **P2.2** Gate có điều kiện theo quan hệ: `G = σ(W[h_rgb ; h_depth ; r])`, khớp giả thuyết "quan hệ không gian cần Depth". So sánh với gate hiện tại.
- [ ] **P2.3** Sửa hoặc mô tả đúng bước cross-attention: với độ dài chuỗi 1, softmax luôn bằng 1, nên text không điều khiển attention. Bước này hiện tương đương `Linear(h_vision)` cộng residual.
- [ ] **P2.4** Log trọng số gate (phần dành cho Depth) theo từng triplet test để phân tích ở P5.
- [ ] **P2.5** *(Tùy chọn)* Modality BBox geometry (x, y, w, h chuẩn hóa, diện tích).
- [ ] **P2.6** Thiếu modality cho GeometryMMKG: vector "thiếu" học được cho RGB và depth, đọc `ent_img_mask.pt`, gate/fusion bỏ qua modality thiếu theo mặt nạ (mục 3.2, ý 4). Có test: thực thể thiếu ảnh cho cùng đầu ra ở bản có và không có depth. Thêm cờ chạy ablation "vector hằng, không mặt nạ" (đọc bảng đã điền như baseline).
- [ ] **P2.7** Notebook Molab cho GeometryMMKG trên dữ liệu v0 (theo mẫu `notebooks/native.py`: tải `v0`, ô HF token, đẩy kết quả sau mỗi lần chạy).

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
| 1 | Chạy lại được số của paper gốc trên dataset gốc bằng script gốc (sai lệch khoảng ±1 MRR), với `-select_on original`. Mục đích: chứng minh code đã sửa (rank_file, ValidSelector) vẫn giống code gốc khi tắt các phần thêm | [ ] NativE trên DB15K (đang chạy), SNAG trên MKG-W |
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
  - [ ] SNAG trên MKG-W: chưa chuẩn bị.
- [x] **P3.0d Mã gốc cho NativE và SNAG** (18/11/2026).
  - NativE: `AdvRelRotatEDB15K.py` giống từng byte với `zjukg/NativE` commit `c72291c` (giữ pooling gắn cứng VGG 4096, chỉ dùng cho P3.0b). Dữ liệu của mình dùng `AdvRelRotatE` (`run_adv_wgan_gp.py`, script gốc `run_mkgy.sh`), vốn tự lấy số chiều từ dữ liệu. Không dùng đường 4-modality (`AdvRelRotatEKuai16K` bắt buộc đủ ảnh, text, audio, video).
  - SNAG: repo GitHub `zjukg/SNAG` không chứa code MKGC; mã gốc là file zip "Dataset and Code" trên Google Drive (`1jIYo7JWXNNi7LZqBIsl103FL8tHcoW-q`, sha256 trong PROVENANCE). `AdvMixRotatE.py` giống từng byte với bản đó. Cờ gốc `use_pool`: khi `use_pool=1` và tên dataset không chứa "MKG", nó pooling cứng theo VGG 4096 chiều, nên dùng `-use_pool 0` cho FB15k-237-IMG và VRD.
  - `notebooks/native.py`: một notebook cho cả 5 dataset (tải `v0` từ HF), ba kịch bản `control` / `dmean` / `dconcat`. Mỗi kịch bản là một `-dataset` riêng, symlink tới cùng benchmark và tới bảng đặc trưng tương ứng. Cell setup kiểm tra bảng đặc trưng (đúng số thực thể, hữu hạn, dòng thiếu ảnh đúng là vector hằng). Tham số mặc định = `run_mkgy.sh` (dim 250, margin 4, batch 1024, neg 128, lr 1e-4, 1000 epoch), đánh giá valid mỗi 50 epoch. Kịch bản đã có rank CSV trên repo kết quả thì bỏ qua. **Chưa chạy trên Molab.**
  - `scripts/prepare_snag_data.py`: dữ liệu cho SNAG dưới tên dataset riêng (ví dụ `MKG-Y-v0`), không dùng tên của tác giả; ghi `benchmarks/<tên>/` (kèm `type_constrain.txt` theo logic `n-n.py` của tác giả, thử trên split MKG-Y của tác giả ra đúng file của họ), embedding `<tên>-{visual,textual}.pth`, và gộp khoá vào `missing.json` (giữ khoá của tác giả). Code gốc đọc `missing.json` khi tên dataset chứa "MKG" và điền `mean + std·randn` vào các dòng có trong danh sách. Trên MKG-Y: **visual** = 519 thực thể không có ảnh (đều nằm trong 756 thực thể thiếu ảnh của tác giả); **textual** = 2.684 thực thể không có mô tả, **trùng khớp danh sách của tác giả** (`entity2id` giống hệt). Như vậy SNAG thay text-tên của 2.684 thực thể này bằng nhiễu, đúng như code gốc làm với dữ liệu của tác giả, trong khi các model khác vẫn thấy text-tên.
- [ ] **P3.1** Feature-TransE (2-modal và 3-modal).
- [ ] **P3.3** SNAG: notebook trên dữ liệu v0 (dùng `prepare_snag_data.py`), chạy nguyên bản trên mọi ô trừ Spatial-VG.
- [ ] **P3.4** MKGformer trên FB15k-237-IMG (dataset gốc của paper, so trực tiếp được với số công bố) và VRD-Trans. Notebook phải dùng ảnh và mô tả thật (`dataset/FB15k-images`, mô tả KG-BERT), không được để thư mục ảnh rỗng hay text giả.
- [ ] **P3.5** Kiểm tra file embedding gốc của tác giả NativE (DB15K) xem họ điền gì cho thực thể thiếu ảnh, ghi vào phụ lục; P3.0b vẫn dùng embedding gốc đó.
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
