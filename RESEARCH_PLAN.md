# Kế Hoạch Nghiên Cứu & Danh Sách Công Việc

**Cập nhật:** 25/11/2026. **Dữ liệu:** v0 (tag `v0` trên HF, mục P0). Mọi thí nghiệm chạy trên dữ liệu v0 và theo giao thức ở mục 4.

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
| MoMoK (ICLR 2025) | Nguyên bản | Nguyên bản | Như NativE | Chỉ +D mean, tầng dữ liệu (không thêm expert). +D concat không chạy được: số chiều gắn cứng theo tên dataset (P3.6) |
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
   - **Thước đo phụ trên MMSG (E1, P6.0)**, chỉ để báo, không dùng chọn checkpoint: xếp hạng trong cùng ảnh (ứng viên
     là các thực thể cùng ảnh với thực thể đã biết; cùng filter và quy tắc bằng điểm; MRR, H@1, H@3, số ứng viên) và
     đoán quan hệ (h, ?, t) trên mọi quan hệ (filtered; MRR, H@1, H@5). Baseline chỉ có file rank toàn cục nên chỉ có
     cận dưới của thước đo trong ảnh (hạng trong ảnh ≤ hạng toàn cục) và không có đoán quan hệ. Thực thể đã biết (h
     khi đoán đuôi, t khi đoán đầu) vẫn là ứng viên như ở thước đo chính; báo thêm tỉ lệ nó vượt đáp án và bản bỏ nó
     khỏi ứng viên.
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
| Spatial-VRD | 76.959 / 8.571 / 21.486 | 93.809 | 52 | `sg_dataset` của VRD + `relation_review.tsv` |
| Spatial-VRD-Trans | 77.083 / 8.550 / 21.383 | 93.809 | 52 | chia lại Spatial-VRD, phân tầng theo quan hệ |
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
  - **Loại triplet tự nối** (chủ thể và đối tượng là cùng một box), cùng quy tắc với bước tiền xử lý VG150 của Xu et al. (CVPR 2017, `data_tools/vg_to_roidb.py`: "sub and obj can't be the same box"), được VG150-curated giữ nguyên. Trong JSON gốc của VRD có 431 quan hệ như vậy (train 347, test 84), chữ đi kèm ghi hai vật khác nhau (ví dụ "man wearing jacket" với `objects: [7, 7]`), tức lỗi chỉ số; sau khi gán nhãn còn 391 (bỏ ngày 21/11/2026, sau bước chia valid theo ảnh nên ảnh của mỗi split không đổi). Lý do bỏ cả ở train: mô hình có quan hệ gần phép đồng nhất (RotatE) xếp hạng 1 các triplet này mà không cần học gì (NativE v0 trước khi sửa: toàn bộ H@1 đến từ 77 triplet tự nối của test). FB15k-237-IMG / FB15K-237-ind giữ nguyên các triplet tự nối của split chuẩn để so được với số công bố.
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

- [x] **P2.1** Cờ ablation (mục 3.1): `use_depth`, `fusion_mode` ∈ {`gate`: G = σ(W_g[x; z]), v = G⊙x + (1−G)⊙z; `concat`: Linear(2d→d); `mean`; `fixed`: một scalar học được α, v = σ(α)x + (1−σ(α))z}, `missing_mode` ∈ {`learned_mask`, `constant`}. Biến thể có tên (`VARIANTS`): `full`, `no_depth` (v = x), `concat`, `mean`, `fixed`, `constant`; thêm `constant_fixed` (22/11/2026), `depth_map_only` và `depth_stats_only` (24/11/2026, P2.10).
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
    Kết quả của chúng đã chuyển sang `geometry-pilot/` trong repo kết quả (19/11/2026), để `geometry/` chỉ chứa lần chạy chính thức.
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
    - **Mặc định trong code = thiết kế đóng băng** (19/11/2026): `train.py` bật sẵn `--input-norm`, `--depth-dropout 0.3`,
      `--text-proj`, `--depth-stats auto` (bảng của dataset trên 3 bộ scene graph, báo lỗi nếu bộ scene graph thiếu bảng; tắt
      trên MMKG). Ablation tắt bằng `--no-input-norm`, `--depth-dropout 0`, `--no-text-proj`, `--depth-stats none`.
      Notebook chọn sẵn các cờ này và luôn truyền tường minh bật / tắt. Kiểm tra: `tests/test_train_cli.py`.
- [ ] **P2.10** Tune GeometryMMKG (quyết định 19/11/2026, người dùng duyệt; thiết kế đã đóng băng ở P2.9).
  - **Ngân sách epoch** (tối đa; checkpoint chọn theo valid): FB15k-237-IMG 200 và FB15K-237-ind 200 (valid mỗi 10 epoch);
    MKG-Y 1000 (valid mỗi 50; 21 batch / epoch, lần chạy thử vẫn tăng mạnh ở epoch 100); Spatial-VRD và VRD-Trans 100 (mỗi 10;
    đỉnh ở epoch 30–70); Spatial-VG 100 (mỗi 10, tập con 5.000 bộ ba). **Luật hội tụ (đặt trước):** nếu epoch tốt nhất của cấu
    hình được chọn nằm trong 10% cuối ngân sách, chạy lại cấu hình đó với ngân sách gấp đôi.
  - **Lưới (6 cấu hình):** lr ∈ {1e-4, 3e-4, 1e-3} × dropout ∈ {0,1; 0,3}. Cố định: margin 5, nhiệt độ self-adversarial 1,
    batch 1024, weight decay 1e-5, `depth_dropout` 0,3. Baseline cũng 6 cấu hình, xuất phát từ config gốc (tiêu chí 5; lưới
    của từng baseline đề xuất riêng).
  - **Phạm vi: cách A**, tune riêng **mọi** biến thể: 6 biến thể (full, no_depth, concat, mean, fixed, constant) trên
    FB15k-237-IMG và Spatial-VRD; full và no_depth trên MKG-Y, FB15K-237-ind, VRD-Trans, Spatial-VG (khoảng 50 giờ GPU cho phần
    tune). Lý do: so sánh fusion (RQ2) công bằng, không chỉ Δdepth.
  - **Giao thức:** seed 42, chế độ dev (chỉ valid, không bao giờ test); mỗi biến thể chọn cấu hình có MRR valid cao nhất; sau
    đó chạy chính thức (3 seed cho các ô chính, mục 4.8), mỗi lần test đúng một lần.
  - **Code:** `train.py --tune --lr --dropout` (bật `--dev`; tên file và thư mục kèm cấu hình `lr<lr>-do<dropout>`; số epoch
    và khoảng valid mặc định theo `EPOCH_BUDGET` của dataset); `scripts/tune_summary.py` chọn cấu hình tốt nhất mỗi biến thể
    (bằng MRR thì lấy lr nhỏ hơn, rồi dropout nhỏ hơn) và áp luật hội tụ; notebook có mục "Tune" chạy cả lưới cho các biến thể
    đã chọn, bỏ qua điểm đã có trên HF (`geometry-tune/<dataset>/<biến thể>/<cấu hình>/`), và bảng tổng hợp. Kiểm tra:
    `tests/test_train_cli.py`; chạy thử 2 điểm lưới × 2 epoch trên VRD trên máy (lr 1e-3 đã cho MRR valid 0,232 sau 2 epoch so
    với 0,162 của lr 1e-4, nên lưới lr có ý nghĩa).
  - **Kết quả Spatial-VRD (20/11/2026, code `9ed7ce5`, 36/36 lần chạy, không lỗi). ĐÃ THAY THẾ:** tính trên dữ liệu trước khi
    bỏ các triplet tự nối (P0.4, 21/11/2026); file đã chuyển sang `superseded/geometry-tune/Spatial-VRD`, chạy lại theo quyết
    định 22/11/2026 bên dưới. Giữ bảng này chỉ để lưu lại. Cấu hình chọn và MRR valid tốt nhất:

    | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất |
    |---|---|---|---|
    | concat | 1e-3 / 0,1 | **0,2620** | 30 |
    | fixed | 1e-3 / 0,1 | **0,2620** | 30 |
    | mean | 1e-3 / 0,1 | 0,2614 | 30 |
    | no_depth | 3e-4 / 0,1 | 0,2527 | 30 |
    | full (gate) | 1e-4 / 0,1 | 0,2465 | 30 |
    | constant (gate, không mặt nạ) | 3e-4 / 0,1 | 0,2463 | 10 |

    Cả 6 biến thể đạt luật hội tụ. Lần chạy `full` lr 1e-4 / dropout 0,1 cho đúng số của vòng 5b (0,2465 so với 0,2466), tức
    quy trình lặp lại được. Nhận xét: (1) depth **giúp** khi gộp đơn giản: concat / fixed / mean hơn `no_depth` khoảng +0,009;
    (2) gate (thiết kế chính) **kém nhất** trong các cách gộp có depth và kém `no_depth` 0,006: valid giảm đều từ epoch 30 trong
    khi phần depth của gate tăng từ 0,16 lên 0,32, tức gate dùng depth để học thuộc; `fixed` thì ngược lại, trọng số depth giảm từ
    0,38 xuống 0,10 và ổn định; (3) dropout 0,3 luôn kém 0,1 khoảng 0,04 và cấu hình chọn của `full` nằm ở biên dưới của lr, tức
    tối ưu có thể nằm ngoài lưới (dropout < 0,1, lr < 1e-4).
  - **Quyết định (20/11/2026, người dùng duyệt, đặt trước khi có kết quả FB15k-237-IMG):**
    - **Chọn cách gộp cho `full`:** cách gộp có MRR valid trung bình cao nhất trên Spatial-VRD và FB15k-237-IMG (mỗi biến thể
      lấy cấu hình đã tune của nó), chọn trong {gate, concat, mean, fixed}, là `full` của 4 dataset còn lại (MKG-Y,
      FB15K-237-ind, VRD-Trans, Spatial-VG). Trong bài: cách gộp được chọn như một siêu tham số trên valid, cùng ngân sách tune;
      nếu không phải gate thì RQ2 trả lời là gate thích nghi không tốt hơn cách gộp tĩnh. Biến thể `constant` giữ cách gộp của
      `full` được chọn.
    - **Luật mở rộng biên (mọi mô hình, kể cả baseline):** thang lr {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, thang dropout
      {0; 0,1; 0,3; 0,5}. Nếu cấu hình chọn của một biến thể nằm ở biên của khoảng đã thử theo một trục (và chưa chạm biên tự
      nhiên, ví dụ dropout 0), thêm điểm một bậc ra ngoài theo trục đó, giữ trục kia ở giá trị đã chọn; lặp lại cho tới khi cấu
      hình chọn nằm bên trong, tối đa 2 bậc ra ngoài lưới gốc mỗi hướng. `tune_summary.py` tính các điểm cần thêm; notebook chạy
      chúng. Điểm thêm cũng chỉ nhìn valid.
  - **Kết quả FB15k-237-IMG (xong 22/11/2026; code `a77d949` / `6d6342b`; 49 lần chạy, tất cả trên GPU).** Gồm lưới gốc
    36 lần, 12 điểm mở rộng biên và 1 lần chạy lại 400 epoch:

    | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất |
    |---|---|---|---|
    | fixed | 1e-3 / 0,1 | **0,3302** | 110 |
    | no_depth | 1e-3 / 0 | **0,3302** | 170 |
    | mean | 1e-3 / 0,3 | 0,3275 | 150 |
    | concat | 1e-3 / 0,1 | 0,3264 | 130 |
    | full (gate) | 1e-4 / 0,1 | 0,3247 | 190 |
    | constant | 1e-4 / 0,1 | 0,3244 | 150 |

    - **Lưới đã bao đủ.** Mọi điểm mở rộng biên đều kém hơn hoặc ngang cấu hình đã chọn. Ngoại lệ duy nhất là `no_depth`, với
      dropout 0 hơn dropout 0,1 khoảng 0,0025; dropout 0 là biên tự nhiên nên không mở rộng thêm.
    - **Luật hội tụ cho `full`:** lần chạy 400 epoch đạt đỉnh ở đúng epoch 190 (0,3247), rồi dao động trong khoảng
      0,322–0,325 đến epoch 400. Như vậy mô hình đã hội tụ, và ngân sách 200 epoch là đủ.
    - **Cách các biến thể xếp hạng:** MRR các biến thể chỉ cách nhau khoảng 0,006. Gate kém hơn ba cách gộp tĩnh, và depth
      không giúp trên FB15k: `no_depth` bằng biến thể có depth tốt nhất.
    - **Cách đặt tên lần chạy lại:** lần chạy lại hội tụ mang hậu tố `-ep<epochs>` trong tên cấu hình (`train.py`
      `config_tag`) để không trùng lần chạy theo ngân sách. `tune_summary.py` đề xuất lần chạy lại đúng một lần, xét hội tụ trên
      lần chạy dài nhất của cấu hình đã chọn, và chọn cấu hình tốt nhất trên mọi lần chạy. Chế độ "Mở rộng biên" của notebook
      chạy cả hai luật.
  - **Kết quả Spatial-VRD trên `v0` mới (xong 22/11/2026; 47 lần, tất cả trên GPU).** Dữ liệu là
    76.959 / 8.571 / 21.486 bộ ba. Gồm lưới gốc 36 lần và 11 điểm mở rộng biên:

    | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất | Tốt nhất trong lưới gốc |
    |---|---|---|---|---|
    | concat | 1e-4 / 0 | **0,2670** | 60 | 0,2586 (1e-4 / 0,1) |
    | fixed | 1e-3 / 0 | 0,2650 | 20 | 0,2576 (1e-3 / 0,1) |
    | mean | 1e-3 / 0 | 0,2643 | 30 | 0,2583 (1e-3 / 0,1) |
    | constant | 1e-4 / 0 | 0,2566 | 30 | 0,2446 (1e-4 / 0,1) |
    | no_depth | 3e-4 / 0 | 0,2560 | 20 | 0,2497 (3e-4 / 0,1) |
    | full (gate) | 1e-4 / 0 | 0,2542 | 30 | 0,2446 (1e-4 / 0,1) |

    - **Dropout 0 thắng ở cả 6 biến thể,** hơn dropout 0,1 khoảng +0,006 đến +0,012. Dropout 0 là biên tự nhiên, nên lưới kết
      thúc ở đó. Các điểm lr một bậc ra ngoài (3e-5, 3e-3) đều kém hơn.
    - **Hạn chế:** ở dropout 0, mỗi biến thể chỉ được thử với đúng một lr (lr đã chọn ở dropout 0,1). Luật mở rộng biên đã chốt
      trước không đòi thử thêm, vì lr đã chọn nằm bên trong khoảng lr đã thử. Phải ghi điều này trong bài.
    - **Thứ hạng giữ nguyên như trên dữ liệu cũ:**
      - các cách gộp tĩnh hơn `no_depth` khoảng +0,008 đến +0,011;
      - gate kém `no_depth` 0,002.
    - **Mức MRR:** ở dropout 0,1, mức thấp hơn dữ liệu cũ khoảng 0,003–0,004, đúng bằng phần MRR mà các triplet tự nối từng
      đóng góp.
    - **`constant` gần như trùng `full`** vì Spatial-VRD chỉ có 2 thực thể thiếu ảnh.
  - **Kết quả luật chọn cách gộp (22/11/2026; luật đặt ngày 20/11):** MRR valid trung bình trên Spatial-VRD và FB15k-237-IMG,
    mỗi biến thể lấy cấu hình đã tune của nó:

    | Cách gộp | Spatial-VRD | FB15k-237-IMG | Trung bình |
    |---|---|---|---|
    | **fixed** | 0,2650 | 0,3302 | **0,2976** |
    | concat | 0,2670 | 0,3264 | 0,2967 |
    | mean | 0,2643 | 0,3275 | 0,2959 |
    | gate | 0,2542 | 0,3247 | 0,2895 |

    - **Kết quả:** `full` của 4 dataset còn lại (MKG-Y, FB15K-237-ind, Spatial-VRD-Trans, Spatial-VG) dùng cách gộp **fixed**,
      tức một trọng số vô hướng học được giữa RGB và depth. Notebook tune `fixed`, `no_depth` và `constant_fixed` ở các bộ này (Spatial-VG không có
      `constant_fixed`, xem bên dưới).
    - **Cách ghi trong bài:**
      - ba cách gộp tĩnh tương đương nhau, vì cách biệt 0,001 nhỏ hơn mức nhiễu;
      - gate thích nghi kém rõ, khoảng −0,007 đến −0,008 (trả lời RQ2);
      - lợi ích của depth chỉ thấy trên scene graph: trên VRD, `fixed` hơn `no_depth` +0,009, còn trên FB15k hai biến thể bằng
        nhau.
    - **Biến thể `constant_fixed` (quyết định 22/11/2026, người dùng duyệt):** theo luật ngày 20/11, biến thể vector hằng
      giữ cách gộp của `full` được chọn. Vì vậy thêm biến thể mới `constant_fixed`: cách gộp `fixed`, vector hằng cho thực
      thể thiếu ảnh, không mặt nạ. Nó khác `fixed` đúng ở cách xử lý thiếu ảnh, nên chênh lệch `fixed` − `constant_fixed` là
      phần lợi của mặt nạ + vector học được.
      - Giữ nguyên `constant` (gate) cùng kết quả đã có, để không lẫn tên.
      - Tune như mọi biến thể khác (lưới gốc + mở rộng biên + luật hội tụ) trên FB15k-237-IMG, Spatial-VRD, MKG-Y,
        FB15K-237-ind và Spatial-VRD-Trans.
      - **Không chạy ở Spatial-VG:** chỉ 6 box thiếu ảnh trên 1,44 triệu thực thể, nên biến thể này gần như trùng `fixed`
        (giống `constant` trùng `full` trên VRD với 2 thực thể thiếu ảnh), trong khi mỗi lần chạy VG rất đắt. Ghi lý do
        này trong bài.
  - **Kết quả tune đến 23/11/2026** (code `c924941` / `69ab9d0`; thời gian mỗi lần đều đặn ở mức GPU: VRD-Trans 8 phút
    cho 100 epoch, MKG-Y 19 phút cho 1000 epoch). FB15K-237-ind và Spatial-VG chưa chạy.
    - **Spatial-VRD, `constant_fixed` (8 lần, xong):** 1e-3 / 0, MRR valid 0,2649, epoch tốt nhất 20. Bằng `fixed` (0,2650),
      đúng như dự kiến vì chỉ có 2 thực thể thiếu ảnh.
    - **Spatial-VRD-Trans (28 lần, xong: lưới gốc, mở rộng biên, luật hội tụ):**

      | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất / ngân sách | Hội tụ |
      |---|---|---|---|---|
      | fixed | 3e-4 / 0 | **0,3611** | 200 / 200 | Chưa, sau khi đã gấp đôi ngân sách |
      | constant_fixed | 1e-3 / 0 | 0,3565 | 90 / 100 | Có |
      | no_depth | 1e-3 / 0 | 0,3503 | 100 / 200 | Có (lần chạy 200 epoch vẫn đạt đỉnh ở epoch 100) |

      - **Dropout 0 thắng ở cả 3 biến thể,** như trên Spatial-VRD.
      - **`fixed` chưa hội tụ sau khi đã gấp đôi.** Luật hội tụ chỉ chạy lại một lần (quyết định 23/11/2026, người dùng
        duyệt: giữ luật, không chạy thêm). Ghi trong bài là hạn chế. Chiều của nó: train thêm chỉ có thể làm `fixed` tăng,
        nên Δdepth đang bị tính thấp đi, không bị thổi phồng.
      - **Δdepth:** `fixed` hơn `no_depth` +0,011. So cùng 100 epoch thì `fixed` 3e-4 / 0 đạt 0,3565, còn khoảng +0,006.
      - **Ngân sách 100 epoch hơi ngắn cho bộ này:** nhiều cấu hình đạt đỉnh đúng ở epoch cuối.
      - **MRR cao hơn hẳn Spatial-VRD** (0,36 so với 0,265) dù mô hình không có embedding theo ID. Giả thuyết: mỗi box có
        một vector đặc trưng riêng, nên khi thực thể đã xuất hiện trong train, mô hình có thể nhớ luôn vector đó, và vector
        này gần như đóng vai một ID. Cần tính đến điều này khi so VRD-Trans với VRD (P5.5).
    - **MKG-Y (25 lần, xong 23/11: lưới gốc, mở rộng biên, luật hội tụ).** 519 / 15.000 thực thể thiếu ảnh:

      | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất / ngân sách | Hội tụ |
      |---|---|---|---|---|
      | fixed | 3e-4 / 0,3 | 0,3785 | 1100 / 2000 | Có (bản 1000 epoch: 0,3784 ở epoch 950) |
      | constant_fixed | 3e-3 / 0,3 | 0,3777 | 800 / 1000 | Có (điểm 1e-2 / 0,3 kém hơn: 0,3696) |
      | no_depth | 1e-3 / 0,3 | 0,3770 | 400 / 1000 | Có |

      - **Ba biến thể cách nhau khoảng 0,0015, tức cỡ nhiễu.** Như vậy depth không giúp, giống FB15k-237-IMG. Mặt nạ và
        vector học được cũng không hơn vector hằng.
      - **Dropout 0,3 thắng ở cả 3 biến thể,** ngược với hai bộ VRD. Các điểm dropout 0,5 đều kém hơn.
    - **FB15k-237-IMG, `constant_fixed` (8 lần, xong 23/11):** 1e-3 / 0,1, MRR valid 0,3300, epoch tốt nhất 70. Bằng
      `fixed` (0,3302) và `no_depth` (0,3302).
    - **FB15K-237-ind (18 lần, xong lưới gốc 24/11; còn mở rộng biên).** 249 / 14.541 thực thể thiếu ảnh; 33 phút mỗi
      lần:

      | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất / 200 | Còn phải chạy |
      |---|---|---|---|---|
      | no_depth | 1e-3 / 0,3 | 0,2557 | 30 | 3e-3 / 0,3; 1e-3 / 0,5 |
      | fixed | 3e-4 / 0,3 | 0,2551 | 50 | 3e-4 / 0,5 |
      | constant_fixed | 1e-3 / 0,3 | 0,2551 | 30 | 3e-3 / 0,3; 1e-3 / 0,5 |

      - **Ba biến thể cách nhau 0,0006, tức cỡ nhiễu:** bộ MMKG thứ ba mà depth không giúp.
      - Epoch tốt nhất rất sớm (20–130 / 200) và dropout 0,3 thắng ở cả 3 biến thể, tức mô hình học thuộc nhanh khi thực
        thể valid chưa từng thấy.
      - Mức tuyệt đối thấp hơn nhiều so với số công bố của CMR (MRR test 0,371).
    - **Spatial-VG (12 lần, xong lưới gốc 24/11; còn mở rộng biên).** Valid là tập con cố định 5.000 bộ ba (mục 4.9);
      35 phút mỗi lần với 1,44 triệu thực thể:

      | Biến thể | lr / dropout | MRR valid | Epoch tốt nhất / 100 | Còn phải chạy |
      |---|---|---|---|---|
      | fixed | 3e-4 / 0,1 | **0,3221** | 70 | 3e-4 / 0 |
      | no_depth | 3e-4 / 0,1 | 0,3105 | 90 | 3e-4 / 0 |

      - **Δdepth = +0,012,** cùng cỡ với hai bộ VRD. `fixed` hơn `no_depth` ở cả 6 cấu hình (+0,010 đến +0,015), nên
        chênh lệch ổn định dù tập valid nhỏ.
      - Dropout 0,1 hơn 0,3 khoảng 0,04 ở mọi lr; luật biên thêm điểm dropout 0 (nơi hai bộ VRD đạt tốt nhất).
    - **Tổng hợp đến 25/11 (MRR valid của cấu hình chọn; FB15K-237-ind và Spatial-VG chưa mở rộng biên):**

      | Dataset | fixed | no_depth | constant_fixed | Δdepth (fixed − no_depth) |
      |---|---|---|---|---|
      | FB15k-237-IMG | 0,3302 | 0,3302 | 0,3300 | 0,000 |
      | MKG-Y | 0,3785 | 0,3770 | 0,3777 | +0,0015 |
      | FB15K-237-ind | 0,2551 | 0,2557 | 0,2551 | −0,0006 |
      | Spatial-VRD | 0,2650 | 0,2560 | 0,2649 | +0,009 |
      | Spatial-VRD-Trans | 0,3611 | 0,3503 | 0,3565 | +0,011 (+0,006 nếu cùng 100 epoch) |
      | Spatial-VG | 0,3221 | 0,3105 | — (không chạy) | +0,012 |

      - **Δdepth chỉ dương trên scene graph,** nhất quán trên cả 6 bộ: 3 bộ MMKG không vượt cỡ nhiễu (−0,0006 đến
        +0,0015), 3 bộ MMSG từ +0,009 đến +0,012. Bảng 2 × 2 bên dưới tách phần lợi này giữa vector depth và số độ sâu
        theo vùng.
      - **`constant_fixed` bằng `fixed` ở mọi bộ:** cơ chế mặt nạ + vector học được cho thực thể thiếu ảnh (P2.6) không
        cho lợi ích đo được, có thể vì ảnh thiếu ít (FB15k 1,7%, MKG-Y 3,5%). Không trình bày nó như một đóng góp. Mặt tốt:
        chênh lệch với baseline không thể quy cho cách xử lý thiếu ảnh.
      - **Dropout tốt nhất khác theo loại dữ liệu:** 0 trên hai bộ VRD (Spatial-VG: 0,1 trong lưới gốc, điểm 0 chưa chạy),
        0,1 trên FB15k-237-IMG, 0,3 trên MKG-Y và FB15K-237-ind. Đây là lý do phải tune riêng từng dataset.
  - **Bảng 2 × 2 của depth trên scene graph (quyết định 24/11/2026, người dùng duyệt).**
    - **Vấn đề:** trên scene graph, "có depth" gồm hai phần: vector depth (DINOv2 của ảnh depth) và các số độ sâu theo
      vùng (`depth_stats`, P2.9). Trên MMKG chỉ có vector depth. Probe P2.9 cho thấy vector depth gần như không thêm gì,
      nên chênh lệch MMKG / MMSG có thể chỉ phản ánh việc có `depth_stats` hay không, và cần biết nhánh vector depth (cùng
      câu hỏi RQ2 về cách gộp RGB với depth) có còn cần khi đã có `depth_stats` hay không.
    - **Thiết kế:** hai biến thể mới, cùng thiết kế đóng băng, tune đủ (lưới gốc + mở rộng biên + luật hội tụ) trên
      Spatial-VRD và Spatial-VRD-Trans:

      | | Không `depth_stats` | Có `depth_stats` |
      |---|---|---|
      | **Không vector depth** | `no_depth` | `depth_stats_only` |
      | **Có vector depth** | `depth_map_only` | `fixed` |

      - `depth_map_only` = `fixed` không có `depth_stats`. `depth_stats_only` = `no_depth` cộng số hạng W_s s. Mọi
        biến thể có depth khác vẫn dùng `depth_stats` trên scene graph như trước (`VARIANT_DEPTH_STATS` trong
        `core/models/end_to_end.py`).
      - `depth_dropout` 0,3 áp dụng cho mọi đầu vào depth của một thực thể: vector và số độ sâu bị bỏ cùng lúc ở
        `fixed`, và bỏ riêng số độ sâu ở `depth_stats_only`. Số lần gọi số ngẫu nhiên không đổi với các biến thể cũ: đã
        kiểm tra trên Spatial-VRD, `fixed` 2 epoch cho cùng loss và MRR với code `69ab9d0`.
      - **Không làm trên MMKG:** số hạng W_s (s_h − s_t) là hiệu độ sâu giữa đầu và đuôi, chỉ có nghĩa khi hai thực thể
        nằm trong cùng một ảnh (cùng không gian nhìn, định nghĩa MMSG). Hai thực thể KG có ảnh khác nhau, nên hiệu này
        không có nghĩa. Bài phải ghi rõ lý do này. **Không tune trên Spatial-VG** (quá đắt).
      - **Đọc kết quả:** `depth_map_only` ≈ `no_depth` nghĩa là vector depth không giúp ở đâu cả. Nếu `depth_stats_only`
        ≈ `fixed` thì nhánh vector depth là thừa, và RQ2 phải viết lại theo hướng đó.
    - **Kết quả Spatial-VRD (xong 25/11/2026; mỗi biến thể mới 8 lần, lưới gốc + mở rộng biên; cả hai hội tụ, không cần
      thêm điểm):** MRR valid của cấu hình chọn.

      | | Không `depth_stats` | Có `depth_stats` | Δ do `depth_stats` |
      |---|---|---|---|
      | **Không vector depth** | `no_depth` 0,2560 (3e-4 / 0, ep 20) | `depth_stats_only` 0,2670 (1e-4 / 0, ep 50) | +0,011 |
      | **Có vector depth** | `depth_map_only` 0,2566 (1e-3 / 0, ep 20) | `fixed` 0,2650 (1e-3 / 0, ep 20) | +0,008 |
      | **Δ do vector depth** | +0,0006 | −0,002 | |

      - **Toàn bộ Δdepth +0,009 của Spatial-VRD đến từ `depth_stats`:** thêm số độ sâu tăng khoảng +0,01 dù có hay không
        có vector depth; thêm vector depth không đổi gì (+0,0006 khi không có số độ sâu, −0,002 khi có). Hai yếu tố gần
        như không tương tác. `depth_map_only` ≈ `no_depth` khớp với probe P2.9.
      - Không phụ thuộc riêng cấu hình được chọn: ở cả 6 cấu hình chung của lưới (lr 1e-4 / 3e-4 / 1e-3 × dropout
        0,1 / 0,3), `depth_map_only` không cao hơn `no_depth` (−0,0007 đến −0,015), còn `depth_stats_only` không thấp
        hơn `fixed` (0 đến +0,012).
      - **Giới hạn:** 1 seed, MRR valid. Chênh lệch dưới khoảng 0,003 (như `depth_stats_only` − `fixed`) chưa phân biệt
        được với nhiễu.
    - **Kết quả Spatial-VRD-Trans (xong 25/11/2026; `depth_map_only` 8 lần, `depth_stats_only` 9 lần; cả hai không cần
      thêm điểm biên):** MRR valid của cấu hình chọn.

      | | Không `depth_stats` | Có `depth_stats` | Δ do `depth_stats` |
      |---|---|---|---|
      | **Không vector depth** | `no_depth` 0,3503 (1e-3 / 0, ep 100; nhân đôi: vẫn ep 100) | `depth_stats_only` 0,3636 (3e-4 / 0, ep 200 / 200) | +0,013 |
      | **Có vector depth** | `depth_map_only` 0,3464 (1e-3 / 0, ep 90) | `fixed` 0,3611 (3e-4 / 0, ep 200 / 200) | +0,015 |
      | **Δ do vector depth** | −0,004 | −0,0025 | |

      - **Lặp lại đúng kết quả Spatial-VRD:** `depth_stats` cho khoảng +0,013 đến +0,015, vector depth không giúp (−0,004
        và −0,0025).
      - `depth_stats_only` và `fixed` vẫn có epoch tốt nhất ở cuối sau khi nhân đôi ngân sách (luật hội tụ chỉ chạy lại
        một lần, báo như giới hạn), nên số của hai ô này là cận dưới. So cùng 100 epoch ở 3e-4 / 0: `depth_stats_only`
        0,3590, `fixed` 0,3565, so với `no_depth` 0,3503 và `depth_map_only` 0,3464. Kết luận không đổi.
      - Ở cả 6 cấu hình chung của lưới (bản 100 epoch), `depth_map_only` thấp hơn `no_depth` (−0,003 đến −0,017), còn
        `depth_stats_only` cao hơn `fixed` (+0,004 đến +0,021).
    - **Tổng hợp 2 bộ (12 cặp cấu hình chung):** thêm vector depth không lần nào làm tăng MRR (0 trên 12, chênh từ
      −0,0007 đến −0,017). Thêm `depth_stats` vào mô hình đã có vector depth không lần nào làm giảm (0 đến +0,021). Như
      vậy Δdepth dương trên scene graph là của số độ sâu theo vùng (hiệu độ sâu giữa đầu và đuôi trong cùng ảnh), còn nhánh
      vector depth (DINOv2 trên ảnh depth) không có ích, thậm chí hơi có hại. Giới hạn: 1 seed, MRR valid, không làm trên
      Spatial-VG.
    - **Mô hình chính:** đã chọn ở P6.0 E2 (26/11/2026): `depth_stats_only` trên MMSG. Cách viết RQ2 và đóng góp về
      depth theo C3 / C4 của P6.0.
  - **Chạy tiếp khi bị ngắt (20/11/2026):** sau mỗi lần valid, `train.py` ghi `resume_<stem>.pt`. File gồm model, optimizer,
    trạng thái RNG, checkpoint tốt nhất và lịch sử, và được ghi nguyên khối nên không bị hỏng nếu bị ngắt giữa lúc ghi. Lần chạy
    sau có cùng stem và cùng tham số huấn luyện sẽ train tiếp từ đó, với cùng batch và dropout như khi không bị ngắt. Nếu tham
    số khác thì từ chối. Các epoch đã chạy tiếp được ghi vào `resumed_at` trong metric, và file bị xoá khi chạy xong. Notebook
    GeometryMMKG đẩy file này cùng log lên repo private `geometry-mmkg-resume`, cách nhau ít nhất N phút (mặc định 10). Khi bấm
    lại, notebook tải về để chạy tiếp. Baseline không có cơ chế này, vì phải sửa code gốc.
  - **Quyết định (22/11/2026, người dùng duyệt), sau khi bỏ triplet tự nối của Spatial-VRD (P0.4, 21/11/2026):**
    - **Giữ thiết kế đóng băng (P2.9).** Dữ liệu dev cũ chỉ khác ở 43/8.614 bộ ba valid (0,5%), nên MRR valid lệch tối đa khoảng
      0,005. Mỗi quyết định thiết kế hoặc được xác nhận thêm trên FB15k-237-IMG (không đổi), như `input_norm`, `depth_dropout`
      và `text_proj`, hoặc có cách biệt lớn hơn mức đó, như số độ sâu theo vùng (+0,010 đến +0,013). `rel_gate` đã bị bỏ nên
      không bị ảnh hưởng.
    - **Tune lại Spatial-VRD từ đầu** (lưới gốc và mở rộng biên) trên `v0` mới. Luật chọn cách gộp dùng kết quả VRD mới này.

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
| 1 | Chạy lại được số của paper gốc trên dataset gốc bằng script gốc (sai lệch khoảng ±1 MRR), với `-select_on original`. Mục đích: chứng minh code đã sửa (rank_file, ValidSelector) vẫn giống code gốc khi tắt các phần thêm | [x] NativE trên DB15K (−0,84), SNAG trên MKG-Y (−0,48) (P3.0b, 19/11/2026) |
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
- [x] **P3.0b Kiểm tra tái lập** (tiêu chí 1). Cả hai đạt (MRR lệch ≤ 1 điểm); còn bản SNAG `select_on=valid`.

  | Seed 42, test | MRR | H@1 | H@3 | H@10 | MR |
  |---|---:|---:|---:|---:|---:|
  | NativE DB15K, paper | 37.16 | 28.01 | 42.25 | 54.13 | |
  | NativE DB15K, mình | 36.32 (−0.84) | 27.28 (−0.73) | 41.35 (−0.90) | 53.02 (−1.11) | 685.3 |
  | SNAG MKG-Y, paper | 39.5 | 35.4 | 41.1 | 47.1 | |
  | SNAG MKG-Y, mình | 39.02 (−0.48) | 34.53 (−0.87) | 41.08 (−0.02) | 46.40 (−0.70) | 1293.4 |

  Số C++ và số Python tính lại từ rank CSV trùng nhau ở cả hai lượt. NativE train đủ 1000 epoch rồi đánh giá test một lần (2 giờ 15 phút trên RTX PRO 6000). SNAG gốc đánh giá test mỗi 100 epoch và early-stop theo MRR test: dừng ở epoch 1900/8000, giữ checkpoint MRR test 0,3902 (19,5 phút).
  - NativE/DB15K: `notebooks/native-db15k-sanity.py` tải embedding gốc từ Google Drive của tác giả (visual `[12842, 4096]`, textual/numeric `[12842, 768]`) và chạy đúng `scripts/run_db15k.sh` với `-select_on=original`, model `AdvRelRotatEDB15K` gốc. Số mục tiêu (arXiv 2406.17605, Table 2/3, NativE "All"): **MRR 37.16 · H@1 28.01 · H@3 42.25 · H@10 54.13**. Kết quả ở `geometry-mmkg-results/native/DB15K-sanity/`.
  - [x] SNAG trên **MKG-Y** (đổi từ MKG-W, 18/11/2026): split và id MKG-Y của tác giả SNAG giống hệt `data/MKG-Y` (và của NativE) theo nội dung, nên lần tái lập này vừa kiểm tra code SNAG vừa kiểm tra dữ liệu v0; sau này SNAG trên MKG-Y v0 chỉ khác đặc trưng (của tác giả so với của mình). Script gốc: `run_base.sh 0 MKG-Y 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 Y001`, thêm `-select_on=original -rank_file`.
    - Số mục tiêu (arXiv 2403.06832, Table 1 "MKGC performance on DB15K, MKG-W and MKG-Y", dòng "SnAg (Ours)", cột MKG-Y; đọc trực tiếp bản v1 11/03/2024 và v4 15/01/2025, hai bản giống nhau): **MRR .395 · H@1 .354 · H@3 .411 · H@10 .471**. Bản không có GMNM ("- w/o GMNM"): .387 / .345 / .407 / .457. Paper ghi "hidden dimension d = 256": khớp script (`EMB_DIM=128`, RotatE dùng `dim_e = 2·dim = 256`). Paper không nói cách chọn checkpoint cho MKGC; code gốc chọn theo MRR test (P3.0a), phải ghi rõ trong bài.
    - `notebooks/snag-mkgy-sanity.py`: tải zip của tác giả từ Drive, kiểm tra sha256 của zip và của `MKG-Y-{visual,textual}.pth` (`[15000, 383]`, `[15000, 384]`), `missing.json` (756 thiếu ảnh, 2.684 thiếu mô tả), chỉ giải nén ba file đó. Lệnh = `run_base.sh` (đã kiểm tra trên máy: mở rộng script bằng một `python` giả, ra đúng 21 tham số giống notebook, `CUDA_VISIBLE_DEVICES=0`), cộng `-select_on=original -rank_file`. Nút thứ hai (tuỳ chọn): `-select_on=valid -valid_steps=100 -patience=10` (luật gốc, chỉ thay test bằng valid) để đo mức lạc quan do chọn theo test. Kết quả về `geometry-mmkg-results/snag/MKG-Y-sanity/`. Bản `original` đã chạy (19/11/2026); **bản `valid` chưa chạy.**
    - Quyết định (19/11/2026): coi là tái lập được khi **MRR lệch ≤ 1 điểm** so với paper (H@k chỉ báo kèm); chạy cả bản `select_on=valid` ngay sau bản gốc. Nếu lệch nhiều: báo nguyên nhân khả dĩ trước, không sửa kiến trúc.
    - **Kết quả `original` (19/11/2026, Molab RTX PRO 6000, torch 2.11, commit `aea72de` + cài `unidecode`, seed 42, 1 lần chạy): MRR 39.02 · H@1 34.53 · H@3 41.08 · H@10 46.40** (MR 1293), lệch so với paper −0.48 · −0.87 · −0.02 · −0.70 điểm → **đạt ngưỡng (|ΔMRR| ≤ 1)**. Rank C++ và Python trùng nhau (2.663 bộ ba test, head + tail). MRR test đạt đỉnh ở epoch 899, dừng sớm ở epoch 1899 (10 lần đánh giá test liên tiếp không vượt), tổng 19,5 phút. Như vậy code (lib C++, patch ghi rank), split MKG-Y và embedding gốc cho ra đúng số của tác giả; checkpoint được chọn theo test như code gốc (P3.0a).
- [x] **P3.0d Mã gốc cho NativE và SNAG** (18/11/2026).
  - NativE: `AdvRelRotatEDB15K.py` giống từng byte với `zjukg/NativE` commit `c72291c` (giữ pooling gắn cứng VGG 4096, chỉ dùng cho P3.0b). Dữ liệu của mình dùng `AdvRelRotatE` (`run_adv_wgan_gp.py`, script gốc `run_mkgy.sh`), vốn tự lấy số chiều từ dữ liệu. Không dùng đường 4-modality (`AdvRelRotatEKuai16K` bắt buộc đủ ảnh, text, audio, video).
  - SNAG: repo GitHub `zjukg/SNAG` không chứa code MKGC; mã gốc là file zip "Dataset and Code" trên Google Drive (`1jIYo7JWXNNi7LZqBIsl103FL8tHcoW-q`, sha256 trong PROVENANCE). `AdvMixRotatE.py` giống từng byte với bản đó. Cờ gốc `use_pool`: khi `use_pool=1` và tên dataset không chứa "MKG", nó pooling cứng theo VGG 4096 chiều, nên dùng `-use_pool 0` cho FB15k-237-IMG và VRD.
  - `notebooks/native.py`: một notebook cho cả 5 dataset (tải `v0` từ HF), ba kịch bản `control` / `dmean` / `dconcat`. Mỗi kịch bản là một `-dataset` riêng, symlink tới cùng benchmark và tới bảng đặc trưng tương ứng. Cell setup kiểm tra bảng đặc trưng (đúng số thực thể, hữu hạn, dòng thiếu ảnh đúng là vector hằng). Tham số mặc định = `run_mkgy.sh` (dim 250, margin 4, batch 1024, neg 128, lr 1e-4, 1000 epoch), đánh giá valid mỗi 50 epoch. Kịch bản đã có rank CSV trên repo kết quả thì bỏ qua. **Chưa chạy trên Molab.**
  - `scripts/prepare_snag_data.py`: dữ liệu cho SNAG dưới tên dataset riêng (ví dụ `MKG-Y-v0`), không dùng tên của tác giả; ghi `benchmarks/<tên>/` (kèm `type_constrain.txt` theo logic `n-n.py` của tác giả, thử trên split MKG-Y của tác giả ra đúng file của họ), embedding `<tên>-{visual,textual}.pth`, và gộp khoá vào `missing.json` (giữ khoá của tác giả). Code gốc đọc `missing.json` khi tên dataset chứa "MKG" và điền `mean + std·randn` vào các dòng có trong danh sách. Trên MKG-Y: **visual** = 519 thực thể không có ảnh (đều nằm trong 756 thực thể thiếu ảnh của tác giả); **textual** = 2.684 thực thể không có mô tả, **trùng khớp danh sách của tác giả** (`entity2id` giống hệt). Như vậy SNAG thay text-tên của 2.684 thực thể này bằng nhiễu, đúng như code gốc làm với dữ liệu của tác giả, trong khi các model khác vẫn thấy text-tên.
- [ ] **P3.1** Feature-TransE (2-modal và 3-modal).
  - **Code (25/11/2026):** `core/models/feature_transe.py`, do một cộng tác viên viết. Bản dùng cho 6 lần chạy ngày 19/11/2026 có sha256 `2b200bc2…`, đã đối chiếu với `config.json` của các lần chạy. Kiến trúc: nối [text; RGB] (bản `depth`: [text; RGB; depth]) → Linear → ReLU → Dropout → Linear → chuẩn hoá L2 về độ dài 1 → điểm TransE L1. Không có embedding theo ID, cổng hay vector học được cho thực thể thiếu ảnh. Thực thể thiếu ảnh đọc vector hằng (mục 3.2, quy tắc 1). Số tham số trên FB15k-237-IMG: 1.953.024 / 2.542.848.
  - Sửa duy nhất: `forward_fusion` nhận và bỏ qua `geo` (từ P2.9, `in_batch_loss` truyền tham số này cho mọi mô hình; thiếu nó thì code gốc lỗi). Phép tính không đổi.
  - Chạy qua `scripts/train.py --model feature_transe --variant {no_depth,depth}`: cùng loss, cách lấy mẫu âm, chọn checkpoint, ngân sách epoch, `--tune`, chạy tiếp khi bị ngắt và `tune_summary.py` với GeometryMMKG; thư mục `results/feature-transe{,-tune,-dev}/`. Các tuỳ chọn của GeometryMMKG bị tắt cho mô hình này. Script huấn luyện riêng của cộng tác viên không đưa vào repo. Kiểm tra: `tests/test_feature_transe.py` (số tham số, bỏ qua `geo` và mặt nạ, CLI, chạy trọn trên dữ liệu nhỏ, từ chối bảng có dòng thiếu ảnh khác vector hằng).
  - **Kết quả của cộng tác viên (19/11/2026, ngoài giao thức tune):** FB15k-237-IMG, 100 epoch, batch 512, lr 1e-4, dropout 0,1; test MRR `no_depth` 0,2825 / 0,2842 / 0,2846 (seed 42 / 43 / 44, trung bình 0,2838), `depth` 0,2801 / 0,2811 / 0,2818 (0,2810). Δdepth −0,0028 [−0,0041; −0,0015]. Chỉ là một điểm lưới với batch khác quy trình chung, nên chỉ dùng tham khảo.
  - **Số độ sâu theo vùng (quyết định 25/11/2026):** trên 3 bộ scene graph, bản `depth` nối thêm 9 số độ sâu (`ent_depth_stats.pt`, chuẩn hoá theo thực thể train có ảnh; thực thể thiếu ảnh nhận 0, tức một hằng số) vào đầu vào, để nhận đúng thông tin depth như GeometryMMKG (`--depth-stats auto`; tag `dstat` trong tên). Trên FB15k / MKG-Y mô hình giữ nguyên từng tham số như bản của cộng tác viên. Kiểm tra trong `tests/test_feature_transe.py`.
  - **Kiểm tra bản đưa vào (25/11/2026, Molab, code `2f12eab`):** chạy lại đúng config của cộng tác viên (FB15k-237-IMG, `no_depth`, seed 42, batch 512, 100 epoch) qua `train.py --model feature_transe`. MRR valid trùng tới 6 chữ số ở cả 10 lần đánh giá, cùng epoch tốt nhất 40 (0,289223); test MRR 0,2825 · MR 180,6 · H@1 0,2004 · H@3 0,3047 · H@10 0,4480, giống hệt lần chạy cũ. Loss in ra lệch ở chữ số thứ tư chỉ do cách lấy trung bình (theo batch / theo bộ ba). 7,6 giây/epoch.
  - Còn làm: thêm Feature-TransE vào notebook tune.
- [ ] **P3.3** SNAG: notebook trên dữ liệu v0 (dùng `prepare_snag_data.py`), chạy nguyên bản trên mọi ô trừ Spatial-VG.
  - Kiểm lại (22/11/2026): embedding ID `nn.Embedding(N, 2·dim)`; số chiều ảnh và text lấy từ tensor (`-use_pool 0` cho tên không chứa "MKG"); bảng đặc trưng đóng băng. Tham số (dim 128, 1 lớp Mformer; dựng trên CPU): FB15k-237-IMG 27,1 triệu tổng / 4,8 triệu train được (+D concat 38,5 / 5,0), Spatial-VRD 169 / 25 triệu. Config gốc `run_base.sh`: 8000 epoch, batch 1024, 32 mẫu âm, margin 12, lr 1e-4, dừng sớm 10 lần đánh giá. Chọn theo valid: đã có (P3.0a).
- [ ] **P3.4** MKGformer trên FB15k-237-IMG (dataset gốc của paper, so trực tiếp được với số công bố) và VRD-Trans. Notebook phải dùng ảnh và mô tả thật (`dataset/FB15k-images`, mô tả KG-BERT), không được để thư mục ảnh rỗng hay text giả.
  - Kiểm lại (22/11/2026): BERT-base + CLIP ViT-B/32, mỗi thực thể và quan hệ là một token mới (`resize_token_embeddings`), không inductive. Tham số: 226,0 triệu gốc, 237,3 triệu trên FB15k-237-IMG, 298,1 triệu trên VRD-Trans. Chọn checkpoint theo valid (`ModelCheckpoint` theo `Eval/hits10`, dừng sớm theo `Eval/mrr` patience 5), không rò test; tiêu chí 2 muốn MRR valid thì đổi `monitor` (thay đổi loại "chọn checkpoint"). Thiếu ảnh: tensor 0.
  - **Phụ thuộc:** `requirements.txt` ghim transformers 4.19, pytorch_lightning 1.3.1, torch 1.7. `modeling_unimo.py` import `apply_chunking_to_forward` từ `transformers.modeling_utils`, không còn ở transformers 5 (Molab: 5.20), và `pl.Trainer.from_argparse_args` không còn ở Lightning 2. Cần môi trường riêng (transformers 4.x, Lightning 1.9) chạy được trên GPU Blackwell; kiểm trước khi viết notebook.
  - `--task_name` chỉ nhận `fb15k-237` / `wn18`: VRD-Trans phải đặt thư mục ảnh theo quy ước `m.xxx` của nhánh `fb15k-237` (tầng dữ liệu).
  - CPU: mỗi ví dụ mở tới 7 ảnh mỗi epoch (1 + 3 aux + 3 rcnn). Đo trên laptop (22/11/2026, ảnh kho MKGformer trung bình 13 KB): khoảng 720 ảnh/giây/nhân, tức khoảng 27 phút/epoch FB15k-237-IMG với 4 worker (config gốc), khoảng 7 phút với 16. Phần GPU chưa đo (`scripts/time_baselines.py mkgformer-gpu`).
- [ ] **P3.5** Kiểm tra file embedding gốc của tác giả NativE (DB15K) xem họ điền gì cho thực thể thiếu ảnh, ghi vào phụ lục; P3.0b vẫn dùng embedding gốc đó.
- [ ] **P3.6** MoMoK (mức 1): ~~clone repo `zjukg/MoMoK` làm submodule~~ (quyết định 25/11/2026: không giữ code trong repo, xem dưới); gắn chọn checkpoint theo valid và ghi rank theo P1; depth vào ở tầng dữ liệu, không thêm expert, không sửa kiến trúc; ở ô inductive chạy nguyên bản như NativE. Kiểm tra tái lập (tiêu chí 1) trên MKG-Y bằng đặc trưng gốc của tác giả.
  - **Đưa vào (25/11/2026), không chép code:** MoMoK và CMR không có giấy phép, repo này public, nên repo chỉ giữ diff của mình (`baselines/patches/MoMoK.diff`, `CMR.diff`); `scripts/setup_baseline.py <tên> --dest DIR` clone repo của tác giả ở commit cố định rồi áp diff (`--no-patch` cho code nguyên bản). Bảng thay đổi trong `baselines/PROVENANCE.md`.
    - Thay đổi MoMoK (đều ngoài kiến trúc): ma trận nhãn dựng trên GPU (giống hệt từng phần tử, giữ hàm gốc `get_batch_numpy` để so); `get_strict_metrics` (rank chặt, ghi rank); `--select_on valid` (mặc định: valid mỗi `eval_freq` epoch và sau epoch cuối, test một lần) / `original` (hành vi gốc); sửa lỗi tuple; `random.seed`.
    - Dữ liệu: `scripts/prepare_momok_data.py` (tên dataset `FB15K-237-branch-<dataset>-v0[-Dmean]`, tức chọn nhánh 768 / 768 của tác giả theo tên; từ chối tên chứa `MKG` và bảng concat).
    - **Chạy tiếp khi bị ngắt (25/11/2026; người dùng duyệt: mọi baseline trừ NativE được thêm nếu làm được):** `--resume FILE` lưu toàn bộ trạng thái sau mỗi lần đánh giá (model, estimator, hai optimizer, lr scheduler, thứ tự hiện tại của `corpus.train_indices`, kết quả tốt nhất, trạng thái ngẫu nhiên). `scripts/momok_resume_check.py`: bản bị ngắt sau epoch 1 rồi chạy tiếp cho dòng metric valid giống hệt bản chạy liền ở cả 4 epoch (MKG-Y thu nhỏ, CPU).
    - Kiểm tra: `tests/test_momok_patch.py` (cần `MOMOK_DIR`): nhãn giống hệt bản numpy; rank chặt khớp cách đếm vét cạn có điểm bằng nhau. Chạy thử 2 epoch MKG-Y trên máy (25/11/2026): valid mỗi epoch, test một lần, 2.663 dòng rank; `compute_metrics.py` cho cùng MRR / MR với log của MoMoK.
    - **Chạy trên Molab (chuẩn bị 26/11/2026): `notebooks/momok.py` + `scripts/momok_molab_run.sh`**, hàng đợi 4 lần chạy, lệnh README của MKG-Y (lr 5e-4, mu 1e-4, dim 300, 2000 epoch, đánh giá mỗi 100 epoch):
      1. MKG-Y, đặc trưng gốc của tác giả (`embeddings/MKG-Y/*.pth` trong file Drive của README, sha256 `215df99c…`), `--select_on original`: kiểm tra tái lập (tiêu chí 1), so với MRR 37,91 / H@1 35,09 của bài báo theo đúng cách báo của code gốc.
      2. Như 1, `--select_on valid`: đặc trưng gốc theo giao thức của mình.
      3. FB15k-237-IMG v0, Control, seed 42; 4. MKG-Y v0, Control, seed 42: lần chạy config gốc để chốt số epoch (P4.0), đồng thời là điểm tune seed 42 của config này (c1).
      - Split và id MKG-Y trong repo MoMoK trùng hoàn toàn với MKG-Y v0 (kiểm 26/11/2026), nên rank của lần 2 tính bằng `compute_metrics.py` trên `data/MKG-Y`. Chạy thử trên máy với đặc trưng gốc (2 epoch, cả hai chế độ chọn): chạy được, rank khớp log của MoMoK.
      - Trạng thái chạy tiếp lên repo private `callmezuy/momok-resume` sau mỗi lần đánh giá (`scripts/momok_hf.py`, khoảng 1,9 GB mỗi lần chạy, chỉ giữ bản mới nhất); kết quả lên `geometry-mmkg-results/momok/<dataset>/<run>/`.
  - **Đọc code (22/11/2026, commit `99c2df1` ngày 12/06/2025, trong scratchpad):**
    - Kiến trúc `RelMoE`: ba bảng theo thực thể đều train được: embedding cấu trúc theo ID (`nn.Embedding(N, dim)`), bảng ảnh và bảng text khởi tạo từ đặc trưng (`nn.Embedding.from_pretrained(..., freeze=False)`). Mỗi modality qua một MoE 3 expert có cổng theo quan hệ, rồi gộp bằng attention. Có 4 bộ chấm điểm TuckER (cấu trúc, ảnh, text, gộp), huấn luyện 1-N (KvsAll, BCE, label smoothing 0,1), cộng loss thông tin tương hỗ (CLUB, trọng số `mu`) giữa các expert. Hệ quả: đặc trưng chỉ là giá trị khởi tạo của một bảng theo thực thể. Ở ô transductive depth chỉ đổi điểm xuất phát; ở ô inductive mọi thực thể, kể cả thực thể mới, đều nhận gradient khi làm mẫu âm.
    - **Số chiều gắn cứng theo tên dataset** (`RelMoE.__init__`). Nhánh `"FB15K-237"` / `"Kuai"`: ảnh và text 768 → (12, 64) → AdaptiveAvgPool (4, 64) → 256. Nhánh `"MKG"`: giữ nguyên ảnh nhưng text phải 384 chiều. Tên không khớp nhánh nào thì `img` không được gán và chương trình lỗi. Bảng sai chiều không báo lỗi mà làm lệch dòng: bảng 1536 chiều qua nhánh FB15K-237 thành 2N dòng (dựng thử: 29.082 dòng cho 14.541 thực thể).
    - **Kết luận:** +D concat (1536) không chạy được ở ô nào, vì không nhánh nào nhận ảnh 1536 cùng text 768. MoMoK chỉ có Control và +D mean.
    - **Cần quyết định:** đặc trưng 768/768 của mình chỉ khớp nhánh `FB15K-237`. MKG-Y, Spatial-VRD và VRD-Trans phải mang tên dataset chứa `FB15K-237` (và không chứa `MKG`) để vào nhánh này. Hai cách: coi tên như một công tắc có sẵn của tác giả (giống `-use_pool 0` của SNAG, ghi trong PROVENANCE), hoặc không chạy MoMoK ở các bộ đó.
    - Tham số (dim 300, r_dim 256, 3 expert, dựng trên CPU): 97,7 triệu (FB15k-237-IMG, MKG-Y), 161,8 triệu (Spatial-VRD / VRD-Trans, trong đó 76 triệu là các bảng theo thực thể). Bốn tensor lõi TuckER chiếm 83,6 triệu.
    - Config gốc (README, MKG-Y): lr 5e-4, mu 1e-4, dim 300, 2000 epoch, batch 1024. Dropout cố định trong các lớp. Không có dừng sớm (`patience` khai báo nhưng không dùng). Không dùng mẫu âm (huấn luyện 1-N).
    - **Chọn checkpoint rò test nặng hơn SNAG:** mỗi 100 epoch đánh giá trên **test**, rồi báo giá trị lớn nhất của **từng metric riêng rẽ** qua các lần đánh giá (MRR và H@1 có thể lấy từ hai epoch khác nhau). Bài báo không nói chọn trên tập nào. Ngoài ra `get_validation_pred` trả về tuple còn `train.py` đọc như dict, nên code phát hành lỗi ngay lần đánh giá đầu (đã xác nhận khi chạy thử). Rank phá hoà ngẫu nhiên. Cần thêm (ngoài kiến trúc): đánh giá valid mỗi N epoch và giữ trọng số tốt nhất, rank theo luật bằng điểm chặt, ghi rank.
    - Thiếu ảnh: tiền xử lý gốc điền ảnh trung bình, khớp quy tắc vector hằng của mình.
    - Tái lập: cần đặc trưng gốc trên Google Drive của tác giả. Mục tiêu MKG-Y: MRR 37,91, H@1 35,09 (Table 1). Bản `original` chỉ tái lập được con số "max từng metric trên test".
    - Phụ thuộc: torch 2.0, numpy, scikit-learn, tqdm. `torch.where` với điều kiện uint8 chỉ cảnh báo trên torch 2.5; cần kiểm trên torch 2.11.
    - **Thời gian:** mỗi batch dựng ma trận nhãn B × N trên CPU (Spatial-VRD: 1024 × 93.809, khoảng 384 MB mỗi batch). **Đo trên Molab (25/11/2026):** 21,8 giây/epoch trên FB15k-237-IMG và 92,5 giây/epoch trên Spatial-VRD (bộ nhớ GPU đỉnh chỉ 0,8 / 3,2 GB), tức khoảng 12,3 và 52 giờ cho 2000 epoch, đều vượt 12 giờ (P4.0). Đo tách từng phần: 72% / 86% thời gian là `get_batch` dựng ma trận nhãn trên CPU; dựng cùng tensor trên GPU (giống hệt từng phần tử) đưa về 6,4 / 13,6 giây/epoch. **Đã quyết (25/11/2026): dựng nhãn trên GPU** (P4.0).
- [ ] **P3.7** CMR (mức 1): clone repo `OreOZhao/CMR`; chạy nguyên bản trên FB15K-237-ind, đối chiếu số paper (MRR 0,371), đây là kiểm tra tái lập cho ô inductive. Thêm ghi rank theo P1. Chạy trên Spatial-VRD (text = tên lớp vật thể, ảnh = vùng cắt theo box). Code CMR đọc đúng một ảnh mỗi thực thể và không chạy được nếu thực thể thiếu ảnh (P0.2). Cách cấp ảnh cho thực thể thiếu ảnh ở tầng dữ liệu, không sửa code: **đã quyết 25/11/2026, một ảnh nhiễu dùng chung** (bên dưới). Ảnh dùng cho CMR: ảnh `ok` đầu tiên trong `image_index.tsv`, hoặc ảnh CMR chọn nếu lấy được `fb15k_best_img.pickle`.
  - **Đưa vào (25/11/2026), không chép code:** `baselines/patches/CMR.diff` qua `scripts/setup_baseline.py cmr`. Thay đổi: khai báo `--mem`, `--adapter-model` (code gốc lỗi ngay bước train đầu vì thiếu); `--seed` được dùng (code ghi đè 202303); task `SpatialVRD_ind`; `--freeze-vit` (gọi `_freeze_module` có sẵn); `torch.optim.AdamW(eps=1e-6)`; rank chặt cạnh rank gốc, `--rank-file`, `--rank-tensor`, `--rank-lambda`. Đã chạy trên Molab (bên dưới).
    - **Dữ liệu (25/11/2026): `scripts/prepare_cmr_data.py`** ghi `data/<Task>/{train,valid,test}.txt.json`, `relations.json`, `img_entities.json` (file `triplet.EntityDict` đọc; `preprocess.py` phát hành chỉ ghi `entities.json` không có đường dẫn ảnh) và `img_data/<Task>/`. Ứng viên = toàn bộ `entity2id` (mục 4.2).
      - FB15K-237-ind (`FB15k237_ind`): giữ nguyên file bộ ba của tác giả sau khi kiểm trùng đúng split của mình (cùng bộ ba, cùng thứ tự, chuỗi quan hệ khớp `relations.json`); tên và mô tả theo đúng `preprocess.py` (mô tả cắt 50 từ; 14.515 / 14.541 thực thể có mô tả); ảnh = ảnh `ok` đầu tiên trong `image_index.tsv`.
      - Spatial-VRD (`SpatialVRD_ind`): bộ ba từ file v0; text = `entity2text.txt` (cùng text mọi mô hình nhận), không có mô tả; ảnh = vùng cắt theo box, cùng luật box với `extract_features.py`.
      - Thực thể thiếu ảnh: một ảnh nhiễu chung (seed 0, 384 px, PNG, sha256 `1fd6a78e…`); script kiểm tập này trùng đúng `ent_img_mask.pt` (FB15K-237-ind 249, Spatial-VRD 2).
      - Chạy trên máy: cả hai bộ qua mọi phép kiểm; `doc.py` của CMR nạp được 430.164 / 153.918 ví dụ ở chế độ `--mm --prefix 4`.
    - **Lần chạy đầu trên Molab (25/11/2026):** dữ liệu và đường ảnh nạp đúng; 1,03 giây/bước, khoảng 12,5 phút/epoch (train + valid). Sandbox Molab khởi động lại hai lần (sau khoảng 37 phút và 2 giờ 47 phút), lần chạy dừng ở epoch 13; valid tăng đều (Acc@1 trong batch 4,4 → 10,5; MMAcc@1 19,4 → 24,5 ở epoch 0 → 12). Log đã lên `geometry-mmkg-results/cmr/FB15K-237-ind/train/` (checkpoint đã xoá ngày 26/11/2026, sau lần chạy chính thức). Code gốc không chạy tiếp được, nên lần chạy chính thức bắt đầu lại.
    - **Chạy tiếp khi bị ngắt (25/11/2026):** `--resume <model_dir>` (diff của mình): sau mỗi epoch lưu model, optimizer, lr scheduler, GradScaler, bộ đệm âm mẫu pre-batch, kết quả tốt nhất, trạng thái ngẫu nhiên vào `resume.pt`. Lúc đánh giá dùng `--workers` tiến trình nạp dữ liệu thay cho 1–2 (thứ tự giữ nguyên). Kiểm tra `scripts/cmr_resume_check.py` (CPU, 64 bộ ba, 3 epoch: bản bị ngắt + chạy tiếp phải trùng từng dòng valid với bản chạy liền); script chạy trên Molab chạy bước này trước lần train mới. **Kết quả trên Molab (25/11/2026): bản bị ngắt sau epoch 0 rồi chạy tiếp trùng từng dòng valid với bản chạy liền ở cả 3 epoch.**
    - **Notebook `notebooks/cmr.py`** + `scripts/cmr_molab_run.sh`: cài đặt, khôi phục từ repo private `callmezuy/cmr-resume` (chỉ giữ bản mới nhất, lịch sử được gộp sau mỗi lần tải), train, đánh giá (λ chọn trên valid, test một lần cho bản không kNN và bản kNN), đổi rank sang id của mình (`scripts/cmr_ranks.py`) và tải kết quả lên `geometry-mmkg-results`. Sandbox khởi động lại thì mở lại notebook, nhập token, bấm chạy.
      - Dung lượng HF: gộp lịch sử không giải phóng chỗ (HF vẫn giữ và tính file lớn của các commit cũ, khoảng 4 GB mỗi epoch), nên lần chạy chính thức chạm giới hạn private storage ở epoch 6. `scripts/cmr_hf.py` giờ xoá hẳn các file lưu mà bản mới nhất không dùng, sau mỗi lần gộp.
    - **Lần chạy chính thức trên FB15K-237-ind (25–26/11/2026, Molab, config README + ViT đóng băng, 1 lần chạy):**
      - Train mất 6 giờ 56 phút, khoảng 12,5 phút/epoch (P4.0 ước lượng 2–6,5 giờ). Dừng sớm sau epoch 32; epoch tốt nhất là 28 (valid Acc@1 trong batch 11,69, MMAcc@1 24,35). Đánh giá mất 2 giờ 24 phút, trong đó kNN khoảng 2 giờ.
      - λ = 0,975 chọn trên valid (MRR valid 0,368, bảng text-mm).
      - Test (rank chặt của mình, filtered, head + tail, mọi thực thể, 52.870 bộ ba):

        | Bản | MRR | MR | H@1 | H@3 | H@10 |
        |---|---|---|---|---|---|
        | CMR + kNN (bản của bài báo) | **0,3681** | 138,9 | 0,2665 | 0,4158 | 0,5667 |
        | CMR không kNN | 0,2581 | 203,4 | 0,1753 | 0,2827 | 0,4219 |
        | Số trong bài báo | 0,371 | | 0,269 | | 0,567 |

      - **Tái lập đạt:** MRR lệch −0,003 so với bài báo, trong ngưỡng 1 điểm. Kết quả này đạt được dù dùng kho ảnh MKGformer và ảnh nhiễu chung, không dùng ảnh crawl bù của tác giả.
      - Bộ nhớ kNN (chứa cả bộ ba test, như code) đóng góp +0,11 MRR.
      - Rank, metric, log và checkpoint nằm ở `geometry-mmkg-results/cmr/FB15K-237-ind/fb15k237_ind-20261125-182037/`.
    - **Spatial-VRD (chuẩn bị 26/11/2026):** `scripts/cmr_molab_run.sh` nhận `CMR_TASK=SpatialVRD_ind` (notebook chọn dataset). Cùng lệnh train của README, ViT đóng băng, cùng quy trình đánh giá. Ảnh vùng cắt lấy từ file zip ảnh VRD của Kaggle (sha256 `b858225e…`, bản sao private `callmezuy/vrd-images-raw` cho Molab). Mỗi dataset có repo lưu trạng thái riêng (`callmezuy/cmr-resume-spatial-vrd`), nên hai lần chạy không gộp lịch sử hay dọn file của nhau.
    - **Thực thể thiếu ảnh (quyết định 25/11/2026): một ảnh nhiễu dùng chung** cho mọi thực thể thiếu ảnh (249 ở FB15K-237-ind; 2 box ở Spatial-VRD), sinh một lần với seed cố định và lưu kèm mã hash. Lý do: gần nhất với câu "padded with random values" của bài báo, và vẫn là một hằng số như quy tắc chung cho baseline (mục 3.2), nên không thành một tín hiệu định danh riêng cho từng thực thể. Không dùng bộ ảnh BaiduNetDisk của tác giả cho lần chạy chính: README nói tác giả crawl bù ảnh từ Wikidata và Google / Bing, bộ ảnh được upload ngày 20/02/2025 (sau bài báo), không có hash; cho riêng CMR thêm ảnh thì không còn cùng kho ảnh với các mô hình khác. Chỉ dùng nếu tải được, và chỉ cho lần kiểm tái lập (ghi rõ khác biệt về ảnh khi so với MRR 0,371).
  - **Đọc code (22/11/2026, commit `bcd23a6` ngày 20/02/2025, trong scratchpad):**
    - Kiến trúc: bi-encoder theo SimKGC. Một BERT-base mã hoá truy vấn (h, r), một BERT-base khác mã hoá thực thể. ViT-B/16 (`vit-base-patch16-224`) mã hoá đúng một ảnh mỗi thực thể, ánh xạ thành 4 token tiền tố cho BERT thực thể. Loss InfoNCE trong batch theo cả hai chiều, cộng pre-batch. Lúc suy luận cộng thêm điểm kNN từ bộ nhớ FAISS các vector truy vấn. **Không có embedding ID**, nên chạy được inductive. 307,3 triệu tham số, như nhau ở mọi dataset. Code để ViT train được (hàm đóng băng có nhưng không được gọi), trong khi bài báo nói ViT đóng băng.
    - Dữ liệu: mỗi ví dụ mở hai ảnh từ file mỗi epoch (`Image.open` → 384 px → processor 224). Text = tên + mô tả, cộng tên hàng xóm trong đồ thị train khi mô tả dưới 20 từ (`--use-link-graph`). `entities.json` cần trường `entity_imgpath`; `preprocess.py` phát hành đã comment phần đọc `fb15k_best_img.pickle`, nên phải tự ghi `entities.json` ở tầng dữ liệu.
    - Config gốc (README): batch 768, lr 1e-5, tối đa 50 epoch, AMP, pre-batch 2, prefix 4, warmup 400, weight decay 1e-4. Dừng sớm sau 3 epoch không tăng Acc@1 trong batch trên valid (bài báo gọi là "Hits@1 on the validation set"). Suy luận: k = 32, λ = 0,95; bài báo nói λ và k được tìm trên valid.
    - Chọn checkpoint theo valid, không rò test lúc train. **Rủi ro lúc suy luận:** `evaluate.py --is-test` với task có đuôi `_ind` nạp cả **bộ ba test** vào bộ nhớ kNN (chỉ bỏ hàng xóm có cùng (h, r) với câu hỏi), dùng lại chỉ mục đã cache nếu có, và in metric của 42 giá trị λ trên test. Bài báo ghi "memorized all the triples with both seen and unseen entities in KS". Đề xuất: chọn λ và việc có dùng kNN hay không trên valid; báo hai số: CMR không kNN (`--knn_topk 0`, cờ có sẵn) và CMR + kNN đúng như code, ghi rõ bộ nhớ chứa bộ ba test.
    - Lỗi trong code phát hành, sửa ngoài kiến trúc: `trainer.py` đọc `args.mem` nhưng `config.py` không khai báo (lỗi ngay bước train đầu); `knowledge_store.py` và `predict.py` đọc `args.adapter_model` không khai báo; `from transformers import AdamW` không còn ở transformers 5 (thay bằng `torch.optim.AdamW`); seed bị ghi đè cứng 202303; `--task` bị giới hạn bằng `choices`, nên Spatial-VRD cần thêm một tên task (đuôi `_ind` để giữ cùng hành vi suy luận).
    - Thời gian (ước lượng): giải mã ảnh đo trên laptop khoảng 260 ảnh/giây/nhân, tức khoảng 20 phút/epoch FB15K-237-ind với 4 worker và khoảng 5 phút với 16. Phần GPU chưa đo (`scripts/time_baselines.py cmr-gpu`). Một lần chạy FB15K-237-ind ước lượng 2–12 giờ, tuỳ số epoch trước khi dừng sớm. CMR không chạy tiếp được khi bị ngắt (chỉ nạp được trọng số, mất optimizer và lịch lr).
    - **Đo trên Molab (25/11/2026, phần GPU, đầu vào tổng hợp):** batch 768 với ViT train được (đúng code phát hành) **hết bộ nhớ 96 GB**; batch 384 cần 80 GB, 10,3 phút/epoch FB15K-237-ind; batch 768 với ViT đóng băng (như bài báo) cần 48,1 GB, khớp GPU A6000 48 GB của tác giả, 6,5 phút/epoch. Nhiều khả năng tác giả chạy với ViT đóng băng. **Đã quyết (25/11/2026): đóng băng ViT theo bài báo** (P4.0).
- [ ] **P3.8** *(Mức 2, chỉ làm nếu còn GPU)* AdaMF-MAT, IKRL, TBKGC, BLP-TransE; MyGO nếu code tách token đã phát hành.

### P4. Chạy thí nghiệm (Molab)

- [ ] **P4.0 Ngân sách GPU cho baseline (lập 22/11/2026, đo trên Molab 25/11/2026, chưa quyết).** Số tham số và tổng giờ GPU dưới đây tính bằng hai script dùng một lần (dựng model trên CPU; nhân số lần chạy với thời gian mỗi lần), không lưu trong repo; mọi kết quả của chúng đã ghi ở đây và ở P3.3–P3.7. Công cụ đo thời gian: `scripts/time_baselines.py` (MoMoK, SNAG, NativE chạy thật vài epoch; CMR, MKGformer đo riêng phần GPU và phần giải mã ảnh).
  - **Đo trên Molab (25/11/2026, RTX PRO 6000 96 GB, `time_baselines.py`, 4 epoch mỗi phép đo):**

    | Baseline, dataset | Giây / epoch | Một lần đánh giá valid | Ghi chú |
    |---|---|---|---|
    | MoMoK, FB15k-237-IMG | 21,8 | 32 giây | Bộ nhớ GPU đỉnh 0,8 GB: phần chậm là dựng ma trận nhãn B × N trên CPU |
    | MoMoK, Spatial-VRD | 92,5 | 86 giây | Như trên, N = 93.809 |
    | SNAG, FB15k-237-IMG | 7,0 | 59 giây | |
    | SNAG, Spatial-VRD-Trans | 2,3 | 186 giây | ValidSelector duyệt từng bộ ba trên 93.809 thực thể |
    | NativE, MKG-Y | 2,0 | 8 giây | |
    | MKGformer, FB15k-237-IMG (chỉ GPU, fp32, batch 96) | 26,4 phút | | 0,28 giây / bước; CPU đo trên laptop khoảng 7 phút / epoch với 16 worker, nên GPU là nút thắt |
    | CMR, batch 768, ViT train được (code phát hành) | **hết bộ nhớ** | | Không vừa 96 GB |
    | CMR, batch 384, ViT train được | 10,3 phút (FB15K-237-ind) | | Bộ nhớ đỉnh 80 GB |
    | CMR, batch 768, ViT đóng băng (như bài báo) | 6,5 phút (FB15K-237-ind) | | Bộ nhớ đỉnh **48,1 GB**, khớp GPU A6000 48 GB của tác giả |

    Ngoài ra, MoMoK `get_validation_pred` trả về tuple trên cả hai bộ, xác nhận `train.py` gốc lỗi ở lần đánh giá đầu.
  - **Hệ quả:**
    - **MoMoK theo config gốc (2000 epoch) không chạy xong trong một phiên 12 giờ ở bộ nào trừ MKG-Y và FB15K-237-ind:** FB15k-237-IMG khoảng 12,3 giờ, Spatial-VRD khoảng 52 giờ, VRD-Trans khoảng 55 giờ. Kể cả 400 epoch, VRD vẫn khoảng 10,5 giờ.
    - **CMR không chạy được đúng như code phát hành:** batch 768 với ViT train được không vừa 96 GB. Chỉ có hai cách: ViT đóng băng như bài báo (một thay đổi so với code, cần quyết định), hoặc giảm batch (đổi config gốc, ảnh hưởng số mẫu âm trong batch).
    - SNAG dừng sớm theo valid. Nếu chạy đủ 8000 epoch thì FB15k-237-IMG mất khoảng 17 giờ, nên phải dựa vào dừng sớm (MKG-Y: dừng ở epoch 1899).
  - **Bảng tài nguyên hiện hành (25/11/2026, theo các quyết định đã chốt; thay hai bảng thời gian / tổng giờ phía dưới, được giữ làm lịch sử).** Molab RTX PRO 6000 96 GB, một lần chạy phải xong trong 12 giờ. **đo** = chạy trọn trên Molab; *suy* = suy từ giây/epoch đo được; còn lại là ước lượng.

    | Baseline (config chạy) | Tham số FB15k / VRD | FB15k-237-IMG | MKG-Y | VRD-Trans | FB15K-237-ind | Spatial-VRD | Spatial-VG | Ghi chú |
    |---|---|---|---|---|---|---|---|---|
    | NativE (1000 epoch) | 31,2 / 192,5 triệu | **7,7–8,2 đo** | 0,6 *suy* | ≈ 3,3 | 5,5–7 | **3,3 đo** (đã xong) | không chạy | 27,6 s/epoch FB15k; chạy song song không nhanh hơn |
    | SNAG (dừng sớm theo valid; 2000–8000 epoch) | 27,1 / 169 triệu (train được 4,8 / 25) | 4,2–16,8 *suy* | **0,33 đo** | 2,3–9,4 *suy* | 3,8–15,4 *suy* | 2,3–9,4 *suy* | không chạy | 7,0 / 2,3 s/epoch; mỗi lần valid 59 s (FB15k) / 186 s (VRD-Trans) |
    | MoMoK (nhãn dựng trên GPU, 2000 epoch) | 97,7 / 161,8 triệu | 3,7 *suy* | ≈ 0,7 *suy* | ≈ 8,5 *suy* | ≈ 3,3 *suy* | ≈ 8,1 *suy* | không chạy | 6,4 / 13,6 s/epoch; chỉ Control và +D mean |
    | MKGformer (15 + 12 epoch, fp32) | 237,3 / 298,1 triệu | 5,3–7 *suy* | — | 1,5–3 | — | — | — | GPU 26 phút/epoch FB15k, 17 GB; CPU nạp ảnh nhanh hơn GPU |
    | CMR (ViT đóng băng, ≤ 50 epoch, dừng sớm 3) | 307,3 triệu | — | — | — | 2–6,5 *suy* | 0,6–2 *suy* | — | GPU 6,5 phút/epoch FB15K-237-ind, 48 GB |
    | Feature-TransE (như GeometryMMKG) | 1,95 / 2,54 triệu (FB15k, không / có depth) | ≈ 0,7 | ≈ 0,3 | ≈ 0,13 | ≈ 0,5 | ≈ 0,13 | ≈ 0,6 | 7,5 s/epoch FB15k ở batch 512 (lần chạy của cộng tác viên) |

    **Kế hoạch chạy theo phương án đã chọn** ((a) + (c1) + (c2); ngân sách epoch đoán trước: NativE 400, SNAG trần 2000, MoMoK 1000, chốt lại sau một lần chạy config gốc cho mỗi (baseline, dataset); không tính lợi của (f)):

    | Baseline | Tune | Chính thức (thêm seed 43, 44) | Ô inductive (1 seed, config gốc) | Giờ GPU (giữa) |
    |---|---|---|---|---|
    | NativE | 72 (24 × FB15k, MKG-Y, VRD-Trans) | 12 | 3 (FB15K-237-ind; VRD đã xong) | khoảng 160 |
    | SNAG | 72 | 12 | 6 | khoảng 235 |
    | MoMoK | 48 (16 × 3 bộ; 2 kịch bản) | 8 | 4 | khoảng 155 |
    | MKGformer | 0 (config gốc) | 0 | — | khoảng 8 (FB15k-237-IMG, VRD-Trans) |
    | CMR | 0 (config gốc) | 0 | 2 lần (FB15K-237-ind, Spatial-VRD) | khoảng 5 |
    | Feature-TransE | 96 (16 × 6 bộ) | 16 | — (không có ID: tune ở mọi ô) | khoảng 45 |
    | **Tổng** | | | | **khoảng 610 giờ** (500–1.050), khoảng 55 phiên Molab |

    SNAG chiếm nhiều nhất vì chưa biết nó dừng sớm ở epoch nào trên FB15k-237-IMG. Lần chạy config gốc sẽ cho biết, và đánh giá valid theo lô (f) giảm phần 59–186 giây mỗi lần valid.
  - *(Lịch sử, 25/11/2026 trước khi dựng nhãn MoMoK trên GPU; đã thay bằng bảng trên.)* Thời gian mỗi lần chạy, config gốc (giờ; **đo** = chạy trọn trên Molab, *suy* = suy từ giây/epoch đo được, còn lại là ước lượng):

    | Baseline | FB15k-237-IMG | MKG-Y | VRD-Trans | FB15K-237-ind | Spatial-VRD |
    |---|---|---|---|---|---|
    | NativE (1000 epoch) | **7,7–8,2 đo** | 0,6 *suy* | ≈ 3,3 (VRD đo) | 5,5–7 | **3,3 đo** |
    | SNAG (dừng sớm ở 2000 / 3000 / đủ 8000 epoch) | 4,2 / 6,3 / 16,8 *suy* | **0,33 đo** (dừng theo test) | 2,3 / 3,5 / 9,4 *suy* | 3,8 / 5,8 / 15,4 *suy* | 2,3 / 3,5 / 9,4 *suy* |
    | MoMoK (2000 epoch) | **12,3 *suy*** | 1,5–2,5 *suy* | **≈ 55 *suy*** | 10,2 *suy* | **≈ 52 *suy*** |
    | MKGformer (15 + 12 epoch) | 5,3–7 *suy* | — | 1,5–3 | — | — |
    | CMR (ViT đóng băng, ≤ 50 epoch, dừng sớm 3) | — | — | — | 2–6,5 *suy* | 0,6–2 *suy* |
    | Feature-TransE (như GeometryMMKG) | ≈ 0,7 | ≈ 0,3 | ≈ 0,13 | ≈ 0,5 | ≈ 0,13 |

  - **Số lần chạy theo tiêu chí 5:** mỗi (mô hình, dataset, kịch bản) khoảng 8 lần tune (lưới 6 + điểm biên, như GeometryMMKG: 49 và 47 lần cho 6 biến thể) cộng 3 seed chính thức ở FB15k-237-IMG và VRD-Trans, 1 seed ở MKG-Y. Kịch bản: NativE, SNAG 3; MoMoK 2; MKGformer, CMR 1; Feature-TransE 2. Baseline có ID ở ô inductive: 1 seed, config gốc, không tune.
  - Tổng giờ GPU sau khi có số đo (thấp / giữa / cao; NativE đã xong 3 lần VRD và 3 lần FB15k):

    | Phương án | Tổng | Trong đó MoMoK (giữa) | Điểm yếu khi bị phản biện |
    |---|---|---|---|
    | Đúng tiêu chí 5 | 2.360 / **2.550** / 3.330 | 1.640 | Không khả thi; MoMoK ở FB15k và hai bộ VRD không chạy xong trong 12 giờ |
    | (a) Giảm ngân sách epoch (NativE 1000 → 400, SNAG trần 2000, MoMoK 2000 → 400), chốt trước từ đường valid của một lần chạy config gốc, vẫn chọn theo valid, kèm luật hội tụ | 900 / **1.030** / 1.560 | 450 | "Baseline bị train ngắn hơn tác giả": cần dẫn đường valid đã đi ngang (NativE FB15k từ khoảng epoch 150) và luật hội tụ; MoMoK VRD 400 epoch vẫn sát 12 giờ |
    | (b) Chỉ tune Control, kịch bản depth dùng lại cấu hình | 1.510 / **1.630** / 2.090 | 1.090 | Lệch nguyên tắc "tune riêng từng biến thể" đã áp cho GeometryMMKG; Δdepth của baseline có thể bị kéo âm |
    | (a) + (b) | 630 / **720** / 1.040 | 340 | Cả hai điểm yếu trên |
    | (a) + (c) | 760 / **860** / 1.300 | 420 | Như (a); MKGformer và CMR chỉ chạy config gốc |
    | (a) + (b) + (c) | 490 / **550** / 780 | 310 | Như (a) + (b) + (c) |
    | (a) + (c) + (d) | khoảng **520** | khoảng 90 | Như (a) + (c); MoMoK thiếu ô VRD / VRD-Trans |
    | (a) + (b) + (c) + (d) | khoảng **300** | khoảng 65 | Như trên |

    (c) gồm: dùng lần tune seed 42 của cấu hình được chọn làm seed 42 chính thức (test một lần trên checkpoint đã lưu, bớt 1/3 số lần chạy chính thức); MKGformer và CMR chỉ chạy config gốc như mốc tham chiếu (so trực tiếp với số công bố trên đúng dataset gốc), không tune; cache ảnh đã giải mã cho hai mô hình đọc ảnh thô (thay đổi loại "nạp dữ liệu"). Điểm yếu: MKGformer / CMR không được tune trên VRD.
    (d) Không chạy MoMoK trên Spatial-VRD và VRD-Trans, ghi lý do: config gốc cần khoảng 52–55 giờ mỗi lần, code không chạy tiếp được khi bị ngắt và không được sửa.
    (e) Phương án khác cho MoMoK: viết lại phần dựng ma trận nhãn của `ConvECorpus.get_batch` (vòng lặp Python + numpy trên CPU) thành phép tính trên GPU cho ra cùng tensor. Đây là thay đổi loại "nạp dữ liệu", không đụng kiến trúc; phải kiểm bằng test rằng tensor nhãn giống hệt từng phần tử. Có thể đưa MoMoK VRD về mức chạy được, nhưng chưa đo được lợi bao nhiêu.
  - Rủi ro chung: không baseline nào chạy tiếp được khi bị ngắt (không được sửa code), nên mọi lần chạy phải dưới 12 giờ.
  - **Đo tách một bước train MoMoK (25/11/2026, Molab; lặp lại đúng vòng lặp của `train.py`, 30 batch, trung vị):**

    | Dataset | `get_batch` (CPU) | Chép lên GPU | Model + estimator | Phần nạp nhãn | Nhãn dựng trên GPU | Giây / epoch: hiện tại → nhãn trên GPU |
    |---|---|---|---|---|---|---|
    | FB15k-237-IMG | 0,093 s | 0,012 s | 0,041 s | 72% | 0,0024 s | 21,4 → 6,4 |
    | Spatial-VRD | 0,579 s | 0,068 s | 0,107 s | 86% | 0,0038 s | 92,8 → 13,6 |

    Tensor nhãn dựng trên GPU **giống hệt từng phần tử** (`torch.equal`) với tensor của `get_batch` trên cả 60 batch. Với cách này, 2000 epoch (cộng 20 lần đánh giá) còn khoảng 3,7 giờ (FB15k-237-IMG), 8,1 giờ (Spatial-VRD), 8,5 giờ (VRD-Trans), 3,3 giờ (FB15K-237-ind): mọi ô chạy được trong 12 giờ ở đúng config gốc.
  - **Quyết định (25/11/2026, người dùng duyệt):**
    - CMR: **đóng băng ViT** theo mục 5.1.3 của bài báo, bằng hàm `_freeze_module` có sẵn trong code nhưng không được gọi; ghi trong PROVENANCE. Kiểm chứng bằng lần tái lập trên FB15K-237-ind (MRR 0,371, ngưỡng lệch 1 điểm). Không giảm batch.
    - MoMoK: **(e)**, dựng ma trận nhãn trên GPU (thay đổi loại "nạp dữ liệu", kèm test so từng phần tử với `get_batch` gốc); giữ đủ các ô, không cần (d).
    - Ngân sách: **(a) + (c1) + (c2)**, không dùng (b), không cần cache ảnh (c3) vì MKGformer và CMR bị nghẽn ở GPU. Thêm (f): đánh giá valid theo lô trong `ValidSelector` (code của mình; kết quả phải giống hệt, kiểm bằng `tests/test_valid_selector.py`).
    - Tổng (giữa, với ngân sách epoch đoán trước cho MoMoK 2000 → 1000): khoảng **600 giờ GPU** (500–1.035), không lần chạy nào vượt 12 giờ. Ngân sách epoch thật chốt sau một lần chạy config gốc cho mỗi (baseline, dataset).
  - Việc tiếp theo: đưa MoMoK và CMR vào `baselines/` (cách đưa vào: hỏi trước), áp các thay đổi ngoài kiến trúc ở P3.6 / P3.7, chạy tái lập (CMR trên FB15K-237-ind, MoMoK trên MKG-Y bằng đặc trưng gốc), rồi chạy config gốc để chốt ngân sách epoch.

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

- [ ] **P6.0 Kế hoạch bài báo (duyệt 25/11/2026).**
  - **Tên tạm:** *Does depth help multimodal knowledge graph completion? A controlled study across knowledge graphs and
    scene graphs.*
  - **Câu chuyện chính:** depth (ước lượng đơn mắt) có giúp hoàn thiện đồ thị tri thức đa phương thức không? Bài trả lời
    bằng cách đặt MMKG (thực thể độc lập) và MMSG (các vật trong cùng một ảnh) vào cùng một giao thức, tách hai trục loại đồ
    thị và inductive / transductive, và đo Δdepth trong cùng mô hình trên nhiều họ mô hình. Bài là bài trả lời câu hỏi đó,
    không phải bài giới thiệu mô hình.
  - **Khẳng định và bằng chứng:**

    | # | Khẳng định | Bằng chứng | Trạng thái (26/11) |
    |---|---|---|---|
    | C1 | Mô hình MMKG có embedding theo ID thất bại trên MMSG inductive vì là mô hình transductive, không phải vì thiếu depth; mô hình dựa trên đặc trưng chạy được ở cả 4 ô | Baseline và mô hình đề xuất trên cả 4 ô; cặp VRD / VRD-Trans | Một phần (NativE) |
    | C2 | Depth không giúp MMKG | Δdepth trên 3 bộ MMKG trong mức nhiễu, trên ít nhất 2 họ mô hình | 1 seed (mô hình đề xuất); 3 seed Feature-TransE trên FB15k. Chỉ đo qua vector depth, vốn vô dụng cả trên MMSG; E2b kiểm tra luận điểm "depth chỉ giúp khi có thang chung trong cảnh" |
    | C3 | Trên MMSG, phần lợi của depth đến từ số độ sâu theo vùng (hiệu độ sâu đầu − đuôi), không đến từ đặc trưng ảnh depth | Bảng 2 × 2 `depth_map_only` / `depth_stats_only` (P2.10), thước đo phụ (E1) | 1 seed, 2 bộ. MRR chính: đúng. Trong ảnh (E1): `depth_stats` +0,007 đến +0,009 trên cả 2 bộ; vector depth +0,010 / +0,012 trên VRD nhưng ≈ 0 trên VRD-Trans, nên viết "không giúp ở MRR chính, trong ảnh chưa nhất quán" cho đến khi có 3 seed. E2: hình học box 2D (gán tay) là tín hiệu mạnh hơn hẳn (box một mình +0,029 trên cả 2 bộ so với `no_depth`, gấp 2,7 / 2,2 lần `depth_stats`), nên viết "`depth_stats` là đường depth duy nhất có ích, nhưng nhỏ hơn hình học 2D" |
    | C4 | Depth có thêm gì ngoài hình học box 2D không | Đối chứng bằng box (E2) | E2 (1 seed, valid): **có theo luật đặt trước, nhưng nhỏ**: `box_grid9` − `box` +0,005 (VRD, sát ngưỡng nhiễu 0,003) / +0,010 (VRD-Trans). Gần như toàn bộ qua `image_first` (+0,006 / +0,022); trong ảnh ≈ 0 (+0,003 / −0,003); đoán quan hệ giảm (−0,017 / −0,025). Tức depth giúp nhận ra vật thuộc ảnh nào hơn là hiểu quan hệ không gian trong ảnh. Chờ E3 và E6 |
    | C5 | MMSG khác MMKG về cấu trúc, và xếp hạng trên toàn bộ thực thể lẫn việc tìm đúng ảnh | Bảng cấu trúc đồ thị; thước đo trong cùng ảnh và đoán quan hệ (E1) | Có bảng cấu trúc và E1 (1 seed, valid): chỉ 26 % (VRD) / 39 % (VRD-Trans) truy vấn có đáp án đứng trên mọi thực thể ảnh khác; MRR trong ảnh 0,45–0,50 so với MRR chính 0,26–0,36 |
    | C6 | Cách gộp RGB với depth không quan trọng; gate không hơn trọng số cố định | Ablation concat / mean / fixed / gate | Có (1 seed) |

  - **Cấu trúc đồ thị (25/11/2026, tính trên cả ba split):**

    | | FB15k-237-IMG | Spatial-VRD |
    |---|---|---|
    | Bộ ba / thực thể có bộ ba | 310.116 / 14.541 | 107.016 / 83.485 (thêm khoảng 10.300 box không có bộ ba, chỉ là ứng viên) |
    | Bậc trung bình | 42,7 | 2,6 |
    | Thành phần liên thông | 6 (lớn nhất 14.529) | 11.375 (lớn nhất 29, trung vị 5) |

    MMSG chỉ cùng định dạng và giao thức với MMKG, không cùng cấu trúc: mỗi thành phần nằm gọn trong một ảnh, không có cạnh
    nào nối hai ảnh. Không viết "MMSG có cấu trúc tương đồng MMKG".
  - **Đóng góp:** (1) giao thức có kiểm soát 2 × 2 (loại đồ thị × inductive / transductive), cùng đặc trưng, cùng đánh giá;
    (2) ba bộ MMSG (Spatial-VRD, Spatial-VRD-Trans, Spatial-VG) làm bộ chẩn đoán, đã phát hành, kèm mô tả cấu trúc và thước
    đo bổ sung; (3) kết quả phân tích C1–C6, kể cả kết quả âm. Không viết "đầu tiên" khi chưa tìm tài liệu có hệ thống; dùng
    "theo hiểu biết của chúng tôi".
  - **Dàn bài:**
    1. Giới thiệu.
    2. Tổng quan: MMKG completion transductive và inductive (CMR, BLP); scene graph completion (RLSV, Wan et al., IJCAI
       2018: thực thể là loại vật, ứng viên trong cùng ảnh, có toạ độ box tương đối); SGG có depth (Sharifzadeh, ADVRD);
       KG có thực thể thị giác (ImageGraph, Oñoro-Rubio et al.).
    3. Benchmark và giao thức: ma trận 2 × 2, cách dựng MMSG, bảng cấu trúc, thước đo.
    4. Mô hình: mô hình dựa trên đặc trưng và các biến thể, baseline, cách thêm depth ở tầng dữ liệu.
    5. Kết quả theo C1 → C6.
    6. Phân tích: Δdepth theo nhóm quan hệ (GEOMETRIC / POSSESSIVE / SEMANTIC / MISC) và theo từng quan hệ.
    7. Giới hạn: depth đơn mắt; không làm 2 × 2 trên VG; luật hội tụ chỉ chạy lại một lần.
  - **Thí nghiệm cần thêm:**
    - **E1. Thước đo phụ trên MMSG:** xếp hạng chỉ trong cùng ảnh (id ảnh lấy từ tên thực thể, ví dụ `5073960523_0`) và
      đoán quan hệ (h, ?, t) trên 52 quan hệ. Thiết kế (duyệt 25/11/2026):
      `docs/superpowers/specs/2026-11-25-e1-mmsg-aux-metrics-design.md`. Không có checkpoint để dùng lại (lần `--tune`
      không lưu), nên **chạy lại 8 cấu hình 2 × 2 đã chọn bằng `--tune`, chỉ valid** (cách A: việc chọn mô hình chính
      không nhìn test). Code: `evaluate_link_prediction(entity_image=...)`, `train.py` tự bật trên hai bộ VRD; ô
      notebook "E1" (`geometry-e1/<dataset>/...`); bảng `scripts/e1_summary.py` (kèm độ lệch MRR so với P2.10).
      - **Kết quả (xong 26/11/2026; valid, seed 42, epoch tốt nhất theo MRR chính).** Cả 8 lần chạy lặp lại đúng MRR valid
        và epoch tốt nhất của P2.10 (lệch 0,0000); `n_negative_gap` = 0 ở mọi lần.

        | | `no_depth` | `depth_map_only` | `depth_stats_only` | `fixed` |
        |---|---|---|---|---|
        | **Spatial-VRD** MRR chính | 0,2560 | 0,2566 | 0,2670 | 0,2650 |
        | MRR trong ảnh / H@1 | 0,4484 / 0,242 | 0,4581 / 0,255 | 0,4555 / 0,251 | 0,4670 / 0,265 |
        | `image_first` | 0,261 | 0,258 | 0,270 | 0,263 |
        | Đoán quan hệ MRR / H@1 | 0,5082 / 0,368 | 0,4735 / 0,340 | 0,5140 / 0,381 | 0,4816 / 0,355 |
        | **VRD-Trans** MRR chính | 0,3503 | 0,3464 | 0,3636 | 0,3611 |
        | MRR trong ảnh / H@1 | 0,4924 / 0,287 | 0,4922 / 0,286 | 0,5003 / 0,295 | 0,5009 / 0,296 |
        | `image_first` | 0,386 | 0,380 | 0,406 | 0,398 |
        | Đoán quan hệ MRR / H@1 | 0,4648 / 0,299 | 0,4609 / 0,308 | 0,4606 / 0,295 | 0,4448 / 0,290 |

        Trung bình 19,0 (VRD) / 17,8 (VRD-Trans) ứng viên mỗi truy vấn trong ảnh; MRR ngẫu nhiên 0,204 / 0,214.
      - **Đọc kết quả:**
        - Xếp hạng toàn cục chủ yếu đo việc tìm đúng ảnh: chỉ 26 % / 39 % truy vấn không có thực thể ảnh khác nào vượt
          đáp án; MRR trong ảnh gấp 1,7 (VRD) / 1,4 (VRD-Trans) lần MRR chính (C5).
        - `depth_stats` giúp ở cả hai cách đo: trong ảnh +0,007 / +0,009 (VRD), +0,008 / +0,009 (VRD-Trans), nhỏ hơn ở
          MRR chính (+0,008 đến +0,015). Phần chênh đến từ việc tìm đúng ảnh: `image_first` tăng +0,005 đến +0,020.
        - Vector depth: trong ảnh +0,010 / +0,012 trên VRD nhưng làm `image_first` giảm nhẹ (−0,003 / −0,007), nên ở MRR
          chính hai phần triệt tiêu; trên VRD-Trans ≈ 0 trong ảnh. Chưa nhất quán giữa hai bộ, 1 seed: chờ E3.
        - Đoán quan hệ: vector depth làm giảm (−0,035 / −0,032 trên VRD, −0,004 / −0,016 trên VRD-Trans); `depth_stats`
          +0,006 / +0,008 trên VRD, −0,004 / −0,016 trên VRD-Trans. Mô hình được huấn luyện và chọn theo xếp hạng thực
          thể, không theo đoán quan hệ, nên đây chỉ là số phụ.
        - Thực thể đã biết vượt đáp án ở 56–63 % truy vấn (mọi biến thể): f(h) + r nằm quá gần f(h), điểm yếu của hàm
          điểm kiểu TransE khi mỗi box là một thực thể. Bỏ nó khỏi ứng viên, MRR trong ảnh tăng khoảng 0,09. Đưa vào phần
          thảo luận; giao thức không đổi.
        - Chọn mô hình chính: E1 chưa tách được `fixed` và `depth_stats_only` (`fixed` hơn trong ảnh trên VRD;
          `depth_stats_only` hơn ở MRR chính và đoán quan hệ, và đơn giản hơn). Chờ E2.
    - **E2. Đối chứng bằng box:** `box_stats_only` và box + `depth_stats`, cùng cơ chế với `depth_stats_only` (cộng
      W_b b, khởi tạo 0, cùng chuẩn hoá); b = hình học box chuẩn hoá theo ảnh, lấy từ `sg_*_annotations.json`. Tune đủ trên
      Spatial-VRD và VRD-Trans. Thiết kế chi tiết duyệt riêng trước khi code.
      - **Thiết kế (duyệt 26/11/2026):** `docs/superpowers/specs/2026-11-26-e2-box-control-design.md`. Tầng dữ liệu: biến
        thể `depth_stats_only` có sẵn nhận `--depth-stats <bảng>:box` hoặc `:box_grid9`, mô hình không đổi (cùng lớp
        tuyến tính khởi tạo 0, chuẩn hoá, mặt nạ, depth dropout 0,3). `box` = 9 số mỗi thực thể (x1, y1, x2, y2, w, h, tâm
        x, tâm y, log diện tích, chia kích thước ảnh, cắt box như `extract_features.py`); `box_grid9` = box ghép grid9 của
        dữ liệu (18 số). Bảng dựng bằng `scripts/box_geometry.py` từ annotation Spatial-VRD (VRD-Trans cùng tên thực thể);
        ảnh `9210906893` ghi sai kích thước trong annotation (500 × 374, file thật 683 × 1024), sửa theo file ảnh. Bản dựng
        ở máy: 93.807 / 93.809 thực thể có box (2 box nằm ngoài ảnh), tổng cột ghi trong `EXPECTED` của script; bản dựng
        trên Molab phải khớp. Ô notebook "E2" tune đủ như P2.10 vào `geometry-e2/<dataset>/<nhánh>/depth_stats_only/`;
        bảng `scripts/e2_summary.py` (so với `no_depth` và `depth_stats_only` của E1). Cách đọc kết quả đặt trước ở mục 6
        của thiết kế.
      - **Kết quả (xong 26/11/2026; valid, seed 42, code `d8e426f`).** 30 lần chạy (VRD: `box` 8, `box_grid9` 7;
        VRD-Trans: 7, 8), tất cả `--tune`, không đọc test; bảng box trên Molab có 93.807 thực thể có box như bản dựng ở
        máy. Cả 4 nhánh hội tụ, không còn điểm biên; `box_grid9` trên VRD-Trans nhân đôi ngân sách (epoch tốt nhất 170 /
        200).

        | | `no_depth` | `depth_stats_only` (grid9) | `box` | `box_grid9` |
        |---|---|---|---|---|
        | **Spatial-VRD** cấu hình | 3e-4 / 0 | 1e-4 / 0 | 1e-4 / 0 | 3e-4 / 0 |
        | MRR chính | 0,2560 | 0,2670 | 0,2853 | 0,2905 |
        | MRR trong ảnh / H@1 | 0,4484 / 0,242 | 0,4555 / 0,251 | 0,4761 / 0,269 | 0,4788 / 0,273 |
        | `image_first` | 0,261 | 0,270 | 0,290 | 0,296 |
        | Đoán quan hệ MRR / H@1 | 0,5082 / 0,368 | 0,5140 / 0,381 | 0,5093 / 0,383 | 0,4926 / 0,362 |
        | **VRD-Trans** cấu hình | 1e-3 / 0 | 3e-4 / 0, 200 ep | 3e-4 / 0 | 3e-4 / 0, 200 ep |
        | MRR chính | 0,3503 | 0,3636 | 0,3789 | 0,3894 |
        | MRR trong ảnh / H@1 | 0,4924 / 0,287 | 0,5003 / 0,295 | 0,5275 / 0,329 | 0,5241 / 0,324 |
        | `image_first` | 0,386 | 0,406 | 0,408 | 0,430 |
        | Đoán quan hệ MRR / H@1 | 0,4648 / 0,299 | 0,4606 / 0,295 | 0,4709 / 0,330 | 0,4456 / 0,301 |

      - **Đọc kết quả (theo mục 6 của thiết kế):**
        - `box` − `depth_stats_only`: +0,018 (VRD) / +0,015 (VRD-Trans) ở MRR chính, +0,021 / +0,027 trong ảnh. Hình học
          2D mạnh hơn grid9 ở mọi thước đo trừ đoán quan hệ trên VRD (−0,005).
        - `box_grid9` − `box`: +0,005 / +0,010, vượt ngưỡng 0,003 trên cả hai bộ → luật thứ hai: depth thêm thông tin ngoài
          box, **C4 = có, nhỏ**. Trên VRD mức tăng sát ngưỡng, kết luận cuối chờ E3.
        - Phần thêm đó nằm ở `image_first` (+0,006 / +0,022), không ở xếp hạng trong ảnh (+0,003 / −0,003), và đoán quan hệ
          giảm (−0,017 / −0,025): grid9 giúp tìm đúng ảnh hơn là hiểu quan hệ trong ảnh (C5).
        - Giới hạn (mục 7 của thiết kế): box là box gán tay, và nhiều quan hệ VRD (left of, above, …) được gán từ chính vị
          trí box, nên mức tăng của box có thể một phần là học lại luật gán nhãn. E6 tách nhóm GEOMETRIC để đo phần này.
      - **Quyết định (26/11/2026, người dùng duyệt):**
        - **Không đưa box vào mô hình chính** (giữ nguyên tắc 19/11/2026). `box` và `box_grid9` là dòng tham chiếu "box gán
          tay (oracle)" trong bảng MMSG, chạy cùng E3. Lý do: nhãn quan hệ sinh từ box; box gán tay chính xác còn depth là
          ước lượng; box chỉ có ở 3/6 bộ và baseline không nhận box; đổi mô hình chính sau khi xem valid làm yếu lập luận
          "đóng băng trước khi tune".
        - **Mô hình chính trên MMSG: `depth_stats_only`** (MRR chính cao hơn `fixed` trên cả hai bộ ở P2.10 / E1, đoán quan
          hệ tốt hơn, đơn giản hơn; E2 cho thấy phần depth có ích nằm ở grid9). Trên MMKG không có `depth_stats` (P2.9),
          nên biến thể có depth ở đó vẫn là `fixed` (vector depth).
    - **E2b. Lợi ích của `depth_stats` có cần thang độ sâu chung không** (cho C2 / C3). Trên MMKG chỉ thử được vector
      depth, mà vector depth vô dụng ngay trên MMSG, nên C2 rỗng có thể chỉ do cách biểu diễn. E2b giữ grid9, chỉ đổi khung
      chuẩn hoá depth từ ảnh sang **từng box** (mỗi vật một thang riêng, như thực thể MMKG), chạy `depth_stats_only` như E2
      trên hai bộ VRD, chỉ valid.
      - **Thiết kế (duyệt 26/11/2026):** `docs/superpowers/specs/2026-11-26-e2b-shared-scale-design.md`; cách đọc kết quả
        đặt trước ở mục 6 (ngưỡng 0,003; `boxnorm` − `no_depth` ≤ 0,003 → cần thang chung; `depth_stats_only` − `boxnorm`
        ≤ 0,003 → không cần, C2 phải viết lại).
      - **Bảng (dựng ở máy 26/11/2026, `scripts/depth_stats_boxnorm.py`):** cùng lượt ZipDepth tính lại grid9 theo ảnh,
        khớp `ent_depth_stats.pt` của dữ liệu tuyệt đối (sai lệch 0) ở cả hai bộ. 93.807 box, 9 box có depth hằng (9 số 0),
        2 box ngoài ảnh. sha256 `depth_stats_boxnorm.pt`: VRD `894bc531…5c990`, VRD-Trans `94caf01b…564bc` (đủ trong
        hằng `E2B_SHA256` của notebook). Kiểm tra nhanh trên bộ ba train: tỉ lệ vật đầu gần hơn vật đuôi (trung bình 9
        số) với "in front of" 0,710 theo ảnh → 0,512 theo box, "behind" 0,292 → 0,439, "under" 0,827 → 0,543: thông tin
        gần / xa giữa hai vật gần như mất, đúng như thiết kế.
      - Ô notebook "E2b" tải bảng từ `geometry-e2b/<dataset>/`, kiểm sha256, tune vào
        `geometry-e2b/<dataset>/boxnorm/depth_stats_only/`; bảng `scripts/e2_summary.py --e2b` (thêm dòng `boxnorm`, hai
        dòng hiệu và phần lợi giữ lại).
    - **E3. Chạy chính thức 3 seed trên test:** `no_depth` và `fixed` trên 3 bộ MMKG; trên MMSG `no_depth` và
      `depth_stats_only` (mô hình chính, quyết định ở E2); bảng 2 × 2 (`depth_map_only`, `fixed`) và hai nhánh box
      (`box`, `box_grid9`, cấu hình chọn ở E2) trên 2 bộ VRD.
    - **E4. Kiểm định:** bootstrap ghép cặp theo truy vấn trong từng seed, kiểm định ghép cặp giữa các seed.
    - **E5. Baseline** theo P3 / P4, có bản +D. Không cấp box cho baseline (box chỉ là đối chứng, E2). Còn quyết định: có
      cấp thêm `depth_stats` ở tầng dữ liệu không.
    - **E6. Phân tích theo nhóm quan hệ** (P5). Tách riêng nhóm GEOMETRIC cho hai nhánh box của E3: nếu mức tăng của box so
      với `depth_stats_only` tập trung ở nhóm này, đó là phần học lại luật gán nhãn từ vị trí box (giới hạn của E2).
  - **Cắt bỏ:** MoMoK trên hai bộ VRD (khoảng 52 giờ mỗi lần); 2 × 2 và đối chứng box trên Spatial-VG; gate chỉ là
    ablation, không phải đóng góp; phần LLM và việc gộp MMKG với MMSG chỉ ở mục hướng phát triển.
  - **Thứ tự:** E1 → E2 → chọn mô hình chính → E3 + E4 → E5 → E6 → viết bài.
  - **Quyết định còn mở:** baseline có nhận `depth_stats` ở tầng dữ liệu không. (Mô hình chính đã chọn ở E2:
    `depth_stats_only` trên MMSG; box không vào mô hình chính và không cấp cho baseline.)
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
