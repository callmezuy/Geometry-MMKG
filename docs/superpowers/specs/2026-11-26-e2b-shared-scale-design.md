# E2b: lợi ích của `depth_stats` có cần thang độ sâu chung không — thiết kế

Ngày: 26/11/2026. Thuộc RESEARCH_PLAN P6.0, cho C2 và C3. Trạng thái: đã duyệt 26/11/2026 (cách A).

## 1. Mục đích

Trên MMKG chỉ có một đường đưa depth vào là vector depth. Nhưng vector depth đã vô dụng ngay trên MMSG (bảng 2 × 2, P2.10),
nên kết quả rỗng trên MMKG (C2) có thể chỉ do cách biểu diễn kém. Luận điểm thay thế: depth chỉ giúp dưới dạng **độ sâu
tương đối giữa các vật cùng cảnh** (grid9 trên depth map chuẩn hoá theo **ảnh**, nên hai vật cùng ảnh so được với nhau),
còn MMKG không có thang chung đó theo cách dựng dữ liệu (mỗi thực thể một ảnh riêng, depth map chuẩn hoá theo ảnh của
chính nó).

E2b kiểm tra vế "cần thang chung" trên MMSG: giữ nguyên grid9, chỉ đổi khung chuẩn hoá từ ảnh sang **từng box**. Mỗi vật
khi đó có thang riêng, giống tình huống của thực thể MMKG. Nếu mức tăng của `depth_stats_only` mất đi, lợi ích cần thang
chung và C2 có cơ sở cơ chế. Nếu còn, luận điểm sai và phải viết lại C2.

## 2. Mục tiêu, không làm, phải tránh

**Mục tiêu**
- Một nhánh mới `boxnorm` trên Spatial-VRD và Spatial-VRD-Trans: biến thể `depth_stats_only` có sẵn, chỉ khác bảng 9 số.
- Tune đủ như P2.10 / E2 (cùng lưới gốc, luật biên, luật hội tụ), chỉ valid, seed 42.
- Bảng 3 ô mỗi bộ: `no_depth`, `depth_stats_only` (grid9 chuẩn hoá theo ảnh, từ E1), `boxnorm`; MRR chính và thước đo phụ
  E1.

**Không làm**
- Không sửa mô hình, `load_depth_stats`, `scripts/depth_stats.py` (pipeline của dữ liệu v0), cách chọn checkpoint.
- Không đổi dữ liệu v0 trên HF.
- Không chạy grid9 trên ảnh MMKG (chỉ cân nhắc nếu kết quả rơi vào nhánh "không cần thang chung", mục 6).
- Không thêm nhánh "chỉ mức sâu trung bình của box" hay `boxnorm` + box.

**Phải tránh**
- Đọc test. Ghi đè `geometry-tune/`, `geometry-e1/`, `geometry-e2/`. Đặt tiêu chí sau khi xem kết quả (mục 6 ghi trước).

## 3. Ràng buộc

Không sửa `baselines/`. Lần chạy dài trên Molab do người dùng bấm. Code, comment, commit message bằng tiếng Anh. Chỉ
commit / push / upload khi người dùng bảo, liệt kê thay đổi trước.

## 4. Cách tiếp cận

- **A (chọn): chuẩn hoá min-max theo box trên cùng depth map.** Cùng ZipDepth, cùng ảnh 384 px, cùng box (cắt / nới như
  `extract_features.py`), cùng 32 × 32 mẫu `roi_align`, cùng 3 × 3 trung bình. Khác đúng một chỗ: trước khi lấy trung bình
  ô, các mẫu của mỗi box được đưa về [0, 1] bằng min / max **của box đó** thay vì của ảnh. Chỉ đổi khung chuẩn hoá, nên mọi
  chênh lệch so với `depth_stats_only` là của thang chung. Min-max cũng chính là cách depth map của thực thể MMKG được
  chuẩn hoá (theo ảnh của riêng nó).
- B (bỏ): chạy ZipDepth trên mảnh cắt của box (giống MMKG hơn về ngữ cảnh ước lượng depth). Đổi hai thứ một lúc (ngữ
  cảnh của mô hình depth và khung chuẩn hoá), nên không tách được nguyên nhân.
- C (bỏ): chỉ trừ trung bình của box (giữ biên độ). Còn giữ độ tương phản theo thang của ảnh, nên chưa phải "thang riêng".

## 5. Thiết kế

### 5.1 Bảng (`scripts/depth_stats_boxnorm.py`, mới)

- Dùng lại `depth_stats.DepthSceneSet`, `ef.vrd_scenes`, `ef.load_zipdepth`, `fill_table`; không sửa `depth_stats.py`.
- Một lượt ZipDepth cho cả hai cách chuẩn hoá:
  - `image`: đúng `depth_stats.grid_means` trên depth map chuẩn hoá theo ảnh (để kiểm tra);
  - `boxnorm`: `roi_align` như trên, rồi mỗi box (g − min) / max(max − min, 1e-6) trên 32 × 32 mẫu của nó, rồi 3 × 3 trung
    bình. Box có depth gần như hằng (max − min < 1e-6) cho 9 số 0; đếm và ghi số box này.
- Kiểm tra tái lập: bảng `image` phải khớp `ent_depth_stats.pt` của dữ liệu ở mọi thực thể có ảnh, sai lệch tuyệt đối tối
  đa ≤ 1e-3 (fp16 trên GPU); in sai lệch thật. Sai thì dừng, không ghi file.
- VRD-Trans: cùng `entity2id` với VRD (như `depth_stats.py fill`), lấy bảng thô của VRD, điền thực thể thiếu ảnh theo
  mặt nạ / train của VRD-Trans (`fill_table`). Kiểm tra `image` với `ent_depth_stats.pt` của VRD-Trans.
- Đầu ra mỗi bộ: `depth_stats_boxnorm.pt` = `{"grid9": [N, 9] (boxnorm), "grid9_image": [N, 9], "model": "zipdepth",
  "norm": "box", "source": …}`, float32, kèm `depth_stats_boxnorm.json` (số ảnh, số box, số box hằng, sai lệch kiểm tra,
  sha256 của file `.pt`, commit code).
- Dựng ở máy (RTX 3050, khoảng 1 phút mỗi bộ). Đưa lên repo kết quả private `geometry-mmkg-results/geometry-e2b/<dataset>/`
  (upload khi người dùng duyệt). Notebook tải về và kiểm sha256 ghi trong hằng của notebook; sai thì dừng.

### 5.2 Chạy

- `train.py --tune --variant depth_stats_only --depth-stats <file>:grid9`. Tag tự sinh
  `inorm-ddrop0.3-tproj-dstat_boxnorm_grid9`, khác các lần chạy cũ.
- Thư mục: `geometry-e2b/<dataset>/boxnorm/depth_stats_only/<cấu hình>/`. `tune_summary.py --root
  geometry-e2b/<dataset>/boxnorm` dùng nguyên.
- Thước đo phụ E1 tự bật.

### 5.3 Notebook (`notebooks/geometry.py`, ô "E2b")

- Tải bảng của bộ đang chọn từ repo kết quả, kiểm sha256, rồi tune như ô E2 (lưới gốc 6 điểm, luật biên / hội tụ tối đa 6
  vòng, bỏ qua điểm đã có, đẩy kết quả và trạng thái resume). Nên dùng chung hàm với ô E2 thay vì chép.
- Ước lượng: mỗi bộ 7–8 lần, tổng khoảng 15 lần, 2–3 giờ GPU.

### 5.4 Tổng hợp

- `scripts/e2_summary.py` thêm `--e2b <root>` (tuỳ chọn): khi có, bảng thêm dòng `boxnorm` (cấu hình chọn bởi
  `tune_summary.choose`) và dòng hiệu `boxnorm` − `no_depth`, `depth_stats_only` − `boxnorm`. Không có `--e2b` thì bảng
  như cũ. Thêm cột (hoặc dòng chú thích) "phần lợi giữ lại" = (`boxnorm` − `no_depth`) / (`depth_stats_only` − `no_depth`)
  theo MRR chính.

### 5.5 Test

1. Hàm chuẩn hoá theo box: depth giả với hai box; biến đổi a·d + b (a > 0) riêng trong một box không đổi 9 số của box đó;
   box hằng cho 0; nhánh `image` trùng `depth_stats.grid_means`.
2. Kiểm tra tái lập: bảng giả lệch 2e-3 thì báo lỗi, lệch 1e-4 thì qua.
3. `train.py` chạy nhỏ với `--depth-stats <file>:grid9` trên dữ liệu giả: tag kết thúc bằng `dstat_boxnorm_grid9`,
   `depth_stats_dim` 9, có `in_image`, không có test.
4. `e2_summary --e2b`: thư mục giả, có dòng `boxnorm` và hai dòng hiệu; không có `--e2b` thì bảng giống hệt bản cũ.

## 6. Cách đọc kết quả (đặt trước khi chạy)

Một seed, MRR valid, ngưỡng nhiễu 0,003 (như P2.10). Mức tăng của grid9 chuẩn hoá theo ảnh: +0,0110 (VRD), +0,0133
(VRD-Trans).
- `boxnorm` − `no_depth` ≤ 0,003 trên cả hai bộ: lợi ích **cần thang chung**. C2 / C3 viết thành một luận điểm: depth như
  đặc trưng ảnh của từng thực thể không giúp ở loại đồ thị nào; depth chỉ giúp dưới dạng độ sâu tương đối trong cùng cảnh,
  điều MMKG không có theo cách dựng.
- `depth_stats_only` − `boxnorm` ≤ 0,003 trên cả hai bộ: **không cần thang chung**; lợi ích là hình dạng depth bên trong
  box. Luận điểm trên sai; C2 phải viết lại, và cân nhắc grid9 trên ảnh MMKG (thiết kế riêng).
- Hai hiệu đều > 0,003: một phần. Báo "phần lợi giữ lại" và viết C2 / C3 theo tỉ lệ đó.
- Hai bộ cho kết luận khác nhau: chưa nhất quán, chờ E3.
- Đọc thêm `image_first` và MRR trong ảnh: E2 cho thấy phần grid9 thêm ngoài box nằm ở `image_first`; nếu `boxnorm` mất
  chủ yếu ở `image_first`, thang chung của ảnh đang giúp nhận ra ảnh (ghi vào C5).

## 7. Giới hạn (ghi vào bài)

- Min-max theo box cũng kéo giãn box có depth gần hằng (vật phẳng, xa); box hằng hoàn toàn thành 0.
- Kiểm tra trên MMSG, suy ra cho MMKG: lập luận dựa trên việc thực thể MMKG có thang riêng, không phải thí nghiệm trên
  MMKG.

## 8. Sau khi xong

RESEARCH_PLAN P6.0: dòng E2b (kết quả, sha256 hai bảng), C2 / C3 theo mục 6; E3 thêm `boxnorm` (3 seed) trên hai bộ VRD
nếu kết quả được dùng trong bài.

## 9. Để người viết code tự quyết

Tên hàm phụ, cách dùng chung code giữa ô E2 và E2b, định dạng in, vị trí hằng sha256.
