# E2: đối chứng bằng box — thiết kế

Ngày: 26/11/2026. Thuộc RESEARCH_PLAN P6.0, khẳng định C4 (và cách viết C3). Trạng thái: đã duyệt 26/11/2026 (cách A,
tầng dữ liệu).

## 1. Mục đích

Trên hai bộ VRD, phần lợi của depth đến từ `depth_stats` (grid9 = trung bình độ sâu của 3 × 3 vùng con trong box, depth
map chuẩn hoá theo ảnh), không đến từ vector depth (P2.10, E1). Câu hỏi còn lại (C4): lợi ích đó là thông tin **độ
sâu**, hay chỉ là thông tin **vị trí box** mà grid9 mang theo (vùng con nào gần / xa phụ thuộc chỗ box nằm trong ảnh)?
Probe P2.9 gợi ý là trùng: box + `dstat` so với box một mình −0,013 đến +0,004. E2 trả lời câu này trên bài toán xếp hạng
của KG, cùng cơ chế với `depth_stats_only`.

Box chỉ là **đối chứng**. Nguyên tắc 19/11/2026 (không dùng toạ độ box làm đặc trưng của mô hình chính) giữ nguyên.

## 2. Mục tiêu, không làm, phải tránh

**Mục tiêu**
- Hai nhánh mới trên Spatial-VRD và Spatial-VRD-Trans, cùng mô hình `depth_stats_only`, chỉ khác bảng số đưa vào:
  - `box`: 9 số hình học box của mỗi thực thể;
  - `box_grid9`: 9 số box ghép với 9 số grid9 của dữ liệu (18 số).
- Tune đủ như P2.10 (cùng lưới gốc, luật biên, luật hội tụ), chỉ valid, seed 42.
- Bảng so sánh 4 ô mỗi bộ: `no_depth`, `depth_stats_only` (grid9), `box`, `box_grid9`. Báo MRR valid chính và thước đo
  phụ E1 (trong ảnh, `image_first`, đoán quan hệ).

**Không làm**
- Không sửa mô hình, fusion, Feature-TransE, `load_depth_stats`, cách chọn checkpoint.
- Không chạy box với vector depth (`fixed` + box), không chạy trên Spatial-VG, MMKG hay baseline.
- Không đổi dữ liệu v0 trên HF (không thêm file vào repo dataset).

**Phải tránh**
- Đọc test (mọi lần chạy là `--tune`).
- Ghi đè kết quả `geometry-tune/` hoặc `geometry-e1/`.
- Đọc kết quả rồi mới đặt tiêu chí: tiêu chí đọc kết quả ghi sẵn ở mục 6.

## 3. Ràng buộc

- Không sửa `baselines/`. Lần chạy dài trên Molab do người dùng bấm. Code, comment, commit message bằng tiếng Anh. Chỉ
  commit / push khi người dùng bảo, liệt kê thay đổi trước.

## 4. Cách tiếp cận

- **A (chọn): tầng dữ liệu.** Dựng bảng số theo đúng định dạng `PATH:KEY` mà `--depth-stats` đã nhận, chạy biến thể
  `depth_stats_only` có sẵn. Mô hình không đổi nên "cùng cơ chế" đúng theo cách dựng: cùng lớp tuyến tính khởi tạo 0, cùng
  chuẩn hoá theo thực thể train có ảnh, cùng mặt nạ thiếu ảnh, cùng depth dropout 0,3 (bỏ cả số hạng cho 30 % thực thể khi
  train, như grid9). Với bảng ghép, W·[b; s] = W_b b + W_s s. Trong TransE số hạng này thành W(b_h − b_t): hình học
  tương đối của hai box, gần với RLSV.
- B (bỏ): biến thể mới trong mô hình với đầu vào riêng. Tên rõ hơn nhưng phải sửa fusion, train, Feature-TransE, và tính
  "cùng cơ chế" phải chứng minh bằng test.

## 5. Thiết kế

### 5.1 Bảng box (`scripts/box_geometry.py`)

- Đầu vào: `entity2id.txt` của dataset, `sg_train_annotations.json` và `sg_test_annotations.json` của Spatial-VRD
  (VRD-Trans không có annotation nhưng `entity2id.txt` trùng hệ với VRD, như `depth_stats.py fill` đã kiểm), và
  `ent_depth_stats.pt` của dataset (cho bảng ghép).
- Thực thể `<photo_id>_<k>` = vật thứ k của ảnh, cùng quy tắc với `extract_features.vrd_scenes`.
- Box: lấy `bbox` (x, y, w, h) của annotation, kích thước ảnh W, H từ annotation, trừ một ảnh có kích thước ghi sai
  (`9210906893`: annotation 500 × 374, file ảnh 683 × 1024, box tới y = 981; kiểm cả 5.000 ảnh ngày 26/11/2026, chỉ
  ảnh này lệch), sửa bằng hằng `IMAGE_SIZE_FIXES` cho khớp `extract_features.py` (dùng kích thước file ảnh). Cắt như `extract_features.py`: box nằm
  hẳn ngoài ảnh → NaN; còn lại kẹp vào [0, W] × [0, H], cạnh nhỏ hơn `MIN_BOX` = 2 px thì nới quanh tâm. Chia cho W, H.
- 9 số mỗi thực thể (như probe P2.9): x1, y1, x2, y2, w, h, tâm x, tâm y, log(w·h). Có phụ thuộc tuyến tính (w = x2 − x1, …);
  không sao với một lớp tuyến tính, giữ để khớp probe.
- Đầu ra: một file `box_geometry.pt` = `{"box": [N, 9], "box_grid9": [N, 18], "model": "box", "source": …}`, float32,
  NaN ở thực thể không có box. `box_grid9` = `box` ghép `ent_depth_stats.pt` (cột 10–18 trùng hệt bảng grid9 của dữ liệu).
- In và ghi kèm (`box_geometry.json`) số thực thể có box, số box bị NaN, và tổng từng cột (float64, bỏ NaN). Không
  dùng sha256: `log` có thể lệch một ULP giữa hai CPU. Tổng cột dựng ở máy ghi vào script (hằng `EXPECTED`) và
  RESEARCH_PLAN; bản dựng trên Molab phải khớp (sai số tương đối 1e-6, số đếm khớp tuyệt đối).
- Thực thể không có ảnh (`ent_img_mask.pt` False, 2 box ở VRD) bị `load_depth_stats` đặt 0 như grid9.

### 5.2 Chạy

- `train.py --tune --variant depth_stats_only --depth-stats <file>:box` (hoặc `:box_grid9`). Tag tự sinh:
  `inorm-ddrop0.3-tproj-dstat_box_geometry_box` / `…_box_grid9`, nên stem khác hẳn các lần chạy cũ.
- Thư mục: `geometry-e2/<dataset>/<nhánh>/depth_stats_only/<cấu hình>/` (local `results/` và HF). Mỗi nhánh một gốc
  riêng, nên `scripts/tune_summary.py --root geometry-e2/<dataset>/<nhánh>` dùng được nguyên (nó gom theo tên biến thể,
  hai nhánh cùng tên `depth_stats_only`).
- Thước đo phụ E1 tự bật (hai bộ VRD).

### 5.3 Notebook (`notebooks/geometry.py`, ô "E2")

- Tải `sg_*_annotations.json` từ repo dataset `callmezuy/spatial-vrd` (tag v0) cho cả hai bộ, dựng `box_geometry.pt`
  vào `WORK_DIR/e2/<dataset>/`, kiểm số đếm và tổng cột với hằng `EXPECTED` trong script, sai thì dừng.
- Chọn nhánh (`box`, `box_grid9`, mặc định cả hai). Chạy lưới gốc rồi lặp luật biên / hội tụ như ô tune (tối đa 6 vòng),
  bỏ qua điểm đã có trên HF, đẩy kết quả và trạng thái resume như ô tune.
- Ước lượng: mỗi nhánh mỗi bộ 6 điểm gốc + 2–4 điểm biên / hội tụ; tổng khoảng 36–40 lần, 5–6 giờ GPU.

### 5.4 Tổng hợp (`scripts/e2_summary.py`)

- Mỗi bộ một bảng 4 dòng: `no_depth` và `depth_stats_only` lấy từ `geometry-e1/` (cấu hình chọn ở P2.10, đã có thước đo
  phụ), `box` và `box_grid9` lấy cấu hình chọn bởi `tune_summary.choose` trên `geometry-e2/`.
- Cột: cấu hình, epoch tốt nhất, MRR valid chính, MRR / H@1 trong ảnh, `image_first`, MRR / H@1 đoán quan hệ, cờ "chưa hội
  tụ sau khi nhân đôi" nếu có.
- Dòng hiệu: `box_grid9 − box` (depth thêm gì ngoài box) và `box − depth_stats_only`.

### 5.5 Test

1. `box_geometry`: annotation dựng tay (2 ảnh), kiểm 9 số tính tay, box tràn mép bị kẹp, box nhỏ được nới, box ngoài ảnh
   → NaN, tên không có trong `entity2id` bị bỏ, `box_grid9[:, 9:]` = bảng grid9.
2. Dataset có `entity2id` khác VRD (thiếu thực thể) vẫn dựng đúng theo tên.
3. `train.py` chạy nhỏ với `--depth-stats <file>:box` trên dữ liệu giả `Spatial-VRD`: tag và `model_kwargs.depth_stats_dim`
   = 9 (18 với `box_grid9`), metrics có `in_image`.
4. `e2_summary`: thư mục giả, kiểm bảng và dòng hiệu; nhánh thiếu thì ghi "missing".

## 6. Cách đọc kết quả (đặt trước khi chạy)

Một seed, MRR valid; chênh dưới khoảng 0,003 coi như bằng (như P2.10). Kết luận cuối chờ E3 (3 seed, có kiểm định).
- `box_grid9` ≈ `box` và `box` ≥ `depth_stats_only` − 0,003: lợi ích của `depth_stats` trùng với hình học box 2D. C3
  viết lại: "phần lợi trên MMSG đến từ hình học tương đối của vật, depth đơn mắt không thêm gì ngoài box". C4 = không.
- `box_grid9` > `box` + 0,003 trên cả hai bộ: depth thêm thông tin ngoài box. C4 = có, kèm độ lớn.
- `box` < `depth_stats_only` − 0,003: grid9 mang thứ box không có; xét thêm `box_grid9`.
- Hai bộ cho kết luận khác nhau: báo là chưa nhất quán, chờ E3.
- Đọc thêm `image_first` và MRR trong ảnh: box có thể giúp tìm đúng ảnh nhiều hơn grid9 (vị trí tuyệt đối khác nhau giữa ảnh).

## 7. Giới hạn (ghi vào bài)

- Box của annotation là box vẽ tay, chính xác; depth là ước lượng đơn mắt.
- Nhiều quan hệ VRD (left of, above, …) được gán từ chính vị trí box, nên box có lợi thế sẵn ở các quan hệ đó; phân tích
  theo nhóm quan hệ (E6) cần tách nhóm GEOMETRIC.
- Box chỉ có trên scene graph; baseline KGC không nhận box.

## 8. Sau khi xong

RESEARCH_PLAN: dòng E2 ở P6.0 (kết quả, sha256 bảng box), dòng C3 / C4, và "quyết định còn mở" (mô hình chính).

## 9. Để người viết code tự quyết

Tên hàm phụ, cách chia ô notebook (một ô chạy hay tách nút cho mỗi nhánh), định dạng in, có dùng lại hàm của
`depth_probe.load_boxes` hay viết lại (phải thêm phần kẹp / nới box nên nhiều khả năng viết lại).
