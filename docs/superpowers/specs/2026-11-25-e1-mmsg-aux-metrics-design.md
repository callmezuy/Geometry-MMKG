# E1: thước đo phụ trên MMSG — thiết kế

Ngày: 25/11/2026. Thuộc RESEARCH_PLAN P6.0, khẳng định C5. Trạng thái: đã duyệt 25/11/2026 (cách A);
bổ sung bản "bỏ thực thể đã biết" (mục 5.2) duyệt 26/11/2026.

## 1. Mục đích

Trên MMSG mỗi box là một thực thể, và thước đo chính xếp hạng trên toàn bộ N thực thể của mọi ảnh (Spatial-VRD:
93.809). MRR chính vì vậy trộn hai việc: tìm đúng ảnh, và chọn đúng vật trong ảnh. E1 thêm hai thước đo theo cách đánh
giá của RLSV (Wan et al., IJCAI 2018) để tách hai việc này, làm bằng chứng cho C5. Thước đo chính (mục 4) giữ nguyên.

Số đo trên dữ liệu (25/11/2026), Spatial-VRD và VRD-Trans như nhau: 4.999 ảnh, trung bình 18,8 box mỗi ảnh (trung vị 18,
tối đa 48); **0** bộ ba nối hai ảnh ở cả 3 split; 5.346 / 101.074 cặp (h, t) có nhiều hơn một quan hệ.

## 2. Mục tiêu, không làm, phải tránh

**Mục tiêu**
- Xếp hạng trong cùng ảnh, filtered, cả head và tail: MRR, H@1, H@3, số ứng viên trung bình mỗi truy vấn.
- Đoán quan hệ (h, ?, t) trên toàn bộ R quan hệ, filtered: MRR, H@1, H@5.
- Cả hai chỉ tính trên MMSG (dataset có id ảnh), vì chỉ C5 dùng tới.
- Có số cho bảng 2 × 2 (`no_depth`, `depth_map_only`, `depth_stats_only`, `fixed`) trên Spatial-VRD và VRD-Trans, chỉ
  trên valid.

**Không làm**
- Không đổi thước đo chính, cách chọn checkpoint, định dạng `ranks_<stem>_test.csv`, `scripts/compute_metrics.py`.
- Không làm Spatial-VG trong E1 (chỉ chừa chỗ).
- Không tính thước đo phụ cho baseline ngoài `train.py` (xem mục 6).
- Không tính thước đo phụ trên MMKG.
- Không hỗ trợ mô hình relation-conditioned (`--rel-gate`, không thuộc thiết kế).

**Phải tránh**
- Nhìn số test trước khi chọn mô hình chính (`fixed` hay `depth_stats_only`).
- Thước đo phụ làm lệch dù một hạng của thước đo chính.
- Ghi đè kết quả tune P2.10 đang có.

## 3. Ràng buộc

- Không sửa code trong `baselines/`.
- Lần chạy dài trên Molab do người dùng tự bấm; không chạy qua execute-code.
- Code, comment, commit message bằng tiếng Anh. Chỉ commit / push khi người dùng bảo, liệt kê thay đổi trước.
- Working tree có thay đổi của phiên khác (RESEARCH_PLAN khoảng dòng 733, `baselines/PROVENANCE.md`,
  `baselines/patches/*.diff`, `scripts/cmr_ranks.py`): không đưa vào commit của E1. Khi sửa RESEARCH_PLAN, chỉ stage
  đúng đoạn của E1.
- Giữ cách xử lý bằng điểm hiện tại: rank = 1 + số ứng viên có điểm tốt hơn hẳn; đáp án đúng và các đáp án đúng khác
  bị loại khỏi phép so sánh (không trừ sau).

## 4. Cách tiếp cận

**Đặt thước đo ở đâu**
- (Chọn) Trong `evaluate_link_prediction`, bật bằng tham số. Dùng lại bảng biểu diễn đã fuse; mỗi truy vấn chỉ thêm
  khoảng 19 ứng viên và R = 52 quan hệ, nên gần như không tốn thêm. Mọi lần đánh giá của `train.py` (valid mỗi lần,
  test ở lần chính thức) đều ghi ra.
- (Bỏ) Script riêng chạy trên checkpoint: không có checkpoint nào để dùng lại (lần `--tune` không lưu, `results/` chỉ
  có `probe`), nên script không tiết kiệm được gì và thêm một đường code đánh giá thứ hai.

**Lấy số cho 2 × 2** (người dùng chọn A, 25/11/2026)
- **A (chọn):** chạy lại 8 cấu hình đã chọn ở P2.10 bằng `--tune` (chỉ valid). Khoảng 1,5–2 giờ GPU. Việc chọn mô hình
  chính chỉ dựa trên valid; kiểm tra thêm được MRR valid có lặp lại số P2.10 không.
- B (bỏ): chạy chính thức seed 42 (có test, dùng lại cho E3, đỡ 8 lần chạy). Bỏ vì phải thấy số test 2 × 2 trước khi
  chọn mô hình chính.

## 5. Thiết kế

### 5.1 Id ảnh của thực thể

`entity_image_ids(data_dir)` trong `core/data_engineering/fast_dataset.py`, trả `LongTensor [N]` (id ảnh đánh số liên
tục 0..n_img−1 theo thứ tự xuất hiện) hoặc `None`.
- Spatial-VRD, Spatial-VRD-Trans: phần trước dấu `_` cuối cùng của tên trong `entity2id.txt`
  (`5073960523_0` → `5073960523`). Tên không có `_` → lỗi.
- Chỉ áp cho danh sách tên dataset cố định (`IMAGE_ID_FROM_NAME = {"Spatial-VRD", "Spatial-VRD-Trans"}`), không đoán
  theo dạng tên: tên thực thể MMKG cũng có thể chứa `_`.
- Spatial-VG: trả `None` kèm thông báo "not implemented (needs VG-SGG.h5 img_to_first_box / img_to_last_box)".
  Làm sau, ngoài E1.
- Bộ MMKG: `None`.
- Khoảng 10.300 box không có bộ ba vẫn có id ảnh và vẫn là ứng viên.

### 5.2 Xếp hạng trong cùng ảnh

Với truy vấn tail (h, r, ?): ứng viên C = các thực thể cùng ảnh với h. Truy vấn head (?, r, t): cùng ảnh với t.

    rank_img = 1 + #{e ∈ C : e ≠ đáp án, e không phải đáp án đúng khác (train ∪ valid ∪ test), s(e) > s(đáp án)}

- Điểm như thước đo chính: s(e) = −‖q − f(e)‖₁ với q = f(h) + r (tail) hoặc f(t) − r (head); điểm của đáp án tính
  giống `_rank_batch` (`true_t`, `true_h`).
- Cài đặt: ma trận đệm ảnh → thực thể `[n_img, M]` (M = số box lớn nhất, 48 trên VRD), gather theo ảnh của từng truy
  vấn, tính khoảng cách `[B, M]`, che ô đệm, đáp án và đáp án đúng khác rồi đếm. Không chạy qua N.
- Nếu một bộ ba được đánh giá có h, t khác ảnh: dừng với lỗi (giả định "mọi đáp án đúng nằm trong cùng ảnh").

Số báo (`result["in_image"]`, gộp head + tail như OpenKE, kèm `head` / `tail` riêng):
- `mrr`, `hits@1`, `hits@3`, `n`;
- `n_cand`: số ứng viên trung bình trước filter (= số box của ảnh, gồm đáp án);
- `n_cand_filtered`: sau khi bỏ đáp án đúng khác (gồm đáp án);
- `mrr_random`: MRR kỳ vọng nếu xếp ngẫu nhiên, trung bình của H(n)/n với n = `n_cand_filtered` của từng truy vấn
  (H là số điều hoà). Làm mức sàn để đọc số.
- `image_first`: tỉ lệ truy vấn không có thực thể ảnh khác nào vượt đáp án, tức `rank − rank_img == 0`. Vì mọi đáp án
  đúng nằm trong ảnh, hạng toàn cục = hạng trong ảnh + số thực thể ảnh khác có điểm cao hơn; con số này đo trực tiếp
  phần "tìm đúng ảnh". Hai đường tính dùng phép cộng số thực khác nhau (cdist toàn cục và gather), nên ở trường hợp gần
  bằng điểm hiệu có thể âm: kẹp về 0 và ghi số lần kẹp vào `n_negative_gap` (kỳ vọng 0).
- **Thực thể đã biết** (h với truy vấn tail, t với truy vấn head) nằm trong ứng viên, như ở thước đo chính, và không bao
  giờ là đáp án đúng (dữ liệu không có bộ ba tự nối). Với TransE, f(h) + r có thể gần f(h), nên h có thể vượt đáp án.
  **Số chính giữ nó làm ứng viên** (cùng quy tắc với thước đo chính, nên phép tách hạng toàn cục = hạng trong ảnh + số
  thực thể ảnh khác vượt đáp án vẫn đúng). Báo thêm:
  - `known_beats_truth`: tỉ lệ truy vấn mà thực thể đã biết vượt đáp án;
  - `excl_known`: cùng thước đo khi bỏ thực thể đã biết khỏi ứng viên (`mrr`, `hits@*`, `mrr_random` với n − 1 ứng
    viên).

### 5.3 Đoán quan hệ

Với mỗi bộ ba (h, r, t): xếp r trong toàn bộ R quan hệ.

    s(r') = −‖f(h) + r' − f(t)‖₁ ;  rank_rel = 1 + #{r' ≠ r, (h, r', t) không đúng : s(r') > s(r)}

- Filter set `{(h, t): {r}}` dựng từ `all_true_triplets` (train ∪ valid ∪ test). Đáp án và quan hệ đúng khác bị loại
  khỏi phép so sánh.
- Tính bằng `cdist(f(h) − f(t), −R_emb)`, `[B, R]`.
- Số báo (`result["relation"]`): `mrr`, `hits@1`, `hits@5`, `mr`, `n`.

### 5.4 Giao diện

`evaluate_link_prediction(..., entity_image=None)`:
- `entity_image=None` (mặc định): hành vi và kết quả như hiện tại, các test cũ không đổi.
- `entity_image` là `LongTensor [N]`: thêm `result["in_image"]` và `result["relation"]`.
- `entity_image` khác `None` với mô hình relation-conditioned: lỗi rõ ràng (không hỗ trợ).
- `return_ranks=True`: `ranks` có thêm `rank_head_img`, `rank_tail_img`, `rank_rel`.
- Thước đo chính tính đúng như cũ; test kiểm tra hạng chính không đổi khi bật `aux`.

### 5.5 `train.py`

- Truyền `entity_image_ids(data_dir)` khi kết quả khác `None` và mô hình không relation-conditioned; ngược lại
  `None` (MMKG, Spatial-VG, `--rel-gate` chạy như cũ, không có thước đo phụ).
- Mỗi lần valid in thêm một dòng thước đo phụ; `history[*]["valid"]`, `best_valid` và `test` mang `in_image` và
  `relation`.
- Lần chính thức trên MMSG ghi thêm `ranks_<stem>_test_aux.csv`, cột
  `h_id,r_id,t_id,rank_head_img,rank_tail_img,rank_rel`, dùng cho bootstrap ở E4. File rank chung không đổi.
- Chọn checkpoint vẫn theo MRR valid chính.
- Feature-TransE đi qua cùng hàm đánh giá nên tự có thước đo phụ.

### 5.6 Lần chạy cho E1 (cách A)

8 lần `--tune`, seed 42, cấu hình đã chọn ở P2.10. Ghi vào thư mục riêng bằng `--out
results/geometry-e1/<dataset>/<variant>/<config>` để không ghi đè kết quả tune P2.10 (notebook bỏ qua điểm lưới đã có
trên HF theo đường dẫn tune).

| Dataset | Biến thể | lr / dropout | Epoch | MRR valid P2.10 |
|---|---|---|---|---|
| Spatial-VRD | `no_depth` | 3e-4 / 0 | 100 | 0,2560 |
| Spatial-VRD | `depth_map_only` | 1e-3 / 0 | 100 | 0,2566 |
| Spatial-VRD | `depth_stats_only` | 1e-4 / 0 | 100 | 0,2670 |
| Spatial-VRD | `fixed` | 1e-3 / 0 | 100 | 0,2650 |
| VRD-Trans | `no_depth` | 1e-3 / 0 | 100 | 0,3503 |
| VRD-Trans | `depth_map_only` | 1e-3 / 0 | 100 | 0,3464 |
| VRD-Trans | `depth_stats_only` | 3e-4 / 0 | 200 (`-ep200`) | 0,3636 |
| VRD-Trans | `fixed` | 3e-4 / 0 | 200 (`-ep200`) | 0,3611 |

- Một ô notebook (mục "E1") chạy 8 lần; người dùng bấm.
- Kiểm tra lặp lại: MRR valid tốt nhất và epoch tốt nhất so với P2.10. Lần lặp `full` trước đây lệch 0,0001; nếu lệch
  quá 0,002 hoặc epoch tốt nhất khác thì dừng lại tìm nguyên nhân trước khi dùng số.
- Số báo E1 lấy ở epoch tốt nhất theo MRR valid chính (`best_valid`).

### 5.7 Tổng hợp

`scripts/e1_summary.py`: đọc các `metrics_*.json` của 8 lần chạy, in bảng markdown mỗi dataset, mỗi ô 2 × 2 một dòng:
MRR chính; trong ảnh MRR / H@1 / H@3 / `image_first`; đoán quan hệ MRR / H@1 / H@5; kèm `n_cand`, `n_cand_filtered`,
`mrr_random`, và độ lệch MRR chính so với P2.10.

### 5.8 Test (`tests/test_aux_metrics.py`)

1. **Bộ dựng tay:** 2 ảnh × 3 thực thể, d = 2, R = 2, vector số nguyên, mô hình đồ chơi trả bảng cố định. Hạng trong
   ảnh và hạng quan hệ tính bằng tay, ghi trong comment. Có ít nhất: một trường hợp bằng điểm (không tính là tốt hơn),
   một đáp án đúng khác bị filter ở trong ảnh, một cặp (h, t) có hai quan hệ đúng, và một thực thể ảnh khác có điểm cao
   hơn đáp án (hạng toàn cục > hạng trong ảnh, `image_first` = 0 cho truy vấn đó).
2. **Một ảnh = toàn bộ:** mọi thực thể cùng một id ảnh, vector số nguyên ngẫu nhiên (phép cộng chính xác nên so khớp
   tuyệt đối được, kể cả bằng điểm): `rank_head_img == rank_head`, `rank_tail_img == rank_tail`.
3. **Đoán quan hệ so với brute force** trên dữ liệu ngẫu nhiên có filter.
4. **Không làm lệch thước đo chính:** có và không có `entity_image` cho cùng hạng chính.
5. **Bộ ba nối hai ảnh → lỗi; mô hình relation-conditioned kèm `entity_image` → lỗi.**
6. **`entity_image_ids`:** tách đúng tên kiểu `5073960523_0`; dataset ngoài danh sách → `None`; tên thiếu `_` → lỗi.
7. **`train.py`:** mở rộng `tests/test_train_cli.py` để lần chạy nhỏ ghi `in_image` / `relation` vào metrics và ghi
   `ranks_*_test_aux.csv` ở lần chính thức (dùng dữ liệu giả có tên thực thể dạng `<ảnh>_<box>`).

## 6. Giới hạn (ghi vào bài)

- **Baseline chỉ có file rank toàn cục** (NativE, SNAG, MoMoK, CMR): hạng trong ảnh không tính chính xác được. Chỉ biết
  hạng trong ảnh ≤ hạng toàn cục, nên MRR chính là cận dưới (yếu) của MRR trong ảnh. Đoán quan hệ không tính được. Muốn
  có số chính xác phải sửa phần đánh giá của baseline để xuất điểm, ngoài phạm vi (không sửa `baselines/`).
- Thước đo phụ có cho GeometryMMKG và Feature-TransE (hai họ mô hình).
- Spatial-VG chưa có thước đo phụ.

## 7. Sau khi xong

Cập nhật RESEARCH_PLAN: dòng E1 ở P6.0 (bỏ "dùng lại checkpoint", ghi cách A và kết quả), bảng C5, và mục 4 (thêm hai
thước đo phụ, ghi rõ chỉ để báo, không dùng chọn checkpoint).

## 8. Để người viết code tự quyết

Tên hàm phụ, cách chia batch và cấu trúc ma trận đệm, định dạng dòng in, nội dung ô notebook.
