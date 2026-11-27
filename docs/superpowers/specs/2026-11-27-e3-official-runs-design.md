# E3: chạy chính thức trên test (kèm kiểm định E4) — thiết kế

Ngày: 27/11/2026. Thuộc RESEARCH_PLAN P6.0 (C2, C3, C4, E2b), giao thức mục 4 (4.7, 4.8, 4.10). Trạng thái: chờ duyệt.
Các điểm ở mục 5 đã được người dùng chốt từng điểm trong phiên 27/11/2026.

## 1. Mục đích

Mọi kết quả đến nay là valid, 1 seed. E3 chạy các ô dùng để kết luận trong bài với cấu hình đã chọn khi tune, theo mục
4: chọn checkpoint theo MRR valid, đánh giá test đúng một lần, 3 seed (42, 43, 44) cho các cấu hình chính, 1 seed (42)
cho MKG-Y và Spatial-VG. Kết quả là số test trong bảng chính của bài, kèm kiểm định (E4) và các luật đọc đặt trước (mục
6), để C2, C3, C4 và E2b được kết luận trên test thay cho valid.

## 2. Mục tiêu, không làm, phải tránh

**Mục tiêu**
- 59 lần chạy chính thức (mục 5.1), mỗi lần lưu metrics, file rank test (hai bộ VRD thêm file rank phụ E1), checkpoint
  và log lên `geometry-e3/`.
- Tune `depth_stats_only` trên Spatial-VG (P2.10 đầy đủ), để mô hình chính giống nhau trên cả ba bộ MMSG.
- Seed 42 chính thức phải lặp lại lần tune trước khi được đánh giá test (mục 5.3).
- Script tổng hợp `scripts/e3_summary.py`: bảng mean ± std, Δ theo seed, bootstrap ghép cặp theo truy vấn trong từng
  seed, t-test ghép cặp giữa seed, phán quyết theo luật mục 6, bảng nhóm theo ảnh cho MMKG.
- Đánh giá che ảnh (mục 4.10) trên checkpoint E3 của MMKG, luật chốt ở mục 6.4.
- Ghi vào RESEARCH_PLAN kết quả mở rộng biên của FB15K-237-ind và Spatial-VG (đã xong trên HF, chưa ghi; mục 8).

**Không làm (không tự thêm)**
- `constant_fixed`, các cách gộp `concat` / `mean` / `full` (C6 dựa trên valid 1 seed; muốn số test cho C6 phải hỏi
  lại).
- Baseline (E5). Bảng 2 × 2 và box trên Spatial-VG.
- Không sửa mô hình, `baselines/`, cách chọn checkpoint, lưới tune, dữ liệu v0.
- Không chạy lại theo luật hội tụ ở seed 43, 44 (mục 5.4).
- Phân tích vật lý / trừu tượng trên FB15k-237-IMG (cuối 4.10) và phân tích theo nhóm quan hệ: để E6.

**Phải tránh**
- Đánh giá test trước khi kiểm seed 42 (mục 5.3). Dùng test để chọn bất cứ thứ gì.
- Ghi đè `geometry/`, `geometry-tune/` (trừ thư mục mới `geometry-tune/Spatial-VG/depth_stats_only/`), `geometry-e1/`,
  `geometry-e2/`, `geometry-e2b/`.
- Đặt hoặc đổi luật đọc sau khi đã thấy số test.

## 3. Ràng buộc

Code, comment, commit message tiếng Anh. Lần chạy dài trên Molab do người dùng bấm trên giao diện. Chỉ commit / push /
upload khi người dùng bảo, liệt kê thay đổi trước. Không đưa thông tin cá nhân hay đường dẫn máy vào file upload.

## 4. Cách tiếp cận

- **A (chọn): ô notebook "E3" mới**, đọc cấu hình chọn của từng ô bằng `tune_summary.choose`, chạy `train.py` không
  `--tune` cho từng seed, dùng lại `run_resumable` / `push_results` / `remote_files` của notebook. Phần logic không cần
  giao diện (danh sách ô, đọc cấu hình, kiểm `e3_cell.json`) nằm trong một module script để test được ngoài notebook.
- B (bỏ): script chạy độc lập từ terminal Molab. Lần chạy dài phải bấm từ giao diện, và cơ chế chạy tiếp khi bị ngắt
  cùng việc đẩy lên HF đang nằm trong notebook; viết lại là chép code.
- C (bỏ): dùng ô train thủ công có sẵn. Ô đó dùng một lr / dropout cho mọi biến thể và ghi vào `geometry/`, trong khi
  mỗi ô của E3 có cấu hình riêng.

## 5. Thiết kế

### 5.1 Các ô và cấu hình

Cấu hình lấy bằng `tune_summary.choose` từ thư mục nguồn, kiểm lại trên HF ngày 27/11 (mọi ô `final`). "ep" là ngân
sách của lần được chọn; mặc định là `EPOCH_BUDGET` của dataset.

| Dataset | Seed | Ô (nguồn) | lr / dropout, ep (MRR valid, epoch tốt nhất) |
|---|---|---|---|
| FB15k-237-IMG | 42, 43, 44 | `no_depth`, `fixed` (tune) | 1e-3 / 0 (0,3302, 170); 1e-3 / 0,1 (0,3302, 110) |
| FB15K-237-ind | 42, 43, 44 | `no_depth`, `fixed` (tune) | 1e-3 / 0,3 (0,2557, 30); 3e-4 / 0,5 (0,2563, 80) |
| MKG-Y | 42 | `no_depth`, `fixed` (tune) | 1e-3 / 0,3 (0,3770, 400); 3e-4 / 0,3, ep 2000 (0,3785, 1100) |
| Spatial-VRD | 42, 43, 44 | `no_depth`, `depth_stats_only`, `depth_map_only`, `fixed` (tune); `box`, `box_grid9` (E2); `boxnorm` (E2b) | 3e-4 / 0 (0,2560, 20); 1e-4 / 0 (0,2670, 50); 1e-3 / 0 (0,2566, 20); 1e-3 / 0 (0,2650, 20); 1e-4 / 0 (0,2853, 60); 3e-4 / 0 (0,2905, 30); 3e-4 / 0 (0,2579, 20) |
| Spatial-VRD-Trans | 42, 43, 44 | như Spatial-VRD | 1e-3 / 0 (0,3503, 100); 3e-4 / 0, ep 200 (0,3636, 200) †; 1e-3 / 0 (0,3464, 90); 3e-4 / 0, ep 200 (0,3611, 200) †; 3e-4 / 0 (0,3789, 90); 3e-4 / 0, ep 200 (0,3894, 170); 3e-4 / 0, ep 200 (0,3550, 200) † |
| Spatial-VG | 42 | `no_depth`, `fixed` (tune); `depth_stats_only` (tune mới, mục 5.2) | 3e-4 / 0, ep 200 (0,3216, 170); 3e-4 / 0 (0,3314, 70); chờ tune |

† epoch tốt nhất ở cuối sau khi đã nhân đôi ngân sách: số của ô là **cận dưới** (mục 5.4).

Thư mục nguồn: tune `geometry-tune/<ds>/` (biến thể = ô); box `geometry-e2/<ds>/<ô>/`, boxnorm
`geometry-e2b/<ds>/boxnorm/` (biến thể `depth_stats_only`). Cờ `--depth-stats` của ô box là `<bảng box>:box` hoặc
`:box_grid9`, của `boxnorm` là `<depth_stats_boxnorm.pt>:grid9`, như E2 / E2b. Ô nào `choose` trả về không `final`
thì dừng ô đó.

Tổng 6 + 6 + 2 + 21 + 21 + 3 = 59 lần chạy.

### 5.2 Spatial-VG: tune `depth_stats_only` (chốt điểm 1: phương án a)

- Dùng ô "Tune" có sẵn (lưới gốc, rồi "Mở rộng biên" = luật biên + luật hội tụ), biến thể `depth_stats_only`, bảng
  `ent_depth_stats.pt` của VG (cờ `--depth-stats auto`). Kết quả vào thư mục mới
  `geometry-tune/Spatial-VG/depth_stats_only/<cấu hình>/`. Chỉ valid, seed 42.
- Ô Tune hiện đã cho chọn biến thể này trên VG (VG có `depth_stats`); chỉ cần cập nhật chú thích của ô (comment hiện
  ghi "VG quá đắt để tune") và, tuỳ người viết code, giá trị mặc định cho VG.
- Ước lượng 7–9 lần (6 điểm lưới, khoảng 1–2 điểm biên, có thể 1 lần 200 epoch), khoảng 4,5–5,5 giờ.
- Cách ghi trong bài: VG có `no_depth`, `depth_stats_only` (mô hình chính) và `fixed` (đã tune, thêm một ô có depth);
  không có 2 × 2 và box trên VG.

### 5.3 Seed 42 phải lặp lại lần tune (chốt điểm 2)

- **Tiêu chí:** epoch tốt nhất trùng đúng, |MRR valid tốt nhất − MRR của lần tune| ≤ 0,0005. Lệch thật được ghi vào
  metrics và bảng tổng hợp.
- **`train.py --expect-valid <MRR>:<epoch>`** (mới, mặc định tắt; không đổi gì khi không truyền):
  - Sau vòng train, trước khi đánh giá test, so `best["mrr"]` và `best["epoch"]` với giá trị kỳ vọng.
  - Trùng: chạy tiếp như cũ (đánh giá test một lần, ghi rank, checkpoint).
  - Lệch: ghi checkpoint tốt nhất và metrics **không có `test`**, xoá file resume, thoát với mã khác 0. Test không được
    đánh giá.
  - Metrics ghi `expect_valid = {"mrr", "epoch", "got_mrr", "got_epoch", "tol": 0.0005, "ok"}` trong cả hai trường hợp.
  - Tham số này không ảnh hưởng huấn luyện; người viết code quyết có đưa nó vào `RESUME_IGNORED_ARGS` hay không (đề
    xuất: có).
- **Thứ tự:** theo từng ô: seed 42 (có `--expect-valid`, giá trị lấy từ lần được `choose` chọn), kiểm, rồi seed 43, 44
  (không có gì để so, không truyền `--expect-valid`). Ô lệch thì dừng ô đó, ô khác vẫn chạy. Không ai nhập tay số kỳ vọng.
- Cơ sở: E1 chạy lại 8 cấu hình và trùng tuyệt đối (lệch 0,0000). Các lần tune dùng 9 commit code khác nhau, nên phép
  kiểm này bắt cả trường hợp code đã đổi kết quả.

### 5.4 Số epoch (chốt điểm 3)

- Mỗi ô chạy đúng ngân sách của lần được chọn (`--epochs` truyền tường minh; `--valid-every` để mặc định, như lần
  tune). Ngân sách gốc sẽ làm seed 42 của các ô đã nhân đôi không thể trùng.
- Cận dưới: chỉ VRD-Trans `depth_stats_only`, `fixed`, `boxnorm` (200 / 200). VRD-Trans `box_grid9` (170 / 200), VG
  `no_depth` (170 / 200) và MKG-Y `fixed` (1100 / 2000) đã hội tụ.
- Seed 43, 44 không chạy lại theo luật hội tụ (luật hội tụ thuộc bước tune; áp lại ở đây là chọn theo test). Bảng tổng
  hợp đếm số seed có epoch tốt nhất trong 10 % cuối ngân sách, mỗi ô.

### 5.5 Thư mục, tên, `e3_cell.json` (chốt điểm 4)

- Kết quả: `geometry-e3/<dataset>/<ô>/` trên repo HF private. `<ô>` ∈ {`no_depth`, `fixed`, `depth_map_only`,
  `depth_stats_only`, `box`, `box_grid9`, `boxnorm`}. Không dùng `geometry/` (ô train thủ công ghi vào đó với một cấu
  hình chung); RESEARCH_PLAN P2.7 ghi rằng `geometry/` không dùng cho số trong bài.
- Stem do `train.py` sinh như hiện nay (`geometry_<ds>_<biến thể>_<tag>_seed<seed>`, không có cấu hình); tag của box /
  boxnorm đã khác nên tên không trùng.
- Mỗi seed: `metrics_*.json`, `ranks_*_test.csv`, `ranks_*_test_aux.csv` (hai bộ VRD), `ckpt_*.pt`, log.
- **`e3_cell.json`** mỗi ô, ghi trước lần chạy đầu và đẩy lên cùng thư mục: dataset, ô, biến thể, spec `--depth-stats`
  (tên file, không đường dẫn máy), sha256 bảng (ô box / boxnorm), lr, dropout, epochs, thư mục và file tune nguồn, commit
  của lần tune, số cấu hình đã thử, MRR valid và epoch kỳ vọng, cờ cận dưới, commit code E3. Nếu file đã có trên HF và
  khác nội dung (trừ commit code E3) thì dừng ô đó: cấu hình hoặc bảng đã đổi giữa các seed.
- **Bỏ qua lần đã có:** một seed xong khi metrics của nó có `test`. Metrics có mà không có `test` (lệch ở 5.3): ô ghi
  "lệch, dừng", không tự chạy lại, không chạy seed tiếp theo.
- Kiểm thêm sau mỗi lần chạy: `options_tag` của lần chính thức bằng `options_tag` của lần tune nguồn.

### 5.6 Bảng box và boxnorm trên Molab (chốt điểm 5)

- `box`, `box_grid9`: dựng bằng `scripts/box_geometry.py --check` từ annotation tag v0, phải khớp `EXPECTED`, như ô E2.
  Không khớp thì dừng hai ô box của bộ đó.
- `boxnorm`: tải `geometry-e2b/<ds>/depth_stats_boxnorm.pt`, kiểm `E2B_SHA256`, như ô E2b. Sai thì dừng ô `boxnorm`.
- Tách hai bước này thành hàm chung trong notebook (ô E2, E2b, E3 cùng gọi). Hành vi của ô E2 / E2b không đổi.
- sha256 của bảng đã dùng ghi vào `e3_cell.json` (với box: bản vừa dựng trên Molab), nên mọi seed của một ô dùng cùng
  một bảng (5.5).

### 5.7 Ô notebook "E3"

- Theo `DATASET` đang chọn: danh sách ô và seed của dataset đó (mục 5.1), chọn được một phần qua giao diện, mặc định tất
  cả. Nút ▶ Chạy E3.
- Mỗi ô: đọc cấu hình (5.1), chuẩn bị bảng nếu cần (5.6), ghi / kiểm `e3_cell.json` (5.5), chạy seed 42 với
  `--expect-valid`, rồi 43, 44; `run_resumable` + `push_results` như các ô khác. Trạng thái mỗi ô / seed in ra cuối ô:
  "đã có", "xong", "lệch, dừng (MRR …, epoch …)", "lỗi (exit …)".
- Một ô thứ hai hiển thị bảng `e3_summary.py` của dataset đang chọn (tải metrics và file rank từ `geometry-e3/<ds>/`).
- Thứ tự chạy đề xuất (người dùng quyết khi bấm): Spatial-VRD trước (rẻ, thử cả đường chạy và phép kiểm seed 42), rồi
  VRD-Trans, FB15k-237-IMG, FB15K-237-ind, MKG-Y, tune VG, E3 VG.

### 5.8 Đánh giá che ảnh trên MMKG (chốt điểm 6)

- Script mới (gợi ý `scripts/e3_mask_eval.py`): nạp checkpoint E3, đánh giá test với `all_mask` = 0 cho **toàn bộ**
  bảng thực thể (mọi thực thể dùng `v_miss`), cùng filter, cùng quy tắc bằng điểm; ghi
  `ranks_<stem>_test_maskall.csv` và một file metrics nhỏ. Che riêng hai đầu của bộ ba thì đáp án mất ảnh còn các ứng
  viên khác vẫn có ảnh, kéo hạng xuống giả tạo, nên không làm.
- Phạm vi: FB15k-237-IMG, FB15K-237-ind, MKG-Y; `no_depth` và `fixed`; mọi seed. Chạy sau khi có checkpoint (ở máy hoặc
  Molab, người viết code chọn), đẩy kết quả vào cùng thư mục `geometry-e3/<ds>/<ô>/`.
- Đây là một phép phân tích đặt trước trên cùng checkpoint, không chọn gì theo nó; bài ghi rõ test được đánh giá thêm
  lần này.

### 5.9 Tổng hợp và kiểm định (chốt điểm 7; E4 gộp vào đây)

`scripts/e3_summary.py --root <geometry-e3>/<ds> --data data/<ds> [--out ...]`, chỉ đọc metrics, file rank và
`e3_cell.json`:
- Mỗi ô: MRR, MR, H@1/3/10 test, mean ± std qua seed (std với ddof 1), số seed, cờ cận dưới, số seed có epoch tốt nhất
  trong 10 % cuối. Hai bộ VRD: thước đo phụ E1 trên test (MRR / H@1 trong ảnh, `image_first`, đoán quan hệ MRR / H@1),
  mean ± std, chỉ báo.
- Mỗi cặp A − B (mục 6): Δ_s theo từng seed; Δ̄; bootstrap ghép cặp theo truy vấn trong từng seed
  (`compute_metrics.paired_bootstrap`, RR mỗi bộ ba = trung bình head và tail, 10.000 lần lấy mẫu, seed bootstrap cố
  định); t-test ghép cặp trên Δ_s giữa seed (bậc tự do 2, chỉ báo); phán quyết (6.1). Kiểm mọi file rank của một dataset
  có cùng tập bộ ba.
- MMKG: bảng theo số đầu bộ ba có ảnh (cả hai / một / không, theo `ent_img_mask.pt`): số bộ ba, MRR mỗi ô, Δdepth,
  bootstrap mỗi seed. Nhóm dưới 500 bộ ba chỉ báo, không phán quyết (ngưỡng là đề xuất của người viết thiết kế; FB15k
  nhóm "không" có 14, MKG-Y có 2). Có file `maskall` thì thêm dòng che ảnh (6.4).
- Đầu ra markdown (in và ghi file) và json (số liệu để viết bài).

## 6. Cách đọc kết quả (đặt trước khi chạy)

Chỉ MRR test chính dùng để phán quyết; các metric khác chỉ báo. Ngưỡng 0,003 như P2.10.

### 6.1 Phán quyết cho một cặp A − B trên một dataset

- 3 seed:
  - **"có"**: Δ̄ > 0,003, Δ_s > 0 ở cả 3 seed, và khoảng tin cậy 95 % bootstrap loại 0 ở cả 3 seed. (Tương tự cho
    chiều âm: "có, âm".)
  - **"bằng"**: |Δ̄| ≤ 0,003.
  - **"chưa rõ"**: còn lại.
- 1 seed (MKG-Y, Spatial-VG): "có" khi |Δ| > 0,003 và khoảng tin cậy loại 0; "bằng" khi |Δ| ≤ 0,003; còn lại "chưa rõ".
  Bảng ghi "1 seed".
- t-test giữa seed không dùng để phán quyết (2 bậc tự do).
- Cặp có ô cận dưới (VRD-Trans `depth_stats_only`, `fixed`, `boxnorm`): bảng ghi chú ô nào là cận dưới và phán quyết có
  thể đổi theo chiều nào nếu ô đó tăng.

### 6.2 Luật cho từng khẳng định

- **C2 (MMKG):** `fixed` − `no_depth` "bằng" trên cả 3 bộ MMKG → C2 đúng với GeometryMMKG (họ thứ hai ở E5). Bộ nào ra
  "có" (dương hoặc âm) → viết lại C2 cho bộ đó. Nếu "chưa rõ": báo nguyên trạng, không viết "trong mức nhiễu".
- **Δdepth trên MMSG:** `depth_stats_only` − `no_depth` trên Spatial-VRD, VRD-Trans (3 seed), Spatial-VG (1 seed).
- **C3 (2 × 2, hai bộ VRD):** C3 đúng khi phần lợi của `depth_stats` (`depth_stats_only` − `no_depth` và `fixed` −
  `depth_map_only`) "có" ở cả hai cặp trên cả hai bộ, và phần lợi của vector depth (`depth_map_only` − `no_depth` và
  `fixed` − `depth_stats_only`) không "có" (dương) ở cặp nào. Trường hợp khác: viết C3 theo đúng các phán quyết.
- **C4:** `box_grid9` − `box` "có" trên cả hai bộ → C4 = có, kèm độ lớn; "bằng" trên cả hai bộ → C4 = không; còn lại →
  chưa nhất quán. Đọc kèm `image_first` và MRR trong ảnh (E2: phần thêm nằm ở `image_first`). Báo thêm `box` −
  `depth_stats_only`.
- **E2b:** luật mục 6 của thiết kế E2b áp nguyên văn lên Δ̄ (3 seed): `boxnorm` − `no_depth` ≤ 0,003 trên cả hai bộ →
  cần thang chung; `depth_stats_only` − `boxnorm` ≤ 0,003 trên cả hai bộ → không cần thang chung; cả hai > 0,003 → một
  phần, báo phần lợi giữ lại (`boxnorm` − `no_depth`) / (`depth_stats_only` − `no_depth`); hai bộ khác nhau → kết luận
  cuối "chưa nhất quán", không chạy thêm thí nghiệm. Báo kèm khoảng tin cậy.

### 6.3 Nhóm theo ảnh (MMKG, 4.10)

Δdepth theo nhóm, phán quyết theo 6.1 cho nhóm ≥ 500 bộ ba. Nhóm so sánh chính là "một" (4.10). Nhóm "không" tự nhiên
không đủ lớn ở FB15k-237-IMG và MKG-Y.

### 6.4 Che ảnh lúc test (MMKG)

Δ_mask = `fixed` − `no_depth` khi che toàn bộ ảnh, Δ̄ qua seed (MKG-Y 1 seed), kèm bootstrap.
- Phán quyết 6.1 của Δ_mask không phải "có" (tức |Δ̄_mask| ≤ 0,003, hoặc có seed mà khoảng tin cậy chứa 0 hay Δ_s
  khác dấu): đối chứng sạch.
- Phán quyết "có" (dương hoặc âm): hai biến thể đã khác nhau ở phần không thuộc ảnh (text, quan hệ); Δdepth trên test đầy đủ của bộ đó phải
  đọc so với Δ_mask, và bài ghi rõ.
- Giới hạn: mô hình được train với ảnh, nên che lúc test là đầu vào lệch phân phối; MRR tuyệt đối giảm, chỉ Δ giữa hai
  biến thể có nghĩa.

## 7. Ước lượng GPU (từ `total_sec` của lần tune được chọn)

| Dataset | Lần chạy | Mỗi lần | Tổng |
|---|---|---|---|
| FB15k-237-IMG | 2 × 3 | 42 phút | 4,2 giờ |
| FB15K-237-ind | 2 × 3 | 33 phút | 3,3 giờ |
| MKG-Y | 2 × 1 | 19 / 38 phút | 1,0 giờ |
| Spatial-VRD | 7 × 3 | 8 phút | 2,8 giờ |
| Spatial-VRD-Trans | 7 × 3 | 8 phút (3 ô), 16 phút (4 ô) | 4,4 giờ |
| Spatial-VG | 3 × 1 | 70 / 35 / 35–70 phút | 2,3–2,9 giờ |
| Tune VG `depth_stats_only` | 7–9 | 35 / 70 phút | 4,5–5,5 giờ |

Tổng khoảng 22–24 giờ GPU. Molab: phiên tối đa 12 giờ, `train.py` chạy tiếp được khi bị ngắt.

## 8. Ghi vào RESEARCH_PLAN

- P2.10: kết quả mở rộng biên đã có trên HF (kiểm 27/11: mọi lần là `--tune`, không có `test`, commit `55a27f7`
  FB15K-237-ind / `2f12eab` Spatial-VG):
  - FB15K-237-ind: `fixed` 3e-4 / 0,5 → 0,2563 (epoch 80 / 200), thành cấu hình chọn (dropout 0,5 là đầu thang); `no_depth`
    hai điểm biên kém hơn (1e-3 / 0,5: 0,2539; 3e-3 / 0,3: 0,2521), giữ 1e-3 / 0,3 (0,2557). Δ (valid) +0,0006.
  - Spatial-VG: dropout 0 thắng ở cả hai: `fixed` 3e-4 / 0 → 0,3314 (epoch 70 / 100); `no_depth` 3e-4 / 0 → 0,3214 ở
    epoch 100 / 100, chạy lại 200 epoch → 0,3216 (epoch 170). Δdepth (valid) +0,0098. Như VRD, ở dropout 0 chỉ thử một
    lr.
  - Bảng tổng hợp P2.10 cập nhật số mới.
- P2.10: kết quả tune `depth_stats_only` trên VG khi xong.
- P2.7: `geometry/` không dùng cho số trong bài; số chính thức ở `geometry-e3/`.
- Mục 4.10: quyết định che ảnh (6.4), "chạy trên checkpoint E3".
- P6.0: dòng E3 (thiết kế này), E4 gộp vào E3; sau khi chạy: kết quả và phán quyết theo mục 6; C2, C3, C4 cập nhật.

## 9. Test

1. `train.py --expect-valid` trên dữ liệu giả (như `tests/test_train_cli.py`): giá trị đúng → có `test`, có rank, có
   checkpoint, `expect_valid.ok` true; sai MRR (> 0,0005) hoặc sai epoch → không có `test`, không có file rank, có
   checkpoint, mã thoát khác 0, `expect_valid.ok` false; không truyền → metrics như cũ.
2. Module ô E3: danh sách ô / seed đúng mục 5.1; đọc cấu hình từ thư mục tune giả (chọn đúng, cờ cận dưới đúng); ô không
   `final` → lỗi; `e3_cell.json` khác nội dung → lỗi, chỉ khác commit code E3 → qua.
3. `e3_summary`: hàm phán quyết (biên 0,003; một seed khác dấu → "chưa rõ"; một seed có khoảng tin cậy chứa 0 → "chưa
   rõ"; 1 seed); file rank giả → bảng có đủ dòng; nhóm theo ảnh đếm đúng; tập bộ ba khác nhau → cảnh báo hoặc lỗi.
4. Che ảnh: với checkpoint giả, kết quả không phụ thuộc giá trị RGB / depth của bảng; `no_depth` và `fixed` cùng tham số
   chung cho đầu ra giống hệt (cùng lập luận với `tests/test_missing_modality.py`).
5. `marimo check` sạch; ô E2 / E2b sau khi tách hàm chung gọi đúng như trước.

## 10. Để người viết code tự quyết

Tên module và hàm phụ, cách chia ô notebook, định dạng in, mã thoát khi lệch, vị trí hằng (dung sai 0,0005, ngưỡng 500
bộ ba), seed bootstrap, có đưa `expect_valid` vào `RESUME_IGNORED_ARGS` không, chạy che ảnh ở máy hay Molab.
