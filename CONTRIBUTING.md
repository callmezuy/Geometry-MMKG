# Hướng dẫn cho collaborator

Đọc file này trước khi sửa code. Mục tiêu: nắm được repo trong khoảng 30 phút, và không làm hỏng tính so sánh giữa các mô hình.

## 1. Đọc theo thứ tự

| # | File | Để biết |
|---|---|---|
| 1 | [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Câu hỏi nghiên cứu, thiết kế 2×2 (MMKG/MMSG × transductive/inductive), giao thức đánh giá (mục 4), danh sách task P0–P6 và trạng thái. **Đây là tài liệu chính**; task nào xong thì đánh dấu `[x]` ở đây. |
| 2 | [data/README.md](data/README.md) | Bốn bộ dữ liệu v2, repo HF và tag, lệnh dựng lại dữ liệu. |
| 3 | [CONVERSATION_SUMMARY.md](CONVERSATION_SUMMARY.md) | Cách chạy trên Molab, kết quả đã có, quy tắc làm việc. |
| 4 | Code chính | `notebooks/geometry-*.py` → `core/models/end_to_end.py` → `core/fusion/geometry_guided_fusion.py` → `core/models/evaluation.py` → `scripts/compute_metrics.py` |
| 5 | Baseline (khi cần) | `baselines/{NativE,SNAG/SNAG_MKGC}/mmkgc/config/{Tester,ValidSelector}.py`, `scripts/patch_mkgformer.py` |

`experiment_results/` chỉ chứa kết quả **v1, chạy theo giao thức cũ** (xem [experiment_results/README.md](experiment_results/README.md)). Không lấy số ở đó để so sánh.

## 2. Bản đồ repo

```
core/
  data_engineering/fast_dataset.py   FastMMSGDataset (đặc trưng .pt), get_all_true_triplets
  features/encoders.py               encoder Text / RGB / Depth (chỉ dùng khi không có .pt)
  fusion/geometry_guided_fusion.py   gate RGB–Depth + cross-attention với Text
  models/end_to_end.py               GeometryMMKGModel
  models/kg_reasoning.py             TransE scorer, JointLoss
  models/evaluation.py               đánh giá chung (head + tail, filtered), save_ranks
  reproducibility.py                 set_seed, seed_worker
notebooks/                           Marimo notebook chạy trên Molab (tải dữ liệu từ HF)
scripts/
  train.py                           train GeometryMMKG local (Spatial-VRD)
  compute_metrics.py                 NGUỒN DUY NHẤT của số liệu báo cáo
  clean_splits.py, build_vrd_relation_review.py, make_transductive_split.py   dựng dữ liệu v2
  patch_mkgformer.py                 patch submodule MKGformer lúc chạy
  legacy/                            script cũ, không dùng cho thí nghiệm mới
baselines/
  NativE/, SNAG/                     code vendored, đã sửa (rank_file, ValidSelector, depth cho SNAG)
  MKGformer/                         git submodule (upstream zjunlp), KHÔNG sửa trực tiếp
tests/                               test_evaluation.py, test_valid_selector.py
data/                                gitignored, trừ data/README.md (tải từ HF)
ZipDepth/                            mô hình ước lượng depth dùng để trích đặc trưng depth
```

## 3. Bắt đầu

```bash
git clone --recurse-submodules https://github.com/callmezuy/Geometry-MMKG.git
cd Geometry-MMKG
pip install -r requirements.txt
python tests/test_evaluation.py && python tests/test_valid_selector.py
```

Dữ liệu: notebook tự tải về bằng `snapshot_download`. Để chạy local, tải repo HF vào `data/<tên>/` (bảng trong `data/README.md`). Muốn số liệu tái lập được thì ghim `revision="v2"`.

Notebook trên Molab tải code từ GitHub `main`, nên **code chỉ có tác dụng trên Molab sau khi đã push**.

## 4. Quy tắc bắt buộc (giữ tính so sánh)

Mọi mô hình (GeometryMMKG, NativE, SNAG, MKGformer, baseline mới) phải theo đúng các quy tắc sau:

1. **Đánh giá:** filtered link prediction, xếp hạng **cả head lẫn tail** trên **toàn bộ** entity, `type_constrain=False`. Bằng điểm thì xếp chặt: rank = 1 + số ứng viên có điểm *tốt hơn hẳn*. Filter set gồm train ∪ valid ∪ test.
2. **Chọn checkpoint theo MRR trên valid; test chỉ chạy đúng một lần.** NativE/SNAG: `-select_on valid` (mặc định). `-select_on original` chỉ dùng khi kiểm tra lại số của paper gốc (P3.0b).
3. **Số liệu báo cáo chỉ lấy từ `scripts/compute_metrics.py`** chạy trên file rank CSV (`h_id,r_id,t_id,rank_head,rank_tail`), không lấy số in trong log.
   - GeometryMMKG: `save_ranks()` (notebook đã có sẵn).
   - NativE/SNAG: `-rank_file=<path>`.
   - MKGformer: biến môi trường `MKG_RANK_FILE` (sau `patch_mkgformer.py`).
   ```bash
   python scripts/compute_metrics.py --data data/Spatial-VRD \
       --ranks full=ranks_full.csv nodepth=ranks_nodepth.csv --reference nodepth --by-relation
   ```
4. **Δdepth** = (có depth) − (không depth) **trong cùng một mô hình và cùng thiết lập**, báo kèm khoảng tin cậy bootstrap (`--reference`). Không so depth giữa các mô hình khác nhau.
5. **Seed:** 42 cho mọi lần chạy (`core.reproducibility.set_seed`); cấu hình chính chạy 3 seed.
6. **Tên baseline:** `SNAG` (gốc, 3 modality) / `SNAG+D` (mình thêm depth) / `SNAG-FO` (feature-only cho inductive). NativE đặt tên tương tự.

## 5. Bẫy thường gặp

- **Thứ tự cột khác nhau:** `train.txt` là `h \t r \t t`, còn `train2id.txt` (OpenKE) là `h \t t \t r`. Dòng đầu của cả hai là số dòng.
- **VRD/VG là inductive:** 100% entity ở valid/test không có trong train (chia theo ảnh). Các mô hình học embedding theo ID (NativE, SNAG, MKGformer) không dự đoán được entity mới, nên cần bản feature-only (P3.2, P3.3).
- **Spatial-VG lấy từ VG150-curated `no_part_whole`**, không phải VG150 chuẩn; có 1,44M entity, mỗi file `.pt` khoảng 4,4 GB, cần nạp bằng `torch.load(..., mmap=True)`.
- **SNAG trên dữ liệu của mình:** truyền `-use_pool 0` (đặc trưng 768 chiều), và tên dataset không được chứa chữ `MKG` (nếu chứa, code sẽ đọc `missing.json`).
- **`baselines/MKGformer` là submodule:** thay đổi chỉ qua `scripts/patch_mkgformer.py`. Không commit con trỏ submodule bị "dirty".
- **PyTorch ≥ 2.6:** khi `torch.load` file chứa object pickle, cần thêm `weights_only=False`.
- **Marimo:** tên biến không được trùng giữa các cell; biến cục bộ trong cell đặt tên bắt đầu bằng `_`.

## 6. Quy ước làm việc

- Dùng **đường dẫn tương đối** trong mọi script.
- **Không ghi đè dữ liệu gốc:** script làm sạch ghi ra `--dst` khác `--src`. Dữ liệu thay đổi thì đẩy lên HF kèm tag phiên bản mới, rồi cập nhật `data/README.md`.
- Sửa code đánh giá hoặc baseline thì phải chạy lại hai file test. Thêm baseline mới thì phải ghi file rank theo đúng định dạng ở mục 4.
- Trước khi push: liệt kê thay đổi trong commit để người khác review. Commit message ghi rõ task (P1.x, P3.x…), và cập nhật trạng thái trong `RESEARCH_PLAN.md`.
