# Hướng dẫn cho collaborator

Đọc file này trước khi sửa code. Mục tiêu: nắm được repo trong khoảng 30 phút, và không làm hỏng tính so sánh giữa các mô hình.

## 1. Đọc theo thứ tự

| # | File | Để biết |
|---|---|---|
| 1 | [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Câu hỏi nghiên cứu, thiết kế 2×2 (MMKG/MMSG × transductive/inductive), giao thức đánh giá (mục 4), danh sách task P0–P6 và trạng thái. **Đây là tài liệu chính**; task nào xong thì đánh dấu `[x]` ở đây. |
| 2 | [data/README.md](data/README.md) | Sáu bộ dữ liệu v0, repo HF, lệnh dựng lại dữ liệu và đặc trưng. |
| 3 | Code chính | `core/models/end_to_end.py` → `core/fusion/geometry_guided_fusion.py` → `core/models/evaluation.py` → `scripts/compute_metrics.py` |
| 4 | Baseline (khi cần) | [baselines/PROVENANCE.md](baselines/PROVENANCE.md) (mã gốc và mọi thay đổi), `notebooks/native.py`, `scripts/prepare_snag_data.py`, `scripts/patch_mkgformer.py` |

`experiment_results/` hiện chỉ có thống kê dữ liệu; chưa có kết quả thí nghiệm nào.

## 2. Bản đồ repo

```
core/
  data_engineering/fast_dataset.py   FastMMSGDataset (đặc trưng .pt), get_all_true_triplets
  features/encoders.py               encoder Text / RGB / Depth train end-to-end (không dùng để trích đặc trưng)
  fusion/geometry_guided_fusion.py   gate RGB–Depth + cross-attention với Text
  models/end_to_end.py               GeometryMMKGModel
  models/kg_reasoning.py             TransE scorer, JointLoss
  models/evaluation.py               đánh giá chung (head + tail, filtered), save_ranks
  reproducibility.py                 set_seed, seed_worker
notebooks/                           Marimo notebook chạy trên Molab (tải dữ liệu từ HF, tag v0)
  native.py                          NativE: Control / +D mean / +D concat trên 5 dataset
  native-db15k-sanity.py             tái lập số paper NativE trên DB15K (P3.0b)
  extract-vg-features.py             trích đặc trưng Spatial-VG
scripts/
  train.py                           train GeometryMMKG local (Spatial-VRD)
  compute_metrics.py                 NGUỒN DUY NHẤT của số liệu báo cáo
  build_fb15k237_img.py, build_fb15k237_ind.py, build_mkgy.py                 dựng dữ liệu MMKG
  build_vrd_relation_review.py, clean_splits.py, make_transductive_split.py   dựng dữ liệu scene graph
  extract_features.py, feature_quality.py                                    đặc trưng và kiểm tra chất lượng
  prepare_snag_data.py               dữ liệu cho SNAG (tầng dữ liệu, kể cả missing.json)
  patch_mkgformer.py                 patch submodule MKGformer lúc chạy
baselines/
  NativE/, SNAG/                     mã gốc của tác giả + thay đổi được phép (baselines/PROVENANCE.md, patches/)
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

Dữ liệu: notebook tự tải về bằng `snapshot_download`. Để chạy local, tải repo HF vào `data/<tên>/` (bảng trong `data/README.md`). Muốn số liệu tái lập được thì ghim `revision="v0"`.

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
6. **Baseline giữ đúng kiến trúc gốc, không sửa code kiến trúc** (lớp, pooling, fusion, expert, hàm score, khởi tạo embedding, kể cả số chiều gắn cứng). Chỉ được sửa ngoài kiến trúc: nạp dữ liệu, đánh giá và ghi rank, chọn checkpoint theo valid, seed, lỗi tương thích thư viện; mỗi thay đổi ghi diff vào phụ lục. Depth và thực thể thiếu ảnh chỉ xử lý ở tầng dữ liệu. Chi tiết: `RESEARCH_PLAN.md` P3.0.
7. **Tên baseline:** `SNAG` (gốc) / `SNAG+D(mean)`, `SNAG+D(concat)` (depth trung bình hoặc nối với RGB rồi đưa vào ô ảnh, kiến trúc giữ nguyên). NativE, MoMoK đặt tên tương tự. Không có bản feature-only.

## 5. Bẫy thường gặp

- **Thứ tự cột khác nhau:** `train.txt` là `h \t r \t t`, còn `train2id.txt` (OpenKE) là `h \t t \t r`. Dòng đầu của cả hai là số dòng.
- **VRD/VG là inductive:** 100% entity ở valid/test không có trong train (chia theo ảnh). Các mô hình học embedding theo ID (NativE, SNAG, MoMoK) chạy nguyên bản: entity chưa thấy giữ embedding ID ngẫu nhiên, chỉ phần đặc trưng mang thông tin, nên số thấp là đúng bản chất. Không viết bản feature-only (sửa kiến trúc). Không chạy các mô hình này trên Spatial-VG.
- **Spatial-VG lấy từ VG150-curated `no_part_whole`**, không phải VG150 chuẩn; có 1,44M entity, mỗi file `.pt` khoảng 4,4 GB, cần nạp bằng `torch.load(..., mmap=True)`.
- **SNAG trên dữ liệu của mình:** chuẩn bị bằng `scripts/prepare_snag_data.py`. Tên dataset không chứa `MKG` thì truyền `-use_pool 0`. Tên chứa `MKG` (ví dụ `MKG-Y-v0`) thì code gốc đọc `missing.json`; script đã tạo khoá cho tên đó từ mặt nạ ảnh của mình, cùng định dạng với file của tác giả. Không sửa code để bỏ qua.
- **`baselines/MKGformer` là submodule:** thay đổi chỉ qua `scripts/patch_mkgformer.py`. Không commit con trỏ submodule bị "dirty".
- **PyTorch ≥ 2.6:** khi `torch.load` file chứa object pickle, cần thêm `weights_only=False`.
- **Marimo:** tên biến không được trùng giữa các cell; biến cục bộ trong cell đặt tên bắt đầu bằng `_`.

## 6. Quy ước làm việc

- Dùng **đường dẫn tương đối** trong mọi script.
- **Không ghi đè dữ liệu gốc:** script làm sạch ghi ra `--dst` khác `--src`. Dữ liệu thay đổi thì đẩy lên HF kèm tag phiên bản mới, rồi cập nhật `data/README.md`.
- Sửa code đánh giá hoặc baseline thì phải chạy lại hai file test. Thêm baseline mới thì phải ghi file rank theo đúng định dạng ở mục 4.
- Trước khi push: liệt kê thay đổi trong commit để người khác review. Commit message ghi rõ task (P1.x, P3.x…), và cập nhật trạng thái trong `RESEARCH_PLAN.md`.
