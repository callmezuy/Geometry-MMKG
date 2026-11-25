# E1: thước đo phụ trên MMSG — kế hoạch code

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** thêm xếp hạng trong cùng ảnh và đoán quan hệ vào đánh giá của `train.py` trên Spatial-VRD / VRD-Trans, rồi
có ô notebook chạy lại 8 cấu hình 2 × 2 (chỉ valid) và script tổng hợp.

**Architecture:** `evaluate_link_prediction` nhận thêm `entity_image` (id ảnh mỗi thực thể); khi có, dùng lại bảng
biểu diễn đã fuse để tính hai thước đo phụ, không đổi thước đo chính. `train.py` truyền `entity_image` cho MMSG.
`scripts/e1_summary.py` giữ danh sách 8 cấu hình và in bảng; notebook import danh sách đó.

**Tech Stack:** Python, PyTorch, marimo (notebook), test chạy bằng `python tests/<file>.py` (máy không có pytest).

**Spec:** `docs/superpowers/specs/2026-11-25-e1-mmsg-aux-metrics-design.md`

## Global Constraints

- Không sửa gì trong `baselines/`. Không đổi `scripts/compute_metrics.py` và định dạng `ranks_<stem>_test.csv`.
- Không truyền `entity_image` thì `evaluate_link_prediction` cho kết quả y như hiện tại (test cũ phải qua).
- Quy tắc bằng điểm: rank = 1 + số ứng viên có điểm **tốt hơn hẳn**; đáp án và đáp án đúng khác (train ∪ valid ∪ test)
  bị loại khỏi phép so sánh, không trừ sau.
- Chọn checkpoint vẫn theo MRR valid chính.
- Code, comment, commit message bằng tiếng Anh.
- **Commit:** mỗi bước "Commit" là đề xuất: liệt kê file và nội dung thay đổi, chờ người dùng đồng ý rồi mới chạy.
  Không push. Không stage thay đổi của phiên khác (RESEARCH_PLAN khoảng dòng 733, `baselines/PROVENANCE.md`,
  `baselines/patches/*.diff`, `scripts/cmr_ranks.py`).
- Không chạy lần dài trên Molab; ô notebook do người dùng bấm.

## Review Focus

1. Dict kết quả có `in_image` / `relation` phải `json.dump` được (không lọt `np.int64`): `train.py` ghi thẳng vào
   `metrics_*.json`. Test ở Task 2.
2. `--data data/Spatial-VRD/` (có dấu `/` cuối) vẫn nhận ra dataset. Test ở Task 1.
3. Bảng ứng viên nằm trên CPU (`keep_on_gpu=False`) trong khi truy vấn ở GPU: gather theo `table.device` rồi mới
   `.to(device)`, giống `_rank_batch`. Test ở Task 2 chạy với `keep_on_gpu=False`.
4. `e1_summary.py` khi mới chạy xong một phần (một dataset, hoặc thiếu ô): in "missing", không dừng. Test ở Task 4.
5. Feature-TransE đi qua cùng đường đánh giá (spec 6 nói có thước đo phụ cho hai họ mô hình). Test ở Task 2.

---

### Task 1: id ảnh của thực thể

**Files:**
- Modify: `core/data_engineering/fast_dataset.py`
- Test: `tests/test_aux_metrics.py` (tạo mới; theo mẫu `tests/test_evaluation.py`, có khối `__main__` gọi mọi test)

**Interfaces:**
- Produces: `IMAGE_ID_FROM_NAME = frozenset({"Spatial-VRD", "Spatial-VRD-Trans"})`;
  `entity_image_ids(data_dir: str) -> torch.LongTensor | None` — `[N]`, id ảnh đánh số liên tục theo thứ tự xuất
  hiện đầu tiên trong `entity2id.txt`; `None` với dataset ngoài danh sách. Dataset = `basename(normpath(data_dir))`.

- [ ] **Step 1: Viết test**

```python
def _write_entities(root, name, names):
    d = os.path.join(root, name); os.makedirs(d)
    with open(os.path.join(d, "entity2id.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(names)}\n" + "".join(f"{n}\t{i}\n" for i, n in enumerate(names)))
    return d

def test_entity_image_ids():
    with tempfile.TemporaryDirectory() as root:
        d = _write_entities(root, "Spatial-VRD", ["5073960523_0", "5073960523_1", "77_a_b_0", "5073960523_2"])
        assert entity_image_ids(d).tolist() == [0, 0, 1, 0]          # last '_' splits; "77_a_b" is one image
        assert entity_image_ids(d + os.sep).tolist() == [0, 0, 1, 0]  # trailing separator
        assert entity_image_ids(_write_entities(root, "FB15k-237-IMG", ["/m/0_x"])) is None
        bad = _write_entities(root, "Spatial-VRD-Trans", ["5073960523"])
        try:
            entity_image_ids(bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass
```

- [ ] **Step 2: Chạy, phải lỗi import**

Run: `python tests/test_aux_metrics.py` — Expected: `ImportError: cannot import name 'entity_image_ids'`.

- [ ] **Step 3: Viết `entity_image_ids`** trong `fast_dataset.py`. Docstring ghi Spatial-VG chưa hỗ trợ (cần
  `VG-SGG.h5` `img_to_first_box` / `img_to_last_box`) và trả `None`; tên không có `_` → `ValueError` nêu tên đó.

- [ ] **Step 4: Chạy lại** `python tests/test_aux_metrics.py` — Expected: không lỗi.

- [ ] **Step 5: Commit** (đề xuất, chờ duyệt): `git add core/data_engineering/fast_dataset.py tests/test_aux_metrics.py`,
  message `Add entity_image_ids for the scene-graph datasets (E1)`.

---

### Task 2: xếp hạng trong cùng ảnh và đoán quan hệ trong `evaluate_link_prediction`

**Files:**
- Modify: `core/models/evaluation.py`
- Test: `tests/test_aux_metrics.py`

**Interfaces:**
- Consumes: `entity_image_ids` (chỉ trong test dữ liệu thật thì không cần; test dựng tay tự tạo tensor).
- Produces:
  - `evaluate_link_prediction(..., entity_image: torch.LongTensor | None = None)`; khi khác `None`:
    - `result["in_image"]` = `_metrics(pooled)` cộng `n_cand`, `n_cand_filtered`, `mrr_random`, `image_first`,
      `n_negative_gap`, và `head` / `tail` (mỗi cái `_metrics` của một hướng). Mọi giá trị là `float` / `int` Python.
    - `result["relation"]` = `{"mrr", "mr", "hits@1", "hits@5", "n"}`.
    - với `return_ranks=True`: `ranks` có thêm `rank_head_img`, `rank_tail_img`, `rank_rel` (numpy int64, cùng thứ tự
      với `h`, `r`, `t`).
  - `save_aux_ranks(ranks: dict, path: str) -> None`: CSV header `h_id,r_id,t_id,rank_head_img,rank_tail_img,rank_rel`.

Quy ước tính (từ spec 5.2, 5.3):
- Trong ảnh, truy vấn tail: ứng viên = thực thể có `entity_image == entity_image[h]`; head: `== entity_image[t]`.
  Ma trận đệm ảnh → thực thể `[n_img, M]` (đệm `-1`) dựng một lần mỗi lần gọi. Điểm ứng viên bằng
  `torch.cdist(q[:, None], cand_emb, p=1)`; điểm đáp án dùng lại `true_t` / `true_h` của `_rank_batch`. Loại đáp án và
  đáp án đúng khác bằng chính `_excluded_pairs` của thước đo chính.
- `n_cand` = số box của ảnh (gồm đáp án); `n_cand_filtered` = số ứng viên còn lại sau filter (gồm đáp án);
  `mrr_random` = trung bình `H(n)/n` với `n = n_cand_filtered` của từng truy vấn, `H(n) = Σ_{k=1..n} 1/k`;
  gap = `rank − rank_img` theo từng truy vấn và hướng; `image_first` = tỉ lệ gap ≤ 0; `n_negative_gap` = số gap < 0.
  Các số trung bình gộp head + tail như `mrr`.
- Đoán quan hệ: điểm ứng viên `-torch.cdist(f(h) − f(t), −R_all, p=1)`, `R_all = model.relation_embeddings.weight`;
  điểm đáp án dùng `true_t`. Filter `{(h, t): {r}}` dựng từ `all_true_triplets`.
- Lỗi `ValueError` nếu một bộ ba được đánh giá có `entity_image[h] != entity_image[t]`; lỗi `ValueError` nếu
  `entity_image` khác `None` và `model.relation_conditioned` đúng. Kiểm tra cả hai trước khi fuse.
- Gather từ `table` ở `table.device`, chuyển sang `device` sau.

- [ ] **Step 1: Viết test bộ dựng tay.** Mô hình đồ chơi trả nguyên `text`:

```python
class IdentityModel(nn.Module):
    def __init__(self, rel):
        super().__init__()
        self.embed_dim = rel.shape[1]
        self.relation_embeddings = nn.Embedding.from_pretrained(rel, freeze=True)
    def forward_fusion(self, text, rgb, depth):
        return text

# image 0: e0, e1, e2, e6 (e6 has no triple, still a candidate); image 1: e3, e4, e5
E = torch.tensor([(0, 0), (1, 0), (1, 2), (5, 5), (1, 1), (6, 5), (3, 0)], dtype=torch.float32)
IMG = torch.tensor([0, 0, 0, 1, 1, 1, 0])
REL = torch.tensor([(1, 0), (0, 2), (0, 1)], dtype=torch.float32)
TRAIN, TEST = [(0, 0, 1)], [(0, 0, 2), (0, 1, 2), (3, 0, 5), (4, 2, 3)]
```

`test_hand_built_ranks` chạy `evaluate_link_prediction(..., entity_image=IMG, return_ranks=True,
keep_on_gpu=False)` với filter = TRAIN ∪ TEST và assert (comment trong test ghi khoảng cách L1 tính tay cho từng truy
vấn):

```python
assert ranks["rank_tail"].tolist() == [3, 1, 1, 6]      # main metric unchanged
assert ranks["rank_head"].tolist() == [2, 2, 1, 5]
assert ranks["rank_tail_img"].tolist() == [2, 1, 1, 2]  # (0,0,2): e1 filtered, e6 ties (not better), e0 better
assert ranks["rank_head_img"].tolist() == [2, 2, 1, 3]  # (0,0,2): e4 ties across images, outside the image anyway
assert ranks["rank_rel"].tolist() == [1, 1, 1, 2]       # (0,?,2): r1 better but true -> filtered; (4,?,3): r1 better, r0 ties
ii, rel = res["in_image"], res["relation"]
assert abs(ii["mrr"] - 2 / 3) < 1e-12 and ii["hits@1"] == 3 / 8 and ii["hits@3"] == 1.0
assert ii["n_cand"] == 3.5 and ii["n_cand_filtered"] == 27 / 8
assert abs(ii["image_first"] - 5 / 8) < 1e-12 and ii["n_negative_gap"] == 0
H = lambda n: sum(1 / k for k in range(1, n + 1))
assert abs(ii["mrr_random"] - np.mean([H(n) / n for n in [4, 4, 3, 3, 3, 4, 3, 3]])) < 1e-12
assert abs(rel["mrr"] - 0.875) < 1e-12 and rel["hits@1"] == 0.75 and rel["hits@5"] == 1.0
json.dumps({k: v for k, v in res.items() if k != "ranks"})  # Review Focus 1
```

- [ ] **Step 2: Viết các test còn lại**
  - `test_single_image_matches_main_ranks`: N = 200, R = 5, d = 6, `text` là số nguyên ngẫu nhiên trong [−5, 5]
    (float32, tổng chính xác), quan hệ số nguyên, khoảng 600 bộ ba ngẫu nhiên có nhiều (h, r) / (r, t) dùng chung (như
    `test_ranks_match_brute_force`), `entity_image = zeros(N)`, `entity_chunk_size=37`, batch 64. Assert
    `rank_tail_img == rank_tail`, `rank_head_img == rank_head`, `in_image["image_first"] == 1.0`.
  - `test_relation_ranks_match_brute_force`: dữ liệu float ngẫu nhiên, một số cặp (h, t) mang 2–3 quan hệ; brute force
    trong test (một biểu thức cho cả ứng viên và đáp án, như `brute_force` của `tests/test_evaluation.py`).
  - `test_main_metrics_unchanged`: cùng dữ liệu, có và không có `entity_image` → `rank_head`, `rank_tail`, `mrr`
    giống hệt.
  - `test_cross_image_triple_raises`: bộ dựng tay với `IMG` đổi e2 sang ảnh 1 → `ValueError`.
  - `test_relation_conditioned_raises`: mô hình có `relation_conditioned = True` kèm `entity_image` → `ValueError`.
  - `test_feature_transe_runs`: `FeatureTransE(3, feature_dim=2, embed_dim=2, hidden_dim=4, use_depth=False)` trên bộ
    dựng tay, chỉ assert có `in_image` và `relation` với `n == 8` và `n == 4`.
  - `test_save_aux_ranks`: ghi rồi đọc lại CSV, header và 4 dòng đúng với ranks của bộ dựng tay.

- [ ] **Step 3: Chạy, phải lỗi** `python tests/test_aux_metrics.py` — Expected: `TypeError` vì
  `evaluate_link_prediction` chưa nhận `entity_image`.

- [ ] **Step 4: Cài đặt** theo quy ước ở trên: thêm hàm phụ cho hạng trong ảnh và hạng quan hệ (gọi từ cùng chỗ với
  `_rank_batch` ở nhánh không relation-conditioned), `save_aux_ranks`, cập nhật docstring đầu file (thêm hai thước đo
  phụ, chỉ để báo) và docstring của `evaluate_link_prediction`.

- [ ] **Step 5: Chạy lại cả test mới và test cũ**

Run: `python tests/test_aux_metrics.py; python tests/test_evaluation.py` — Expected: không lỗi, test cũ in `OK`.

- [ ] **Step 6: Commit** (đề xuất, chờ duyệt): `core/models/evaluation.py`, `tests/test_aux_metrics.py`, message
  `Add in-image ranking and relation prediction to the evaluation (E1)`.

---

### Task 3: `train.py` ghi thước đo phụ

**Files:**
- Modify: `scripts/train.py` (hàm `evaluate` khoảng dòng 384–390, in valid khoảng dòng 443, phần test khoảng dòng
  474–477, docstring đầu file mục Output)
- Modify: `core/models/evaluation.py` (thêm `format_aux_metrics`)
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: `entity_image_ids`, `evaluate_link_prediction(..., entity_image=...)`, `save_aux_ranks`.
- Produces:
  - `aux_entity_image(data_dir: str, model) -> torch.LongTensor | None` trong `train.py`: `entity_image_ids(data_dir)`
    nếu mô hình không relation-conditioned, ngược lại `None`.
  - `format_aux_metrics(m: dict) -> str` trong `evaluation.py`: một dòng, ví dụ
    `in-image MRR 0.6667 | H@1 0.3750 | H@3 1.0000 | image-first 0.625 | cand 3.5 (3.4 filtered, random MRR 0.577) || relation MRR 0.8750 | H@1 0.7500 | H@5 1.0000`;
    trả `""` khi `m` không có `in_image`.
  - Lần chính thức trên MMSG ghi `ranks_<stem>_test_aux.csv` cạnh `ranks_<stem>_test.csv`.

- [ ] **Step 1: Viết test** `test_aux_entity_image` trong `tests/test_train_cli.py`: thư mục tạm `Spatial-VRD` với
  `entity2id.txt` 3 tên → tensor `[0, 0, 1]` với mô hình giả `relation_conditioned = False`; `None` với
  `relation_conditioned = True`; `None` với thư mục `FB15k-237-IMG`. Thêm `format_aux_metrics({...})` của bộ dựng tay
  (dùng kết quả cố định viết trong test) chứa `"in-image MRR 0.6667"` và `"relation MRR 0.8750"`; `format_aux_metrics({})
  == ""`.

- [ ] **Step 2: Chạy, phải lỗi** `python tests/test_train_cli.py` — Expected: `ImportError` / `AttributeError`.

- [ ] **Step 3: Cài đặt.** Tính `entity_image = aux_entity_image(data_dir, model)` một lần sau khi dựng mô hình, in
  một dòng cho biết thước đo phụ bật hay tắt và lý do; truyền vào `evaluate`; sau dòng in valid, in
  `format_aux_metrics(m)` nếu khác rỗng; lần chính thức gọi `save_aux_ranks` khi `entity_image` khác `None`.
  `history`, `best_valid`, `test` tự mang `in_image` / `relation` qua `strip_ranks` (không đổi hàm này).

- [ ] **Step 4: Chạy test** `python tests/test_train_cli.py; python tests/test_aux_metrics.py` — Expected: không lỗi.

- [ ] **Step 5: Chạy thử thật trên máy** (ngắn, không phải Molab):

Run: `python scripts/train.py --data data/Spatial-VRD --variant no_depth --dev --epochs 1 --valid-max 500 --out <scratchpad>/e1-smoke --no-resume`
Expected: dòng `in-image MRR ... || relation MRR ...` sau valid; `metrics_*.json` có `best_valid.in_image` và
`best_valid.relation`; `in_image.n_negative_gap` = 0; `n_cand` khoảng 19. Chạy thêm với `--data data/FB15k-237-IMG
--epochs 1 --valid-max 500 --dev`: dòng báo thước đo phụ tắt, `best_valid` không có `in_image`.

- [ ] **Step 6: Commit** (đề xuất, chờ duyệt): `scripts/train.py`, `core/models/evaluation.py`,
  `tests/test_train_cli.py`, message `Report the E1 auxiliary metrics in every train.py evaluation on scene graphs`.

---

### Task 4: danh sách cấu hình và bảng tổng hợp E1

**Files:**
- Create: `scripts/e1_summary.py`
- Test: `tests/test_e1_summary.py`

**Interfaces:**
- Consumes: `tune_summary.load_runs(root)` (đọc `<root>/<variant>/<config>/metrics_*.json` của lần `--tune`).
- Produces:
  - `E1_CONFIGS: dict[str, list[tuple[str, float, float, int | None]]]` — (biến thể, lr, dropout, epochs; `None` =
    ngân sách mặc định), giá trị chép từ spec 5.6:
    `"Spatial-VRD": [("no_depth", 3e-4, 0.0, None), ("depth_map_only", 1e-3, 0.0, None), ("depth_stats_only", 1e-4, 0.0, None), ("fixed", 1e-3, 0.0, None)]`,
    `"Spatial-VRD-Trans": [("no_depth", 1e-3, 0.0, None), ("depth_map_only", 1e-3, 0.0, None), ("depth_stats_only", 3e-4, 0.0, 200), ("fixed", 3e-4, 0.0, 200)]`.
  - `config_dir(lr, dropout, epochs) -> str`: tên thư mục cấu hình giống `train.py config_tag()`, ví dụ
    `lr0.0003-do0-ep200`.
  - `summarize(dataset: str, e1_root: str, reference_root: str | None) -> str`: bảng markdown, một dòng mỗi cấu hình
    theo thứ tự `E1_CONFIGS`; cột: biến thể, cấu hình, epoch tốt nhất, MRR chính (valid), Δ so với P2.10, epoch tốt nhất
    P2.10, MRR / H@1 / H@3 / image-first trong ảnh, MRR / H@1 / H@5 đoán quan hệ, `n_cand`, `n_cand_filtered`,
    `mrr_random`. Cấu hình chưa có file → dòng ghi `missing`. Δ > 0,002 hoặc epoch tốt nhất khác → thêm `⚠` (spec 5.6).
  - CLI: `python scripts/e1_summary.py --dataset Spatial-VRD --root <e1 root> [--reference <tune root>] [--out <file.md>]`.

- [ ] **Step 1: Viết test** với hai `metrics_*.json` giả (chỉ các khóa `load_runs` và `summarize` đọc): một ô E1 có
  `in_image` / `relation`, một ô tune tham chiếu lệch 0,003. Assert: `config_dir(3e-4, 0.0, 200) ==
  "lr0.0003-do0-ep200"`, `config_dir(1e-3, 0.0, None) == "lr0.001-do0"`; bảng có 4 dòng dữ liệu, 3 dòng `missing`,
  dòng có số chứa `⚠`; `--reference` bỏ trống vẫn chạy (cột Δ để `–`).

- [ ] **Step 2: Chạy, phải lỗi** `python tests/test_e1_summary.py` — Expected: `ModuleNotFoundError: e1_summary`.

- [ ] **Step 3: Cài đặt** `scripts/e1_summary.py`; `config_dir` phải cho cùng chuỗi với `config_tag` của `train.py`
  (gọi lại `config_tag` nếu import được không vướng, nếu không thì cùng định dạng `f"lr{lr:g}-do{do:g}"` + `-ep<epochs>`).

- [ ] **Step 4: Chạy test** `python tests/test_e1_summary.py` — Expected: không lỗi.

- [ ] **Step 5: Commit** (đề xuất, chờ duyệt): `scripts/e1_summary.py`, `tests/test_e1_summary.py`, message
  `Add the E1 configuration list and summary table`.

---

### Task 5: ô notebook E1

**Files:**
- Modify: `notebooks/geometry.py` (thêm hai ô sau ô bảng tune, khoảng dòng 670–690; thêm một dòng vào phần hướng dẫn
  đầu notebook)

**Interfaces:**
- Consumes: `E1_CONFIGS`, `config_dir`, `summarize` từ `scripts/e1_summary.py`; `run_resumable`, `push_results`,
  `remote_files`, `hf_token_ui`, `resume_min_ui`, `DATASET`, `DATA_DIR`, `REPO_ROOT`, `WORK_DIR`, `CODE_COMMIT` của
  notebook.

- [ ] **Step 1: Ô chạy.** Nút `▶ Chạy E1`; dataset ngoài `E1_CONFIGS` → `mo.stop` với thông báo. Mỗi cấu hình của
  `DATASET`: giống `_run_point` của ô tune, nhưng `--out` = `WORK_DIR/results/geometry-e1/<DATASET>/<biến thể>/<cấu hình>`,
  thư mục HF `geometry-e1/<DATASET>/<biến thể>/<cấu hình>`, thêm `--epochs` khi cấu hình có. Bỏ qua cấu hình đã có
  `metrics_*.json` trên HF hoặc trên máy. Markdown ô ghi: chỉ valid, seed 42, 8 cấu hình từ P2.10, dùng để chọn mô hình
  chính, không đụng test.

- [ ] **Step 2: Ô bảng.** Tải `geometry-e1/<DATASET>/*/*/metrics_*.json` và `geometry-tune/<DATASET>/*/*/metrics_*.json`
  từ HF, gọi `summarize(DATASET, e1_root, tune_root)`, hiện bảng bằng `mo.md`.

- [ ] **Step 3: Kiểm tra cú pháp**

Run: `python -m py_compile notebooks/geometry.py` — Expected: không lỗi. (Máy không có marimo; người dùng mở notebook
trên Molab để kiểm tra giao diện.)

- [ ] **Step 4: Commit** (đề xuất, chờ duyệt): `notebooks/geometry.py`, message `Notebook: E1 runs and summary`.

---

### Task 6: cập nhật RESEARCH_PLAN và kiểm tra toàn bộ

**Files:**
- Modify: `RESEARCH_PLAN.md` (dòng E1 trong P6.0 khoảng dòng 931–932; mục 4 sau mục 4.5)

- [ ] **Step 1: Sửa dòng E1:** bỏ "dùng lại checkpoint"; ghi: không có checkpoint (lần tune không lưu), chạy lại 8 cấu
  hình đã chọn bằng `--tune` (chỉ valid, cách A, duyệt 25/11/2026), link spec, ô notebook "E1", `scripts/e1_summary.py`.
  Thêm vào mục 4 một ý: trên MMSG báo thêm hai thước đo phụ (trong cùng ảnh, đoán quan hệ), chỉ để báo, không dùng chọn
  checkpoint; baseline chỉ có cận dưới (spec 6).

- [ ] **Step 2: Chạy toàn bộ test**

Run: `for f in tests/test_*.py; do python "$f" || echo "FAIL $f"; done` — Expected: không có dòng `FAIL`.

- [ ] **Step 3: Commit** (đề xuất, chờ duyệt): chỉ stage đoạn E1 và mục 4 của `RESEARCH_PLAN.md` (dựng patch từ
  `git diff RESEARCH_PLAN.md` chỉ gồm các hunk đó rồi `git apply --cached`; kiểm tra bằng `git diff --cached` rằng đoạn
  Feature-TransE khoảng dòng 733 không có trong đó). Kèm file spec và file kế hoạch này. Message
  `RESEARCH_PLAN: E1 design and protocol note`.

---

## Sau khi code xong (người dùng làm)

Trên Molab: chọn Spatial-VRD, bấm `▶ Chạy E1`; rồi Spatial-VRD-Trans. Xem bảng ở ô E1. Nếu có `⚠` (lệch > 0,002 so với
P2.10 hoặc khác epoch tốt nhất), dừng lại tìm nguyên nhân trước khi dùng số.
