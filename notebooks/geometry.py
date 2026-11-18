import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import json
    import subprocess
    import marimo as mo
    import torch
    from huggingface_hub import HfApi, snapshot_download

    mo.md(
        """
        # GeometryMMKG trên dữ liệu v0 (Molab)
        RESEARCH_PLAN.md mục 3.1, P2.1 / P2.6 / P2.7. Mỗi biến thể là một lần chạy `scripts/train.py`:

        | Biến thể | RGB–depth | Thiếu ảnh |
        |---|---|---|
        | `full` | gate `G = σ(W_g[rgb; depth])` | vector học được + mặt nạ |
        | `no_depth` | chỉ RGB (w/o Depth) | vector học được + mặt nạ |
        | `concat` | `Linear(2d→d)` | vector học được + mặt nạ |
        | `mean` | `(rgb + depth) / 2` | vector học được + mặt nạ |
        | `fixed` | một trọng số vô hướng học được | vector học được + mặt nạ |
        | `constant` | gate | vector hằng của bảng đặc trưng, không mặt nạ (như baseline) |

        Δdepth = `full` − `no_depth` (cùng dataset, cùng seed, cùng tham số). Giao thức mục 4: chọn checkpoint theo
        MRR valid, test một lần, ghi rank từng bộ ba cho `scripts/compute_metrics.py`. Spatial-VG đánh giá valid trên
        tập con 5.000 bộ ba cố định.

        **Thứ tự dùng:** chọn dataset (cell dữ liệu tự tải) → dán HF token, nhấn Enter → chọn biến thể, tham số →
        bấm ▶. Mỗi biến thể xong là đẩy log, rank, metric và checkpoint lên
        `geometry-mmkg-results/geometry/<dataset>/<variant>/`; biến thể đã có metric của seed đó thì bỏ qua,
        nên session tắt giữa chừng thì chỉ cần bấm lại.
        """
    )
    return HfApi, json, mo, os, snapshot_download, subprocess, sys, torch


@app.cell
def _(mo):
    DATA_REVISION = "v0"
    RESULTS_REPO = "callmezuy/geometry-mmkg-results"  # dataset repo private, tự tạo nếu chưa có
    # tên -> (repo HF, chế độ, số seed theo kế hoạch: 3 ở các ô chính, 1 ở MKG-Y và Spatial-VG; mục 4.8)
    DATASETS = {
        "FB15k-237-IMG": ("callmezuy/fb15k237-img", "transductive", 3),
        "MKG-Y": ("callmezuy/mkg-y", "transductive", 1),
        "Spatial-VRD-Trans": ("callmezuy/spatial-vrd-trans", "transductive", 3),
        "FB15K-237-ind": ("callmezuy/fb15k237-ind", "inductive", 3),
        "Spatial-VRD": ("callmezuy/spatial-vrd", "inductive", 3),
        "Spatial-VG": ("callmezuy/spatial-vg", "inductive", 1),
    }
    VARIANT_NAMES = ["full", "no_depth", "concat", "mean", "fixed", "constant"]
    dataset_ui = mo.ui.dropdown(options=list(DATASETS), value="Spatial-VRD", label="Dataset")
    dataset_ui
    return DATASETS, DATA_REVISION, RESULTS_REPO, VARIANT_NAMES, dataset_ui


@app.cell
def _(os, subprocess):
    # Mã nguồn Geometry-MMKG (một lần cho cả session)
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "scripts", "train.py")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "scripts", "train.py")):
        REPO_ROOT = os.path.dirname(WORK_DIR)
    else:
        REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
        if not os.path.exists(REPO_ROOT):
            subprocess.run(["git", "clone", "https://github.com/callmezuy/Geometry-MMKG.git", REPO_ROOT], check=True)
        else:
            subprocess.run(["git", "-C", REPO_ROOT, "pull"], check=False)
    CODE_COMMIT = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    print(f"Geometry-MMKG code at commit {CODE_COMMIT}")
    return CODE_COMMIT, REPO_ROOT, WORK_DIR


@app.cell
def _(DATASETS, DATA_REVISION, WORK_DIR, dataset_ui, mo, os, snapshot_download, torch):
    # Dữ liệu của dataset đang chọn: tải từ HF (tag v0), chỉ các file train.py và compute_metrics.py cần.
    DATASET = dataset_ui.value
    _repo, SETTING, PLANNED_SEEDS = DATASETS[DATASET]
    DATA_DIR = os.path.join(WORK_DIR, "data", DATASET)
    snapshot_download(
        repo_id=_repo, repo_type="dataset", revision=DATA_REVISION, local_dir=DATA_DIR,
        allow_patterns=[
            "entity2id.txt", "relation2id.txt", "train2id.txt", "valid2id.txt", "test2id.txt",
            "ent_text_emb.pt", "ent_rgb_emb.pt", "ent_depth_emb.pt", "ent_img_mask.pt", "ent_fill.pt",
            "relation_category.tsv", "feature_stats.json",
        ],
    )

    # Kiểm tra bảng đặc trưng: đúng số thực thể, hữu hạn; dòng thiếu ảnh của rgb / depth = vector hằng (P0.7)
    with open(os.path.join(DATA_DIR, "entity2id.txt"), encoding="utf-8") as _f:
        N_ENT = int(_f.readline())
    _load = lambda _fn: torch.load(os.path.join(DATA_DIR, _fn), map_location="cpu", weights_only=True)
    _mask, _fill = _load("ent_img_mask.pt"), _load("ent_fill.pt")
    assert _mask.dtype == torch.bool and _mask.shape == (N_ENT,), (_mask.dtype, tuple(_mask.shape))
    _miss = ~_mask
    _rows = []
    for _k in ("text", "rgb", "depth"):
        _fn = f"ent_{_k}_emb.pt"
        _x = _load(_fn)
        assert _x.dim() == 2 and _x.shape[0] == N_ENT and _x.dtype == torch.float32, (_fn, tuple(_x.shape))
        assert torch.isfinite(_x).all(), f"{_fn}: non-finite values"
        if _k in _fill and _miss.any():
            assert torch.allclose(_x[_miss], _fill[_k].expand(int(_miss.sum()), -1), atol=1e-5), \
                f"{_fn}: missing-image rows are not the constant fill"
        _rows.append(f"| `{_fn}` | {tuple(_x.shape)} |")
        del _x

    mo.md("\n".join([
        f"### Dữ liệu `{DATASET}` ({SETTING}, `{_repo}@{DATA_REVISION}`)",
        f"{N_ENT:,} thực thể, {int(_miss.sum()):,} thiếu ảnh (vector hằng đã kiểm tra). "
        f"Kế hoạch: **{PLANNED_SEEDS} seed** ({'42, 43, 44' if PLANNED_SEEDS == 3 else '42'}).",
        "",
        "| Bảng | Kích thước |",
        "|---|---|",
        *_rows,
    ]))
    return DATASET, DATA_DIR, N_ENT, PLANNED_SEEDS, SETTING


@app.cell(hide_code=False)
def _(mo):
    hf_token_ui = mo.ui.text(kind="password", label="HF token (quyền write)", full_width=True)
    mo.vstack([
        mo.md("Dán HF token có quyền **write**, nhấn **Enter**, rồi mới bấm nút chạy. Token chỉ nằm trong bộ nhớ "
              "kernel, không ghi vào file notebook. Ô này chỉ phụ thuộc `mo`, nên chạy lại các cell khác không xoá token."),
        hf_token_ui,
    ])
    return (hf_token_ui,)


@app.cell
def _(HfApi, RESULTS_REPO, os, subprocess, sys):
    def run_streaming(cmd, cwd, log_path):
        """Chạy lệnh, in log theo thời gian thực và ghi đồng thời vào log_path."""
        print("$ " + " ".join(cmd), flush=True)
        with open(log_path, "w", encoding="utf-8") as _log:
            _proc = subprocess.Popen(
                [sys.executable if cmd[0] == "python" else cmd[0]] + cmd[1:],
                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                for _line in _proc.stdout:
                    if "it/s]" in _line or "s/it]" in _line:  # tqdm progress: keep the log short
                        continue
                    print(_line, end="", flush=True)
                    _log.write(_line)
                    _log.flush()
            except BaseException:  # interrupt/stop: kill training so it does not keep running unattended
                _proc.kill()
                _proc.wait()
                raise
            return _proc.wait()

    def push_results(folder, subdir, token):
        """Đẩy folder lên RESULTS_REPO/subdir. Lỗi upload không làm hỏng kết quả local."""
        try:
            _api = HfApi(token=token)
            _api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
            _api.upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path=folder,
                               path_in_repo=subdir, commit_message=f"Add {subdir} results")
            print(f"Uploaded {folder} -> {RESULTS_REPO}/{subdir}")
            return True
        except Exception as _e:
            print(f"Upload lên HF thất bại ({_e}). Hãy tải thủ công thư mục {folder} trước khi tắt session.")
            return False

    def remote_files(prefix, token):
        """Đường dẫn (tương đối với prefix) các file đã có trên RESULTS_REPO/prefix (rỗng nếu repo chưa có)."""
        try:
            _fs = HfApi(token=token).list_repo_files(RESULTS_REPO, repo_type="dataset")
        except Exception:
            return set()
        return {_p[len(prefix) + 1:] for _p in _fs if _p.startswith(prefix + "/")}

    return push_results, remote_files, run_streaming


@app.cell
def _(DATASET, PLANNED_SEEDS, VARIANT_NAMES, mo):
    # Mặc định = scripts/train.py. Tuning (quyết định (d), 18/11/2026): lưới lr × margin trên valid của
    # no_depth, seed 42; cấu hình thắng dùng cho mọi biến thể. Ngân sách epoch chốt sau lần chạy thử.
    seed_ui = mo.ui.dropdown(options=["42", "43", "44"], value="42", label="Seed")
    variants_ui = mo.ui.multiselect(options=VARIANT_NAMES, value=["full", "no_depth"], label="Biến thể")
    epochs_ui = mo.ui.number(start=1, stop=2000, step=1, value=100, label="Epochs")
    batch_ui = mo.ui.dropdown(options=["512", "1024", "2048", "4096"], value="1024", label="Batch")
    lr_ui = mo.ui.text(value="1e-4", label="LR")
    margin_ui = mo.ui.text(value="5", label="Margin")
    adv_ui = mo.ui.text(value="1.0", label="Nhiệt độ self-adversarial")
    dropout_ui = mo.ui.text(value="0.1", label="Dropout")
    wd_ui = mo.ui.text(value="1e-5", label="Weight decay")
    valid_every_ui = mo.ui.number(start=1, stop=500, step=1, value=10, label="Đánh giá valid mỗi N epoch")
    save_ckpt_ui = mo.ui.checkbox(value=True, label="Lưu checkpoint lên HF")
    train_btn = mo.ui.run_button(label="▶ Train các biến thể đã chọn")
    mo.vstack([
        mo.md(f"`{DATASET}`: kế hoạch **{PLANNED_SEEDS} seed**. Mỗi lần bấm chạy một seed; đổi tham số **không** "
              "tự chạy lại training. Dùng cùng tham số cho mọi biến thể (Δdepth so trong cùng config)."
              + (" Valid dùng tập con 5.000 bộ ba (mục 4.9)." if DATASET == "Spatial-VG" else "")),
        mo.hstack([seed_ui, variants_ui]),
        mo.hstack([epochs_ui, batch_ui, lr_ui, margin_ui]),
        mo.hstack([adv_ui, dropout_ui, wd_ui, valid_every_ui]),
        save_ckpt_ui,
        train_btn,
    ])
    return (
        adv_ui,
        batch_ui,
        dropout_ui,
        epochs_ui,
        lr_ui,
        margin_ui,
        save_ckpt_ui,
        seed_ui,
        train_btn,
        valid_every_ui,
        variants_ui,
        wd_ui,
    )


@app.cell
def _(
    CODE_COMMIT,
    DATASET,
    DATA_DIR,
    REPO_ROOT,
    WORK_DIR,
    adv_ui,
    batch_ui,
    dropout_ui,
    epochs_ui,
    hf_token_ui,
    lr_ui,
    margin_ui,
    mo,
    os,
    push_results,
    remote_files,
    run_streaming,
    save_ckpt_ui,
    seed_ui,
    train_btn,
    valid_every_ui,
    variants_ui,
    wd_ui,
):
    mo.stop(not train_btn.value, mo.md("⏸ Bấm nút ▶ ở trên để bắt đầu."))
    _token = hf_token_ui.value.strip()
    mo.stop(not _token, mo.md("⚠ Chưa nhập HF token ở ô phía trên: nhập, nhấn Enter, rồi bấm chạy lại."))
    _seed = int(seed_ui.value)
    _prefix = f"geometry/{DATASET}"
    RESULTS_DIR = os.path.join(WORK_DIR, "results", "geometry", DATASET)
    _done_remote = remote_files(_prefix, _token)
    train_status = {}
    for _v in variants_ui.value:
        _stem = f"geometry_{DATASET}_{_v}_seed{_seed}"
        _metrics = f"metrics_{_stem}.json"
        _out = os.path.join(RESULTS_DIR, _v)
        os.makedirs(_out, exist_ok=True)
        if f"{_v}/{_metrics}" in _done_remote or os.path.exists(os.path.join(_out, _metrics)):
            print(f"=== {_v}: đã có {_metrics}, bỏ qua ===")
            train_status[_v] = "đã có"
            continue
        print(f"=== {_v}, seed {_seed}, code {CODE_COMMIT} ===", flush=True)
        _cmd = [
            "python", os.path.join(REPO_ROOT, "scripts", "train.py"),
            "--data", DATA_DIR,
            "--variant", _v,
            "--seed", str(_seed),
            "--epochs", str(int(epochs_ui.value)),
            "--batch-size", str(int(batch_ui.value)),
            "--lr", str(float(lr_ui.value)),
            "--margin", str(float(margin_ui.value)),
            "--adv-temp", str(float(adv_ui.value)),
            "--dropout", str(float(dropout_ui.value)),
            "--weight-decay", str(float(wd_ui.value)),
            "--valid-every", str(int(valid_every_ui.value)),
            "--out", _out,
        ]
        if not save_ckpt_ui.value:
            _cmd.append("--no-ckpt")
        _rc = run_streaming(_cmd, REPO_ROOT, os.path.join(_out, f"{_stem}.log"))
        train_status[_v] = "xong" if _rc == 0 else f"lỗi (exit {_rc})"
        push_results(_out, f"{_prefix}/{_v}", _token)
    mo.md("### Kết quả train, seed {}\n\n".format(_seed)
          + "\n".join(f"- `{_k}`: {_v}" for _k, _v in train_status.items()))
    return RESULTS_DIR, train_status


@app.cell
def _(DATASET, DATA_DIR, REPO_ROOT, RESULTS_DIR, VARIANT_NAMES, mo, os, seed_ui, subprocess, sys, train_status):
    # Metric từ rank CSV của seed này (một nguồn duy nhất: scripts/compute_metrics.py), so với no_depth (Δdepth).
    _ = train_status
    _seed = int(seed_ui.value)
    _ranks = []
    for _v in VARIANT_NAMES:
        _p = os.path.join(RESULTS_DIR, _v, f"ranks_geometry_{DATASET}_{_v}_seed{_seed}_test.csv")
        if os.path.exists(_p):
            _ranks.append(f"{_v}={_p}")
    mo.stop(not _ranks, mo.md("Chưa có rank CSV nào trên máy cho seed này."))
    _out = os.path.join(RESULTS_DIR, f"metrics_geometry_{DATASET}_seed{_seed}.md")
    _cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "compute_metrics.py"), "--data", DATA_DIR,
            "--ranks", *_ranks, "--out", _out]
    if any(_r.startswith("no_depth=") for _r in _ranks) and len(_ranks) > 1:
        _cmd += ["--reference", "no_depth"]
    _res = subprocess.run(_cmd, text=True, capture_output=True)
    print(_res.stderr[-2000:])
    mo.md(open(_out, encoding="utf-8").read() if _res.returncode == 0 and os.path.exists(_out)
          else f"compute_metrics lỗi (exit {_res.returncode}), xem stderr ở trên.")
    return


if __name__ == "__main__":
    app.run()
