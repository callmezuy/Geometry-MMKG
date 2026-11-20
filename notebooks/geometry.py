import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import json
    import shutil
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
        nên session tắt giữa chừng thì chỉ cần bấm lại. Lần chạy dở dang cũng không mất: `train.py` lưu trạng thái
        huấn luyện sau mỗi lần valid, notebook đẩy nó (kèm log) lên repo private `geometry-mmkg-resume` và bấm lại
        thì chạy tiếp từ epoch đã lưu (xong thì file đó bị xoá).
        """
    )
    return HfApi, json, mo, os, shutil, snapshot_download, subprocess, sys, torch


@app.cell
def _(mo):
    DATA_REVISION = "v0"
    RESULTS_REPO = "callmezuy/geometry-mmkg-results"  # dataset repo private, tự tạo nếu chưa có
    # trạng thái huấn luyện dở dang (train.py resume_<stem>.pt), chỉ là file tạm: repo riêng để dọn lịch sử thoải mái
    RESUME_REPO = "callmezuy/geometry-mmkg-resume"
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
    return DATASETS, DATA_REVISION, RESULTS_REPO, RESUME_REPO, VARIANT_NAMES, dataset_ui


@app.cell
def _(os, shutil, subprocess):
    # Mã nguồn Geometry-MMKG (một lần cho cả session)
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "scripts", "train.py")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "scripts", "train.py")):
        REPO_ROOT = os.path.dirname(WORK_DIR)
    else:
        REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
        # Molab giữ lại thư mục làm việc giữa các session nhưng chỉ còn file .py/.md (mất .git, mã C++, checkpoint),
        # nên thiếu .git thì xoá và clone lại.
        if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
            shutil.rmtree(REPO_ROOT, ignore_errors=True)
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
            "ent_depth_stats.pt", "depth_stats.json",  # scene graphs only (P2.9)
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
    HAS_DEPTH_STATS = os.path.exists(os.path.join(DATA_DIR, "ent_depth_stats.pt"))
    if HAS_DEPTH_STATS:
        _x = _load("ent_depth_stats.pt")
        assert _x.dim() == 2 and _x.shape[0] == N_ENT and torch.isfinite(_x).all(), tuple(_x.shape)
        _rows.append(f"| `ent_depth_stats.pt` | {tuple(_x.shape)} |")

    mo.md("\n".join([
        f"### Dữ liệu `{DATASET}` ({SETTING}, `{_repo}@{DATA_REVISION}`)",
        f"{N_ENT:,} thực thể, {int(_miss.sum()):,} thiếu ảnh (vector hằng đã kiểm tra). "
        f"Kế hoạch: **{PLANNED_SEEDS} seed** ({'42, 43, 44' if PLANNED_SEEDS == 3 else '42'}).",
        "",
        "| Bảng | Kích thước |",
        "|---|---|",
        *_rows,
    ]))
    return DATASET, DATA_DIR, HAS_DEPTH_STATS, N_ENT, PLANNED_SEEDS, SETTING


@app.cell(hide_code=False)
def _(mo):
    hf_token_ui = mo.ui.text(kind="password", label="HF token (quyền write)", full_width=True)
    mo.vstack([
        mo.md("Dán HF token có quyền **write**, nhấn **Enter**, rồi mới bấm nút chạy. Token chỉ nằm trong bộ nhớ "
              "kernel, không ghi vào file notebook. Ô này chỉ phụ thuộc `mo`, nên chạy lại các cell khác không xoá token."),
        hf_token_ui,
    ])
    return (hf_token_ui,)


@app.cell(hide_code=False)
def _(mo):
    resume_min_ui = mo.ui.number(start=0, stop=240, step=1, value=10,
                                 label="Đẩy trạng thái dở dang lên HF tối đa mỗi N phút (0 = không đẩy, chỉ giữ trên máy)")
    mo.vstack([
        mo.md("**Chạy tiếp khi bị ngắt.** `train.py` ghi trạng thái huấn luyện (model, optimizer, RNG, checkpoint tốt "
              "nhất, lịch sử) sau mỗi lần valid; notebook đẩy nó cùng log lên `geometry-mmkg-resume`, cách nhau ít "
              "nhất N phút. Bấm chạy lại thì notebook tải về và train tiếp từ epoch đã lưu, cùng batch và dropout như "
              "lúc chưa ngắt; metric ghi lại các epoch đã chạy tiếp (`resumed_at`). Bị ngắt thì mất tối đa N phút "
              "cộng một khoảng valid."),
        resume_min_ui,
    ])
    return (resume_min_ui,)


@app.cell
def _(HfApi, RESULTS_REPO, RESUME_REPO, os, shutil, subprocess, sys):
    import time as _time
    from huggingface_hub import CommitOperationAdd as _CommitOperationAdd, hf_hub_download as _hf_hub_download

    def run_streaming(cmd, cwd, log_path, on_line=None, append=False):
        """Chạy lệnh, in log theo thời gian thực và ghi đồng thời vào log_path (append: nối vào log của phần trước)."""
        print("$ " + " ".join(cmd), flush=True)
        with open(log_path, "a" if append else "w", encoding="utf-8") as _log:
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
                    if on_line is not None:
                        on_line(_line)
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
                               path_in_repo=subdir, commit_message=f"Add {subdir} results",
                               ignore_patterns=["resume_*.pt", "*.tmp"])
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

    def run_resumable(cmd, cwd, out, subdir, stem, token, minutes):
        """run_streaming cho một lần train.py, chạy tiếp được khi bị ngắt:
        1. nếu RESUME_REPO/subdir có resume_<stem>.pt (lần trước bị ngắt) thì tải nó và log của phần trước về `out`;
        2. mỗi khi train.py báo `resume state saved`, đẩy resume_<stem>.pt + log lên RESUME_REPO/subdir (một commit),
           nếu lần đẩy trước đã cách ít nhất `minutes` phút (0 = không đẩy);
        3. chạy xong (exit 0) thì xoá chúng trên RESUME_REPO. Lỗi mạng không bao giờ làm dừng training."""
        _api = HfApi(token=token)
        _names = [f"resume_{stem}.pt", f"{stem}.log"]
        _log = os.path.join(out, _names[1])
        _append = os.path.exists(os.path.join(out, _names[0]))   # cùng session: file vẫn còn trên máy
        if not _append:
            try:
                _remote = set(_api.list_repo_files(RESUME_REPO, repo_type="dataset"))
            except Exception:
                _remote = set()   # repo chưa có
            if f"{subdir}/{_names[0]}" in _remote:
                for _n in _names:
                    if f"{subdir}/{_n}" in _remote:
                        _src = _hf_hub_download(RESUME_REPO, f"{subdir}/{_n}", repo_type="dataset", token=token)
                        shutil.copyfile(_src, os.path.join(out, _n))
                _append = True
                print(f"[resume] tải trạng thái dở dang từ {RESUME_REPO}/{subdir}, train tiếp từ đó", flush=True)
        _last = [0.0]

        def _on_line(line):
            if minutes <= 0 or not line.startswith("resume state saved: ") or _time.time() - _last[0] < minutes * 60:
                return
            try:
                _api.create_repo(RESUME_REPO, repo_type="dataset", private=True, exist_ok=True)
                _api.create_commit(RESUME_REPO, repo_type="dataset", commit_message=f"{subdir}: {line.strip()[20:]}",
                                   operations=[_CommitOperationAdd(f"{subdir}/{_n}", os.path.join(out, _n))
                                               for _n in _names])
                _last[0] = _time.time()
                print(f"[resume] đã đẩy trạng thái lên HF ({line.strip().rsplit('(', 1)[-1]}", flush=True)
            except Exception as _e:
                print(f"[resume] đẩy lên HF thất bại ({_e}); training vẫn chạy tiếp", flush=True)

        _rc = run_streaming(cmd, cwd, _log, on_line=_on_line, append=_append)
        if _rc == 0:
            try:
                _remote = set(_api.list_repo_files(RESUME_REPO, repo_type="dataset"))
                for _n in _names:
                    if f"{subdir}/{_n}" in _remote:
                        _api.delete_file(f"{subdir}/{_n}", RESUME_REPO, repo_type="dataset",
                                         commit_message=f"{subdir}: finished")
            except Exception:
                pass   # repo chưa có hoặc lỗi mạng: file tạm còn lại không ảnh hưởng kết quả
        return _rc

    return push_results, remote_files, run_resumable, run_streaming


@app.cell
def _(DATASET, HAS_DEPTH_STATS, PLANNED_SEEDS, VARIANT_NAMES, mo):
    # Mặc định = thiết kế đóng băng (P2.8 / P2.9, 19/11/2026): input_norm + depth_dropout 0.3 + text_proj, cộng
    # depth_stats trên bộ scene graph. Bỏ chọn một cờ = chạy ablation không có cờ đó (truyền --no-... cho train.py).
    # Chế độ dev chỉ đánh giá valid, không bao giờ chạy test.
    seed_ui = mo.ui.dropdown(options=["42", "43", "44"], value="42", label="Seed")
    variants_ui = mo.ui.multiselect(options=VARIANT_NAMES, value=["full", "no_depth"], label="Biến thể")
    dev_ui = mo.ui.checkbox(value=True, label="Chế độ dev (P2.8): chỉ valid, không chạy test")
    options_ui = mo.ui.multiselect(
        options=["rel_gate", "input_norm", "depth_dropout", "text_proj", "depth_stats"],
        value=["input_norm", "depth_dropout", "text_proj"] + (["depth_stats"] if HAS_DEPTH_STATS else []),
        label="Cờ (mặc định = thiết kế đóng băng; depth_stats chỉ có ở bộ scene graph)")
    ddrop_ui = mo.ui.text(value="0.3", label="Xác suất depth_dropout")
    # P2.9: số độ sâu theo vùng (scripts/vrd_depth_stats.py), file nằm trong repo kết quả private, dạng PATH[:KEY]
    dstat_ui = mo.ui.text(value="", label="Số độ sâu từ file thử nghiệm (P2.9, file trong repo kết quả, ví dụ "
                                         "probe/vrd_depth_stats_dav2.pt:grid9; để trống = tắt; khác cờ depth_stats "
                                         "là bảng của dataset)", full_width=True)
    epochs_ui = mo.ui.number(start=0, stop=4000, step=1, value=0, label="Epochs (0 = ngân sách của dataset, P2.10)")
    batch_ui = mo.ui.dropdown(options=["512", "1024", "2048", "4096"], value="1024", label="Batch")
    lr_ui = mo.ui.text(value="1e-4", label="LR")
    margin_ui = mo.ui.text(value="5", label="Margin")
    adv_ui = mo.ui.text(value="1.0", label="Nhiệt độ self-adversarial")
    dropout_ui = mo.ui.text(value="0.1", label="Dropout")
    wd_ui = mo.ui.text(value="1e-5", label="Weight decay")
    valid_every_ui = mo.ui.number(start=0, stop=500, step=1, value=0, label="Valid mỗi N epoch (0 = theo ngân sách)")
    save_ckpt_ui = mo.ui.checkbox(value=True, label="Lưu checkpoint lên HF (không áp dụng cho dev)")
    train_btn = mo.ui.run_button(label="▶ Train các biến thể đã chọn")
    mo.vstack([
        mo.md(f"`{DATASET}`: kế hoạch **{PLANNED_SEEDS} seed**. Mỗi lần bấm chạy một seed; đổi tham số **không** "
              "tự chạy lại training. Dùng cùng tham số và cùng cờ cho mọi biến thể (Δdepth so trong cùng config); "
              "biến thể nào không dùng được một cờ (ví dụ `no_depth` với `rel_gate`) thì tự bỏ cờ đó."
              + (" Valid dùng tập con 5.000 bộ ba (mục 4.9)." if DATASET == "Spatial-VG" else "")),
        mo.hstack([seed_ui, variants_ui]),
        mo.hstack([dev_ui, options_ui, ddrop_ui]),
        dstat_ui,
        mo.hstack([epochs_ui, batch_ui, lr_ui, margin_ui]),
        mo.hstack([adv_ui, dropout_ui, wd_ui, valid_every_ui]),
        save_ckpt_ui,
        train_btn,
    ])
    return (
        adv_ui,
        batch_ui,
        ddrop_ui,
        dev_ui,
        dropout_ui,
        dstat_ui,
        epochs_ui,
        lr_ui,
        margin_ui,
        options_ui,
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
    HAS_DEPTH_STATS,
    REPO_ROOT,
    RESULTS_REPO,
    WORK_DIR,
    adv_ui,
    batch_ui,
    ddrop_ui,
    dev_ui,
    dropout_ui,
    dstat_ui,
    epochs_ui,
    hf_token_ui,
    lr_ui,
    margin_ui,
    mo,
    options_ui,
    os,
    push_results,
    remote_files,
    resume_min_ui,
    run_resumable,
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
    _dev = bool(dev_ui.value)
    _opts = set(options_ui.value)
    _ddrop = float(ddrop_ui.value) if "depth_dropout" in _opts else 0.0
    # every option explicit, on or off (train.py defaults to the frozen design)
    _flags = (["--rel-gate"] if "rel_gate" in _opts else []) \
        + (["--input-norm"] if "input_norm" in _opts else ["--no-input-norm"]) \
        + ["--depth-dropout", str(_ddrop)] \
        + (["--text-proj"] if "text_proj" in _opts else ["--no-text-proj"])
    _dstat_tag = []
    mo.stop("depth_stats" in _opts and bool(dstat_ui.value.strip()),
            mo.md("⚠ Chọn cờ `depth_stats` (bảng của dataset) **hoặc** điền file thử nghiệm, không cả hai."))
    mo.stop("depth_stats" in _opts and not HAS_DEPTH_STATS,
            mo.md(f"⚠ `{DATASET}` không có `ent_depth_stats.pt` (chỉ bộ scene graph có), bỏ cờ `depth_stats`."))
    if "depth_stats" in _opts:
        _flags += ["--depth-stats", "data"]
        _dstat_tag = ["dstat"]
    elif not dstat_ui.value.strip():
        _flags += ["--depth-stats", "none"]
    else:
        from huggingface_hub import hf_hub_download as _hf_download

        _spec = dstat_ui.value.strip()
        _remote, _key = _spec.rsplit(":", 1) if ":" in _spec else (_spec, "grid9")
        _local = _hf_download(RESULTS_REPO, _remote, repo_type="dataset", token=_token,
                              local_dir=os.path.join(WORK_DIR, "depth_stats"))
        _flags += ["--depth-stats", f"{_local}:{_key}"]
        _name = os.path.splitext(os.path.basename(_remote))[0].split("depth_stats_", 1)[-1]
        _dstat_tag = [f"dstat_{_name}_{_key}"]
    # same tag as scripts/train.py options_tag()
    OPTIONS_TAG = "-".join(
        (["relgate"] if "rel_gate" in _opts else []) + (["inorm"] if "input_norm" in _opts else [])
        + ([f"ddrop{_ddrop:g}"] if _ddrop > 0 else []) + (["tproj"] if "text_proj" in _opts else [])
        + _dstat_tag) or "base"
    # dev: geometry-dev/<dataset>/<tag>/<variant>/ ; chính thức: geometry/<dataset>/<variant>/
    _prefix = f"geometry-dev/{DATASET}/{OPTIONS_TAG}" if _dev else f"geometry/{DATASET}"
    RESULTS_DIR = os.path.join(WORK_DIR, "results", *_prefix.split("/"))
    _done_remote = remote_files(_prefix, _token)
    train_status = {}
    for _v in variants_ui.value:
        _stem = f"geometry_{DATASET}_{_v}{'' if OPTIONS_TAG == 'base' else '_' + OPTIONS_TAG}_seed{_seed}"
        _metrics = f"metrics_{_stem}.json"
        _out = os.path.join(RESULTS_DIR, _v)
        os.makedirs(_out, exist_ok=True)
        if f"{_v}/{_metrics}" in _done_remote or os.path.exists(os.path.join(_out, _metrics)):
            print(f"=== {_v}: đã có {_metrics}, bỏ qua ===")
            train_status[_v] = "đã có"
            continue
        print(f"=== {_v}, seed {_seed}, cờ {OPTIONS_TAG}{' [dev]' if _dev else ''}, code {CODE_COMMIT} ===",
              flush=True)
        _cmd = [
            "python", os.path.join(REPO_ROOT, "scripts", "train.py"),
            "--data", DATA_DIR,
            "--variant", _v,
            "--seed", str(_seed),
            *(["--epochs", str(int(epochs_ui.value))] if int(epochs_ui.value) > 0 else []),
            "--batch-size", str(int(batch_ui.value)),
            "--lr", str(float(lr_ui.value)),
            "--margin", str(float(margin_ui.value)),
            "--adv-temp", str(float(adv_ui.value)),
            "--dropout", str(float(dropout_ui.value)),
            "--weight-decay", str(float(wd_ui.value)),
            *(["--valid-every", str(int(valid_every_ui.value))] if int(valid_every_ui.value) > 0 else []),
            "--out", _out,
            *_flags,
        ]
        if _dev:
            _cmd.append("--dev")
        elif not save_ckpt_ui.value:
            _cmd.append("--no-ckpt")
        _rc = run_resumable(_cmd, REPO_ROOT, _out, f"{_prefix}/{_v}", _stem, _token, int(resume_min_ui.value))
        train_status[_v] = "xong" if _rc == 0 else f"lỗi (exit {_rc})"
        push_results(_out, f"{_prefix}/{_v}", _token)
    mo.md("### Kết quả train, seed {}, cờ `{}`{}\n\n".format(_seed, OPTIONS_TAG, " (dev)" if _dev else "")
          + "\n".join(f"- `{_k}`: {_v}" for _k, _v in train_status.items()))
    return OPTIONS_TAG, RESULTS_DIR, train_status


@app.cell
def _(
    DATASET,
    DATA_DIR,
    OPTIONS_TAG,
    REPO_ROOT,
    RESULTS_DIR,
    VARIANT_NAMES,
    dev_ui,
    mo,
    os,
    seed_ui,
    subprocess,
    sys,
    train_status,
):
    # Metric từ rank CSV của seed này (một nguồn duy nhất: scripts/compute_metrics.py), so với no_depth (Δdepth).
    # Chế độ dev không có rank test: xem bảng valid ở cell dưới.
    _ = train_status
    mo.stop(dev_ui.value, mo.md("Chế độ dev: không có rank test (xem bảng valid ở cell dưới)."))
    _seed = int(seed_ui.value)
    _sfx = "" if OPTIONS_TAG == "base" else "_" + OPTIONS_TAG
    _ranks = []
    for _v in VARIANT_NAMES:
        _p = os.path.join(RESULTS_DIR, _v, f"ranks_geometry_{DATASET}_{_v}{_sfx}_seed{_seed}_test.csv")
        if os.path.exists(_p):
            _ranks.append(f"{_v}={_p}")
    mo.stop(not _ranks, mo.md("Chưa có rank CSV nào trên máy cho seed này."))
    _out = os.path.join(RESULTS_DIR, f"metrics_geometry_{DATASET}{_sfx}_seed{_seed}.md")
    _cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "compute_metrics.py"), "--data", DATA_DIR,
            "--ranks", *_ranks, "--out", _out]
    if any(_r.startswith("no_depth=") for _r in _ranks) and len(_ranks) > 1:
        _cmd += ["--reference", "no_depth"]
    _res = subprocess.run(_cmd, text=True, capture_output=True)
    print(_res.stderr[-2000:])
    mo.md(open(_out, encoding="utf-8").read() if _res.returncode == 0 and os.path.exists(_out)
          else f"compute_metrics lỗi (exit {_res.returncode}), xem stderr ở trên.")
    return


@app.cell
def _(DATASET, RESULTS_REPO, WORK_DIR, hf_token_ui, json, mo, os, snapshot_download, train_status):
    # P2.8: bảng valid của mọi lần chạy dev trên dataset này (tải metric JSON từ repo kết quả, nên xem được cả các
    # lần chạy ở session trước). Không có số test nào ở đây.
    _ = train_status
    _token = hf_token_ui.value.strip()
    mo.stop(not _token, mo.md("Nhập HF token để xem bảng valid của các lần chạy dev."))
    _local = os.path.join(WORK_DIR, "hf_results")
    try:
        snapshot_download(RESULTS_REPO, repo_type="dataset", token=_token, local_dir=_local,
                          allow_patterns=[f"geometry-dev/{DATASET}/*/*/metrics_*.json"])
    except Exception as _e:
        print(f"Không tải được metric dev: {_e}")
    _root = os.path.join(_local, "geometry-dev", DATASET)
    _rows = []
    for _tag in sorted(os.listdir(_root)) if os.path.isdir(_root) else []:
        for _v in sorted(os.listdir(os.path.join(_root, _tag))):
            for _fn in sorted(os.listdir(os.path.join(_root, _tag, _v))):
                if not _fn.startswith("metrics_"):
                    continue
                with open(os.path.join(_root, _tag, _v, _fn), encoding="utf-8") as _f:
                    _m = json.load(_f)
                _vals = [_h["valid"]["mrr"] for _h in _m["history"] if "valid" in _h]
                _g = _m.get("best_gate") or {}
                _gate = "" if not _g else f"{_g['mean']:.3f}" + (
                    f" (sd thực thể {_g['std_entities']:.3f}" if "std_entities" in _g else "") + (
                    f", sd quan hệ {_g['std_relations']:.3f}" if "std_relations" in _g else "") + (
                    ")" if "std_entities" in _g else "")
                _rows.append(f"| `{_tag}` | `{_v}` | {_m['seed']} | {_m['args']['epochs']} | {_m['best_epoch']} | "
                             f"{_m['best_valid']['mrr']:.4f} | {_vals[-1]:.4f} | {_gate} |")
    mo.md("\n".join([f"### Các lần chạy dev trên `{DATASET}` (chỉ valid)", "",
                     "| Cờ | Biến thể | Seed | Epochs | Epoch tốt nhất | MRR valid tốt nhất | MRR valid cuối | "
                     "Phần depth của gate |",
                     "|---|---|---|---|---|---|---|---|", *_rows]) if _rows else "Chưa có lần chạy dev nào.")
    return


@app.cell
def _(DATASET, VARIANT_NAMES, mo):
    # P2.10 (quyết định 19/11/2026): lưới lr x dropout, tune riêng mọi biến thể (cách A), seed 42, chỉ valid.
    # Thiết kế đóng băng và ngân sách epoch = mặc định của scripts/train.py (EPOCH_BUDGET), không truyền tay.
    _ablation = DATASET in ("FB15k-237-IMG", "Spatial-VRD")  # 6 biến thể ở đây, full + no_depth ở các bộ khác
    tune_variants_ui = mo.ui.multiselect(options=VARIANT_NAMES,
                                         value=list(VARIANT_NAMES) if _ablation else ["full", "no_depth"],
                                         label="Biến thể tune")
    tune_lrs_ui = mo.ui.text(value="1e-4,3e-4,1e-3", label="LR (lưới)")
    tune_dos_ui = mo.ui.text(value="0.1,0.3", label="Dropout (lưới)")
    tune_mode_ui = mo.ui.radio(options=["Lưới gốc", "Mở rộng biên"], value="Lưới gốc", label="Chế độ (Mở rộng biên = luật biên + luật hội tụ)")
    tune_btn = mo.ui.run_button(label="▶ Chạy tune (P2.10)")
    mo.vstack([
        mo.md(f"### Tune `{DATASET}` (P2.10)\n\nMỗi điểm lưới là một lần `train.py --tune` (chỉ valid, seed 42, "
              "thiết kế đóng băng, số epoch theo ngân sách của dataset). Điểm đã có kết quả trên HF thì bỏ qua, nên "
              "bấm lại sau khi bị ngắt sẽ chạy tiếp phần còn thiếu. Kết quả: `geometry-tune/<dataset>/<biến thể>/<cấu hình>/`.\n\n"
              "**Mở rộng biên** (luật 20/11/2026): sau lưới gốc, `tune_summary.py` tìm các điểm một bậc ra ngoài biên cho "
              "biến thể có cấu hình chọn nằm ở biên; notebook chạy chúng rồi tính lại, lặp tới khi không còn điểm nào."),
        mo.hstack([tune_variants_ui, tune_lrs_ui, tune_dos_ui]),
        tune_mode_ui,
        tune_btn,
    ])
    return tune_btn, tune_dos_ui, tune_lrs_ui, tune_mode_ui, tune_variants_ui


@app.cell
def _(
    CODE_COMMIT,
    DATASET,
    DATA_DIR,
    HAS_DEPTH_STATS,
    REPO_ROOT,
    RESULTS_REPO,
    WORK_DIR,
    hf_token_ui,
    mo,
    os,
    push_results,
    remote_files,
    resume_min_ui,
    run_resumable,
    snapshot_download,
    sys,
    tune_btn,
    tune_dos_ui,
    tune_lrs_ui,
    tune_mode_ui,
    tune_variants_ui,
):
    mo.stop(not tune_btn.value, mo.md("⏸ Bấm **▶ Chạy tune** để bắt đầu."))
    _token = hf_token_ui.value.strip()
    mo.stop(not _token, mo.md("⚠ Chưa nhập HF token ở ô phía trên."))
    # same tag as scripts/train.py options_tag() for the frozen design
    _tag = "inorm-ddrop0.3-tproj" + ("-dstat" if HAS_DEPTH_STATS else "")
    _prefix = f"geometry-tune/{DATASET}"
    tune_status = {}

    def _run_point(_v, _lr, _do, _done, _epochs=None):
        # _epochs: only for the convergence rerun (twice the budget); same '-ep' suffix as train.py config_tag()
        _cfg = f"lr{_lr:g}-do{_do:g}" + (f"-ep{_epochs}" if _epochs else "")
        _stem = f"geometry_{DATASET}_{_v}_{_tag}_{_cfg}_seed42"
        _out = os.path.join(WORK_DIR, "results", "geometry-tune", DATASET, _v, _cfg)
        os.makedirs(_out, exist_ok=True)
        _metrics = f"metrics_{_stem}.json"
        if f"{_v}/{_cfg}/{_metrics}" in _done or os.path.exists(os.path.join(_out, _metrics)):
            tune_status[f"{_v} {_cfg}"] = "đã có"
            return
        print(f"=== tune {_v} {_cfg}, code {CODE_COMMIT} ===", flush=True)
        _rc = run_resumable(
            ["python", os.path.join(REPO_ROOT, "scripts", "train.py"), "--data", DATA_DIR, "--variant", _v,
             "--seed", "42", "--tune", "--lr", str(_lr), "--dropout", str(_do), "--out", _out,
             *(["--epochs", str(_epochs)] if _epochs else [])],
            REPO_ROOT, _out, f"{_prefix}/{_v}/{_cfg}", _stem, _token, int(resume_min_ui.value))
        tune_status[f"{_v} {_cfg}"] = "xong" if _rc == 0 else f"lỗi (exit {_rc})"
        push_results(_out, f"{_prefix}/{_v}/{_cfg}", _token)

    if tune_mode_ui.value == "Lưới gốc":
        _lrs = [float(_x) for _x in tune_lrs_ui.value.split(",") if _x.strip()]
        _dos = [float(_x) for _x in tune_dos_ui.value.split(",") if _x.strip()]
        _done = remote_files(_prefix, _token)
        for _v in tune_variants_ui.value:
            for _lr in _lrs:
                for _do in _dos:
                    _run_point(_v, _lr, _do, _done)
    else:
        # edge rule: recompute the choice from every finished point (HF + this session), run the proposed points, repeat
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        import tune_summary as _ts

        for _round in range(1, 7):
            _hf = os.path.join(WORK_DIR, "hf_results")
            snapshot_download(RESULTS_REPO, repo_type="dataset", token=_token, local_dir=_hf,
                              allow_patterns=[f"{_prefix}/*/*/metrics_*.json"])
            _runs = {}
            for _root in (os.path.join(_hf, *_prefix.split("/")),
                          os.path.join(WORK_DIR, "results", "geometry-tune", DATASET)):
                for _r in _ts.load_runs(_root) if os.path.isdir(_root) else []:
                    _runs[(_r["variant"], _r["config"]["lr"], _r["config"]["dropout"], _r["epochs"])] = _r
            _choice = _ts.choose(list(_runs.values()))
            _todo = [(_v, _p["lr"], _p["dropout"], None) for _v in tune_variants_ui.value
                     for _p in _choice.get(_v, {}).get("edge_points", [])]
            _todo += [(_v, _c["rerun"]["lr"], _c["rerun"]["dropout"], _c["rerun"]["epochs"])
                      for _v, _c in _choice.items() if _v in tune_variants_ui.value and _c.get("rerun")]
            print(f"--- edge / convergence rules, round {_round}: {len(_todo)} point(s) {_todo}", flush=True)
            if not _todo:
                break
            _done = remote_files(_prefix, _token)
            for _v, _lr, _do, _ep in _todo:
                _run_point(_v, _lr, _do, _done, _ep)
    mo.md("### Tune\n\n" + ("\n".join(f"- `{_k}`: {_s}" for _k, _s in tune_status.items())
                             or "Không còn điểm nào cần chạy."))
    return (tune_status,)


@app.cell
def _(DATASET, REPO_ROOT, RESULTS_REPO, WORK_DIR, hf_token_ui, mo, os, snapshot_download, subprocess, sys,
      tune_status):
    # Bảng lưới và cấu hình chọn cho mỗi biến thể (scripts/tune_summary.py, kèm luật hội tụ), từ metric trên HF.
    _ = tune_status
    _token = hf_token_ui.value.strip()
    mo.stop(not _token, mo.md("Nhập HF token để xem bảng tune."))
    _local = os.path.join(WORK_DIR, "hf_results")
    try:
        snapshot_download(RESULTS_REPO, repo_type="dataset", token=_token, local_dir=_local,
                          allow_patterns=[f"geometry-tune/{DATASET}/*/*/metrics_*.json"])
    except Exception as _e:
        print(f"Không tải được metric tune: {_e}")
    _root = os.path.join(_local, "geometry-tune", DATASET)
    mo.stop(not os.path.isdir(_root), mo.md("Chưa có lần chạy tune nào."))
    _out = os.path.join(WORK_DIR, "results", f"tune_{DATASET}")
    _res = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "tune_summary.py"), "--root", _root,
                           "--out", _out], text=True, capture_output=True)
    mo.md(open(_out + ".md", encoding="utf-8").read() if _res.returncode == 0
          else f"tune_summary lỗi (exit {_res.returncode}):\n\n```\n{_res.stderr[-2000:]}\n```")
    return


if __name__ == "__main__":
    app.run()
