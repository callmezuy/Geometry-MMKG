import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import shutil
    import subprocess
    import marimo as mo
    import torch
    from huggingface_hub import HfApi, snapshot_download

    mo.md(
        """
        # NativE trên dữ liệu v0: Control / +D mean / +D concat (Molab)
        RESEARCH_PLAN.md mục 3.2 và P3.0c. NativE chạy **nguyên bản** (`run_adv_wgan_gp.py`, model `AdvRelRotatE`,
        tự lấy số chiều đặc trưng từ dữ liệu). Depth chỉ vào ở **tầng dữ liệu**, qua ô ảnh có sẵn:

        | Kịch bản | Ô ảnh (`-visual.pth`) | Ô text |
        |---|---|---|
        | `control` = NativE | `ent_rgb_emb.pt` (768) | `ent_text_emb.pt` |
        | `dmean` = NativE+D(mean) | `ent_rgb_depth_mean_emb.pt` (768) | như trên |
        | `dconcat` = NativE+D(concat) | `ent_rgb_depth_concat_emb.pt` (1536) | như trên |

        Thực thể thiếu ảnh mang một vector hằng (P0.13). Giao thức (mục 4): chọn checkpoint theo MRR valid,
        test một lần, ghi rank từng bộ ba cho `scripts/compute_metrics.py`. Diff của NativE so với mã gốc:
        `baselines/PROVENANCE.md`.

        **Thứ tự dùng:** chọn dataset (cell setup tự tải dữ liệu) → dán HF token, nhấn Enter → chỉnh tham số →
        bấm ▶. Mỗi kịch bản xong là đẩy log, rank và checkpoint lên repo kết quả; kịch bản đã có rank trên HF
        thì được bỏ qua, nên session tắt giữa chừng thì chỉ cần bấm lại.
        """
    )
    return HfApi, mo, os, shutil, snapshot_download, subprocess, sys, torch


@app.cell
def _(mo):
    DATA_REVISION = "v0"
    RESULTS_REPO = "callmezuy/geometry-mmkg-results"  # dataset repo private, tự tạo nếu chưa có
    # tên -> (repo HF, chế độ, số seed theo kế hoạch: 3 ở các ô chính, 1 ở MKG-Y và ô inductive của baseline)
    DATASETS = {
        "FB15k-237-IMG": ("callmezuy/fb15k237-img", "transductive", 3),
        "MKG-Y": ("callmezuy/mkg-y", "transductive", 1),
        "Spatial-VRD-Trans": ("callmezuy/spatial-vrd-trans", "transductive", 3),
        "FB15K-237-ind": ("callmezuy/fb15k237-ind", "inductive", 1),
        "Spatial-VRD": ("callmezuy/spatial-vrd", "inductive", 1),
    }
    SCENARIOS = {  # tag -> bảng đặc trưng cho ô ảnh
        "control": "ent_rgb_emb.pt",
        "dmean": "ent_rgb_depth_mean_emb.pt",
        "dconcat": "ent_rgb_depth_concat_emb.pt",
    }
    dataset_ui = mo.ui.dropdown(options=list(DATASETS), value="FB15k-237-IMG", label="Dataset")
    dataset_ui
    return DATASETS, DATA_REVISION, RESULTS_REPO, SCENARIOS, dataset_ui


@app.cell
def _(os, subprocess):
    # Mã nguồn Geometry-MMKG + thư viện C++ của NativE (một lần cho cả session)
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "NativE")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "NativE")):
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
    NATIVE_DIR = os.path.join(REPO_ROOT, "baselines", "NativE")
    for _d in ("checkpoint", "embeddings", "benchmarks"):
        os.makedirs(os.path.join(NATIVE_DIR, _d), exist_ok=True)

    _mmkgc_dir = os.path.join(NATIVE_DIR, "mmkgc")
    _make_sh = os.path.join(_mmkgc_dir, "make.sh")
    with open(_make_sh, "r", encoding="utf-8") as _f:
        _sh = _f.read().replace("\r\n", "\n").replace("mkdir release", "mkdir -p release")
    with open(_make_sh, "w", encoding="utf-8") as _f:
        _f.write(_sh)
    subprocess.run(["bash", "make.sh"], cwd=_mmkgc_dir, check=True)
    print("Base.so built")
    return CODE_COMMIT, NATIVE_DIR, REPO_ROOT, WORK_DIR


@app.cell
def _(
    DATASETS,
    DATA_REVISION,
    NATIVE_DIR,
    SCENARIOS,
    WORK_DIR,
    dataset_ui,
    mo,
    os,
    shutil,
    snapshot_download,
    torch,
):
    # Dữ liệu của dataset đang chọn: tải từ HF (tag v0), dựng benchmarks/ và embeddings/ cho NativE.
    DATASET = dataset_ui.value
    _repo, SETTING, PLANNED_SEEDS = DATASETS[DATASET]
    DATA_DIR = os.path.join(WORK_DIR, "data", DATASET)
    _openke = ["entity2id.txt", "relation2id.txt", "train2id.txt", "valid2id.txt", "test2id.txt"]
    snapshot_download(
        repo_id=_repo, repo_type="dataset", revision=DATA_REVISION, local_dir=DATA_DIR,
        allow_patterns=_openke + list(SCENARIOS.values()) + [
            "ent_text_emb.pt", "ent_img_mask.pt", "ent_fill.pt", "relation_category.tsv", "feature_stats.json"],
    )

    # Kiểm tra bảng đặc trưng: đúng số thực thể, hữu hạn; dòng thiếu ảnh = vector hằng (P0.13)
    with open(os.path.join(DATA_DIR, "entity2id.txt"), encoding="utf-8") as _f:
        N_ENT = int(_f.readline())
    _load = lambda _fn: torch.load(os.path.join(DATA_DIR, _fn), map_location="cpu", weights_only=True)
    _mask, _fill = _load("ent_img_mask.pt"), _load("ent_fill.pt")
    _miss = ~_mask
    _const = {
        "control": _fill["rgb"],
        "dmean": (_fill["rgb"] + _fill["depth"]) / 2,
        "dconcat": torch.cat([_fill["rgb"], _fill["depth"]]),
    }
    _rows = []
    for _tag, _fn in list(SCENARIOS.items()) + [("text", "ent_text_emb.pt")]:
        _x = _load(_fn)
        assert _x.dim() == 2 and _x.shape[0] == N_ENT and _x.dtype == torch.float32, (_fn, tuple(_x.shape))
        assert torch.isfinite(_x).all(), f"{_fn}: non-finite values"
        if _tag in _const and _miss.any():
            assert torch.allclose(_x[_miss], _const[_tag].expand(int(_miss.sum()), -1), atol=1e-5), \
                f"{_fn}: missing-image rows are not the constant fill"
        _rows.append(f"| `{_fn}` | {tuple(_x.shape)} |")
        del _x

    # benchmarks/<DATASET>/ = file OpenKE; mỗi kịch bản là một "-dataset" riêng <DATASET>__<tag>, trỏ (symlink)
    # về cùng benchmark, với -visual.pth / -textual.pth là symlink tới bảng đặc trưng tương ứng.
    _bench = os.path.join(NATIVE_DIR, "benchmarks", DATASET)
    os.makedirs(_bench, exist_ok=True)
    for _fn in _openke:
        shutil.copyfile(os.path.join(DATA_DIR, _fn), os.path.join(_bench, _fn))

    def _link(src, dst):
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    RUN_NAMES = {}
    for _tag, _fn in SCENARIOS.items():
        _run = f"{DATASET}__{_tag}"
        _link(_bench, os.path.join(NATIVE_DIR, "benchmarks", _run))
        _link(os.path.join(DATA_DIR, _fn), os.path.join(NATIVE_DIR, "embeddings", f"{_run}-visual.pth"))
        _link(os.path.join(DATA_DIR, "ent_text_emb.pt"), os.path.join(NATIVE_DIR, "embeddings", f"{_run}-textual.pth"))
        RUN_NAMES[_tag] = _run

    mo.md("\n".join([
        f"### Dữ liệu `{DATASET}` ({SETTING}, `{_repo}@{DATA_REVISION}`)",
        f"{N_ENT:,} thực thể, {int(_miss.sum()):,} thiếu ảnh (vector hằng đã kiểm tra). "
        f"Kế hoạch: **{PLANNED_SEEDS} seed** ({'42, 43, 44' if PLANNED_SEEDS == 3 else '42'}).",
        "",
        "| Bảng | Kích thước |",
        "|---|---|",
        *_rows,
    ]))
    return DATASET, DATA_DIR, N_ENT, PLANNED_SEEDS, RUN_NAMES, SETTING


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

    def remote_files(subdir, token):
        """Tên file đã có trên RESULTS_REPO/subdir (rỗng nếu repo chưa có)."""
        try:
            _fs = HfApi(token=token).list_repo_files(RESULTS_REPO, repo_type="dataset")
        except Exception:
            return set()
        return {os.path.basename(_p) for _p in _fs if _p.startswith(subdir + "/")}

    return push_results, remote_files, run_streaming


@app.cell
def _(PLANNED_SEEDS, SCENARIOS, mo):
    # Mặc định = config gốc của NativE cho MKG-Y / MKG-W (baselines/NativE/scripts/run_mkgy.sh).
    # Tuning cùng ngân sách cho mọi model là tiêu chí 5 của P3.0 (làm sau).
    seed_ui = mo.ui.dropdown(options=["42", "43", "44"], value="42", label="Seed")
    scen_ui = mo.ui.multiselect(options=list(SCENARIOS), value=list(SCENARIOS), label="Kịch bản")
    epochs_ui = mo.ui.number(start=10, stop=5000, step=10, value=1000, label="Epochs")
    dim_ui = mo.ui.number(start=50, stop=1000, step=50, value=250, label="Dim")
    margin_ui = mo.ui.number(start=1, stop=24, step=1, value=4, label="Margin")
    batch_ui = mo.ui.dropdown(options=["512", "1024", "2048"], value="1024", label="Batch")
    neg_ui = mo.ui.dropdown(options=["32", "64", "128", "256"], value="128", label="Neg")
    lr_ui = mo.ui.text(value="1e-4", label="LR")
    lrg_ui = mo.ui.text(value="1e-4", label="LR generator")
    mu_ui = mo.ui.text(value="0.0001", label="mu")
    valid_steps_ui = mo.ui.number(start=1, stop=500, step=1, value=50, label="Đánh giá valid mỗi N epoch")
    valid_max_ui = mo.ui.number(start=0, stop=100000, step=500, value=0, label="Số bộ valid (0 = tất cả)")
    save_ckpt_ui = mo.ui.checkbox(value=True, label="Lưu checkpoint lên HF")
    train_btn = mo.ui.run_button(label="▶ Train các kịch bản đã chọn")
    mo.vstack([
        mo.md(f"Kế hoạch cho dataset này: **{PLANNED_SEEDS} seed**. Mỗi lần bấm chạy một seed; đổi tham số "
              "**không** tự chạy lại training. Dùng cùng tham số cho cả 3 kịch bản (Δdepth so trong cùng config)."),
        mo.hstack([seed_ui, scen_ui]),
        mo.hstack([epochs_ui, dim_ui, margin_ui, batch_ui, neg_ui]),
        mo.hstack([lr_ui, lrg_ui, mu_ui, valid_steps_ui, valid_max_ui]),
        save_ckpt_ui,
        train_btn,
    ])
    return (
        batch_ui,
        dim_ui,
        epochs_ui,
        lr_ui,
        lrg_ui,
        margin_ui,
        mu_ui,
        neg_ui,
        save_ckpt_ui,
        scen_ui,
        seed_ui,
        train_btn,
        valid_max_ui,
        valid_steps_ui,
    )


@app.cell
def _(
    CODE_COMMIT,
    DATASET,
    NATIVE_DIR,
    RUN_NAMES,
    WORK_DIR,
    batch_ui,
    dim_ui,
    epochs_ui,
    hf_token_ui,
    lr_ui,
    lrg_ui,
    margin_ui,
    mo,
    mu_ui,
    neg_ui,
    os,
    push_results,
    remote_files,
    run_streaming,
    save_ckpt_ui,
    scen_ui,
    seed_ui,
    shutil,
    train_btn,
    valid_max_ui,
    valid_steps_ui,
):
    mo.stop(not train_btn.value, mo.md("⏸ Bấm nút ▶ ở trên để bắt đầu."))
    _token = hf_token_ui.value.strip()
    mo.stop(not _token, mo.md("⚠ Chưa nhập HF token ở ô phía trên: nhập, nhấn Enter, rồi bấm chạy lại."))
    _seed = int(seed_ui.value)
    _subdir = f"native/{DATASET}"
    RESULTS_DIR = os.path.join(WORK_DIR, "results", "native", DATASET)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _done_remote = remote_files(_subdir, _token)
    train_status = {}
    for _tag in scen_ui.value:
        _run = RUN_NAMES[_tag]
        _stem = f"native_{DATASET}_{_tag}_seed{_seed}"
        _rank = f"ranks_{_stem}_test.csv"
        if os.path.basename(_rank) in _done_remote or os.path.exists(os.path.join(RESULTS_DIR, _rank)):
            print(f"=== {_tag}: đã có {_rank}, bỏ qua ===")
            train_status[_tag] = "đã có"
            continue
        print(f"=== {_tag} ({_run}), seed {_seed}, code {CODE_COMMIT} ===", flush=True)
        _ckpt = os.path.join(NATIVE_DIR, "checkpoint", f"{_stem}.ckpt")
        if os.path.exists(_ckpt):
            os.remove(_ckpt)
        _cmd = [  # = scripts/run_mkgy.sh, cộng chọn checkpoint theo valid và ghi rank
            "python", "run_adv_wgan_gp.py",
            f"-dataset={_run}",
            f"-batch_size={int(batch_ui.value)}",
            f"-margin={int(margin_ui.value)}",
            f"-epoch={int(epochs_ui.value)}",
            f"-dim={int(dim_ui.value)}",
            f"-save={_ckpt}",
            f"-neg_num={int(neg_ui.value)}",
            f"-mu={float(mu_ui.value)}",
            f"-learning_rate={float(lr_ui.value)}",
            f"-lrg={float(lrg_ui.value)}",
            f"-seed={_seed}",
            "-select_on=valid",
            f"-valid_steps={int(valid_steps_ui.value)}",
            f"-valid_max={int(valid_max_ui.value)}",
            f"-rank_file={os.path.join(RESULTS_DIR, _rank)}",
        ]
        _rc = run_streaming(_cmd, NATIVE_DIR, os.path.join(RESULTS_DIR, f"{_stem}.log"))
        if _rc == 0 and save_ckpt_ui.value and os.path.exists(_ckpt):
            shutil.copy2(_ckpt, os.path.join(RESULTS_DIR, os.path.basename(_ckpt)))
        train_status[_tag] = "xong" if _rc == 0 else f"lỗi (exit {_rc})"
        push_results(RESULTS_DIR, _subdir, _token)
    mo.md("### Kết quả train, seed {}\n\n".format(_seed)
          + "\n".join(f"- `{_k}`: {_v}" for _k, _v in train_status.items())
          + "\n\nSo dòng `Rank check ... C++: ... | Python: ...` trong log: hai bộ số phải khớp.")
    return RESULTS_DIR, train_status


@app.cell
def _(DATASET, DATA_DIR, REPO_ROOT, RESULTS_DIR, mo, os, seed_ui, subprocess, sys, train_status):
    # Metric từ rank CSV của seed này (một nguồn duy nhất: scripts/compute_metrics.py), so với control.
    _ = train_status
    _seed = int(seed_ui.value)
    _ranks = []
    for _tag in ("control", "dmean", "dconcat"):
        _p = os.path.join(RESULTS_DIR, f"ranks_native_{DATASET}_{_tag}_seed{_seed}_test.csv")
        if os.path.exists(_p):
            _ranks.append(f"{_tag}={_p}")
    mo.stop(not _ranks, mo.md("Chưa có rank CSV nào trên máy cho seed này."))
    _out = os.path.join(RESULTS_DIR, f"metrics_native_{DATASET}_seed{_seed}.md")
    _cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "compute_metrics.py"), "--data", DATA_DIR,
            "--ranks", *_ranks, "--out", _out]
    if any(_r.startswith("control=") for _r in _ranks) and len(_ranks) > 1:
        _cmd += ["--reference", "control"]
    _res = subprocess.run(_cmd, text=True, capture_output=True)
    print(_res.stderr[-2000:])
    mo.md(open(_out, encoding="utf-8").read() if _res.returncode == 0 and os.path.exists(_out)
          else f"compute_metrics lỗi (exit {_res.returncode}), xem stderr ở trên.")
    return


if __name__ == "__main__":
    app.run()
