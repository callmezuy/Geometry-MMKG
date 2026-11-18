import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import sys
    import csv
    import glob
    import json
    import time
    import shutil
    import hashlib
    import subprocess
    import zipfile
    import marimo as mo
    import torch
    from huggingface_hub import HfApi

    mo.md(
        """
        # 🔁 P3.0b: Kiểm tra SNAG trên MKG-Y so với paper gốc (Molab)
        Mục tiêu (RESEARCH_PLAN.md, P3.0 tiêu chí 1): chạy SNAG nguyên bản trên MKG-Y với đúng dữ liệu, embedding và
        script của tác giả, rồi đặt số của mình cạnh số trong paper. Lần chạy này kiểm tra ba thứ:
        - **code**: pipeline SNAG của dự án (lib C++, patch chỉ thêm ghi rank / ValidSelector) cho ra đúng kết quả gốc;
        - **dữ liệu**: split MKG-Y của dự án đúng là split của tác giả (đã so nội dung: giống hệt);
        - **mốc so sánh đặc trưng** cho P3.3 (SNAG trên MKG-Y v0: cùng split, id, code, chỉ khác đặc trưng).

        - Mã: `baselines/SNAG/SNAG_MKGC` (= zip "Dataset and Code" của tác giả, xem `baselines/PROVENANCE.md`).
        - Embedding gốc: tải zip từ Google Drive của tác giả, kiểm tra sha256, chỉ giải nén
          `MKG-Y-{visual,textual}.pth` và `missing.json`.
        - Lệnh: đúng `bash scripts/run_base.sh 0 MKG-Y 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 Y001`,
          chỉ thêm `-select_on=original` và `-rank_file`.
        - **Hành vi gốc (P3.0a):** cứ 100 epoch đánh giá trên **test**, giữ trọng số có MRR test tốt nhất, dừng sớm
          sau 10 lần không cải thiện. Số trong paper được chọn theo test; lần tái lập giữ nguyên hành vi này.
        - Số mục tiêu: SnAg (Ours) trên MKG-Y, arXiv 2403.06832 (v1 và v4) Table 1:
          **MRR .395 · H@1 .354 · H@3 .411 · H@10 .471**.
        """
    )
    return HfApi, csv, glob, hashlib, json, mo, os, re, shutil, subprocess, sys, time, torch, zipfile


@app.cell
def _(glob, hashlib, json, os, shutil, subprocess, sys, torch, zipfile):
    # 1. Mã nguồn Geometry-MMKG
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "SNAG")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "SNAG")):
        REPO_ROOT = os.path.dirname(WORK_DIR)
    else:
        REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
        # Molab giữ lại thư mục làm việc giữa các session nhưng chỉ còn file .py/.md (mất .git, make.sh, mã C++),
        # nên thiếu .git thì xoá và clone lại.
        if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
            shutil.rmtree(REPO_ROOT, ignore_errors=True)
            subprocess.run(
                ["git", "clone", "https://github.com/callmezuy/Geometry-MMKG.git", REPO_ROOT],
                check=True,
            )
        else:
            subprocess.run(["git", "-C", REPO_ROOT, "pull"], check=False)
    CODE_COMMIT = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    print(f"Geometry-MMKG code at commit {CODE_COMMIT}")
    SNAG_DIR = os.path.join(REPO_ROOT, "baselines", "SNAG", "SNAG_MKGC")
    EMB_DIR = os.path.join(SNAG_DIR, "embeddings")
    BENCH_DIR = os.path.join(SNAG_DIR, "benchmarks", "MKG-Y")
    # run_mmkgc.py writes checkpoints and logs under <SNAG_MKGC>/../../data/mmkgc/ (upstream path logic)
    SNAG_DATA_DIR = os.path.normpath(os.path.join(SNAG_DIR, "..", "..", "data", "mmkgc"))

    # 2. Zip "Dataset and Code" của tác giả (Google Drive, link trong SNAG_MKGC/readme.md)
    ZIP_ID = "1jIYo7JWXNNi7LZqBIsl103FL8tHcoW-q"
    ZIP_SHA256 = "b25d6823868bf56788c7851c9c45294c44c7a4622e2c2d678978b3e1da8991c3"  # baselines/PROVENANCE.md
    # sha256 of the copies on the author's machine == members of the zip (checked 18/11/2026)
    MEMBER_SHA256 = {
        "SNAG_MKGC/embeddings/MKG-Y-visual.pth": "d461c913ef25c744fc473e1f6473e9e3940116d16ae2290ab5972c2321bae4a7",
        "SNAG_MKGC/embeddings/MKG-Y-textual.pth": "a736a6c2506de6dab31ed83552c401f74543e14d2d1ddc650ad27702a69aa37e",
        "SNAG_MKGC/embeddings/missing.json": "5c1c6ae7b6f37cb02110dd62d62875dc85b204f85f056b89085fbfd639f1e2b9",
    }

    def _sha256(path):
        _h = hashlib.sha256()
        with open(path, "rb") as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b""):
                _h.update(_chunk)
        return _h.hexdigest()

    _need = [m for m in MEMBER_SHA256 if not (
        os.path.exists(os.path.join(EMB_DIR, os.path.basename(m)))
        and _sha256(os.path.join(EMB_DIR, os.path.basename(m))) == MEMBER_SHA256[m])]
    if _need:
        try:
            import gdown
        except ImportError:
            if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"]).returncode != 0:
                subprocess.run(["uv", "pip", "install", "--python", sys.executable, "gdown"], check=True)
            import gdown
        _dl_dir = os.path.join(WORK_DIR, "downloads", "snag")
        os.makedirs(_dl_dir, exist_ok=True)
        _zip = os.path.join(_dl_dir, "SNAG_MKGC.zip")
        if not (os.path.exists(_zip) and _sha256(_zip) == ZIP_SHA256):
            gdown.download(id=ZIP_ID, output=_zip, quiet=False)
        _zsha = _sha256(_zip)
        assert _zsha == ZIP_SHA256, f"zip sha256 mismatch: {_zsha}"
        print(f"SNAG_MKGC.zip sha256 OK ({os.path.getsize(_zip):,} bytes)")
        os.makedirs(EMB_DIR, exist_ok=True)
        with zipfile.ZipFile(_zip) as _zf:
            for _m in _need:
                _dst = os.path.join(EMB_DIR, os.path.basename(_m))
                with _zf.open(_m) as _src, open(_dst + ".part", "wb") as _out:
                    shutil.copyfileobj(_src, _out)
                os.replace(_dst + ".part", _dst)
    for _m, _sha in MEMBER_SHA256.items():  # missing.json is also in git: must be the author's file
        _got = _sha256(os.path.join(EMB_DIR, os.path.basename(_m)))
        assert _got == _sha, f"{_m}: sha256 {_got} != {_sha}"
        print(f"{os.path.basename(_m)}: sha256 OK")

    # Kiểm tra: tensor thuần, 15.000 thực thể; missing.json có khoá MKG-Y (756 thiếu ảnh, 2.684 thiếu mô tả)
    with open(os.path.join(BENCH_DIR, "entity2id.txt")) as _f:
        _n_ent = int(_f.readline())
    for _name, _dim in [("MKG-Y-visual", 383), ("MKG-Y-textual", 384)]:
        _t = torch.load(os.path.join(EMB_DIR, f"{_name}.pth"), map_location="cpu", weights_only=True)
        assert tuple(_t.shape) == (_n_ent, _dim), (_name, tuple(_t.shape))
        print(f"{_name}: {tuple(_t.shape)} {_t.dtype}")
        del _t
    with open(os.path.join(EMB_DIR, "missing.json")) as _f:
        _miss = json.load(_f)
    print({k: len(v) for k, v in _miss.items() if k.startswith("MKG-Y")})
    assert len(_miss["MKG-Y-visual.pth"]) == 756 and len(_miss["MKG-Y-textual.pth"]) == 2684
    for _fn, _n in [("train2id.txt", 21310), ("valid2id.txt", 2665), ("test2id.txt", 2663)]:
        with open(os.path.join(BENCH_DIR, _fn)) as _f:
            assert int(_f.readline()) == _n, _fn

    # 3. Biên dịch C++ Data Loader (Base.so)
    _mmkgc_dir = os.path.join(SNAG_DIR, "mmkgc")
    _make_sh = os.path.join(_mmkgc_dir, "make.sh")
    with open(_make_sh, "r", encoding="utf-8") as _f:
        _sh_content = _f.read().replace("\r\n", "\n").replace("mkdir release", "mkdir -p release")
    with open(_make_sh, "w", encoding="utf-8") as _f:
        _f.write(_sh_content)
    subprocess.run(["bash", "make.sh"], cwd=_mmkgc_dir, check=True)
    assert glob.glob(os.path.join(_mmkgc_dir, "release", "Base.so"))
    print("SNAG MKG-Y setup complete!")
    return CODE_COMMIT, EMB_DIR, REPO_ROOT, SNAG_DATA_DIR, SNAG_DIR, WORK_DIR, ZIP_SHA256


@app.cell
def _(
    CODE_COMMIT,
    HfApi,
    SNAG_DATA_DIR,
    WORK_DIR,
    ZIP_SHA256,
    csv,
    glob,
    json,
    os,
    re,
    shutil,
    subprocess,
    sys,
    time,
    torch,
):
    SEED = 42  # args.py default; run_base.sh does not pass -seed
    RESULTS_REPO = "callmezuy/geometry-mmkg-results"  # dataset repo private, tự tạo nếu chưa có
    RESULTS_SUBDIR = "snag/MKG-Y-sanity"
    RESULTS_DIR = os.path.join(WORK_DIR, "results", "snag", "MKG-Y-sanity")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # SnAg (Ours) on MKG-Y, arXiv 2403.06832 Table 1 (same in v1, 11/03/2024, and v4, 15/01/2025)
    PAPER = {"mrr": 0.395, "hit1": 0.354, "hit3": 0.411, "hit10": 0.471}
    PAPER_SOURCE = "arXiv 2403.06832 (v1 and v4), Table 1, row 'SnAg (Ours)', MKG-Y"
    TOL_MRR_POINTS = 1.0  # reproduced if |MRR - paper| <= 1 point (user decision 19/11/2026; H@k reported only)

    # = what `bash scripts/run_base.sh 0 MKG-Y 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 Y001` runs
    # (run with CUDA_VISIBLE_DEVICES=0). Checked against the script by expanding it with a stub `python`.
    RUN_BASE_CMD = [
        "python", "run_mmkgc.py",
        "-dataset=MKG-Y",
        "-num_proj=1",
        "-use_intermediate=0",
        "-joint_way=Mformer_hd_mean",
        "-batch_size=1024",
        "-margin=12",
        "-epoch=8000",
        "-dim=128",
        "-save=MKG-Y-1024-128-32-12-1e-4-MKG-Y",
        "-neg_num=32",
        "-noise_update=epoch",
        "-noise_ratio=0.2",
        "-mask_ratio=0.7",
        "-use_pool=1",
        "-add_noise=1",
        "-learning_rate=1e-4",
        "-num_hidden_layers=1",
        "-num_attention_heads=2",
        "-exp_id=128-32-1_layer_0_inter_Mformer_hd_mean_Y001",
    ]
    SAVE_NAME = "MKG-Y-1024-128-32-12-1e-4-MKG-Y"
    EXP_ID = "128-32-1_layer_0_inter_Mformer_hd_mean_Y001"
    CKPT_DIR = os.path.join(SNAG_DATA_DIR, "checkpoint")
    DUMP_DIR = os.path.join(SNAG_DATA_DIR, "dump")
    _METRIC_RE = re.compile(r"mrr:([-\d.e]+),\s*mr:([-\d.e]+),\s*hit10:([-\d.e]+),\s*hit3:([-\d.e]+),\s*hit1:([-\d.e]+)")

    def run_streaming(cmd, cwd, log_path, env_extra=None, progress_every=60.0):
        """Chạy lệnh, in log theo thời gian thực và ghi vào log_path.
        Thanh tqdm (8000 epoch) chỉ in/ghi tối đa mỗi `progress_every` giây; mọi dòng khác in đủ."""
        print("$ " + " ".join(cmd), flush=True)
        _last = 0.0
        with open(log_path, "w", encoding="utf-8") as _log:
            _proc = subprocess.Popen(
                [sys.executable if cmd[0] == "python" else cmd[0]] + cmd[1:],
                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", **(env_extra or {})},
            )
            try:
                for _line in _proc.stdout:  # text mode: tqdm's '\r' arrives as a line break
                    if "%|" in _line and ("it/s" in _line or "s/it" in _line):
                        if time.time() - _last < progress_every:
                            continue
                        _last = time.time()
                    print(_line, end="", flush=True)
                    _log.write(_line)
                    _log.flush()
            except BaseException:  # interrupt/stop: kill training so it does not keep running unattended
                _proc.kill()
                _proc.wait()
                raise
            return _proc.wait()

    def clean_previous(save_name, exp_id):
        """run_mmkgc.py loads an existing '<save>-MRR<x>' file instead of the new weights, and torchlight
        appends to an existing train.log: remove both before each run."""
        for _p in glob.glob(os.path.join(CKPT_DIR, save_name + "-MRR*")):
            os.remove(_p)
        for _d in glob.glob(os.path.join(DUMP_DIR, "*-KGC_exp-MKG-Y", exp_id)):
            shutil.rmtree(_d)

    def ranks_metrics(rank_csv):
        """Filtered head+tail metrics recomputed from the per-triple rank CSV (Tester rank_file)."""
        _r = []
        with open(rank_csv) as _f:
            for _row in csv.DictReader(_f):
                _r += [int(_row["rank_head"]), int(_row["rank_tail"])]
        _t = torch.tensor(_r, dtype=torch.float64)
        return {"mrr": float((1 / _t).mean()), "mr": float(_t.mean()), "hit10": float((_t <= 10).double().mean()),
                "hit3": float((_t <= 3).double().mean()), "hit1": float((_t <= 1).double().mean()),
                "n_ranks": len(_r)}

    def collect(tag, mode, cmd, rc, log_path, rank_csv, seconds):
        """Copy checkpoint + torchlight params, parse the log, write metrics_<tag>.json."""
        _out = {"tag": tag, "select_on": mode, "exit_code": rc, "command": " ".join(cmd),
                "env": "CUDA_VISIBLE_DEVICES=0", "seed": SEED, "code_commit": CODE_COMMIT,
                "author_zip_sha256": ZIP_SHA256, "seconds": round(seconds, 1),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__, "paper": PAPER, "paper_source": PAPER_SOURCE}
        _ck = sorted(glob.glob(os.path.join(CKPT_DIR, SAVE_NAME + "-MRR*")))
        if _ck:
            _out["checkpoint_name"] = os.path.basename(_ck[-1])
            shutil.copy2(_ck[-1], os.path.join(RESULTS_DIR, f"snag_{tag}.ckpt"))
        for _p in glob.glob(os.path.join(DUMP_DIR, "*-KGC_exp-MKG-Y", EXP_ID, "params.pkl")):
            shutil.copy2(_p, os.path.join(RESULTS_DIR, f"params_snag_{tag}.json"))  # torchlight writes JSON
        with open(log_path, encoding="utf-8") as _f:
            _log = _f.read()
        _found = _METRIC_RE.findall(_log)
        if _found:  # last line of this form = the final test of the selected checkpoint
            _out["test_cpp"] = dict(zip(["mrr", "mr", "hit10", "hit3", "hit1"], map(float, _found[-1])))
        if mode == "original":  # every-100-epoch TEST evaluations (upstream selection on test)
            _ep = [int(e) for e in re.findall(r"Epoch (\d+) \| loss: [^,\n]+, saving", _log)]
            _mr = [float(m[0]) for m in _found[:-1]]
            _out["test_history"] = [{"epoch": e, "test_mrr": m} for e, m in zip(_ep, _mr)]
            _out["early_stop"] = "Early Stop" in _log
            if _mr:
                _b = max(range(len(_mr)), key=_mr.__getitem__)
                _out["selected_epoch"] = _ep[_b] if _b < len(_ep) else None
        else:
            _m = re.findall(r"restored best weights: epoch (\d+), valid MRR ([\d.]+)", _log)
            if _m:
                _out["selected_epoch"], _out["selected_valid_mrr"] = int(_m[-1][0]), float(_m[-1][1])
            _out["early_stop"] = "Early stop on validation MRR" in _log
        if os.path.exists(rank_csv):
            _out["test"] = ranks_metrics(rank_csv)
            _out["delta_vs_paper_points"] = {k: round(100 * (_out["test"][k] - v), 2) for k, v in PAPER.items()}
        with open(os.path.join(RESULTS_DIR, f"metrics_snag_{tag}.json"), "w") as _f:
            json.dump(_out, _f, indent=2)
        return _out

    def comparison_md():
        """Table: paper vs every finished run in RESULTS_DIR (values x100)."""
        _runs = []
        for _p in sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_snag_*.json"))):
            with open(_p) as _f:
                _runs.append(json.load(_f))
        _rows = [f"| | MRR | H@1 | H@3 | H@10 | epoch chọn | MRR lệch ≤ {TOL_MRR_POINTS:g} điểm? |",
                 "|---|---|---|---|---|---|---|",
                 "| Paper (Table 1) | " + " | ".join(f"{100 * PAPER[k]:.1f}" for k in ["mrr", "hit1", "hit3", "hit10"]) + " | | |"]
        for _r in _runs:
            if "test" not in _r:
                _rows.append(f"| {_r['tag']} | (lỗi, exit {_r['exit_code']}) | | | | | |")
                continue
            _cells = [f"{100 * _r['test'][k]:.1f} ({_r['delta_vs_paper_points'][k]:+.1f})" for k in ["mrr", "hit1", "hit3", "hit10"]]
            _ok = abs(_r["delta_vs_paper_points"]["mrr"]) <= TOL_MRR_POINTS
            _rows.append(f"| {_r['tag']} | " + " | ".join(_cells) + f" | {_r.get('selected_epoch')} | {'✅ đạt' if _ok else '❌ không'} |")
        return "\n".join(_rows) + ("\n\nNgưỡng áp dụng cho bản `original` (tái lập paper); bản `valid` chỉ để đo mức "
                                   "lạc quan do chọn checkpoint theo test.")

    def push_results(token):
        """Đẩy RESULTS_DIR lên HF. Lỗi upload không làm hỏng kết quả local."""
        try:
            _api = HfApi(token=token)
            _api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
            _api.upload_folder(
                repo_id=RESULTS_REPO, repo_type="dataset", folder_path=RESULTS_DIR,
                path_in_repo=RESULTS_SUBDIR, commit_message=f"Add {RESULTS_SUBDIR} results",
            )
            print(f"⬆️  Uploaded {RESULTS_DIR} -> {RESULTS_REPO}/{RESULTS_SUBDIR}")
        except Exception as _e:
            print(f"⚠️  Upload lên HF thất bại ({_e}). Hãy tải thủ công thư mục {RESULTS_DIR} trước khi tắt session.")

    return (
        CKPT_DIR,
        DUMP_DIR,
        EXP_ID,
        PAPER,
        PAPER_SOURCE,
        RESULTS_DIR,
        RESULTS_REPO,
        RESULTS_SUBDIR,
        RUN_BASE_CMD,
        SAVE_NAME,
        SEED,
        clean_previous,
        collect,
        comparison_md,
        push_results,
        ranks_metrics,
        run_streaming,
    )


@app.cell(hide_code=False)
def _(mo):
    hf_token_ui = mo.ui.text(kind="password", label="HF token (quyền write)", full_width=True)
    mo.vstack([
        mo.md("Dán HF token có quyền **write**, nhấn **Enter**, rồi mới bấm nút chạy. Kết quả được đẩy lên dataset "
              "repo private `RESULTS_REPO` (cell phía trên; tự tạo nếu chưa có). Token chỉ nằm trong bộ nhớ kernel, "
              "không ghi vào file notebook. Ô này chỉ phụ thuộc `mo`, nên chạy lại các cell khác không xoá token."),
        hf_token_ui,
    ])
    return (hf_token_ui,)


@app.cell
def _(mo):
    run_orig_btn = mo.ui.run_button(label="▶ 1. Chạy SNAG MKG-Y nguyên bản (select_on=original)")
    run_orig_btn
    return (run_orig_btn,)


@app.cell
def _(
    RESULTS_DIR,
    RUN_BASE_CMD,
    EXP_ID,
    SAVE_NAME,
    SEED,
    SNAG_DIR,
    clean_previous,
    collect,
    comparison_md,
    hf_token_ui,
    mo,
    os,
    push_results,
    run_orig_btn,
    run_streaming,
    time,
):
    mo.stop(not run_orig_btn.value, mo.md("⏸ Bấm nút để bắt đầu."))
    mo.stop(not hf_token_ui.value.strip(), mo.md("⚠ Chưa nhập HF token ở ô phía trên: nhập rồi bấm chạy lại (để kết quả không mất khi session tắt)."))
    _tag = f"MKG-Y_original_seed{SEED}"
    _rank = os.path.join(RESULTS_DIR, f"ranks_snag_{_tag}_test.csv")
    _log = os.path.join(RESULTS_DIR, f"snag_{_tag}.log")
    _cmd = RUN_BASE_CMD + [
        "-select_on=original",  # upstream behaviour: test every 100 epochs, keep best test MRR, early stop 10
        f"-rank_file={_rank}",
    ]
    clean_previous(SAVE_NAME, EXP_ID)
    _t0 = time.time()
    orig_rc = run_streaming(_cmd, SNAG_DIR, _log, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    orig_res = collect(_tag, "original", _cmd, orig_rc, _log, _rank, time.time() - _t0)
    push_results(hf_token_ui.value.strip())
    mo.md(
        f"""
        ### {'✅' if orig_rc == 0 else '❌'} Xong (exit code {orig_rc})
        Số ×100, trong ngoặc là chênh so với paper (điểm). Số của mình được tính lại từ file rank (Python);
        dòng `Rank check ... C++ | Python` trong log phải khớp nhau.

        {comparison_md()}
        """
    )
    return orig_rc, orig_res


@app.cell
def _(mo):
    run_valid_btn = mo.ui.run_button(label="▶ 2. (Tuỳ chọn) Chạy lại, chọn checkpoint theo valid (select_on=valid)")
    mo.vstack([
        mo.md("Cùng lệnh, chỉ đổi cách chọn checkpoint: đánh giá **valid** mỗi 100 epoch, dừng sớm sau 10 lần không "
              "cải thiện (đúng luật gốc, chỉ thay test bằng valid), test một lần trên checkpoint tốt nhất. "
              "Chênh lệch với lần 1 = mức lạc quan do chọn checkpoint theo test (P3.0a)."),
        run_valid_btn,
    ])
    return (run_valid_btn,)


@app.cell
def _(
    RESULTS_DIR,
    RUN_BASE_CMD,
    EXP_ID,
    SAVE_NAME,
    SEED,
    SNAG_DIR,
    clean_previous,
    collect,
    comparison_md,
    hf_token_ui,
    mo,
    os,
    push_results,
    run_streaming,
    run_valid_btn,
    time,
):
    mo.stop(not run_valid_btn.value, mo.md("⏸ Bấm nút để bắt đầu."))
    mo.stop(not hf_token_ui.value.strip(), mo.md("⚠ Chưa nhập HF token ở ô phía trên."))
    _tag = f"MKG-Y_valid_seed{SEED}"
    _rank = os.path.join(RESULTS_DIR, f"ranks_snag_{_tag}_test.csv")
    _log = os.path.join(RESULTS_DIR, f"snag_{_tag}.log")
    _cmd = RUN_BASE_CMD + [
        "-select_on=valid",
        "-valid_steps=100",  # same cadence as upstream save_steps=100
        "-patience=10",  # same rule as upstream -early_stop 10
        f"-rank_file={_rank}",
    ]
    clean_previous(SAVE_NAME, EXP_ID)
    _t0 = time.time()
    valid_rc = run_streaming(_cmd, SNAG_DIR, _log, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    valid_res = collect(_tag, "valid", _cmd, valid_rc, _log, _rank, time.time() - _t0)
    push_results(hf_token_ui.value.strip())
    mo.md(
        f"""
        ### {'✅' if valid_rc == 0 else '❌'} Xong (exit code {valid_rc})

        {comparison_md()}
        """
    )
    return valid_rc, valid_res


if __name__ == "__main__":
    app.run()
