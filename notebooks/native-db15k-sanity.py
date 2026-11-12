import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import shutil
    import subprocess
    import zipfile
    import marimo as mo
    import torch
    from huggingface_hub import HfApi

    mo.md(
        """
        # 🔁 P3.0b: Kiểm tra NativE trên DB15K so với paper gốc (Molab)
        Mục tiêu (RESEARCH_PLAN.md, P3.0 tiêu chí 1): chứng minh code NativE **đã sửa** (rank_file, ValidSelector,
        cờ `-orig_pool`) vẫn cho ra số của paper khi tắt các phần mình thêm vào.

        - Dữ liệu: `baselines/NativE/benchmarks/DB15K` (có sẵn trong repo) + embedding gốc của tác giả
          (Google Drive trong README của NativE, folder `191u4WhT_7P9Ze8Q9N3U3rdr8IWEKYMry`).
        - Lệnh: giống `scripts/run_db15k.sh` (3-modal, 1000 epoch), thêm `-select_on=original -orig_pool=1`.
        - Số mục tiêu: NativE "All" trên DB15K, arXiv 2406.17605 Table 2/3:
          **MRR 37.16 · H@1 28.01 · H@3 42.25 · H@10 54.13**. Chấp nhận sai lệch khoảng ±1 MRR.
        """
    )
    return HfApi, mo, os, shutil, subprocess, sys, torch, zipfile


@app.cell
def _(os, shutil, subprocess, sys, torch, zipfile):
    # 1. Mã nguồn Geometry-MMKG
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "NativE")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "NativE")):
        REPO_ROOT = os.path.dirname(WORK_DIR)
    else:
        REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
        if not os.path.exists(REPO_ROOT):
            subprocess.run(
                ["git", "clone", "https://github.com/callmezuy/Geometry-MMKG.git", REPO_ROOT],
                check=True,
            )
        else:
            subprocess.run(["git", "-C", REPO_ROOT, "pull"], check=False)
    _commit = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    print(f"Geometry-MMKG code at commit {_commit}")
    NATIVE_DIR = os.path.join(REPO_ROOT, "baselines", "NativE")
    EMB_DIR = os.path.join(NATIVE_DIR, "embeddings")
    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(os.path.join(NATIVE_DIR, "checkpoint"), exist_ok=True)

    # 2. Embedding gốc DB15K từ Google Drive của tác giả NativE
    try:
        import gdown
    except ImportError:
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"]).returncode != 0:
            subprocess.run(["uv", "pip", "install", "--python", sys.executable, "gdown"], check=True)
        import gdown

    _DRIVE_FILES = {  # file id trong folder Drive công khai của NativE
        "DB15K-visual": "10026x1LtQ9h-9qBvu2PijXnigadVmRrJ",
        "DB15K-textual": "187bG3PzvroHYq8ZX2k5965QcAQTEEqb3",
        "DB15K-numeric": "1qrkO8zjGtwvexPekWv39jPlpn8n7fpJA",
    }
    _dl_dir = os.path.join(WORK_DIR, "downloads", "native-db15k")
    os.makedirs(_dl_dir, exist_ok=True)
    for _name, _fid in _DRIVE_FILES.items():
        _pth = os.path.join(EMB_DIR, f"{_name}.pth")
        if os.path.exists(_pth):
            continue
        _zip = os.path.join(_dl_dir, f"{_name}.zip")
        if not os.path.exists(_zip):
            gdown.download(id=_fid, output=_zip, quiet=False)
        with zipfile.ZipFile(_zip) as _zf:
            assert _zf.namelist() == [f"{_name}.pth"], _zf.namelist()
            with _zf.open(f"{_name}.pth") as _src, open(_pth, "wb") as _dst:
                shutil.copyfileobj(_src, _dst)

    # Kiểm tra: tensor thuần (weights_only=True), 12.842 thực thể; visual 4096-d, textual/numeric 768-d
    with open(os.path.join(NATIVE_DIR, "benchmarks", "DB15K", "entity2id.txt")) as _f:
        _n_ent = int(_f.readline())
    for _name, _dim in [("DB15K-visual", 4096), ("DB15K-textual", 768), ("DB15K-numeric", 768)]:
        _t = torch.load(os.path.join(EMB_DIR, f"{_name}.pth"), map_location="cpu", weights_only=True)
        assert tuple(_t.shape) == (_n_ent, _dim), (_name, tuple(_t.shape))
        print(f"{_name}: {tuple(_t.shape)} {_t.dtype}")
        del _t

    # 3. Biên dịch C++ Data Loader (Base.so)
    _mmkgc_dir = os.path.join(NATIVE_DIR, "mmkgc")
    _make_sh = os.path.join(_mmkgc_dir, "make.sh")
    with open(_make_sh, "r", encoding="utf-8") as _f:
        _sh_content = _f.read().replace("\r\n", "\n").replace("mkdir release", "mkdir -p release")
    with open(_make_sh, "w", encoding="utf-8") as _f:
        _f.write(_sh_content)
    subprocess.run(["bash", "make.sh"], cwd=_mmkgc_dir, check=True)
    print("DB15K setup complete!")
    return EMB_DIR, NATIVE_DIR, REPO_ROOT, WORK_DIR


@app.cell
def _(HfApi, WORK_DIR, os, subprocess, sys):
    SEED = 42
    RESULTS_REPO = "callmezuy/geometry-mmkg-results"  # dataset repo private, tự tạo nếu chưa có
    RESULTS_DIR = os.path.join(WORK_DIR, "results", "native", "DB15K-sanity")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def run_streaming(cmd, cwd, log_path):
        """Chạy lệnh, in log ra màn hình theo thời gian thực và ghi đồng thời vào log_path."""
        print("$ " + " ".join(cmd), flush=True)
        with open(log_path, "w", encoding="utf-8") as _log:
            _proc = subprocess.Popen(
                [sys.executable if cmd[0] == "python" else cmd[0]] + cmd[1:],
                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for _line in _proc.stdout:
                print(_line, end="", flush=True)
                _log.write(_line)
                _log.flush()
            return _proc.wait()

    def push_results(subdir):
        """Đẩy RESULTS_DIR lên HF. Lỗi upload không làm hỏng kết quả local."""
        try:
            _api = HfApi()
            _api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
            _api.upload_folder(
                repo_id=RESULTS_REPO, repo_type="dataset", folder_path=RESULTS_DIR,
                path_in_repo=subdir, commit_message=f"Add {subdir} results",
            )
            print(f"⬆️  Uploaded {RESULTS_DIR} -> {RESULTS_REPO}/{subdir}")
        except Exception as _e:
            print(f"⚠️  Upload lên HF thất bại ({_e}). Hãy tải thủ công thư mục {RESULTS_DIR} trước khi tắt session.")

    return RESULTS_DIR, RESULTS_REPO, SEED, push_results, run_streaming


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="▶ Chạy NativE DB15K (1000 epoch, config gốc)")
    run_btn
    return (run_btn,)


@app.cell
def _(NATIVE_DIR, RESULTS_DIR, SEED, mo, os, push_results, run_btn, run_streaming, shutil):
    mo.stop(not run_btn.value, mo.md("⏸ Bấm nút để bắt đầu."))
    _tag = f"DB15K_original_seed{SEED}"
    _ckpt = os.path.join(NATIVE_DIR, "checkpoint", f"native_{_tag}.ckpt")
    _cmd = [  # = scripts/run_db15k.sh, cộng các cờ kiểm tra
        "python", "run_adv_wgan_gp_3modal.py",
        "-dataset=DB15K",
        "-batch_size=1024",
        "-margin=12",
        "-epoch=1000",
        "-dim=250",
        "-adv_num=1",
        f"-save={_ckpt}",
        "-neg_num=128",
        "-mu=0.0001",
        "-learning_rate=1e-4",
        "-lrg=1e-4",
        f"-seed={SEED}",
        "-select_on=original",  # hành vi gốc: train cố định số epoch, test checkpoint cuối
        "-orig_pool=1",         # pooling gốc của AdvRelRotatEDB15K (4096/768 -> 256)
        f"-rank_file={os.path.join(RESULTS_DIR, f'ranks_native_{_tag}_test.csv')}",
    ]
    db15k_rc = run_streaming(_cmd, NATIVE_DIR, os.path.join(RESULTS_DIR, f"native_{_tag}.log"))
    if db15k_rc == 0 and os.path.exists(_ckpt):
        shutil.copy2(_ckpt, os.path.join(RESULTS_DIR, os.path.basename(_ckpt)))
    push_results("native/DB15K-sanity")
    mo.md(
        f"""
        ### {'✅' if db15k_rc == 0 else '❌'} Xong (exit code {db15k_rc})
        So dòng `Rank check ... C++: ... | Python: ...` (MRR, MR, H@10, H@3, H@1) trong log với paper:
        **MRR 37.16 · H@1 28.01 · H@3 42.25 · H@10 54.13**. Ghi kết quả vào RESULTS_DIR và vào RESEARCH_PLAN.md (P3.0b).
        """
    )
    return (db15k_rc,)


if __name__ == "__main__":
    app.run()
