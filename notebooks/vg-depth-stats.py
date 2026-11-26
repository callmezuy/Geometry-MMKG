import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    import marimo as mo
    from huggingface_hub import HfApi, snapshot_download

    mo.md(
        """
        # Số độ sâu theo vùng cho Spatial-VG (Molab GPU)

        Chạy `scripts/depth_stats.py vg` (RESEARCH_PLAN.md P2.9): `ent_depth_stats.pt` [1.441.984, 9] = trung bình
        lưới 3×3 của depth map ZipDepth (cùng depth map, cùng box như `ent_depth_emb.pt` của v0) trong box của mỗi
        thực thể; thực thể thiếu ảnh = trung bình của thực thể train có ảnh. Chỉ chạy ZipDepth, không chạy DINOv2.

        Cần: code mới nhất (`scripts/depth_stats.py`) đã push lên GitHub; khoảng 16 GB đĩa cho ảnh VG.
        """
    )
    return HfApi, hashlib, json, mo, os, shutil, snapshot_download, subprocess, sys


@app.cell
def _(hashlib, os, shutil, subprocess):
    # 1. Code: repo dự án (scripts/ + bản sao ZipDepth/ kèm checkpoint)
    WORK_DIR = os.path.abspath(".")
    REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
    # Molab giữ lại thư mục làm việc giữa các session nhưng chỉ còn file .py/.md, nên thiếu .git thì clone lại.
    if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
        shutil.rmtree(REPO_ROOT, ignore_errors=True)
        subprocess.run(["git", "clone", "https://github.com/callmezuy/Geometry-MMKG.git", REPO_ROOT], check=True)
    else:
        subprocess.run(["git", "-C", REPO_ROOT, "pull"], check=False)
    CODE_COMMIT = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert os.path.exists(os.path.join(REPO_ROOT, "scripts", "depth_stats.py")), (
        "scripts/depth_stats.py chưa có trên GitHub: hãy push code mới nhất trước"
    )
    ZIPDEPTH_MD5 = "23f5f042af37939afa80b0f056246df5"  # checkpoints/zipdepth_base.pth, như notebooks/extract-vg-features.py
    with open(os.path.join(REPO_ROOT, "ZipDepth", "checkpoints", "zipdepth_base.pth"), "rb") as _f:
        _md5 = hashlib.md5(_f.read()).hexdigest()
    assert _md5 == ZIPDEPTH_MD5, f"zipdepth_base.pth md5 {_md5} != {ZIPDEPTH_MD5}"
    try:
        import h5py  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "h5py"], check=True)
    print(f"code {CODE_COMMIT[:10]} (ZipDepth checkpoint md5 ok)")
    return CODE_COMMIT, REPO_ROOT, WORK_DIR


@app.cell
def _(WORK_DIR, os, snapshot_download):
    # 2. Spatial-VG v0 từ HF: chỉ split, H5, mặt nạ ảnh (không cần bảng đặc trưng 13 GB)
    HF_REPO_ID = "callmezuy/spatial-vg"
    DATA_REVISION = "v0"
    DATA_DIR = os.path.join(WORK_DIR, "data", "Spatial-VG")
    snapshot_download(
        repo_id=HF_REPO_ID, repo_type="dataset", revision=DATA_REVISION, local_dir=DATA_DIR,
        allow_patterns=["entity2id.txt", "relation2id.txt", "train2id.txt", "VG-SGG.h5", "ent_img_mask.pt"],
    )
    with open(os.path.join(DATA_DIR, "entity2id.txt"), encoding="utf-8") as _f:
        N_ENT = int(_f.readline())
    assert N_ENT == 1441984, N_ENT
    print(f"{HF_REPO_ID}@{DATA_REVISION} -> {DATA_DIR}: {N_ENT:,} entities")
    return DATA_DIR, HF_REPO_ID


@app.cell(hide_code=False)
def _(mo):
    hf_token_ui = mo.ui.text(kind="password", label="HF token", full_width=True)
    mo.vstack([
        mo.md("Dán HF token: cần quyền **write** để mở pull request lên `callmezuy/spatial-vg` (ảnh VG tải từ bản sao "
              "public `callmezuy/vg-images-raw`). Token chỉ nằm trong bộ nhớ kernel."),
        hf_token_ui,
    ])
    return (hf_token_ui,)


@app.cell
def _(WORK_DIR, hf_token_ui, mo, os, snapshot_download):
    # 3. Ảnh VG gốc từ bản sao public, kiểm tra đúng kích thước gốc (chờ token ở ô trên rồi mới tải)
    RAW_DIR = os.path.join(WORK_DIR, "data", "raw", "vg")
    os.makedirs(RAW_DIR, exist_ok=True)
    _token = hf_token_ui.value.strip() or None
    mo.stop(not _token, mo.md("⏸ Dán HF token ở ô phía trên để tải ảnh VG."))
    _files = {"images.zip": 9731705982, "images2.zip": 5471658058, "image_data.json.zip": 1780854}
    for _name, _size in _files.items():
        _p = os.path.join(RAW_DIR, _name)
        if not (os.path.exists(_p) and os.path.getsize(_p) == _size):
            print(f"downloading {_name} ({_size / 2**30:.1f} GB) from HF callmezuy/vg-images-raw...", flush=True)
            if os.path.exists(_p):
                os.remove(_p)
            snapshot_download(repo_id="callmezuy/vg-images-raw", repo_type="dataset", local_dir=RAW_DIR,
                              allow_patterns=[_name], token=_token)
        assert os.path.getsize(_p) == _size, f"{_name}: {os.path.getsize(_p)} != {_size}"
    print("VG images ok")
    return (RAW_DIR,)


@app.cell
def _(mo):
    img_batch_ui = mo.ui.dropdown(options=["16", "32", "64", "128"], value="64", label="ZipDepth batch (ảnh)")
    run_btn = mo.ui.run_button(label="▶ Tính số độ sâu VG")
    mo.vstack([mo.md("Bấm nút để bắt đầu (khoảng 15–30 phút trên GPU Molab)."), img_batch_ui, run_btn])
    return img_batch_ui, run_btn


@app.cell
def _(os, subprocess, sys):
    def run_streaming(cmd, cwd, log_path):
        """Chạy lệnh, in log theo thời gian thực và ghi vào log_path; ngắt cell thì kill tiến trình con."""
        print("$ " + " ".join(cmd), flush=True)
        with open(log_path, "w", encoding="utf-8") as _log:
            _proc = subprocess.Popen(
                [sys.executable] + cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                for _line in _proc.stdout:
                    if "UserWarning" in _line:
                        continue
                    print(_line, end="", flush=True)
                    _log.write(_line)
                    _log.flush()
            except BaseException:
                _proc.kill()
                _proc.wait()
                raise
            return _proc.wait()

    return (run_streaming,)


@app.cell
def _(DATA_DIR, RAW_DIR, REPO_ROOT, WORK_DIR, img_batch_ui, json, mo, os, run_btn, run_streaming):
    mo.stop(not run_btn.value, mo.md("⏸ Bấm **▶ Tính số độ sâu VG** để bắt đầu."))
    OUT_DIR = os.path.join(WORK_DIR, "features", "Spatial-VG-depth-stats")
    os.makedirs(OUT_DIR, exist_ok=True)
    _rc = run_streaming(
        [
            os.path.join(REPO_ROOT, "scripts", "depth_stats.py"), "vg",
            "--data", DATA_DIR, "--out", OUT_DIR,
            "--zip", os.path.join(RAW_DIR, "images.zip"), os.path.join(RAW_DIR, "images2.zip"),
            "--image-data", os.path.join(RAW_DIR, "image_data.json.zip"),
            "--workers", str(min(16, os.cpu_count() or 1)), "--img-batch", img_batch_ui.value,
        ],
        cwd=REPO_ROOT, log_path=os.path.join(OUT_DIR, "depth_stats.log"),
    )
    assert _rc == 0, f"depth_stats.py exited with {_rc}; see {OUT_DIR}/depth_stats.log"
    with open(os.path.join(OUT_DIR, "depth_stats.json"), encoding="utf-8") as _f:
        _info = json.load(_f)
    mo.md(
        f"""
        ### Kết quả
        - Ảnh: {_info['n_images']:,}; box có số độ sâu: {_info['n_boxes']:,}; ảnh lỗi: {_info['n_unreadable']};
          box ngoài ảnh: {_info['n_outside']}; thực thể điền trung bình train: {_info['filled_entities']:,}
        - Thời gian: {_info['sec'] / 60:.1f} phút; code `{_info['code_commit']}`
        """
    )
    return (OUT_DIR,)


@app.cell
def _(mo):
    upload_btn = mo.ui.run_button(label="⬆ Mở pull request lên HF")
    mo.vstack([
        mo.md("Đẩy `ent_depth_stats.pt`, `depth_stats.json`, `depth_stats.log` lên `callmezuy/spatial-vg` dưới dạng "
              "**pull request** (không đổi `main`); duyệt, merge rồi mới dời tag `v0`. Chỉ bấm sau khi đã xem kết quả."),
        upload_btn,
    ])
    return (upload_btn,)


@app.cell
def _(HF_REPO_ID, HfApi, OUT_DIR, hf_token_ui, mo, upload_btn):
    mo.stop(not upload_btn.value, mo.md("⏸ Chưa đẩy lên HF."))
    mo.stop(not hf_token_ui.value.strip(), mo.md("⚠ Chưa nhập HF token ở ô phía trên."))
    _info = HfApi(token=hf_token_ui.value.strip()).upload_folder(
        repo_id=HF_REPO_ID, repo_type="dataset", folder_path=OUT_DIR, create_pr=True,
        allow_patterns=["ent_depth_stats.pt", "depth_stats.json", "depth_stats.log"],
        commit_message="Data v0: depth statistics table (ZipDepth 3x3 grid; RESEARCH_PLAN.md P2.9)",
    )
    mo.md(f"Pull request: {_info.pr_url}")
    return


if __name__ == "__main__":
    app.run()
