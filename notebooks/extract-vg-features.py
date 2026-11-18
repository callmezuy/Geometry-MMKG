import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import os
    import subprocess
    import sys

    import marimo as mo
    from huggingface_hub import HfApi, snapshot_download

    mo.md(
        """
        # Đặc trưng cho Spatial-VG (Molab GPU)

        Chạy `scripts/extract_features.py vg` (RESEARCH_PLAN.md P0.7) trên 108.073 ảnh / 1.441.984 thực thể
        của Spatial-VG: ZipDepth (384 px, min-max theo ảnh) → cắt box bằng `roi_align` → DINOv2-base cho RGB và
        depth; RoBERTa-base cho text. Mỗi thực thể một box nên không cần lấy trung bình nhiều ảnh.

        **Không** ghi `ent_rgb_depth_mean_emb.pt` / `ent_rgb_depth_concat_emb.pt` (người dùng duyệt 18/11/2026):
        NativE, SNAG, MoMoK không chạy trên VG, và bảng concat một mình khoảng 8,9 GB.

        Cần: code mới nhất của `scripts/extract_features.py` và `scripts/feature_quality.py` đã được push lên
        GitHub; khoảng 40 GB đĩa (ảnh 15 GB, bảng tạm 9 GB, đầu ra khoảng 13 GB).
        """
    )
    return HfApi, hashlib, json, mo, os, snapshot_download, subprocess, sys


@app.cell
def _(hashlib, os, subprocess):
    # 1. Code: repo dự án (scripts/ + bản sao ZipDepth/ kèm checkpoint)
    WORK_DIR = os.path.abspath(".")
    REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
    if not os.path.exists(REPO_ROOT):
        subprocess.run(["git", "clone", "https://github.com/callmezuy/Geometry-MMKG.git", REPO_ROOT], check=True)
    else:
        subprocess.run(["git", "-C", REPO_ROOT, "pull"], check=False)
    CODE_COMMIT = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    _script = open(os.path.join(REPO_ROOT, "scripts", "extract_features.py"), encoding="utf-8").read()
    assert "def kg_scenes" in _script and "ent_img_mask" in _script, (
        "scripts/extract_features.py trên GitHub còn là bản cũ: hãy push code mới nhất trước"
    )

    ZIPDEPTH_COMMIT = "91f3fd21e131641f51e8d35736d1958350180e3a"  # upstream fabiotosi92/ZipDepth mà ZipDepth/ trong repo dự án sao lại (đối chiếu 18/11/2026)
    ZIPDEPTH_MD5 = "23f5f042af37939afa80b0f056246df5"  # checkpoints/zipdepth_base.pth
    _zd = os.path.join(REPO_ROOT, "ZipDepth")  # đã nằm trong repo dự án, không clone riêng
    assert os.path.exists(os.path.join(_zd, "zipdepth", "model", "architecture.py")), f"thiếu {_zd}"
    with open(os.path.join(_zd, "checkpoints", "zipdepth_base.pth"), "rb") as _f:
        _md5 = hashlib.md5(_f.read()).hexdigest()
    assert _md5 == ZIPDEPTH_MD5, f"zipdepth_base.pth md5 {_md5} != {ZIPDEPTH_MD5}"
    try:
        import h5py  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "h5py"], check=True)
    print(f"code {CODE_COMMIT[:10]}, ZipDepth {ZIPDEPTH_COMMIT[:10]} (checkpoint md5 ok)")
    return CODE_COMMIT, REPO_ROOT, WORK_DIR, ZIPDEPTH_COMMIT


@app.cell
def _(WORK_DIR, os, snapshot_download):
    # 2. Split Spatial-VG từ HF (repo v0, P0.8): commit chỉ có split + H5; tag v0 gắn sau khi đặc trưng được merge
    HF_REPO_ID = "callmezuy/spatial-vg"
    DATA_REVISION = "b1e85c0ecd40a721eddb51e8323303dbbb4a2e53"  # "Data v0: structure" (18/11/2026)
    DATA_DIR = os.path.join(WORK_DIR, "data", "Spatial-VG")
    snapshot_download(
        repo_id=HF_REPO_ID, repo_type="dataset", revision=DATA_REVISION, local_dir=DATA_DIR,
        allow_patterns=["*.txt", "*.h5", "*.json", "*.tsv"],
    )
    with open(os.path.join(DATA_DIR, "entity2id.txt"), encoding="utf-8") as _f:
        N_ENT = int(_f.readline())
    assert N_ENT == 1441984, N_ENT
    print(f"{HF_REPO_ID}@{DATA_REVISION} -> {DATA_DIR}: {N_ENT:,} entities")
    return DATA_DIR, DATA_REVISION, HF_REPO_ID


@app.cell(hide_code=False)
def _(mo):
    hf_token_ui = mo.ui.text(kind="password", label="HF token", full_width=True)
    mo.vstack([
        mo.md("Dán HF token: cần quyền **read** repo private `callmezuy/vg-images-raw` (ảnh VG, tải nhanh) và quyền "
              "**write** để mở pull request lên `callmezuy/spatial-vg`. Token chỉ nằm trong bộ nhớ kernel, "
              "không được ghi vào file notebook. Để trống thì ảnh tải từ Stanford (rất chậm, khoảng 1 MB/s)."),
        hf_token_ui,
    ])
    return (hf_token_ui,)


@app.cell
def _(WORK_DIR, hf_token_ui, mo, os, snapshot_download, subprocess):
    # 3. Ảnh VG gốc, kiểm tra đúng kích thước gốc. VG_SOURCE = "hf": bản sao private callmezuy/vg-images-raw
    # (chép nguyên file từ máy 18/11/2026, tải vài phút, cần token ở ô trên); "stanford": Stanford / UW (khoảng 1 MB/s,
    # curl -C - tải tiếp nếu bị ngắt)
    VG_SOURCE = "hf"
    RAW_DIR = os.path.join(WORK_DIR, "data", "raw", "vg")
    os.makedirs(RAW_DIR, exist_ok=True)
    VG_IMAGES_REPO = "callmezuy/vg-images-raw"
    _token = hf_token_ui.value.strip() or None
    mo.stop(VG_SOURCE == "hf" and not _token, mo.md("⏸ Dán HF token ở ô phía trên để tải ảnh VG từ HF."))
    _files = {
        "images.zip": ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", 9731705982),
        "images2.zip": ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", 5471658058),
        "image_data.json.zip": ("https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip", 1780854),
    }
    for _name, (_url, _size) in _files.items():
        _p = os.path.join(RAW_DIR, _name)
        if not (os.path.exists(_p) and os.path.getsize(_p) == _size):
            if VG_SOURCE == "hf":
                print(f"downloading {_name} ({_size / 2**30:.1f} GB) from HF {VG_IMAGES_REPO}...", flush=True)
                if os.path.exists(_p):
                    os.remove(_p)  # partial file from Stanford
                snapshot_download(repo_id=VG_IMAGES_REPO, repo_type="dataset", local_dir=RAW_DIR,
                                  allow_patterns=[_name], token=_token)
            else:
                print(f"downloading {_name} ({_size / 2**30:.1f} GB) from {_url}...", flush=True)
                subprocess.run(["curl", "-sSL", "-C", "-", "-o", _p, _url], check=True)
        assert os.path.getsize(_p) == _size, f"{_name}: {os.path.getsize(_p)} != {_size}"
    print("VG images ok")
    return (RAW_DIR,)


@app.cell
def _(mo):
    enc_batch_ui = mo.ui.dropdown(options=["256", "512", "1024", "2048"], value="1024", label="DINOv2 batch (crops, x2)")
    img_batch_ui = mo.ui.dropdown(options=["16", "32", "64", "128"], value="64", label="ZipDepth batch (images)")
    extract_btn = mo.ui.run_button(label="▶ Trích đặc trưng VG")
    mo.vstack([
        mo.md("DINOv2 batch x2 (RGB + depth) phải vừa VRAM: batch 256 cần khoảng 4,6 GB, nên 1024 khoảng 18 GB và "
              "2048 khoảng 37 GB. GPU Molab có 96 GB nên mặc định 1024 / 64; GPU 4 GB (RTX 3050) dùng 64 / 16. "
              "Đổi tham số **không** tự chạy lại; bấm nút để bắt đầu."),
        mo.hstack([enc_batch_ui, img_batch_ui]),
        extract_btn,
    ])
    return enc_batch_ui, extract_btn, img_batch_ui


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
                    if "crops.append" in _line or "UserWarning" in _line:
                        continue  # PIL non-writable-array warning, one per DataLoader worker
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
def _(
    DATA_DIR,
    RAW_DIR,
    REPO_ROOT,
    WORK_DIR,
    enc_batch_ui,
    extract_btn,
    img_batch_ui,
    mo,
    os,
    run_streaming,
):
    mo.stop(not extract_btn.value, mo.md("⏸ Bấm **▶ Trích đặc trưng VG** để bắt đầu."))
    OUT_DIR = os.path.join(WORK_DIR, "features", "Spatial-VG")
    os.makedirs(OUT_DIR, exist_ok=True)
    _rc = run_streaming(
        [
            os.path.join(REPO_ROOT, "scripts", "extract_features.py"), "vg",
            "--data", DATA_DIR, "--out", OUT_DIR,
            "--zip", os.path.join(RAW_DIR, "images.zip"), os.path.join(RAW_DIR, "images2.zip"),
            "--image-data", os.path.join(RAW_DIR, "image_data.json.zip"),
            "--workers", str(min(16, os.cpu_count() or 1)),
            "--img-batch", img_batch_ui.value, "--enc-batch", enc_batch_ui.value,
            # no --mean-concat on VG (see the first cell)
        ],
        cwd=REPO_ROOT, log_path=os.path.join(OUT_DIR, "extract.log"),
    )
    assert _rc == 0, f"extract_features.py exited with {_rc}; see {OUT_DIR}/extract.log"
    return (OUT_DIR,)


@app.cell
def _(CODE_COMMIT, DATA_DIR, OUT_DIR, REPO_ROOT, ZIPDEPTH_COMMIT, json, mo, os, run_streaming):
    # 4. Chất lượng đặc trưng (scripts/feature_quality.py): cosine, PC1, probe 150 lớp vật thể VG, 10-NN depth vs RGB
    _rc = run_streaming(
        [os.path.join(REPO_ROOT, "scripts", "feature_quality.py"), "--data", DATA_DIR, "--feat", OUT_DIR, "--labels", "vg"],
        cwd=REPO_ROOT, log_path=os.path.join(OUT_DIR, "quality.log"),
    )
    assert _rc == 0, f"feature_quality.py exited with {_rc}"
    _stats_path = os.path.join(OUT_DIR, "feature_stats.json")
    with open(_stats_path, encoding="utf-8") as _f:
        _stats = json.load(_f)
    _stats.update({"code_commit": CODE_COMMIT, "zipdepth_commit": ZIPDEPTH_COMMIT, "run_on": "molab"})
    with open(_stats_path, "w", encoding="utf-8") as _f:
        json.dump(_stats, _f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "feature_quality.json"), encoding="utf-8") as _f:
        _q = json.load(_f)
    mo.md(
        f"""
        ### Kết quả
        - Thực thể có ảnh: {_stats['entities_with_image']:,} / {_stats['n_entities']:,}; ảnh lỗi: {_stats['n_unreadable']}
        - Probe 150 lớp (đa số {_q['rgb']['majority_acc']:.3f}): RGB {_q['rgb']['probe_acc']:.3f} ·
          depth {_q['depth']['probe_acc']:.3f} · text {_q['text']['probe_acc']:.3f}
        - Cosine trung bình: RGB {_q['rgb']['mean_cos']:.3f} · depth {_q['depth']['mean_cos']:.3f} · text {_q['text']['mean_cos']:.3f}
        - 10-NN depth trùng RGB: {_q['knn10_overlap_depth_rgb']:.3f} (ngẫu nhiên {_q['chance']:.6f})
        """
    )
    return


@app.cell
def _(mo):
    upload_btn = mo.ui.run_button(label="⬆ Mở pull request lên HF")
    mo.vstack([
        mo.md("Đẩy đặc trưng lên `callmezuy/spatial-vg` dưới dạng **pull request** (không đổi `main`; "
              "duyệt, merge rồi mới gắn tag `v0`, P0.8). Chỉ bấm sau khi đã xem kết quả ở trên."),
        upload_btn,
    ])
    return (upload_btn,)


@app.cell
def _(HF_REPO_ID, HfApi, OUT_DIR, hf_token_ui, mo, upload_btn):
    mo.stop(not upload_btn.value, mo.md("⏸ Chưa đẩy lên HF."))
    mo.stop(not hf_token_ui.value.strip(), mo.md("⚠ Chưa nhập HF token ở ô phía trên."))
    _info = HfApi(token=hf_token_ui.value.strip()).upload_folder(
        repo_id=HF_REPO_ID, repo_type="dataset", folder_path=OUT_DIR, create_pr=True,
        allow_patterns=["ent_rgb_emb.pt", "ent_depth_emb.pt", "ent_text_emb.pt", "ent_img_mask.pt", "ent_fill.pt",
                        "feature_stats.json", "feature_quality.json", "extract.log", "quality.log"],
        commit_message="Data v0: features (DINOv2 RGB/depth, RoBERTa text; RESEARCH_PLAN.md P0.7)",
    )
    mo.md(f"Pull request: {_info.pr_url}")
    return


if __name__ == "__main__":
    app.run()
