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
    from huggingface_hub import HfApi, HfFileSystem, snapshot_download

    mo.md(
        """
        # 🧪 Reproduce NativE Baseline & Ablations on Unified-MMKG (Molab + HF Remote Storage)
        Notebook này kết nối tới **Hugging Face Remote Storage** (`callmezuy/unified-mmkg`, ghim tag `v2`) và chạy 3 kịch bản thực nghiệm của **NativE**:
        1. **Control (2-Modal):** Visual (`RGB`) + Textual (`Text`), không dùng Depth.
        2. **Ablation 1 (Early Fusion Mean):** Visual (`Mean(RGB + Depth)`) + Textual (`Text`).
        3. **Ablation 2 (Late Fusion 3-Modal):** Visual (`RGB`) + Textual (`Text`) + Numeric (`Depth`).

        Giao thức (CONTRIBUTING.md mục 4): chọn checkpoint theo MRR valid (`-select_on=valid`), test một lần,
        ghi rank từng triplet ra CSV cho `scripts/compute_metrics.py`. Log, rank CSV và checkpoint của mỗi kịch bản
        được **đẩy lên HF ngay khi chạy xong** (cần secret `HF_TOKEN` có quyền ghi).
        """
    )
    return HfApi, HfFileSystem, mo, os, shutil, snapshot_download, subprocess, sys, torch


@app.cell
def _(HfFileSystem, mo, os):
    # Khởi tạo HfFileSystem ở biến toàn cục để Marimo tự động nhận diện vào tab Remote Storage
    hf_fs = HfFileSystem()
    HF_REPO_ID = "callmezuy/unified-mmkg"
    DATA_REVISION = "v2"  # ghim phiên bản dữ liệu để tái lập được (data/README.md)
    HF_STORAGE_PATH = f"datasets/{HF_REPO_ID}"

    _remote_files = [os.path.basename(p) for p in hf_fs.ls(HF_STORAGE_PATH, detail=False)]
    mo.md(
        f"""
        ### 🔗 Kết nối Remote Storage thành công: `{HF_STORAGE_PATH}` (revision `{DATA_REVISION}`)
        **Các tệp trên Hugging Face:** `{', '.join(_remote_files)}`
        """
    )
    return DATA_REVISION, HF_REPO_ID, HF_STORAGE_PATH, hf_fs


@app.cell
def _(DATA_REVISION, HF_REPO_ID, os, shutil, snapshot_download, subprocess, torch):
    # 1. Đảm bảo mã nguồn Geometry-MMKG (chứa baselines/NativE) có mặt trên Molab
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "NativE")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "NativE")):
        REPO_ROOT = os.path.dirname(WORK_DIR)
    else:
        REPO_ROOT = os.path.join(WORK_DIR, "Geometry-MMKG")
        if not os.path.exists(REPO_ROOT):
            print("Cloning Geometry-MMKG repository...")
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
    os.makedirs(os.path.join(NATIVE_DIR, "checkpoint"), exist_ok=True)

    # 2. Tải dữ liệu Unified-MMKG (v2) từ Hugging Face
    DATA_DIR = os.path.join(WORK_DIR, "data", "Unified-MMKG")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Syncing dataset {HF_REPO_ID}@{DATA_REVISION} to {DATA_DIR}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=DATA_REVISION,
        local_dir=DATA_DIR,
    )

    # 3. Chuẩn bị benchmarks/Unified-MMKG và embeddings/ trong NativE
    DATASET_NAME = "Unified-MMKG"
    BENCHMARK_DIR = os.path.join(NATIVE_DIR, "benchmarks", DATASET_NAME)
    EMB_DIR = os.path.join(NATIVE_DIR, "embeddings")
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

    for _fname in os.listdir(DATA_DIR):
        if _fname.endswith(".txt"):
            shutil.copy2(os.path.join(DATA_DIR, _fname), os.path.join(BENCHMARK_DIR, _fname))
    for _split in ["train2id.txt", "valid2id.txt", "test2id.txt"]:
        assert os.path.exists(os.path.join(BENCHMARK_DIR, _split)), f"Thiếu {_split} trong {HF_REPO_ID}@{DATA_REVISION}"

    # 4. Embedding .pth theo tên NativE yêu cầu (-visual.pth, -textual.pth, -numeric.pth)
    print("Preparing NativE modality embeddings (.pth)...")
    _rgb_path = os.path.join(DATA_DIR, "ent_rgb_emb.pt")
    _txt_path = os.path.join(DATA_DIR, "ent_text_emb.pt")
    _dep_path = os.path.join(DATA_DIR, "ent_depth_emb.pt")

    shutil.copy2(_rgb_path, os.path.join(EMB_DIR, f"{DATASET_NAME}-rgb-visual.pth"))
    shutil.copy2(_rgb_path, os.path.join(EMB_DIR, f"{DATASET_NAME}-visual.pth"))
    shutil.copy2(_txt_path, os.path.join(EMB_DIR, f"{DATASET_NAME}-textual.pth"))
    shutil.copy2(_dep_path, os.path.join(EMB_DIR, f"{DATASET_NAME}-numeric.pth"))

    _rgb_tensor = torch.load(_rgb_path, map_location="cpu", weights_only=False).float()
    _dep_tensor = torch.load(_dep_path, map_location="cpu", weights_only=False).float()
    _mean_tensor = (_rgb_tensor + _dep_tensor) / 2.0
    torch.save(_mean_tensor, os.path.join(EMB_DIR, f"{DATASET_NAME}-mean-visual.pth"))
    del _rgb_tensor, _dep_tensor, _mean_tensor

    # 5. Biên dịch C++ Data Loader (Base.so)
    _mmkgc_dir = os.path.join(NATIVE_DIR, "mmkgc")
    _make_sh = os.path.join(_mmkgc_dir, "make.sh")
    with open(_make_sh, "r", encoding="utf-8") as _f:
        _sh_content = _f.read().replace("\r\n", "\n").replace("mkdir release", "mkdir -p release")
    with open(_make_sh, "w", encoding="utf-8") as _f:
        _f.write(_sh_content)
    print("Compiling C++ OpenKE Base.so in mmkgc...")
    subprocess.run(["bash", "make.sh"], cwd=_mmkgc_dir, check=True)
    # Các patch cũ (bỏ pooling, type_constrain=False, load_checkpoint) đã nằm sẵn trong repo, không cần sửa lúc chạy.

    print("NativE environment & Unified-MMKG data setup complete!")
    return BENCHMARK_DIR, DATASET_NAME, DATA_DIR, EMB_DIR, NATIVE_DIR, REPO_ROOT, WORK_DIR


@app.cell
def _(DATASET_NAME, HfApi, WORK_DIR, os, subprocess, sys):
    # Kết quả: log + rank CSV + checkpoint của từng kịch bản, đẩy lên HF ngay sau khi chạy xong.
    SEED = 42
    RESULTS_REPO = "callmezuy/geometry-mmkg-results"  # dataset repo private, tự tạo nếu chưa có
    RESULTS_DIR = os.path.join(WORK_DIR, "results", "native", DATASET_NAME)
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
        """Đẩy RESULTS_DIR lên HF (results/native/<dataset>/). Lỗi upload không làm hỏng kết quả local."""
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
    epochs_ui = mo.ui.number(start=10, stop=2000, step=10, value=1000, label="Epochs")
    batch_size_ui = mo.ui.dropdown(options=[256, 512, 1024, 2048], value=1024, label="Batch Size")
    dim_ui = mo.ui.dropdown(options=[100, 128, 200, 250], value=250, label="Embedding Dim")
    margin_ui = mo.ui.number(start=2, stop=24, step=1, value=12, label="Margin")
    neg_num_ui = mo.ui.dropdown(options=[32, 64, 128], value=128, label="Negative Samples")
    train_btn = mo.ui.run_button(label="▶ Train 3 kịch bản")

    mo.vstack([
        mo.md("Mặc định = config gốc của NativE cho DB15K (`baselines/NativE/scripts/run_db15k.sh`). "
              "Đặt xong tham số rồi mới bấm nút: đổi tham số **không** tự chạy lại training."),
        mo.hstack([epochs_ui, batch_size_ui, dim_ui, margin_ui, neg_num_ui]),
        train_btn,
    ])
    return batch_size_ui, dim_ui, epochs_ui, margin_ui, neg_num_ui, train_btn


@app.cell
def _(
    DATASET_NAME,
    EMB_DIR,
    NATIVE_DIR,
    RESULTS_DIR,
    SEED,
    batch_size_ui,
    dim_ui,
    epochs_ui,
    margin_ui,
    neg_num_ui,
    os,
    push_results,
    run_streaming,
    shutil,
):
    def run_native(variant, script, visual_src):
        """Một kịch bản NativE: chọn file visual, train (chọn checkpoint theo valid), test một lần, đẩy kết quả."""
        shutil.copy2(
            os.path.join(EMB_DIR, f"{DATASET_NAME}-{visual_src}.pth"),
            os.path.join(EMB_DIR, f"{DATASET_NAME}-visual.pth"),
        )
        _tag = f"{DATASET_NAME}_{variant}_seed{SEED}"
        _ckpt = os.path.join(NATIVE_DIR, "checkpoint", f"native_{_tag}.ckpt")
        _cmd = [
            "python", script,
            f"-dataset={DATASET_NAME}",
            f"-batch_size={int(batch_size_ui.value)}",
            f"-margin={int(margin_ui.value)}",
            f"-epoch={int(epochs_ui.value)}",
            f"-dim={int(dim_ui.value)}",
            "-adv_num=1",
            f"-save={_ckpt}",
            f"-rank_file={os.path.join(RESULTS_DIR, f'ranks_native_{_tag}_test.csv')}",
            "-select_on=valid", "-valid_steps=10",  # checkpoint = best validation MRR (ValidSelector.py)
            f"-seed={SEED}",
            f"-neg_num={int(neg_num_ui.value)}",
            "-mu=0.0001",
            "-learning_rate=1e-4",
            "-lrg=1e-4",
        ]
        _rc = run_streaming(_cmd, NATIVE_DIR, os.path.join(RESULTS_DIR, f"native_{_tag}.log"))
        if _rc == 0 and os.path.exists(_ckpt):
            shutil.copy2(_ckpt, os.path.join(RESULTS_DIR, os.path.basename(_ckpt)))
        else:
            print(f"❌ {variant} thất bại (exit code {_rc}), xem log ở trên.")
        push_results(f"native/{DATASET_NAME}")
        return _rc

    return (run_native,)


@app.cell
def _(mo, run_native, train_btn):
    mo.stop(not train_btn.value, mo.md("⏸ Bấm **▶ Train 3 kịch bản** để bắt đầu."))
    print("=== [1/3] CONTROL NativE (2 Modalities: Visual=RGB, Textual=Text) ===")
    ctrl_rc = run_native("Control", "run_adv_wgan_gp.py", "rgb-visual")
    mo.md(f"### {'✅' if ctrl_rc == 0 else '❌'} Kịch bản 1: Control (exit code {ctrl_rc})")
    return (ctrl_rc,)


@app.cell
def _(ctrl_rc, mo, run_native, train_btn):
    mo.stop(not train_btn.value)
    _ = ctrl_rc  # chạy sau kịch bản 1 (các kịch bản dùng chung file -visual.pth)
    print("=== [2/3] ABLATION 1 (Early Fusion: Visual=Mean(RGB+Depth), Textual=Text) ===")
    mean_rc = run_native("Ablation-Mean", "run_adv_wgan_gp.py", "mean-visual")
    mo.md(f"### {'✅' if mean_rc == 0 else '❌'} Kịch bản 2: Ablation 1 - Early Fusion Mean (exit code {mean_rc})")
    return (mean_rc,)


@app.cell
def _(mean_rc, mo, run_native, train_btn):
    mo.stop(not train_btn.value)
    _ = mean_rc
    print("=== [3/3] ABLATION 2 (Late Fusion 3-Modal: Visual=RGB, Textual=Text, Numeric=Depth) ===")
    modal3_rc = run_native("Ablation-3Modal", "run_adv_wgan_gp_3modal.py", "rgb-visual")
    mo.md(f"### {'✅' if modal3_rc == 0 else '❌'} Kịch bản 3: Ablation 2 - Late Fusion 3-Modal (exit code {modal3_rc})")
    return (modal3_rc,)


@app.cell
def _(RESULTS_DIR, RESULTS_REPO, mo, modal3_rc):
    _ = modal3_rc
    mo.md(
        f"""
        ### 📦 Kết quả
        Local: `{RESULTS_DIR}` · HF: `{RESULTS_REPO}` (thư mục `native/Unified-MMKG/`).

        1. Kiểm tra dòng `Rank check ... C++: ... | Python: ...` trong log: hai bộ số phải khớp (P1.4).
        2. Tính số báo cáo trên máy local (CONTRIBUTING.md mục 4):
        ```
        python scripts/compute_metrics.py --data data/Merged-MMKG \\
            --ranks control=ranks_native_Unified-MMKG_Control_seed42_test.csv \\
                    mean=ranks_native_Unified-MMKG_Ablation-Mean_seed42_test.csv \\
                    3modal=ranks_native_Unified-MMKG_Ablation-3Modal_seed42_test.csv \\
            --reference control --by-relation
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
