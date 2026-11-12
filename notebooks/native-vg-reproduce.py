import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import sys
    import shutil
    import subprocess
    import marimo as mo
    import torch
    import torch.nn as nn
    from huggingface_hub import HfFileSystem, snapshot_download

    mo.md(
        """
        # 🧪 Reproduce NativE Baseline & Ablations on Spatial-VG (Molab + HF Remote Storage)
        Notebook này kết nối trực tiếp tới **Hugging Face Remote Storage** (`callmezuy/spatial-vg`) thông qua `HfFileSystem` và thực thi trọn bộ 3 kịch bản thực nghiệm của **NativE** trên bộ dữ liệu quy mô lớn **Spatial-VG v2** (VG150-curated; 1,441,984 thực thể, 50 quan hệ; split chính thức theo ảnh, inductive):
        - Sử dụng cơ chế `mmap=True` kết hợp `AdaptiveAvgPool1d(256)` theo từng chunk trên CPU trước khi nạp vào NativE (như kiến trúc gốc của `AdvRelRotatEDB15K` giảm chiều về `256`), giúp giảm dung lượng mỗi bảng embedding từ `4.43 GB` xuống `1.47 GB` để tránh tràn VRAM/RAM khi huấn luyện 3-Modal trên 1.44 triệu thực thể.
        """
    )
    return (
        HfFileSystem,
        mo,
        nn,
        os,
        re,
        shutil,
        snapshot_download,
        subprocess,
        sys,
        torch,
    )


@app.cell
def _(HfFileSystem, mo, os):
    # Khởi tạo HfFileSystem ở biến toàn cục để Marimo tự động nhận diện vào tab Remote Storage
    hf_fs = HfFileSystem()
    HF_REPO_ID = "callmezuy/spatial-vg"
    HF_STORAGE_PATH = f"datasets/{HF_REPO_ID}"

    _remote_files = [os.path.basename(p) for p in hf_fs.ls(HF_STORAGE_PATH, detail=False)]
    mo.md(
        f"""
        ### 🔗 Kết nối Remote Storage thành công: `{HF_STORAGE_PATH}`
        **Các tệp trên Hugging Face:** `{', '.join(_remote_files)}`
        """
    )
    return HF_REPO_ID, HF_STORAGE_PATH, hf_fs


@app.cell
def _(HF_REPO_ID, nn, os, re, shutil, snapshot_download, subprocess, torch):
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

    NATIVE_DIR = os.path.join(REPO_ROOT, "baselines", "NativE")

    # 2. Tải dữ liệu Spatial-VG từ Hugging Face Remote Storage
    DATA_DIR = os.path.join(WORK_DIR, "data", "Spatial-VG")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Syncing dataset from {HF_REPO_ID} to {DATA_DIR}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )

    # 3. Chuẩn bị thư mục benchmarks/Spatial-VG và embeddings/ trong NativE
    DATASET_NAME = "Spatial-VG"
    BENCHMARK_DIR = os.path.join(NATIVE_DIR, "benchmarks", DATASET_NAME)
    EMB_DIR = os.path.join(NATIVE_DIR, "embeddings")
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

    for _fname in os.listdir(DATA_DIR):
        if _fname.endswith(".txt"):
            shutil.copy2(os.path.join(DATA_DIR, _fname), os.path.join(BENCHMARK_DIR, _fname))

    # Tự động sinh train2id.txt, valid2id.txt, test2id.txt nếu chưa có
    if not os.path.exists(os.path.join(BENCHMARK_DIR, "train2id.txt")):
        print("Generating OpenKE *2id.txt files from (h, r, t) text files...")
        _ent2id = {}
        with open(os.path.join(BENCHMARK_DIR, "entity2id.txt"), "r", encoding="utf-8") as _f:
            for _line in _f.readlines()[1:]:
                _parts = _line.strip().split("\t")
                if len(_parts) >= 2:
                    _ent2id[_parts[0]] = _parts[1]
        _rel2id = {}
        with open(os.path.join(BENCHMARK_DIR, "relation2id.txt"), "r", encoding="utf-8") as _f:
            for _line in _f.readlines()[1:]:
                _parts = _line.strip().split("\t")
                if len(_parts) >= 2:
                    _rel2id[_parts[0]] = _parts[1]
        for _split in ["train", "valid", "test"]:
            _rows = []
            with open(os.path.join(BENCHMARK_DIR, f"{_split}.txt"), "r", encoding="utf-8") as _f:
                for _line in _f:
                    _p = _line.strip().split("\t")
                    if len(_p) == 3 and _p[0] in _ent2id and _p[1] in _rel2id and _p[2] in _ent2id:
                        _rows.append(f"{_ent2id[_p[0]]}\t{_ent2id[_p[2]]}\t{_rel2id[_p[1]]}\n")
            with open(os.path.join(BENCHMARK_DIR, f"{_split}2id.txt"), "w", encoding="utf-8") as _f:
                _f.write(f"{len(_rows)}\n")
                _f.writelines(_rows)

    # 4. Chuẩn bị các file embedding .pth (Pool 768 -> 256 theo từng chunk với mmap=True để tiết kiệm RAM/VRAM)
    def _pool_and_save(src_pt, dst_pth, target_dim=256, chunk_size=100000):
        if os.path.exists(dst_pth):
            return
        print(f"Pooling {os.path.basename(src_pt)} -> {os.path.basename(dst_pth)} (dim={target_dim})...")
        _src = torch.load(src_pt, map_location="cpu", mmap=True, weights_only=False)
        _num_ent = _src.shape[0]
        _out = torch.empty((_num_ent, target_dim), dtype=torch.float32)
        _pool = nn.AdaptiveAvgPool1d(target_dim)
        with torch.no_grad():
            for _start in range(0, _num_ent, chunk_size):
                _end = min(_start + chunk_size, _num_ent)
                _chunk = _src[_start:_end].float().unsqueeze(1)
                _out[_start:_end] = _pool(_chunk).squeeze(1)
        torch.save(_out, dst_pth)
        del _src, _out

    _rgb_src = os.path.join(DATA_DIR, "ent_rgb_emb.pt")
    _txt_src = os.path.join(DATA_DIR, "ent_text_emb.pt")
    _dep_src = os.path.join(DATA_DIR, "ent_depth_emb.pt")

    _rgb_dst = os.path.join(EMB_DIR, f"{DATASET_NAME}-rgb-visual.pth")
    _vis_dst = os.path.join(EMB_DIR, f"{DATASET_NAME}-visual.pth")
    _txt_dst = os.path.join(EMB_DIR, f"{DATASET_NAME}-textual.pth")
    _num_dst = os.path.join(EMB_DIR, f"{DATASET_NAME}-numeric.pth")
    _mean_dst = os.path.join(EMB_DIR, f"{DATASET_NAME}-mean-visual.pth")

    _pool_and_save(_rgb_src, _rgb_dst, target_dim=256)
    shutil.copy2(_rgb_dst, _vis_dst)
    _pool_and_save(_txt_src, _txt_dst, target_dim=256)
    _pool_and_save(_dep_src, _num_dst, target_dim=256)

    if not os.path.exists(_mean_dst):
        print("Creating Early Fusion Mean (RGB + Depth) embedding...")
        _r = torch.load(_rgb_dst, map_location="cpu", weights_only=False)
        _d = torch.load(_num_dst, map_location="cpu", weights_only=False)
        torch.save((_r + _d) / 2.0, _mean_dst)
        del _r, _d

    # 5. Biên dịch C++ Data Loader (Base.so) cho kiến trúc CPU của Molab
    _mmkgc_dir = os.path.join(NATIVE_DIR, "mmkgc")
    _make_sh = os.path.join(_mmkgc_dir, "make.sh")
    with open(_make_sh, "r", encoding="utf-8") as _f:
        _sh_content = _f.read().replace("\r\n", "\n").replace("mkdir release", "mkdir -p release")
    with open(_make_sh, "w", encoding="utf-8") as _f:
        _f.write(_sh_content)
    print("Compiling C++ OpenKE Base.so in mmkgc...")
    subprocess.run(["bash", "make.sh"], cwd=_mmkgc_dir, check=True)

    # 6. Đảm bảo patch tương thích chiều embedding & PyTorch 2.6+ trong NativE
    _adv_db15k_py = os.path.join(_mmkgc_dir, "module", "model", "AdvRelRotatEDB15K.py")
    with open(_adv_db15k_py, "r", encoding="utf-8") as _f:
        _code = _f.read()
    if "img_pool = torch.nn.AvgPool2d" in _code:
        _code = re.sub(
            r"self\.img_dim = 256.*?self\.img_embeddings = nn\.Embedding\.from_pretrained",
            "self.img_dim = img_emb.shape[1]\n        self.text_dim = text_emb.shape[1]\n        self.numeric_dim = numeric_emb.shape[1]\n\n        self.img_embeddings = nn.Embedding.from_pretrained",
            _code,
            flags=re.DOTALL,
        )
        _code = _code.replace(
            "self.numeric_proj = nn.Sequential(\n            nn.Linear(self.text_dim, self.dim_e)",
            "self.numeric_proj = nn.Sequential(\n            nn.Linear(self.numeric_dim, self.dim_e)",
        )
        with open(_adv_db15k_py, "w", encoding="utf-8") as _f:
            _f.write(_code)

    for _run_script in ["run_adv_wgan_gp.py", "run_adv_wgan_gp_3modal.py"]:
        _script_path = os.path.join(NATIVE_DIR, _run_script)
        with open(_script_path, "r", encoding="utf-8") as _f:
            _sc = _f.read()
        _sc = _sc.replace('"link")', '"link", False)')
        _sc = _sc.replace("# kge_score.load_checkpoint(args.save)", "kge_score.load_checkpoint(args.save)")
        with open(_script_path, "w", encoding="utf-8") as _f:
            _f.write(_sc)

    print("NativE environment & Spatial-VG data setup complete!")
    return BENCHMARK_DIR, DATA_DIR, DATASET_NAME, EMB_DIR, NATIVE_DIR, REPO_ROOT, WORK_DIR


@app.cell
def _(mo):
    epochs_ui = mo.ui.number(start=5, stop=200, step=5, value=50, label="Epochs")
    batch_size_ui = mo.ui.dropdown(options=[512, 1024, 2048, 4096], value=2048, label="Batch Size")
    dim_ui = mo.ui.dropdown(options=[64, 100, 128, 200], value=128, label="Embedding Dim")
    margin_ui = mo.ui.number(start=2, stop=24, step=1, value=12, label="Margin")
    neg_num_ui = mo.ui.dropdown(options=[16, 32, 64, 128], value=64, label="Negative Samples")

    mo.hstack([epochs_ui, batch_size_ui, dim_ui, margin_ui, neg_num_ui])
    return batch_size_ui, dim_ui, epochs_ui, margin_ui, neg_num_ui


@app.cell
def _(
    DATASET_NAME,
    EMB_DIR,
    NATIVE_DIR,
    batch_size_ui,
    dim_ui,
    epochs_ui,
    margin_ui,
    mo,
    neg_num_ui,
    os,
    shutil,
    subprocess,
):
    print("=== [1/3] Running CONTROL NativE on Spatial-VG (2 Modalities: Visual=RGB, Textual=Text) ===")
    shutil.copy2(
        os.path.join(EMB_DIR, f"{DATASET_NAME}-rgb-visual.pth"),
        os.path.join(EMB_DIR, f"{DATASET_NAME}-visual.pth"),
    )

    _cmd_ctrl = [
        "python",
        "run_adv_wgan_gp.py",
        f"-dataset={DATASET_NAME}",
        f"-batch_size={int(batch_size_ui.value)}",
        f"-margin={int(margin_ui.value)}",
        f"-epoch={int(epochs_ui.value)}",
        f"-dim={int(dim_ui.value)}",
        "-adv_num=1",
        f"-save={DATASET_NAME}-Control",
        f"-neg_num={int(neg_num_ui.value)}",
        "-mu=0.0001",
        "-learning_rate=1e-4",
        "-lrg=1e-4",
    ]
    ctrl_res = subprocess.run(_cmd_ctrl, cwd=NATIVE_DIR, text=True, capture_output=True)
    print(ctrl_res.stdout)
    if ctrl_res.returncode != 0:
        print("STDERR:", ctrl_res.stderr)

    mo.md(f"### ✅ Hoàn tất kịch bản 1: Control (`{DATASET_NAME}-Control`)")
    return (ctrl_res,)


@app.cell
def _(
    DATASET_NAME,
    EMB_DIR,
    NATIVE_DIR,
    batch_size_ui,
    dim_ui,
    epochs_ui,
    margin_ui,
    mo,
    neg_num_ui,
    os,
    shutil,
    subprocess,
):
    print("=== [2/3] Running ABLATION 1 on Spatial-VG (Early Fusion: Visual=Mean(RGB+Depth)) ===")
    shutil.copy2(
        os.path.join(EMB_DIR, f"{DATASET_NAME}-mean-visual.pth"),
        os.path.join(EMB_DIR, f"{DATASET_NAME}-visual.pth"),
    )

    _cmd_mean = [
        "python",
        "run_adv_wgan_gp.py",
        f"-dataset={DATASET_NAME}",
        f"-batch_size={int(batch_size_ui.value)}",
        f"-margin={int(margin_ui.value)}",
        f"-epoch={int(epochs_ui.value)}",
        f"-dim={int(dim_ui.value)}",
        "-adv_num=1",
        f"-save={DATASET_NAME}-Ablation-Mean",
        f"-neg_num={int(neg_num_ui.value)}",
        "-mu=0.0001",
        "-learning_rate=1e-4",
        "-lrg=1e-4",
    ]
    mean_res = subprocess.run(_cmd_mean, cwd=NATIVE_DIR, text=True, capture_output=True)
    print(mean_res.stdout)
    if mean_res.returncode != 0:
        print("STDERR:", mean_res.stderr)

    mo.md(f"### ✅ Hoàn tất kịch bản 2: Ablation 1 - Early Fusion Mean (`{DATASET_NAME}-Ablation-Mean`)")
    return (mean_res,)


@app.cell
def _(
    DATASET_NAME,
    EMB_DIR,
    NATIVE_DIR,
    batch_size_ui,
    dim_ui,
    epochs_ui,
    margin_ui,
    mo,
    neg_num_ui,
    os,
    shutil,
    subprocess,
):
    print("=== [3/3] Running ABLATION 2 on Spatial-VG (Late Fusion 3-Modal: Visual=RGB, Textual=Text, Numeric=Depth) ===")
    shutil.copy2(
        os.path.join(EMB_DIR, f"{DATASET_NAME}-rgb-visual.pth"),
        os.path.join(EMB_DIR, f"{DATASET_NAME}-visual.pth"),
    )

    _cmd_3modal = [
        "python",
        "run_adv_wgan_gp_3modal.py",
        f"-dataset={DATASET_NAME}",
        f"-batch_size={int(batch_size_ui.value)}",
        f"-margin={int(margin_ui.value)}",
        f"-epoch={int(epochs_ui.value)}",
        f"-dim={int(dim_ui.value)}",
        "-adv_num=1",
        f"-save={DATASET_NAME}-Ablation-3Modal",
        f"-neg_num={int(neg_num_ui.value)}",
        "-mu=0.0001",
        "-learning_rate=1e-4",
        "-lrg=1e-4",
    ]
    modal3_res = subprocess.run(_cmd_3modal, cwd=NATIVE_DIR, text=True, capture_output=True)
    print(modal3_res.stdout)
    if modal3_res.returncode != 0:
        print("STDERR:", modal3_res.stderr)

    mo.md(f"### ✅ Hoàn tất kịch bản 3: Ablation 2 - Late Fusion 3-Modal (`{DATASET_NAME}-Ablation-3Modal`)")
    return (modal3_res,)


if __name__ == "__main__":
    app.run()
