import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import glob
    import sys
    import subprocess
    import marimo as mo
    from huggingface_hub import HfFileSystem, snapshot_download

    mo.md(
        """
        # 🧪 Reproduce MKGformer Baseline on Unified-MMKG (Molab + HF Remote Storage)
        Notebook này kết nối trực tiếp tới **Hugging Face Remote Storage** (`callmezuy/unified-mmkg`) thông qua `HfFileSystem`, tự động chuyển đổi định dạng sang cấu trúc của **MKGformer** (`train.tsv`, `dev.tsv`, `test.tsv`, `entities.txt`, `relations.txt`, `entity2text.txt`, `relation2text.txt`), và thực thi tự động cả 2 pha:
        1. **Phase 1 (`pretrain=1`):** Entity Modeling Pretraining.
        2. **Phase 2 (`pretrain=0`):** Tự động dò tìm checkpoint tốt nhất từ Phase 1 và chạy Fine-tuning đánh giá Link Prediction.
        """
    )
    return HfFileSystem, glob, mo, os, snapshot_download, subprocess, sys


@app.cell
def _(HfFileSystem, mo, os):
    # Khởi tạo HfFileSystem ở biến toàn cục để Marimo tự động nhận diện vào tab Remote Storage
    hf_fs = HfFileSystem()
    HF_REPO_ID = "callmezuy/unified-mmkg"
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
def _(HF_REPO_ID, os, snapshot_download, subprocess, sys):
    # 1. Đảm bảo mã nguồn Geometry-MMKG (chứa baselines/MKGformer) có mặt trên Molab
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "baselines", "MKGformer")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "baselines", "MKGformer")):
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

    MKG_DIR = os.path.join(REPO_ROOT, "baselines", "MKGformer", "MKG")
    if not os.path.exists(MKG_DIR):
        subprocess.run(["git", "-C", REPO_ROOT, "submodule", "update", "--init", "--recursive"], check=True)

    # Vá MKGformer (submodule upstream): weights_only=False cho PyTorch 2.6+, và ghi rank theo
    # từng triplet (head + tail, luật đếm chặt như OpenKE) khi đặt biến môi trường MKG_RANK_FILE.
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "patch_mkgformer.py"), MKG_DIR], check=True)

    # 2. Tải dữ liệu Unified-MMKG từ Hugging Face Remote Storage
    DATA_DIR = os.path.join(WORK_DIR, "data", "Unified-MMKG")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Syncing dataset from {HF_REPO_ID} to {DATA_DIR}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )

    # 3. Chuyển đổi Unified-MMKG sang định dạng MKGformer (dataset/FB15k-237 & dataset/FB15k-images)
    MKG_DATA_DIR = os.path.join(MKG_DIR, "dataset", "FB15k-237")
    MKG_IMG_DIR = os.path.join(MKG_DIR, "dataset", "FB15k-images")
    os.makedirs(MKG_DATA_DIR, exist_ok=True)
    os.makedirs(MKG_IMG_DIR, exist_ok=True)

    _id2ent = {}
    with open(os.path.join(DATA_DIR, "entity2id.txt"), "r", encoding="utf-8") as _f:
        for _line in _f.readlines()[1:]:
            _parts = _line.strip().split("\t")
            if len(_parts) >= 2:
                _id2ent[_parts[1]] = _parts[0]

    _id2rel = {}
    with open(os.path.join(DATA_DIR, "relation2id.txt"), "r", encoding="utf-8") as _f:
        for _line in _f.readlines()[1:]:
            _parts = _line.strip().split("\t")
            if len(_parts) >= 2:
                _id2rel[_parts[1]] = _parts[0]

    for _split, _tsv_name in [("train", "train.tsv"), ("valid", "dev.tsv"), ("test", "test.tsv")]:
        _src_txt = os.path.join(DATA_DIR, f"{_split}.txt")
        _dst_tsv = os.path.join(MKG_DATA_DIR, _tsv_name)
        with open(_src_txt, "r", encoding="utf-8") as _fin, open(_dst_tsv, "w", encoding="utf-8") as _fout:
            for _line in _fin:
                _p = _line.strip().split("\t")
                if len(_p) == 3:
                    # train.txt đã ở chuẩn (head, relation, tail)
                    _fout.write(f"{_p[0]}\t{_p[1]}\t{_p[2]}\n")

    with open(os.path.join(MKG_DATA_DIR, "entities.txt"), "w", encoding="utf-8") as _f:
        for _eid in range(len(_id2ent)):
            _f.write(f"{_id2ent[str(_eid)]}\n")

    with open(os.path.join(MKG_DATA_DIR, "relations.txt"), "w", encoding="utf-8") as _f:
        for _rid in range(len(_id2rel)):
            _f.write(f"{_id2rel[str(_rid)]}\n")

    with open(os.path.join(MKG_DATA_DIR, "entity2text.txt"), "w", encoding="utf-8") as _f:
        for _eid in range(len(_id2ent)):
            _e_str = _id2ent[str(_eid)]
            _f.write(f"{_e_str}\tEntity {_e_str}\n")

    with open(os.path.join(MKG_DATA_DIR, "relation2text.txt"), "w", encoding="utf-8") as _f:
        for _rid in range(len(_id2rel)):
            _r_str = _id2rel[str(_rid)]
            _f.write(f"{_r_str}\tRelation {_r_str}\n")

    print(f"Converted Unified-MMKG to MKGformer format at {MKG_DATA_DIR}!")
    return DATA_DIR, MKG_DATA_DIR, MKG_DIR, MKG_IMG_DIR, REPO_ROOT, WORK_DIR


@app.cell
def _(mo):
    pretrain_epochs_ui = mo.ui.number(start=1, stop=50, step=1, value=15, label="Phase 1 Epochs (Pretrain)")
    finetune_epochs_ui = mo.ui.number(start=1, stop=50, step=1, value=12, label="Phase 2 Epochs (Link Prediction)")
    pretrain_bs_ui = mo.ui.dropdown(options=[32, 64, 96, 128], value=64, label="Phase 1 Batch Size")
    finetune_bs_ui = mo.ui.dropdown(options=[32, 64, 96, 128], value=96, label="Phase 2 Batch Size")

    mo.hstack([pretrain_epochs_ui, finetune_epochs_ui, pretrain_bs_ui, finetune_bs_ui])
    return finetune_bs_ui, finetune_epochs_ui, pretrain_bs_ui, pretrain_epochs_ui


@app.cell
def _(MKG_DIR, glob, mo, os, pretrain_bs_ui, pretrain_epochs_ui, subprocess):
    print("=== PHASE 1: MKGformer Entity Modeling Pretraining (pretrain=1) ===")
    _cmd_phase1 = [
        "python",
        "main.py",
        "--gpus",
        "0,",
        f"--max_epochs={int(pretrain_epochs_ui.value)}",
        "--num_workers=4",
        "--model_name_or_path",
        "bert-base-uncased",
        "--accumulate_grad_batches",
        "1",
        "--model_class",
        "UnimoKGC",
        "--batch_size",
        str(int(pretrain_bs_ui.value)),
        "--pretrain",
        "1",
        "--bce",
        "0",
        "--check_val_every_n_epoch",
        "1",
        "--overwrite_cache",
        "--data_dir",
        "dataset/FB15k-237",
        "--task_name",
        "fb15k-237",
        "--eval_batch_size",
        "64",
        "--max_seq_length",
        "64",
        "--lr",
        "5e-4",
    ]
    phase1_res = subprocess.run(_cmd_phase1, cwd=MKG_DIR, text=True, capture_output=True)
    print(phase1_res.stdout)
    if phase1_res.returncode != 0:
        print("STDERR:", phase1_res.stderr)

    _ckpt_candidates = sorted(
        glob.glob(os.path.join(MKG_DIR, "output", "**", "*.ckpt"), recursive=True),
        key=os.path.getmtime,
    )
    PRETRAIN_CKPT = _ckpt_candidates[-1] if _ckpt_candidates else None
    print("Discovered Phase 1 Checkpoint:", PRETRAIN_CKPT)

    mo.md(f"### ✅ Hoàn tất Phase 1! Checkpoint phát hiện: `{PRETRAIN_CKPT}`")
    return PRETRAIN_CKPT, phase1_res


@app.cell
def _(MKG_DIR, PRETRAIN_CKPT, WORK_DIR, finetune_bs_ui, finetune_epochs_ui, mo, os, subprocess):
    print("=== PHASE 2: MKGformer Missing Entity Prediction / Link Prediction (pretrain=0) ===")
    _cmd_phase2 = [
        "python",
        "main.py",
        "--gpus",
        "0,",
        f"--max_epochs={int(finetune_epochs_ui.value)}",
        "--num_workers=4",
        "--model_name_or_path",
        "bert-base-uncased",
        "--accumulate_grad_batches",
        "1",
        "--model_class",
        "UnimoKGC",
        "--batch_size",
        str(int(finetune_bs_ui.value)),
        "--label_smoothing",
        "0.3",
        "--pretrain",
        "0",
        "--bce",
        "1",
        "--check_val_every_n_epoch",
        "1",
        "--overwrite_cache",
        "--data_dir",
        "dataset/FB15k-237",
        "--task_name",
        "fb15k-237",
        "--eval_batch_size",
        "128",
        "--max_seq_length",
        "64",
        "--lr",
        "4e-5",
    ]
    if PRETRAIN_CKPT:
        _cmd_phase2.extend(["--checkpoint", PRETRAIN_CKPT])

    # Rank theo từng triplet cho scripts/compute_metrics.py (do scripts/patch_mkgformer.py thêm vào)
    mkg_ranks_path = os.path.join(WORK_DIR, "ranks_mkgformer_mmkg_test.csv")
    phase2_res = subprocess.run(
        _cmd_phase2, cwd=MKG_DIR, text=True, capture_output=True,
        env={**os.environ, "MKG_RANK_FILE": mkg_ranks_path},
    )
    print(phase2_res.stdout)
    if phase2_res.returncode != 0:
        print("STDERR:", phase2_res.stderr)

    mo.md("### 🏆 Hoàn tất Phase 2 (Đánh giá Link Prediction cho MKGformer trên Unified-MMKG)!")
    return mkg_ranks_path, phase2_res


if __name__ == "__main__":
    app.run()
