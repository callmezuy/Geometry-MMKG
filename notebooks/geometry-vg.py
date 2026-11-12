import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import subprocess
    import marimo as mo
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from huggingface_hub import HfFileSystem, snapshot_download

    mo.md(
        """
        # 🚀 Train GeometryMMKGModel on Spatial-VG (1.44M Entities - Molab + HF Remote Storage)
        Notebook này kết nối trực tiếp tới **Hugging Face Remote Storage** (`callmezuy/spatial-vg`) qua `HfFileSystem`, sử dụng `mmap=True` để nạp 3 ma trận đặc trưng 4.4 GB mà không tràn RAM, và huấn luyện **GeometryMMKG** trên tập `Spatial-VG` **v2** (VG150-curated `no_part_whole`; 1.441.984 thực thể, 50 quan hệ; train 319.069 / valid 26.372 / test 137.984 theo split chính thức theo ảnh, inductive).
        """
    )
    return (
        DataLoader,
        HfFileSystem,
        Subset,
        mo,
        optim,
        os,
        snapshot_download,
        subprocess,
        sys,
        torch,
    )


@app.cell
def _(HfFileSystem, mo, os):
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
def _(HF_REPO_ID, os, snapshot_download, subprocess, sys):
    WORK_DIR = os.path.abspath(".")
    if os.path.exists(os.path.join(WORK_DIR, "core")):
        REPO_ROOT = WORK_DIR
    elif os.path.exists(os.path.join(os.path.dirname(WORK_DIR), "core")):
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

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    DATA_DIR = os.path.join(WORK_DIR, "data", "Spatial-VG")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Syncing dataset from {HF_REPO_ID} to {DATA_DIR}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print("Dataset ready at:", DATA_DIR)
    return DATA_DIR, REPO_ROOT, WORK_DIR


@app.cell
def _(mo):
    epochs_ui = mo.ui.number(start=5, stop=100, step=5, value=30, label="Số Epochs")
    batch_size_ui = mo.ui.dropdown(options=[512, 1024, 2048, 4096], value=2048, label="Batch Size")
    lr_ui = mo.ui.number(start=1e-5, stop=1e-2, step=1e-4, value=2e-4, label="Learning Rate")
    margin_ui = mo.ui.number(start=1.0, stop=20.0, step=0.5, value=5.0, label="Margin (TransE)")
    lambda_align_ui = mo.ui.number(start=0.0, stop=1.0, step=0.05, value=0.1, label="Lambda Align")

    mo.hstack([epochs_ui, batch_size_ui, lr_ui, margin_ui, lambda_align_ui])
    return batch_size_ui, epochs_ui, lambda_align_ui, lr_ui, margin_ui


@app.cell
def _(
    DATA_DIR,
    DataLoader,
    Subset,
    WORK_DIR,
    batch_size_ui,
    epochs_ui,
    lambda_align_ui,
    lr_ui,
    margin_ui,
    mo,
    optim,
    os,
    torch,
):
    from core.data_engineering.fast_dataset import FastMMSGDataset, get_all_true_triplets
    from core.models.end_to_end import GeometryMMKGModel
    from core.models.kg_reasoning import JointLoss
    from core.models.evaluation import evaluate_link_prediction, format_metrics, save_ranks
    from core.reproducibility import set_seed, seed_worker

    SEED = 42  # same as NativE (-seed 42); change for the 3-seed runs (P4.5)
    _gen = set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Memory-mapping pre-extracted features (1.44M entities) via mmap=True...")
    all_text = torch.load(
        os.path.join(DATA_DIR, "ent_text_emb.pt"), map_location="cpu", mmap=True, weights_only=False
    )
    all_rgb = torch.load(
        os.path.join(DATA_DIR, "ent_rgb_emb.pt"), map_location="cpu", mmap=True, weights_only=False
    )
    all_depth = torch.load(
        os.path.join(DATA_DIR, "ent_depth_emb.pt"), map_location="cpu", mmap=True, weights_only=False
    )

    print("Initializing Fast Datasets (train / valid / test)...")
    train_dataset = FastMMSGDataset(DATA_DIR, split="train")
    valid_dataset = FastMMSGDataset(DATA_DIR, split="valid")
    test_dataset = FastMMSGDataset(DATA_DIR, split="test")

    _bs = int(batch_size_ui.value)
    train_loader = DataLoader(
        train_dataset, batch_size=_bs, shuffle=True, num_workers=2,
        generator=_gen, worker_init_fn=seed_worker,
    )
    # Dùng 5.000 mẫu Validation (lấy ngẫu nhiên, seed cố định) trong các vòng lặp trung gian để kiểm tra nhanh.
    # Lấy ngẫu nhiên vì valid v2 được sắp theo ảnh: range(5000) sẽ chỉ lấy vài trăm ảnh đầu.
    import random as _random
    _val_idx = sorted(_random.Random(42).sample(range(len(valid_dataset)), min(5000, len(valid_dataset))))
    fast_val_subset = Subset(valid_dataset, _val_idx)
    valid_loader = DataLoader(fast_val_subset, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    all_true_triplets = get_all_true_triplets(DATA_DIR)

    num_relations = len(train_dataset.relation2id)
    model = GeometryMMKGModel(num_relations=num_relations, embed_dim=768, use_encoders=False).to(device)

    criterion = JointLoss(margin=float(margin_ui.value), lambda_align=float(lambda_align_ui.value))
    optimizer = optim.AdamW(model.parameters(), lr=float(lr_ui.value), weight_decay=1e-5)

    EPOCHS = int(epochs_ui.value)
    best_val_mrr = 0.0
    best_ckpt_path = os.path.join(WORK_DIR, "best_vg_model.pth")

    for _epoch in range(EPOCHS):
        model.train()
        _total_loss = 0.0

        for _batch in train_loader:
            optimizer.zero_grad()
            _h_ids = _batch["h_id"]
            _r_ids = _batch["r_id"].to(device)
            _t_ids = _batch["t_id"]

            _h_t = all_text[_h_ids].to(device)
            _h_r = all_rgb[_h_ids].to(device)
            _h_d = all_depth[_h_ids].to(device)

            _t_t = all_text[_t_ids].to(device)
            _t_r = all_rgb[_t_ids].to(device)
            _t_d = all_depth[_t_ids].to(device)

            _h_entity = model.forward_fusion(_h_t, _h_r, _h_d)
            _t_entity = model.forward_fusion(_t_t, _t_r, _t_d)

            _r_emb = model.relation_embeddings(_r_ids)
            _hr_entity = _h_entity + _r_emb

            _dist = torch.cdist(_hr_entity, _t_entity, p=1)
            _scores = -_dist

            _pos_score = _scores.diag()
            _mask = torch.eye(_scores.size(0), device=device).bool()
            _scores.masked_fill_(_mask, -1e9)
            _neg_score = _scores.max(dim=1)[0]

            _loss, _, _ = criterion(_pos_score, _neg_score, _h_t, _h_r, _h_d)
            _loss.backward()
            optimizer.step()
            _total_loss += _loss.item()

        print(f"Epoch {_epoch + 1:03d}/{EPOCHS} | Train Loss: {_total_loss / len(train_loader):.4f}")

        if (_epoch + 1) % 5 == 0:
            print(f"--- Evaluating on VALIDATION subset at Epoch {_epoch + 1} ---")
            _val = evaluate_link_prediction(
                model, all_text, all_rgb, all_depth, valid_loader, all_true_triplets, device, desc="Valid"
            )
            print(f"Val Metrics -> {format_metrics(_val)}")
            if _val["mrr"] > best_val_mrr:
                best_val_mrr = _val["mrr"]
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"Saved new best model to {best_ckpt_path}!")

    print("\n=== Evaluating Best Checkpoint on TEST Set ===")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=False))

    test_metrics = evaluate_link_prediction(
        model, all_text, all_rgb, all_depth, test_loader, all_true_triplets, device, return_ranks=True, desc="Test"
    )
    print(f"FINAL TEST METRICS -> {format_metrics(test_metrics)}")
    ranks_path = os.path.join(WORK_DIR, f"ranks_geometry_vg_seed{SEED}_test.csv")
    save_ranks(test_metrics["ranks"], ranks_path)  # input for scripts/compute_metrics.py

    mo.md(
        f"""
        ## 🏆 Kết Quả Đánh Giá Cuối Cùng Trên Tập Test (`Spatial-VG`)
        | Metric | Giá trị |
        | :--- | :--- |
        | **MRR** (head + tail) | `{test_metrics['mrr']:.4f}` |
        | **MR** | `{test_metrics['mr']:.1f}` |
        | **Hits@1** | `{test_metrics['hits@1']:.4f}` |
        | **Hits@3** | `{test_metrics['hits@3']:.4f}` |
        | **Hits@10** | `{test_metrics['hits@10']:.4f}` |
        | MRR tail / head | `{test_metrics['tail']['mrr']:.4f}` / `{test_metrics['head']['mrr']:.4f}` |

        Ranks theo từng triplet: `{ranks_path}`
        """
    )
    return (
        EPOCHS,
        all_depth,
        all_rgb,
        all_text,
        all_true_triplets,
        best_ckpt_path,
        best_val_mrr,
        criterion,
        device,
        evaluate_link_prediction,
        fast_val_subset,
        model,
        num_relations,
        optimizer,
        test_dataset,
        test_loader,
        test_metrics,
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
    )


if __name__ == "__main__":
    app.run()
