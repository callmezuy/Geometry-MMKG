"""
E3 image-masking control (RESEARCH_PLAN section 4.10, decided 27/11/2026;
docs/superpowers/specs/2026-11-27-e3-official-runs-design.md 5.8 and 6.4): re-evaluate the official E3 checkpoints of
the knowledge graphs on test with the image mask set to 0 for EVERY entity, so every entity uses the learned missing
vector v_miss and no RGB / depth feature reaches the model. Same filter, same tie rule; checkpoint selection is not
touched (this is a pre-registered analysis of the checkpoints chosen on valid).

  python scripts/e3_mask_eval.py --cell-dir results/geometry-e3/FB15k-237-IMG/fixed --data data/FB15k-237-IMG

For every ckpt_<stem>.pt in --cell-dir: ranks_<stem>_test_maskall.csv and metrics_<stem>_maskall.json.
Only FB15k-237-IMG, FB15K-237-ind and MKG-Y, variants no_depth and fixed.
"""
import argparse
import glob
import json
import os
import sys

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "scripts"))

from core.data_engineering.fast_dataset import (  # noqa: E402
    TripleBatches, get_all_true_triplets, load_entity_features, load_triples, read_count)
from core.models.end_to_end import GeometryMMKGModel  # noqa: E402
from core.models.evaluation import evaluate_link_prediction, format_metrics, save_ranks  # noqa: E402

MMKG = ("FB15k-237-IMG", "FB15K-237-ind", "MKG-Y")
MASKED_VARIANTS = ("no_depth", "fixed")


def load_model(ckpt, n_rel, embed_dim, device):
    model = GeometryMMKGModel(n_rel, embed_dim=embed_dim, dropout=ckpt["args"]["dropout"], **ckpt["model_kwargs"],
                              **ckpt["options"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def masked_eval(model, feats, test, all_true, device, eval_batch_size=512):
    """Test ranks with every image masked (the data mask itself is not changed)."""
    return evaluate_link_prediction(
        model, feats.text, feats.rgb, feats.depth, TripleBatches(test, eval_batch_size), all_true, device,
        return_ranks=True, desc="test (all images masked)", all_mask=torch.zeros_like(feats.mask))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-dir", required=True, help="folder with the ckpt_<stem>.pt of one E3 cell")
    p.add_argument("--data", required=True, help="dataset folder, e.g. data/FB15k-237-IMG")
    p.add_argument("--eval-batch-size", type=int, default=512)
    a = p.parse_args(argv)
    dataset = os.path.basename(os.path.normpath(a.data))
    if dataset not in MMKG:
        raise SystemExit(f"{dataset}: the masking control is defined on the knowledge graphs only ({', '.join(MMKG)})")
    ckpts = sorted(glob.glob(os.path.join(a.cell_dir, "ckpt_*.pt")))
    if not ckpts:
        raise SystemExit(f"no ckpt_*.pt in {a.cell_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = load_entity_features(a.data, device)
    n_rel = read_count(os.path.join(a.data, "relation2id.txt"))
    test, all_true = load_triples(a.data, "test"), get_all_true_triplets(a.data)
    from train import code_commit
    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["variant"] not in MASKED_VARIANTS:
            raise SystemExit(f"{os.path.basename(path)}: variant {ckpt['variant']} is not one of {MASKED_VARIANTS}")
        stem = os.path.basename(path)[len("ckpt_"):-len(".pt")]
        model = load_model(ckpt, n_rel, feats.text.shape[1], device)
        m = masked_eval(model, feats, test, all_true, device, a.eval_batch_size)
        print(f"{stem}: {format_metrics(m)}", flush=True)
        save_ranks(m["ranks"], os.path.join(a.cell_dir, f"ranks_{stem}_test_maskall.csv"))
        out = {"masked": "all entities", "dataset": dataset, "variant": ckpt["variant"], "ckpt": os.path.basename(path),
               "best_epoch": ckpt["epoch"], "test": {k: v for k, v in m.items() if k != "ranks"},
               "code_commit": code_commit()}
        with open(os.path.join(a.cell_dir, f"metrics_{stem}_maskall.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
