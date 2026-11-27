"""
E3 image-masking control (scripts/e3_mask_eval.py; docs/superpowers/specs/2026-11-27-e3-official-runs-design.md 5.8):
an official checkpoint re-evaluated on test with every entity's image masked.
Run from Code/:  python tests/test_e3_mask_eval.py
"""
import json
import os
import sys
import tempfile

import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "scripts"))
sys.path.insert(0, os.path.join(CODE, "tests"))
sys.path.insert(0, CODE)
import e3_mask_eval as mev  # noqa: E402
import train  # noqa: E402
from test_feature_transe import _tiny_dataset  # noqa: E402

STEM = "geometry_FB15k-237-IMG_fixed_inorm-ddrop0.3-tproj_seed42"


def _official_run(root, name="FB15k-237-IMG"):
    data = os.path.join(root, name)
    os.makedirs(data)
    _tiny_dataset(data)
    cell = os.path.join(root, "geometry-e3", name, "fixed")
    train.main(["--data", data, "--variant", "fixed", "--epochs", "2", "--valid-every", "1", "--batch-size", "8",
                "--out", cell, "--no-resume"])
    return data, cell


def test_cli_writes_masked_ranks():
    with tempfile.TemporaryDirectory() as root:
        data, cell = _official_run(root)
        mev.main(["--cell-dir", cell, "--data", data])
        ranks = os.path.join(cell, f"ranks_{STEM}_test_maskall.csv")
        lines = open(ranks).read().splitlines()
        assert lines[0] == "h_id,r_id,t_id,rank_head,rank_tail" and len(lines) == 1 + 6   # 6 test triples
        m = json.load(open(os.path.join(cell, f"metrics_{STEM}_maskall.json"), encoding="utf-8"))
        assert m["masked"] == "all entities" and m["ckpt"] == f"ckpt_{STEM}.pt" and "mrr" in m["test"]
        assert "ranks" not in m["test"]
        # the unmasked official ranks are untouched
        assert os.path.exists(os.path.join(cell, f"ranks_{STEM}_test.csv"))


def test_masked_eval_ignores_image_features():
    from core.data_engineering.fast_dataset import get_all_true_triplets, load_entity_features, load_triples, read_count
    with tempfile.TemporaryDirectory() as root:
        data, cell = _official_run(root)
        device = torch.device("cpu")
        ckpt = torch.load(os.path.join(cell, f"ckpt_{STEM}.pt"), map_location="cpu", weights_only=False)
        feats = load_entity_features(data, device, keep_on_gpu=False)
        model = mev.load_model(ckpt, read_count(os.path.join(data, "relation2id.txt")), feats.text.shape[1], device)
        test, all_true = load_triples(data, "test"), get_all_true_triplets(data)
        a = mev.masked_eval(model, feats, test, all_true, device)
        g = torch.Generator().manual_seed(5)
        feats.rgb = torch.randn(feats.rgb.shape, generator=g)
        feats.depth = torch.randn(feats.depth.shape, generator=g)
        b = mev.masked_eval(model, feats, test, all_true, device)
        for k in ("rank_head", "rank_tail"):
            assert list(a["ranks"][k]) == list(b["ranks"][k]), k
        assert feats.mask.sum() > 0      # the data mask itself is left as it is


def test_refuses_scene_graphs_and_other_variants():
    with tempfile.TemporaryDirectory() as root:
        data = os.path.join(root, "Spatial-VRD")
        os.makedirs(data)
        cell = os.path.join(root, "cell")
        os.makedirs(cell)
        try:
            mev.main(["--cell-dir", cell, "--data", data])
            raise AssertionError("a scene graph must be refused")
        except SystemExit as e:
            assert "knowledge graph" in str(e.code)
        data, cell = _official_run(os.path.join(root, "kg"))
        ck = torch.load(os.path.join(cell, f"ckpt_{STEM}.pt"), weights_only=False)
        ck["variant"] = "concat"
        torch.save(ck, os.path.join(cell, f"ckpt_{STEM}.pt"))
        try:
            mev.main(["--cell-dir", cell, "--data", data])
            raise AssertionError("a variant outside no_depth / fixed must be refused")
        except SystemExit as e:
            assert "concat" in str(e.code)


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all E3 masking tests passed")
