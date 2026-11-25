"""
Feature-TransE (RESEARCH_PLAN.md P3.1) through scripts/train.py --model feature_transe.
Run from Code/:  python tests/test_feature_transe.py      (pytest also works)
"""
import json
import os
import sys
import tempfile

import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "scripts"))
sys.path.insert(0, CODE)
import train  # noqa: E402
from core.models.feature_transe import FeatureTransE  # noqa: E402


def test_parameter_counts_match_the_collaborator_runs():
    # FB15k-237-IMG (237 relations, 768-d features): the counts printed in the 19/11/2026 train logs.
    assert sum(p.numel() for p in FeatureTransE(237, use_depth=False).parameters()) == 1_953_024
    assert sum(p.numel() for p in FeatureTransE(237, use_depth=True).parameters()) == 2_542_848


def test_forward_fusion_ignores_geo_and_mask_and_is_unit_norm():
    torch.manual_seed(0)
    m = FeatureTransE(3, feature_dim=8, embed_dim=6, hidden_dim=5, use_depth=True).eval()
    t, x, z = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    a = m.forward_fusion(t, x, z)
    b = m.forward_fusion(t, x, z, torch.zeros(4, dtype=torch.bool), geo=torch.randn(4, 9))
    assert torch.equal(a, b)
    assert torch.allclose(a.norm(dim=1), torch.ones(4), atol=1e-6)
    nd = FeatureTransE(3, feature_dim=8, embed_dim=6, hidden_dim=5, use_depth=False).eval()
    assert torch.equal(nd.forward_fusion(t, x, z), nd.forward_fusion(t, x, torch.randn(4, 8)))  # depth unused


def test_cli_variants_and_options():
    a = train.parse_args(["--data", "x", "--model", "feature_transe"])
    assert a.variant == "no_depth"
    assert (a.rel_gate, a.input_norm, a.depth_dropout, a.text_proj, a.depth_stats) == (False, False, 0.0, False,
                                                                                       "none")
    a.depth_stats = None
    assert train.options_tag(a) == "base"
    assert train.parse_args(["--data", "x"]).variant == "full"            # GeometryMMKG default unchanged
    for argv in (["--model", "feature_transe", "--variant", "full"], ["--variant", "depth"]):
        try:
            train.parse_args(["--data", "x"] + argv)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{argv} must be refused")


def _tiny_dataset(d, n=12, r=2, dim=8, missing=(3, 7), bad_fill=False):
    g = torch.Generator().manual_seed(0)
    with open(os.path.join(d, "entity2id.txt"), "w") as f:
        f.write(f"{n}\n" + "".join(f"e{i}\t{i}\n" for i in range(n)))
    with open(os.path.join(d, "relation2id.txt"), "w") as f:
        f.write(f"{r}\n" + "".join(f"r{i}\t{i}\n" for i in range(r)))
    trip = [(h, (h + 1 + k) % n, k % r) for h in range(n) for k in range(3)]   # (h, t, r), OpenKE order
    splits = {"train": trip[:24], "valid": trip[24:30], "test": trip[30:]}
    for s, rows in splits.items():
        with open(os.path.join(d, f"{s}2id.txt"), "w") as f:
            f.write(f"{len(rows)}\n" + "".join(f"{h} {t} {rr}\n" for h, t, rr in rows))
    mask = torch.ones(n, dtype=torch.bool)
    mask[list(missing)] = False
    fill = {"rgb": torch.randn(dim, generator=g), "depth": torch.randn(dim, generator=g)}
    for k in ("text", "rgb", "depth"):
        x = torch.randn(n, dim, generator=g)
        if k in fill:
            x[~mask] = fill[k]
            if bad_fill and k == "rgb":
                x[missing[0]] += 1.0
        torch.save(x, os.path.join(d, f"ent_{k}_emb.pt"))
    torch.save(mask, os.path.join(d, "ent_img_mask.pt"))
    torch.save(fill, os.path.join(d, "ent_fill.pt"))


def test_end_to_end_run_writes_metrics_and_ranks():
    with tempfile.TemporaryDirectory() as root:
        data = os.path.join(root, "Tiny")
        os.makedirs(data)
        _tiny_dataset(data)
        out = os.path.join(root, "out")
        train.main(["--data", data, "--model", "feature_transe", "--variant", "depth", "--epochs", "2",
                    "--valid-every", "1", "--batch-size", "8", "--out", out, "--no-ckpt"])
        stem = "feature-transe_Tiny_depth_seed42"
        m = json.load(open(os.path.join(out, f"metrics_{stem}.json")))
        assert m["model"] == "Feature-TransE" and m["options"] == {} and m["options_tag"] == "base"
        assert m["model_kwargs"]["use_depth"] is True and "test" in m and m["best_gate"] is None
        assert os.path.exists(os.path.join(out, f"ranks_{stem}_test.csv"))


def test_rows_without_image_must_be_the_constant_fill():
    with tempfile.TemporaryDirectory() as root:
        data = os.path.join(root, "Tiny")
        os.makedirs(data)
        _tiny_dataset(data, bad_fill=True)
        try:
            train.main(["--data", data, "--model", "feature_transe", "--epochs", "1", "--out",
                        os.path.join(root, "out"), "--dev"])
        except ValueError as e:
            assert "ent_fill.pt" in str(e)
        else:
            raise AssertionError("a missing-image row that is not the constant must be refused")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
