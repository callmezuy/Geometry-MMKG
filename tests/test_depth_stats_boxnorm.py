"""
E2b (docs/superpowers/specs/2026-11-26-e2b-shared-scale-design.md): grid9 with the depth min-max normalised per box.
Run from Code/:  python tests/test_depth_stats_boxnorm.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))
sys.path.insert(0, os.path.join(CODE, "tests"))
import depth_stats as ds  # noqa: E402
import depth_stats_boxnorm as bn  # noqa: E402
import train  # noqa: E402

R = 64
ROIS = torch.tensor([[0, 4.0, 4.0, 28.0, 30.0],      # box A of image 0
                     [0, 30.0, 10.0, 60.0, 50.0],    # box B of image 0
                     [1, 0.0, 0.0, 64.0, 64.0]])     # whole image 1


def _depth(seed=0):
    return torch.rand(2, 1, R, R, generator=torch.Generator().manual_seed(seed))


def test_per_box_scale_is_removed():
    d = _depth()
    base, const = bn.box_minmax_grid(d, ROIS)
    assert base.shape == (3, 9) and not const.any()
    assert float(base.min()) >= 0.0 and float(base.max()) <= 1.0
    # change the depth only around box A by a * d + b (a > 0): box A keeps its 9 numbers
    d2 = d.clone()
    d2[0, :, 0:34, 0:34] = 3.0 * d2[0, :, 0:34, 0:34] + 0.7
    moved, _ = bn.box_minmax_grid(d2, ROIS[:1])
    torch.testing.assert_close(moved, base[:1], atol=1e-5, rtol=0)
    # the per-image numbers of box A do change (they keep the shared scale)
    assert not torch.allclose(ds.grid_means(d2, ROIS[:1]), ds.grid_means(d, ROIS[:1]), atol=1e-3)


def test_constant_box_gives_zeros():
    d = _depth()
    d[0, :, 0:34, 0:34] = 0.25
    x, const = bn.box_minmax_grid(d, ROIS)
    assert const.tolist() == [True, False, False]
    assert torch.equal(x[0], torch.zeros(9))


def test_check_image_table():
    ref = np.random.default_rng(0).random((5, 9)).astype(np.float32)
    mask = np.array([True, True, True, False, True])
    off = ref.copy()
    off[3] += 1.0                                   # entity without an image: not compared
    assert bn.check_image_table(off + 1e-4, ref, mask) <= 1.1e-4
    try:
        bn.check_image_table(ref + 2e-3 * mask[:, None], ref, mask)
        raise AssertionError("check must fail")
    except ValueError:
        pass


def _dataset(root, name, n=12, missing=(3, 7)):
    from test_feature_transe import _tiny_dataset
    d = os.path.join(root, name)
    os.makedirs(d)
    _tiny_dataset(d, n=n, missing=missing, scene_graph=True)
    for split in ("train", "valid", "test"):        # tab-separated like data v0 (extract_features.read_count_file)
        p = os.path.join(d, f"{split}2id.txt")
        lines = open(p).read().splitlines()
        with open(p, "w") as f:
            f.write("\n".join([lines[0]] + [ln.replace(" ", "\t") for ln in lines[1:]]) + "\n")
    return d


def test_finish_fill_and_train():
    n = 12
    with tempfile.TemporaryDirectory() as root:
        vrd = _dataset(root, "Spatial-VRD", n)
        trans = _dataset(root, "Spatial-VRD-Trans", n)
        mask = torch.load(os.path.join(vrd, "ent_img_mask.pt"), weights_only=True).numpy()
        rng = np.random.default_rng(1)
        raw_img, raw_box = rng.random((n, 9)).astype(np.float32), rng.random((n, 9)).astype(np.float32)
        raw_img[~mask] = np.nan
        raw_box[~mask] = np.nan
        for d in (vrd, trans):                       # the data table = the per-image numbers (+ any fill)
            torch.save(torch.from_numpy(np.nan_to_num(raw_img, nan=0.5)), os.path.join(d, ds.TABLE))
        info = {"data": "Spatial-VRD", "code_commit": None, "n_images": 1, "n_boxes": 10, "n_constant_boxes": 0,
                "n_outside": 0, "source": {"zip": "vrd.zip", "data": "Spatial-VRD"}}
        out_vrd, out_trans = os.path.join(root, "e2b", "Spatial-VRD"), os.path.join(root, "e2b", "Spatial-VRD-Trans")
        got = bn.finish(raw_img.copy(), raw_box.copy(), vrd, out_vrd, info)
        assert got["check_max_abs_diff"] == 0.0 and got["filled_entities"] == 2 and len(got["sha256"]) == 64
        blob = torch.load(os.path.join(out_vrd, bn.TABLE), weights_only=False)
        assert torch.equal(blob["grid9"][mask], torch.from_numpy(raw_box[mask])) and blob["norm"] == "box"
        # VRD-Trans: refilled from the VRD build
        bn.main(["fill", "--src", out_vrd, "--src-data", vrd, "--data", trans, "--out", out_trans])
        t = torch.load(os.path.join(out_trans, bn.TABLE), weights_only=False)
        assert torch.equal(t["grid9"][mask], blob["grid9"][mask])
        assert json.load(open(os.path.join(out_trans, bn.INFO)))["source"]["refilled_from"] == "Spatial-VRD"
        # a per-image table that does not match the data: nothing is written
        torch.save(torch.from_numpy(np.nan_to_num(raw_img, nan=0.5) + 0.01), os.path.join(trans, ds.TABLE))
        bad = os.path.join(root, "e2b", "bad")
        try:
            bn.main(["fill", "--src", out_vrd, "--src-data", vrd, "--data", trans, "--out", bad])
            raise AssertionError("check must fail")
        except SystemExit as e:
            assert e.code == 1 and not os.path.exists(bad)
        # train.py reads the table like any depth statistics
        out = os.path.join(root, "run")
        train.main(["--data", vrd, "--variant", "depth_stats_only", "--tune", "--lr", "0.001", "--dropout", "0",
                    "--epochs", "1", "--valid-every", "1", "--batch-size", "8", "--depth-stats",
                    f"{os.path.join(out_vrd, bn.TABLE)}:grid9", "--out", out, "--no-resume"])
        (fn,) = [f for f in os.listdir(out) if f.startswith("metrics_")]
        m = json.load(open(os.path.join(out, fn), encoding="utf-8"))
        assert m["options_tag"] == "inorm-ddrop0.3-tproj-dstat_boxnorm_grid9", m["options_tag"]
        assert m["options"]["depth_stats_dim"] == 9 and m["depth_stats"]["model"] == "zipdepth"
        assert "in_image" in m["best_valid"] and "test" not in m


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("OK")
