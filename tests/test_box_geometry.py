"""
E2 box control (docs/superpowers/specs/2026-11-26-e2-box-control-design.md): per-entity box-geometry tables.
Run from Code/:  python tests/test_box_geometry.py
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))
sys.path.insert(0, os.path.join(CODE, "tests"))
import box_geometry as bg  # noqa: E402
import train  # noqa: E402

NAMES = ["10_0", "10_1", "10_2", "10_3", "20_0", "30_0"]
TRAIN_ANN = [
    {"photo_id": "10", "width": 100, "height": 50, "objects": [
        {"bbox": {"x": 10, "y": 5, "w": 20, "h": 10}},     # plain box
        {"bbox": {"x": 90, "y": 0, "w": 20, "h": 50}},     # crosses the right edge -> clamped to x2 = 100
        {"bbox": {"x": 50, "y": 10, "w": 0, "h": 20}},     # zero width -> widened to MIN_BOX = 2 px around x = 50
    ]},                                                    # 10_3: in entity2id, no object -> NaN
    {"photo_id": "20", "width": 40, "height": 40, "objects": [
        {"bbox": {"x": 50, "y": 0, "w": 10, "h": 10}},     # fully outside -> NaN
        {"bbox": {"x": 0, "y": 0, "w": 10, "h": 10}},      # 20_1 not in entity2id -> skipped
    ]},
]
TEST_ANN = [{"photo_id": "30", "width": 10, "height": 10, "objects": [{"bbox": {"x": 0, "y": 0, "w": 10, "h": 10}}]}]
# x1, y1, x2, y2, w, h, cx, cy, log(w h), all divided by the image size
EXPECTED_ROWS = [
    [0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, math.log(0.04)],
    [0.9, 0.0, 1.0, 1.0, 0.1, 1.0, 0.95, 0.5, math.log(0.1)],
    [0.49, 0.2, 0.51, 0.6, 0.02, 0.4, 0.5, 0.4, math.log(0.008)],
    [math.nan] * 9,
    [math.nan] * 9,
    [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0],
]


def _write(root, names=NAMES, train_ann=TRAIN_ANN, test_ann=TEST_ANN, dataset="Spatial-VRD"):
    data, ann = os.path.join(root, dataset), os.path.join(root, "ann")
    os.makedirs(data, exist_ok=True)
    os.makedirs(ann, exist_ok=True)
    with open(os.path.join(data, "entity2id.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(names)}\n" + "".join(f"{n}\t{i}\n" for i, n in enumerate(names)))
    grid9 = torch.arange(len(names) * 9, dtype=torch.float32).reshape(len(names), 9) / 100
    torch.save(grid9, os.path.join(data, "ent_depth_stats.pt"))
    for fn, a in (("sg_train_annotations.json", train_ann), ("sg_test_annotations.json", test_ann)):
        with open(os.path.join(ann, fn), "w", encoding="utf-8") as f:
            json.dump(a, f)
    return data, ann, grid9


def test_box_table_by_hand():
    with tempfile.TemporaryDirectory() as root:
        data, ann, _ = _write(root)
        x = bg.box_table({n: i for i, n in enumerate(NAMES)}, [os.path.join(ann, fn) for fn in bg.ANNOTATIONS])
    assert x.shape == (6, 9) and x.dtype == np.float64
    np.testing.assert_allclose(x, np.array(EXPECTED_ROWS), rtol=0, atol=1e-12)


def test_image_size_fix():
    # the annotated size of photo 9210906893 is wrong (500 x 374); the image file is 683 x 1024
    ann = [{"photo_id": "9210906893", "width": 500, "height": 374,
            "objects": [{"bbox": {"x": 109, "y": 716, "w": 94, "h": 157}}]}]
    with tempfile.TemporaryDirectory() as root:
        data, ann_dir, _ = _write(root, names=["9210906893_0"], train_ann=ann, test_ann=[])
        x = bg.build(data, ann_dir)["box"][0].double().numpy()
    np.testing.assert_allclose(x[:4], [109 / 683, 716 / 1024, 203 / 683, 873 / 1024], atol=1e-7)


def test_build_and_grid9_concat():
    with tempfile.TemporaryDirectory() as root:
        data, ann, grid9 = _write(root)
        out = bg.build(data, ann)
    assert out["box"].dtype == torch.float32 and out["box"].shape == (6, 9)
    assert out["box_grid9"].shape == (6, 18) and out["model"] == "box"
    assert torch.equal(out["box_grid9"][:, 9:], grid9)                      # the dataset's own grid9, unchanged
    same = torch.isnan(out["box"]) | (out["box_grid9"][:, :9] == out["box"])
    assert bool(same.all())
    s = out["summary"]
    assert s["n_entities"] == 6 and s["n_with_box"] == 4 and s["n_nan"] == 2
    assert abs(s["col_sums_box"][8] - (math.log(0.04) + math.log(0.1) + math.log(0.008))) < 1e-6


def test_check():
    with tempfile.TemporaryDirectory() as root:
        data, ann, _ = _write(root)
        s = bg.build(data, ann)["summary"]
    expected = {"Spatial-VRD": json.loads(json.dumps(s))}
    bg.check(s, "Spatial-VRD", expected)
    bad = json.loads(json.dumps(s))
    bad["col_sums_box"][0] += 1e-3
    for wrong, ds in ((bad, "Spatial-VRD"), (s, "Spatial-VG")):
        try:
            bg.check(wrong, ds, expected)          # changed sum; dataset without expected numbers
            raise AssertionError("check must fail")
        except ValueError:
            pass


def test_other_entity_set_is_matched_by_name():
    # a dataset with fewer entities in another order (VRD-Trans shares names with VRD): rows follow its entity2id
    names = ["30_0", "10_1"]
    with tempfile.TemporaryDirectory() as root:
        data, ann, _ = _write(root, names=names, dataset="Spatial-VRD-Trans")
        out = bg.build(data, ann)
    np.testing.assert_allclose(out["box"].double().numpy(), np.array([EXPECTED_ROWS[5], EXPECTED_ROWS[1]]),
                               atol=1e-7)


def test_cli_writes_file_and_summary():
    with tempfile.TemporaryDirectory() as root:
        data, ann, _ = _write(root)
        path = os.path.join(root, "e2", "box_geometry.pt")
        bg.main(["--data", data, "--annotations", ann, "--out", path])
        blob = torch.load(path, weights_only=False)
        assert set(blob) >= {"box", "box_grid9", "model", "source"}
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            assert json.load(f)["n_with_box"] == 4


def test_train_reads_the_box_table():
    from test_feature_transe import _tiny_dataset
    names = [f"1_{i}" for i in range(12)]
    ann = [{"photo_id": "1", "width": 120, "height": 60,
            "objects": [{"bbox": {"x": 5 * i, "y": 2 * i, "w": 10 + i, "h": 8}} for i in range(12)]}]
    with tempfile.TemporaryDirectory() as root:
        data, ann_dir, _ = _write(root, names=names, train_ann=ann, test_ann=[])
        _tiny_dataset(data, scene_graph=True)           # same entity names, adds triples and features
        torch.save(torch.rand(12, 9, generator=torch.Generator().manual_seed(0)),
                   os.path.join(data, "ent_depth_stats.pt"))
        path = os.path.join(root, "box_geometry.pt")
        bg.main(["--data", data, "--annotations", ann_dir, "--out", path])
        for key, dim in (("box", 9), ("box_grid9", 18)):
            out = os.path.join(root, "out", key)
            train.main(["--data", data, "--variant", "depth_stats_only", "--tune", "--lr", "0.001", "--dropout", "0",
                        "--epochs", "1", "--valid-every", "1", "--batch-size", "8", "--depth-stats",
                        f"{path}:{key}", "--out", out, "--no-resume"])
            (fn,) = [f for f in os.listdir(out) if f.startswith("metrics_")]
            m = json.load(open(os.path.join(out, fn), encoding="utf-8"))
            assert m["options"]["depth_stats_dim"] == dim and m["depth_stats"]["key"] == key, m["options"]
            assert m["options"]["depth_dropout"] == 0.3                 # dropped like grid9 (same mechanism)
            assert m["options_tag"].endswith(f"dstat_box_geometry_{key}"), m["options_tag"]
            assert "in_image" in m["best_valid"] and "test" not in m


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("OK")
