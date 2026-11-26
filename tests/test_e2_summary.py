"""
E2 summary (scripts/e2_summary.py): the four cells of one dataset and the difference rows.
Run from Code/:  python tests/test_e2_summary.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e2_summary as e2  # noqa: E402
from e1_summary import E1_CONFIGS, config_dir  # noqa: E402


def _metrics(path, variant, lr, dropout, mrr, in_mrr, rel_mrr, epochs=100, best_epoch=20, tag="t"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = {"tune": True, "variant": variant, "config": {"lr": lr, "dropout": dropout}, "seed": 42, "options_tag": tag,
         "epoch_budget": {"epochs": epochs}, "best_epoch": best_epoch, "code_commit": "x",
         "best_valid": {"mrr": mrr, "hits@1": 0.1, "hits@10": 0.5, "mr": 10.0,
                        "in_image": {"mrr": in_mrr, "hits@1": in_mrr / 2, "image_first": 0.3},
                        "relation": {"mrr": rel_mrr, "hits@1": rel_mrr / 2}}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f)


def _tree(root, with_box_grid9=True):
    ds = "Spatial-VRD"
    e1, e2r = os.path.join(root, "e1"), os.path.join(root, "e2")
    for i, (v, lr, do, ep) in enumerate(E1_CONFIGS[ds]):
        cfg = config_dir(lr, do, ep)
        _metrics(os.path.join(e1, v, cfg, "metrics_a.json"), v, lr, do, 0.25 + i / 100, 0.45, 0.50)
    # box: two grid points, the better one (lr 3e-4) is chosen
    for lr, mrr in ((1e-4, 0.260), (3e-4, 0.270)):
        _metrics(os.path.join(e2r, "box", "depth_stats_only", f"lr{lr:g}-do0", "metrics_b.json"),
                 "depth_stats_only", lr, 0.0, mrr, 0.47, 0.52)
    if with_box_grid9:
        _metrics(os.path.join(e2r, "box_grid9", "depth_stats_only", "lr0.0003-do0", "metrics_c.json"),
                 "depth_stats_only", 3e-4, 0.0, 0.2715, 0.48, 0.51)
    return ds, e1, e2r


def test_summary_table():
    with tempfile.TemporaryDirectory() as root:
        ds, e1, e2r = _tree(root)
        t = e2.summarize(ds, e1, e2r)
    rows = {line.split("|")[1].strip(): line for line in t.splitlines() if line.startswith("| ")}
    for name in ("`no_depth`", "`depth_stats_only`", "`box`", "`box_grid9`", "`box_grid9` − `box`",
                 "`box` − `depth_stats_only`"):
        assert name in rows, (name, t)
    assert "lr0.0003-do0" in rows["`box`"] and "0.2700" in rows["`box`"]
    assert "+0.0015" in rows["`box_grid9` − `box`"]                # 0.2715 − 0.2700
    # depth_stats_only of E1 has valid MRR 0.25 + 2 / 100 = 0.27 -> box − depth_stats_only = 0.0000
    assert "+0.0000" in rows["`box` − `depth_stats_only`"]
    assert "edge / convergence points left" in rows["`box`"]        # lr 3e-4 is at the edge of {1e-4, 3e-4}


def test_missing_arm():
    with tempfile.TemporaryDirectory() as root:
        ds, e1, e2r = _tree(root, with_box_grid9=False)
        t = e2.summarize(ds, e1, e2r)
    line = [x for x in t.splitlines() if x.startswith("| `box_grid9` |")][0]
    assert "missing" in line
    assert all(x.count("|") == t.splitlines()[2].count("|") for x in t.splitlines() if x.startswith("|"))


def test_e2b_rows():
    with tempfile.TemporaryDirectory() as root:
        ds, e1, e2r = _tree(root)
        old = e2.summarize(ds, e1, e2r)
        assert e2.summarize(ds, e1, e2r, None) == old                   # without --e2b: the E2 table unchanged
        e2b = os.path.join(root, "e2b")
        # no_depth (E1) = 0.25, depth_stats_only (E1) = 0.27 -> boxnorm 0.255 keeps (0.255 − 0.25) / 0.02 = 0.25
        _metrics(os.path.join(e2b, "boxnorm", "depth_stats_only", "lr0.0003-do0", "metrics_d.json"),
                 "depth_stats_only", 3e-4, 0.0, 0.255, 0.46, 0.50)
        t = e2.summarize(ds, e1, e2r, e2b)
        missing = e2.summarize(ds, e1, e2r, os.path.join(root, "nothing"))
    rows = {line.split("|")[1].strip(): line for line in t.splitlines() if line.startswith("| ")}
    assert "0.2550" in rows["`boxnorm`"]
    assert "+0.0050" in rows["`boxnorm` − `no_depth`"] and "+0.0150" in rows["`depth_stats_only` − `boxnorm`"]
    assert "Share of the depth_stats gain kept by `boxnorm` (valid MRR): 0.25" in t
    assert set(old.splitlines()) <= set(t.splitlines())                  # every E2 line is kept as it was
    assert "missing" in [x for x in missing.splitlines() if x.startswith("| `boxnorm` |")][0]
    assert "Share" not in missing


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("OK")
