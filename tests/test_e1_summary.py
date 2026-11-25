"""
scripts/e1_summary.py: the E1 configuration list and the summary table.
Run from Code/:  python tests/test_e1_summary.py
"""
import json
import os
import sys
import tempfile

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "scripts"))
import e1_summary  # noqa: E402


def _write(root, variant, cfg, mrr, best_epoch, aux=True):
    d = os.path.join(root, variant, cfg)
    os.makedirs(d)
    best = {"mrr": mrr}
    if aux:
        best["in_image"] = {"mrr": 0.61, "hits@1": 0.4, "hits@3": 0.8, "image_first": 0.3, "n_cand": 19.0,
                            "n_cand_filtered": 18.3, "mrr_random": 0.2, "known_beats_truth": 0.45,
                            "excl_known": {"mrr": 0.77, "hits@1": 0.66}}
        best["relation"] = {"mrr": 0.7, "hits@1": 0.6, "hits@5": 0.9}
    with open(os.path.join(d, f"metrics_geometry_x_{variant}_{cfg}_seed42.json"), "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "tune": True, "best_epoch": best_epoch, "best_valid": best}, f)


def test_config_dir():
    assert e1_summary.config_dir(3e-4, 0.0, 200) == "lr0.0003-do0-ep200"
    assert e1_summary.config_dir(1e-3, 0.0, None) == "lr0.001-do0"
    assert len(e1_summary.E1_CONFIGS["Spatial-VRD"]) == 4 and len(e1_summary.E1_CONFIGS["Spatial-VRD-Trans"]) == 4


def test_summary_table():
    with tempfile.TemporaryDirectory() as root:
        e1, ref = os.path.join(root, "e1"), os.path.join(root, "tune")
        _write(e1, "no_depth", "lr0.0003-do0", 0.2590, 20)
        _write(ref, "no_depth", "lr0.0003-do0", 0.2560, 20, aux=False)   # drift 0.003 > 0.002 -> flagged
        table = e1_summary.summarize("Spatial-VRD", e1, ref)
        rows = [r for r in table.splitlines() if r.startswith("| `")]
        assert len(rows) == 4, table
        assert sum("missing" in r for r in rows) == 3
        filled = next(r for r in rows if "missing" not in r)
        assert "⚠" in filled and "0.2590" in filled and "+0.0030" in filled and "0.6100" in filled, filled
        assert "0.7700" in filled and "0.6600" in filled and "0.450" in filled, filled
        no_ref = e1_summary.summarize("Spatial-VRD", e1, None)
        assert "⚠" not in no_ref and "–" in no_ref


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("OK")
