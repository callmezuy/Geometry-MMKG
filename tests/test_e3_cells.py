"""
E3 cells (scripts/e3_cells.py): cell list, configuration resolution through tune_summary.choose, e3_cell.json,
train.py arguments and seed state (docs/superpowers/specs/2026-11-27-e3-official-runs-design.md 5.1, 5.3-5.5).
Run from Code/:  python tests/test_e3_cells.py
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e3_cells as e3  # noqa: E402


def _run(root, variant, lr, do, mrr, epochs=100, best_epoch=20, tag="inorm-ddrop0.3-tproj-dstat", commit="abc"):
    cfg = f"lr{lr:g}-do{do:g}" + (f"-ep{epochs}" if epochs != 100 else "")
    d = os.path.join(root, variant, cfg)
    os.makedirs(d, exist_ok=True)
    m = {"tune": True, "variant": variant, "config": {"lr": lr, "dropout": do}, "seed": 42, "options_tag": tag,
         "epoch_budget": {"epochs": epochs}, "best_epoch": best_epoch, "code_commit": commit,
         "best_valid": {"mrr": mrr, "hits@1": 0.1, "hits@10": 0.5, "mr": 10.0}}
    with open(os.path.join(d, f"metrics_geometry_x_{variant}_{tag}_{cfg}_seed42.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)


def _grid(root, variant, best, best_mrr=0.30):
    """Base grid with every point at 0.20 except `best` (lr, dropout) at best_mrr."""
    for lr in (1e-4, 3e-4, 1e-3):
        for do in (0.1, 0.3):
            _run(root, variant, lr, do, best_mrr if (lr, do) == best else 0.20)


def test_cells_and_seeds():
    assert sum(len(e3.CELLS[d]) * len(e3.SEEDS[d]) for d in e3.CELLS) == 59
    assert set(e3.CELLS) == set(e3.SEEDS)
    assert e3.SEEDS["MKG-Y"] == (42,) and e3.SEEDS["Spatial-VG"] == (42,) and e3.SEEDS["Spatial-VRD"] == (42, 43, 44)
    assert e3.CELLS["FB15k-237-IMG"] == ("no_depth", "fixed")
    assert e3.CELLS["Spatial-VG"] == ("no_depth", "fixed", "depth_stats_only")
    assert e3.CELLS["Spatial-VRD-Trans"] == ("no_depth", "depth_stats_only", "depth_map_only", "fixed", "box",
                                             "box_grid9", "boxnorm")
    assert e3.cell_variant("boxnorm") == e3.cell_variant("box") == "depth_stats_only"
    assert e3.cell_variant("fixed") == "fixed"


def test_box_source_dirs():
    assert e3.source_dir("Spatial-VRD", "box_grid9") == "geometry-e2/Spatial-VRD/box_grid9"
    assert e3.source_dir("Spatial-VRD", "boxnorm") == "geometry-e2b/Spatial-VRD/boxnorm"
    assert e3.source_dir("Spatial-VRD", "fixed") == "geometry-tune/Spatial-VRD"


def test_resolve_picks_the_choice():
    with tempfile.TemporaryDirectory() as hf:
        root = os.path.join(hf, "geometry-tune", "Spatial-VRD")
        # fixed: 3e-4 / 0 wins, late at 100 epochs, doubled to 200 and still late -> final, lower bound
        _grid(root, "fixed", None)
        _run(root, "fixed", 3e-4, 0.0, 0.31, epochs=100, best_epoch=100)
        _run(root, "fixed", 3e-4, 0.0, 0.33021234, epochs=200, best_epoch=200, commit="def")
        r = e3.resolve(hf, "Spatial-VRD", "fixed")
        assert (r["lr"], r["dropout"], r["epochs"]) == (3e-4, 0.0, 200), r
        assert r["expected"] == {"valid_mrr": 0.33021234, "best_epoch": 200} and r["lower_bound"] is True
        assert r["variant"] == "fixed" and r["cell"] == "fixed" and r["tune_code_commit"] == "def"
        assert r["source_dir"] == "geometry-tune/Spatial-VRD" and r["source_file"].endswith("-ep200_seed42.json")
        assert r["options_tag"] == "inorm-ddrop0.3-tproj-dstat" and r["n_configs"] == 8
        assert not os.path.isabs(r["source_file"])
        # no_depth: best at the lr edge, the edge point is missing -> not final
        _grid(root, "no_depth", (1e-3, 0.3))
        for bad in ("no_depth", "depth_map_only"):            # not final / no runs at all
            try:
                e3.resolve(hf, "Spatial-VRD", bad)
                raise AssertionError(f"{bad} must not resolve")
            except ValueError:
                pass
        # a box cell reads its own E2 folder, variant depth_stats_only
        box = os.path.join(hf, "geometry-e2", "Spatial-VRD", "box")
        _grid(box, "depth_stats_only", (3e-4, 0.1), 0.28)
        _run(box, "depth_stats_only", 3e-4, 0.0, 0.27)
        r = e3.resolve(hf, "Spatial-VRD", "box")
        assert r["variant"] == "depth_stats_only" and (r["lr"], r["dropout"], r["epochs"]) == (3e-4, 0.1, 100)
        assert r["lower_bound"] is False and r["source_dir"] == "geometry-e2/Spatial-VRD/box"


def test_table_digest_ignores_source():
    with tempfile.TemporaryDirectory() as d:
        t = {"box": torch.arange(18, dtype=torch.float32).reshape(2, 9), "box_grid9": torch.ones(2, 18)}
        t["box"][1, 3] = float("nan")
        a, b, c = (os.path.join(d, n) for n in ("a.pt", "b.pt", "c.pt"))
        torch.save({**t, "model": "box", "source": "/machine/one/path"}, a)
        torch.save({**t, "model": "box", "source": "/another/machine"}, b)
        t2 = {k: v.clone() for k, v in t.items()}
        t2["box"][0, 0] = 0.5
        torch.save({**t2, "source": "x"}, c)
        assert e3.table_digest(a, ("box",)) == e3.table_digest(b, ("box",))
        assert len(e3.table_digest(a, ("box",))) == 64
        assert e3.table_digest(a, ("box",)) != e3.table_digest(c, ("box",))
        assert e3.table_digest(a, ("box",)) != e3.table_digest(a, ("box_grid9",))


def _resolved():
    return {"dataset": "Spatial-VRD", "cell": "box", "variant": "depth_stats_only", "lr": 1e-4, "dropout": 0.0,
            "epochs": 100, "source_dir": "geometry-e2/Spatial-VRD/box", "source_file": "depth_stats_only/x/m.json",
            "tune_code_commit": "d8e426f", "options_tag": "inorm-ddrop0.3-tproj-dstat_box_geometry_box",
            "n_configs": 8, "expected": {"valid_mrr": 0.2853, "best_epoch": 60}, "lower_bound": False}


def test_cell_record_check():
    rec = e3.cell_record(_resolved(), "/work/e2/Spatial-VRD/box_geometry.pt:box", "f" * 64, "aaa")
    assert rec["depth_stats"] == "box_geometry.pt:box" and rec["table_sha256"] == "f" * 64
    assert rec["e3_code_commit"] == "aaa" and rec["lr"] == 1e-4 and rec["expected"]["best_epoch"] == 60
    assert "/work" not in json.dumps(rec)
    e3.check_cell_record(rec, {**rec, "e3_code_commit": "bbb"})
    for key, value in (("lr", 3e-4), ("table_sha256", "0" * 64), ("epochs", 200)):
        try:
            e3.check_cell_record(rec, {**rec, key: value})
            raise AssertionError(f"a different {key} must be refused")
        except ValueError as err:
            assert key in str(err)
    plain = e3.cell_record({**_resolved(), "cell": "fixed", "variant": "fixed"}, None, None, "aaa")
    assert plain["depth_stats"] is None and plain["table_sha256"] is None


def test_train_args():
    r = {**_resolved(), "expected": {"valid_mrr": 0.33021234, "best_epoch": 60}}
    a42 = e3.train_args(r, "/data/Spatial-VRD", 42, "/out", "/t/box_geometry.pt:box")
    i = a42.index("--expect-valid")
    mrr, epoch = a42[i + 1].split(":")
    assert float(mrr) == 0.33021234 and int(epoch) == 60
    for flag, value in (("--variant", "depth_stats_only"), ("--seed", "42"), ("--epochs", "100"), ("--out", "/out"),
                        ("--depth-stats", "/t/box_geometry.pt:box"), ("--data", "/data/Spatial-VRD")):
        assert a42[a42.index(flag) + 1] == value, flag
    assert float(a42[a42.index("--lr") + 1]) == 1e-4 and float(a42[a42.index("--dropout") + 1]) == 0.0
    a43 = e3.train_args(r, "/data/Spatial-VRD", 43, "/out")
    assert "--expect-valid" not in a43 and "--depth-stats" not in a43 and a43[a43.index("--epochs") + 1] == "100"
    assert "--tune" not in a42 and "--dev" not in a42
    assert e3.stem("Spatial-VRD", "fixed", "inorm-ddrop0.3-tproj-dstat", 43) == \
        "geometry_Spatial-VRD_fixed_inorm-ddrop0.3-tproj-dstat_seed43"


def test_seed_state():
    assert e3.seed_state(None) == "todo"
    assert e3.seed_state({"test": {"mrr": 0.3}}) == "done"
    assert e3.seed_state({"test": {"mrr": 0.3}, "expect_valid": {"ok": True}}) == "done"
    assert e3.seed_state({"expect_valid": {"ok": False}}) == "mismatch"
    try:
        e3.seed_state({"best_valid": {}})
        raise AssertionError("metrics without test and without expect_valid must be refused")
    except ValueError:
        pass


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all E3 cell tests passed")
