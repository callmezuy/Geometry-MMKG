"""
E3 summary (scripts/e3_summary.py): verdicts, paired statistics, the pre-registered claim rules
(docs/superpowers/specs/2026-11-27-e3-official-runs-design.md section 6), the tables and the CLI.
Run from Code/:  python tests/test_e3_summary.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e3_summary as es  # noqa: E402


def test_constants():
    assert es.THRESHOLD == 0.003 and es.N_BOOT == 10000 and es.MIN_GROUP == 500


def test_verdict_rules():
    pos = [(0.001, 0.006)] * 3
    assert es.verdict([0.0031, 0.0031, 0.0031], pos) == "yes"
    assert es.verdict([0.0030, 0.0030, 0.0030], pos) == "equal"
    assert es.verdict([-0.001, 0.005, 0.006], pos) == "unclear"                     # one seed of the other sign
    assert es.verdict([0.004, 0.005, 0.006], [(0.001, 0.007), (-0.001, 0.009), (0.002, 0.01)]) == "unclear"
    neg = [(-0.006, -0.001)] * 3
    assert es.verdict([-0.004, -0.005, -0.006], neg) == "yes (negative)"
    assert es.verdict([-0.004, -0.005, -0.006], [(-0.006, 0.001)] * 3) == "unclear"
    assert es.verdict([0.002, -0.001, 0.004], pos) == "equal"                       # |mean| 0.0017
    # one seed
    assert es.verdict([0.004], [(0.001, 0.007)]) == "yes"
    assert es.verdict([0.004], [(-0.001, 0.009)]) == "unclear"
    assert es.verdict([-0.004], [(-0.007, -0.001)]) == "yes (negative)"
    assert es.verdict([0.002], [(0.001, 0.003)]) == "equal"


def test_pair_stats():
    rng = np.random.default_rng(0)
    b = {s: rng.random(2000) for s in (42, 43, 44)}
    a = {s: b[s] + 0.01 + 0.001 * rng.standard_normal(2000) for s in b}
    st = es.pair_stats(a, b, n_boot=500)
    assert abs(st["mean"] - 0.01) < 1e-3 and st["verdict"] == "yes"
    assert all(st["seeds"][s]["lo"] > 0 for s in b) and set(st["seeds"]) == {42, 43, 44}
    assert st["std"] is not None and st["t"] is not None and st["p_t"] is not None
    one = es.pair_stats({42: a[42]}, {42: b[42]}, n_boot=500)
    assert one["std"] is None and one["t"] is None and one["p_t"] is None and one["verdict"] == "yes"
    try:
        es.pair_stats({42: a[42], 43: a[43]}, {42: b[42]}, n_boot=10)
        raise AssertionError("different seeds must be refused")
    except ValueError:
        pass


def test_pairs():
    assert es.PAIRS["mmkg"] == [("fixed", "no_depth", "C2")]
    vrd = [(a, b) for a, b, _ in es.PAIRS["vrd"]]
    for p in (("depth_stats_only", "no_depth"), ("fixed", "depth_map_only"), ("depth_map_only", "no_depth"),
              ("fixed", "depth_stats_only"), ("box_grid9", "box"), ("box", "depth_stats_only"),
              ("boxnorm", "no_depth"), ("depth_stats_only", "boxnorm")):
        assert p in vrd, p
    assert [(a, b) for a, b, _ in es.PAIRS["vg"]] == [("depth_stats_only", "no_depth"), ("fixed", "no_depth"),
                                                       ("fixed", "depth_stats_only")]


def test_claim_readers():
    mmkg = ("FB15k-237-IMG", "FB15K-237-ind", "MKG-Y")
    assert es.read_c2({d: "equal" for d in mmkg})[0] == "holds"
    st, text = es.read_c2({**{d: "equal" for d in mmkg}, "MKG-Y": "yes (negative)"})
    assert st == "rewrite" and "MKG-Y" in text
    assert es.read_c2({**{d: "equal" for d in mmkg}, "MKG-Y": "unclear"})[0] == "report as is"
    assert es.read_c2({"FB15k-237-IMG": "equal"})[0] == "incomplete"

    vrd = ("Spatial-VRD", "Spatial-VRD-Trans")
    good = {("depth_stats_only", "no_depth"): "yes", ("fixed", "depth_map_only"): "yes",
            ("depth_map_only", "no_depth"): "equal", ("fixed", "depth_stats_only"): "yes (negative)"}
    assert es.read_c3({d: good for d in vrd})[0] == "holds"
    assert es.read_c3({vrd[0]: good, vrd[1]: {**good, ("fixed", "depth_map_only"): "unclear"}})[0] == "per verdicts"
    assert es.read_c3({vrd[0]: good, vrd[1]: {**good, ("depth_map_only", "no_depth"): "yes"}})[0] == "per verdicts"
    assert es.read_c3({vrd[0]: good})[0] == "incomplete"

    assert es.read_c4({d: "yes" for d in vrd})[0] == "C4 yes"
    assert es.read_c4({d: "equal" for d in vrd})[0] == "C4 no"
    assert es.read_c4({vrd[0]: "yes", vrd[1]: "unclear"})[0] == "inconsistent"
    assert es.read_c4({vrd[0]: "yes"})[0] == "incomplete"

    # (boxnorm - no_depth, depth_stats_only - boxnorm, share)
    assert es.read_e2b({d: (0.002, 0.009, 0.18) for d in vrd})[0] == "needed"
    assert es.read_e2b({d: (0.009, 0.002, 0.82) for d in vrd})[0] == "not needed"
    st, text = es.read_e2b({d: (0.005, 0.008, 0.38) for d in vrd})
    assert st == "partial" and "0.38" in text
    assert es.read_e2b({vrd[0]: (0.002, 0.009, 0.18), vrd[1]: (0.005, 0.008, 0.38)})[0] == "inconsistent"
    assert es.read_e2b({vrd[0]: (0.002, 0.009, 0.18)})[0] == "incomplete"


def test_lower_bound_note():
    assert es.lower_bound_note("fixed", "no_depth", set()) == ""
    assert "grow" in es.lower_bound_note("fixed", "no_depth", {"fixed"})
    assert "shrink" in es.lower_bound_note("depth_stats_only", "boxnorm", {"boxnorm"})
    assert "unknown" in es.lower_bound_note("depth_stats_only", "boxnorm", {"boxnorm", "depth_stats_only"})


N_ENT, N_TRIPLES = 100, 600
TINY = "Tiny-MMKG"


def _keys():
    i = np.arange(N_TRIPLES)
    return np.stack([i % N_ENT, i // N_ENT, (i * 7 + 3) % N_ENT], axis=1)      # (h, r, t), unique


def _write_cell(root, cell, seeds, better=0, drop=None, maskall=False, expect=None):
    import json
    d = os.path.join(root, TINY, cell)
    os.makedirs(d, exist_ok=True)
    rec = {"dataset": TINY, "cell": cell, "variant": cell, "options_tag": "tag", "lower_bound": cell == "fixed"}
    json.dump(rec, open(os.path.join(d, "e3_cell.json"), "w"))
    for s in seeds:
        rng = np.random.default_rng(s)
        keys = _keys()
        rh = rng.integers(1, 50, N_TRIPLES)
        rt = np.maximum(1, rng.integers(1, 50, N_TRIPLES) - better)
        if drop is not None and s == drop:
            keys, rh, rt = keys[1:], rh[1:], rt[1:]
        stem = f"geometry_{TINY}_{cell}_tag_seed{s}"
        m = {"seed": s, "variant": cell, "best_epoch": 190 if s == 44 else 50, "epoch_budget": {"epochs": 200},
             "test": {"mrr": 0.3 + 0.01 * better, "mr": 20.0, "hits@1": 0.2, "hits@3": 0.3, "hits@10": 0.5}}
        if expect is not None and s == 42:
            m["expect_valid"] = {"mrr": 0.25, "epoch": 50, "got_mrr": 0.25 + expect, "got_epoch": 50, "tol": 5e-4,
                                 "ok": abs(expect) <= 5e-4}
            if not m["expect_valid"]["ok"]:
                del m["test"]
        json.dump(m, open(os.path.join(d, f"metrics_{stem}.json"), "w"))
        rows = "".join(f"{h},{r},{t},{a},{b}\n" for (h, r, t), a, b in zip(keys, rh, rt))
        for suffix in ("_test",) + (("_test_maskall",) if maskall else ()):
            with open(os.path.join(d, f"ranks_{stem}{suffix}.csv"), "w") as f:
                f.write("h_id,r_id,t_id,rank_head,rank_tail\n" + rows)
        if maskall:
            json.dump({"masked": "all entities"}, open(os.path.join(d, f"metrics_{stem}_maskall.json"), "w"))


def _data_root(root):
    import torch
    d = os.path.join(root, "data", TINY)
    os.makedirs(d, exist_ok=True)
    mask = torch.ones(N_ENT, dtype=torch.bool)
    mask[::5] = False                                                          # 20% without an image
    torch.save(mask, os.path.join(d, "ent_img_mask.pt"))
    return os.path.join(root, "data"), mask.numpy()


def _summarize(root, data_root):
    return es.summarize(os.path.join(root, "e3"), data_root, [TINY], cells={TINY: ("no_depth", "fixed")},
                        seeds={TINY: (42, 43, 44)}, kinds={TINY: "mmkg"}, n_boot=200)


def test_image_groups():
    keys = np.array([[0, 0, 1], [0, 1, 2], [2, 0, 3], [1, 0, 3]])
    mask = np.array([True, False, True, False])
    assert es.image_groups(keys, mask).tolist() == [1, 2, 1, 0]
    assert es.image_groups(np.array([[0, 0, 2]]), mask).tolist() == [2]


def test_summarize_tiny():
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        data_root, mask = _data_root(root)
        e3root = os.path.join(root, "e3")
        _write_cell(e3root, "no_depth", (42, 43, 44))
        _write_cell(e3root, "fixed", (42, 43, 44), better=3)
        md, data = _summarize(root, data_root)
        d = data[TINY]
        assert set(d["cells"]) == {"no_depth", "fixed"} and d["cells"]["fixed"]["n_seeds"] == 3
        assert d["cells"]["fixed"]["late_seeds"] == 1                          # seed 44: best epoch 190 / 200
        assert d["pairs"]["fixed - no_depth"]["verdict"] in ("yes", "unclear", "equal", "yes (negative)")
        groups = es.image_groups(_keys(), mask)
        assert {g: d["groups"][g]["n"] for g in d["groups"]} == {
            name: int((groups == k).sum()) for k, name in es.GROUP_NAMES.items() if (groups == k).any()}
        assert sum(d["groups"][g]["n"] for g in d["groups"]) == N_TRIPLES
        small = [g for g in d["groups"] if d["groups"][g]["n"] < es.MIN_GROUP]
        assert small and all(d["groups"][g]["pair"]["verdict"] == "—" for g in small)
        assert d["masked"] is None and "`fixed` − `no_depth`" in md and "lower bound" in md
        assert "| `fixed` |" in md and "image group" in md.lower()
        # the masked control appears once every seed of both cells has it
        _write_cell(e3root, "no_depth", (42, 43, 44), maskall=True)
        _write_cell(e3root, "fixed", (42, 43, 44), better=3, maskall=True)
        md, data = _summarize(root, data_root)
        assert data[TINY]["masked"] is not None and "masked" in md.lower()


def test_incomplete_cell_gives_no_verdict():
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        data_root, _ = _data_root(root)
        _write_cell(os.path.join(root, "e3"), "no_depth", (42, 43, 44))
        _write_cell(os.path.join(root, "e3"), "fixed", (42, 43))
        md, data = _summarize(root, data_root)
        assert "fixed - no_depth" not in data[TINY]["pairs"] and "incomplete" in md


def test_triple_sets_must_match():
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        data_root, _ = _data_root(root)
        _write_cell(os.path.join(root, "e3"), "no_depth", (42, 43, 44))
        _write_cell(os.path.join(root, "e3"), "fixed", (42, 43, 44), drop=43)
        try:
            _summarize(root, data_root)
            raise AssertionError("different triple sets must stop the summary")
        except ValueError as e:
            assert "triple" in str(e)


def test_chunked_bootstrap_matches_compute_metrics():
    from compute_metrics import paired_bootstrap
    rng = np.random.default_rng(3)
    a, b = rng.random(700), rng.random(700)
    want = paired_bootstrap(a, b, 300, 0)
    got = es.bootstrap(a - b, 300, 0, chunk=70)
    assert np.allclose(want, got), (want, got)


def test_load_cell_refuses_duplicates_and_other_stems():
    import json
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        e3root = os.path.join(root, "e3")
        _write_cell(e3root, "fixed", (42, 43, 44))
        d = os.path.join(e3root, TINY, "fixed")
        src = os.path.join(d, f"metrics_geometry_{TINY}_fixed_tag_seed43.json")
        shutil.copy(src, os.path.join(d, f"metrics_geometry_{TINY}_fixed_othertag_seed43.json"))
        shutil.copy(os.path.join(d, f"ranks_geometry_{TINY}_fixed_tag_seed43_test.csv"),
                    os.path.join(d, f"ranks_geometry_{TINY}_fixed_othertag_seed43_test.csv"))
        try:
            es.load_cell(os.path.join(e3root, TINY), "fixed")
            raise AssertionError("a metrics file with another stem must be refused")
        except ValueError as e:
            assert "othertag" in str(e)
        # without a record: two files of one seed are refused too
        os.remove(os.path.join(d, "e3_cell.json"))
        try:
            es.load_cell(os.path.join(e3root, TINY), "fixed")
            raise AssertionError("two metrics files of one seed must be refused")
        except ValueError as e:
            assert "seed 43" in str(e)
        json.dumps({})


def test_seed42_check_and_group_mrr_in_the_summary():
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        data_root, _ = _data_root(root)
        e3root = os.path.join(root, "e3")
        _write_cell(e3root, "no_depth", (42, 43, 44), expect=0.0002)
        _write_cell(e3root, "fixed", (42,), expect=0.003)             # stopped by the gate
        md, data = _summarize(root, data_root)
        chk = data[TINY]["cells"]["no_depth"]["seed42_check"]
        assert abs(chk["d_mrr"] - 0.0002) < 1e-12 and chk["got_epoch"] == chk["epoch"] == 50 and chk["ok"] is True
        assert "seed-42 check" in md and "+0.0002" in md
        assert "stopped by the seed-42 check" in md and "fixed" not in data[TINY]["cells"]
        _write_cell(e3root, "fixed", (42, 43, 44), better=3)
        md, data = _summarize(root, data_root)
        g = data[TINY]["groups"]["both"]
        assert set(g["mrr"]) == {"fixed", "no_depth"} and g["mrr"]["fixed"] > g["mrr"]["no_depth"]


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all E3 summary tests passed")
