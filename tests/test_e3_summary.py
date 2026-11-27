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


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all E3 summary tests passed")
