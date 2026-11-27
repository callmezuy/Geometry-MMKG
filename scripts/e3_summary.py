"""
E3 summary and E4 tests (RESEARCH_PLAN P6.0; docs/superpowers/specs/2026-11-27-e3-official-runs-design.md 5.9 and
section 6, rules fixed before the runs). Reads only geometry-e3/<dataset>/<cell>/ (e3_cell.json, metrics, test rank
files, and the image-masked rank files of scripts/e3_mask_eval.py when present).

* Per cell: test MRR / MR / Hits@k mean +- std over seeds; on the two VRD sets also the E1 auxiliary metrics (reported
  only).
* Per pair A - B: delta per seed, mean delta, paired bootstrap over test triples within each seed
  (compute_metrics.paired_bootstrap), paired t-test over seeds (reported only: 2 degrees of freedom), verdict:
  "yes"   mean > 0.003, every seed > 0 and every per-seed 95% CI excludes 0 (one seed: |delta| > 0.003 and CI
          excludes 0); "yes (negative)" mirrored;
  "equal" |mean| <= 0.003;
  "unclear" otherwise.
* Claims C2, C3, C4 and E2b from those verdicts (spec 6.2); image groups (both / one / no end with an image) on the
  knowledge graphs (spec 6.3); the image-masked control (spec 6.4).
  python scripts/e3_summary.py --root results/geometry-e3 --data-root data [--datasets ...] [--out e3]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_metrics import paired_bootstrap  # noqa: E402

THRESHOLD = 0.003
N_BOOT = 10000
BOOT_SEED = 0
MIN_GROUP = 500
EPS = 1e-12               # float slack on the threshold comparisons
MMKG_SETS = ("FB15k-237-IMG", "FB15K-237-ind", "MKG-Y")
VRD_SETS = ("Spatial-VRD", "Spatial-VRD-Trans")
KINDS = {**{d: "mmkg" for d in MMKG_SETS}, **{d: "vrd" for d in VRD_SETS}, "Spatial-VG": "vg"}
DEPTH_STATS_PAIRS = (("depth_stats_only", "no_depth"), ("fixed", "depth_map_only"))
VECTOR_PAIRS = (("depth_map_only", "no_depth"), ("fixed", "depth_stats_only"))
PAIRS = {
    "mmkg": [("fixed", "no_depth", "C2")],
    "vrd": [("depth_stats_only", "no_depth", "depth (main model)"),
            ("fixed", "depth_map_only", "C3: depth statistics, with the depth vector"),
            ("depth_map_only", "no_depth", "C3: depth vector, without depth statistics"),
            ("fixed", "depth_stats_only", "C3: depth vector, with depth statistics"),
            ("box_grid9", "box", "C4: depth beyond the box"),
            ("box", "depth_stats_only", "box vs depth statistics"),
            ("boxnorm", "no_depth", "E2b: per-box scale vs nothing"),
            ("depth_stats_only", "boxnorm", "E2b: shared scale vs per-box scale")],
    "vg": [("depth_stats_only", "no_depth", "depth (main model)"),
           ("fixed", "no_depth", "depth (with the depth vector)"),
           ("fixed", "depth_stats_only", "depth vector, with depth statistics")],
}


def verdict(deltas, cis, thr=THRESHOLD):
    """Verdict of one pair on one dataset (module docstring); deltas / cis: one entry per seed."""
    d = np.asarray(deltas, dtype=np.float64)
    m = float(d.mean())
    if abs(m) <= thr + EPS:
        return "equal"
    lo = np.array([c[0] for c in cis])
    hi = np.array([c[1] for c in cis])
    if m > 0 and (d > 0).all() and (lo > 0).all():
        return "yes"
    if m < 0 and (d < 0).all() and (hi < 0).all():
        return "yes (negative)"
    return "unclear"


def pair_stats(rr_a, rr_b, n_boot=N_BOOT):
    """rr_a / rr_b: {seed: per-triple reciprocal rank}, aligned on the same triples within a seed."""
    if set(rr_a) != set(rr_b):
        raise ValueError(f"seeds differ: {sorted(rr_a)} vs {sorted(rr_b)}")
    seeds = sorted(rr_a)
    per = {}
    for s in seeds:
        if len(rr_a[s]) != len(rr_b[s]):
            raise ValueError(f"seed {s}: {len(rr_a[s])} vs {len(rr_b[s])} triples")
        delta, lo, hi, p = paired_bootstrap(np.asarray(rr_a[s]), np.asarray(rr_b[s]), n_boot, BOOT_SEED)
        per[s] = {"delta": float(delta), "lo": float(lo), "hi": float(hi), "p": float(p)}
    deltas = [per[s]["delta"] for s in seeds]
    out = {"seeds": per, "mean": float(np.mean(deltas)), "std": None, "t": None, "p_t": None}
    if len(seeds) > 1:
        from scipy.stats import ttest_1samp
        out["std"] = float(np.std(deltas, ddof=1))
        t = ttest_1samp(deltas, 0.0)           # = paired t-test of the per-seed MRRs
        out["t"], out["p_t"] = float(t.statistic), float(t.pvalue)
    out["verdict"] = verdict(deltas, [(per[s]["lo"], per[s]["hi"]) for s in seeds])
    return out


def _missing(have, need):
    return [d for d in need if d not in have]


def read_c2(verdicts):
    """{MMKG dataset: verdict of fixed - no_depth} -> (status, text)."""
    miss = _missing(verdicts, MMKG_SETS)
    if miss:
        return "incomplete", f"C2 needs {', '.join(miss)}"
    moved = [d for d in MMKG_SETS if verdicts[d].startswith("yes")]
    if moved:
        return "rewrite", "C2 must be rewritten for " + ", ".join(f"{d} ({verdicts[d]})" for d in moved)
    if all(verdicts[d] == "equal" for d in MMKG_SETS):
        return "holds", "C2 holds for GeometryMMKG: depth is within the noise on all three knowledge graphs"
    return "report as is", "C2: no dataset moves, but not all are within the noise: " + ", ".join(
        f"{d} {verdicts[d]}" for d in MMKG_SETS)


def read_c3(v):
    """{VRD dataset: {(a, b): verdict}} -> (status, text)."""
    miss = _missing(v, VRD_SETS)
    if miss:
        return "incomplete", f"C3 needs {', '.join(miss)}"
    stats_ok = all(v[d].get(p) == "yes" for d in VRD_SETS for p in DEPTH_STATS_PAIRS)
    vector_up = [(d, p) for d in VRD_SETS for p in VECTOR_PAIRS if v[d].get(p) == "yes"]
    if stats_ok and not vector_up:
        return "holds", "C3 holds: depth statistics help on both sets, the depth vector never does"
    parts = [f"{d}: " + ", ".join(f"{a} - {b} {v[d].get((a, b))}" for a, b in DEPTH_STATS_PAIRS + VECTOR_PAIRS)
             for d in VRD_SETS]
    return "per verdicts", "C3 is written from the verdicts: " + "; ".join(parts)


def read_c4(v):
    """{VRD dataset: verdict of box_grid9 - box} -> (status, text)."""
    miss = _missing(v, VRD_SETS)
    if miss:
        return "incomplete", f"C4 needs {', '.join(miss)}"
    if all(v[d] == "yes" for d in VRD_SETS):
        return "C4 yes", "C4 yes: depth adds to the box on both sets"
    if all(v[d] == "equal" for d in VRD_SETS):
        return "C4 no", "C4 no: depth adds nothing beyond the box"
    return "inconsistent", "C4 inconsistent: " + ", ".join(f"{d} {v[d]}" for d in VRD_SETS)


def _e2b_branch(bn_nd, dso_bn, thr=THRESHOLD):
    if bn_nd <= thr + EPS:
        return "needed"
    if dso_bn <= thr + EPS:
        return "not needed"
    return "partial"


def read_e2b(means):
    """{VRD dataset: (mean boxnorm - no_depth, mean depth_stats_only - boxnorm, share kept)} -> (status, text);
    the rule of section 6 of the E2b design on the 3-seed means."""
    miss = _missing(means, VRD_SETS)
    if miss:
        return "incomplete", f"E2b needs {', '.join(miss)}"
    branch = {d: _e2b_branch(means[d][0], means[d][1]) for d in VRD_SETS}
    detail = ", ".join(f"{d} {branch[d]} (share kept {means[d][2]:.2f})" for d in VRD_SETS)
    if len(set(branch.values())) > 1:
        return "inconsistent", f"E2b inconsistent between the two sets (final, no further run): {detail}"
    b = branch[VRD_SETS[0]]
    text = {"needed": "the depth statistics gain needs the shared scale of the image",
            "not needed": "the gain does not need the shared scale (C2 must be rewritten)",
            "partial": "the gain partly needs the shared scale"}[b]
    return b, f"E2b {b}: {text}; {detail}"


def lower_bound_note(a, b, lower_bound_cells):
    la, lb = a in lower_bound_cells, b in lower_bound_cells
    if la and lb:
        return f"`{a}` and `{b}` are lower bounds: direction unknown"
    if la:
        return f"`{a}` is a lower bound: the difference would grow if it were higher"
    if lb:
        return f"`{b}` is a lower bound: the difference would shrink if it were higher"
    return ""
