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

THRESHOLD = 0.003
N_BOOT = 10000
BOOT_SEED = 0
MIN_GROUP = 500
EPS = 1e-12               # float slack on the threshold comparisons
BOOT_CHUNK = 500          # resamples drawn at once (memory: chunk x n triples x 16 bytes)
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


def bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED, chunk=BOOT_CHUNK):
    """compute_metrics.paired_bootstrap on the per-triple differences d, drawn in chunks of resamples from one
    generator (the same index stream as one draw), so 10,000 resamples of 138k triples fit in memory.
    Returns (delta, lo, hi, p)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, n, size=(min(chunk, n_boot - i), n))].mean(axis=1)
                            for i in range(0, n_boot, chunk)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return d.mean(), lo, hi, min(p, 1.0)


def pair_stats(rr_a, rr_b, n_boot=N_BOOT):
    """rr_a / rr_b: {seed: per-triple reciprocal rank}, aligned on the same triples within a seed."""
    if set(rr_a) != set(rr_b):
        raise ValueError(f"seeds differ: {sorted(rr_a)} vs {sorted(rr_b)}")
    seeds = sorted(rr_a)
    per = {}
    for s in seeds:
        if len(rr_a[s]) != len(rr_b[s]):
            raise ValueError(f"seed {s}: {len(rr_a[s])} vs {len(rr_b[s])} triples")
        delta, lo, hi, p = bootstrap(np.asarray(rr_a[s]) - np.asarray(rr_b[s]), n_boot, BOOT_SEED)
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


# ---------------------------------------------------------------- loading, tables, CLI
GROUP_NAMES = {2: "both", 1: "one", 0: "none"}
LATE_FRACTION = 0.9
CELL_COLUMNS = [("MRR", "mrr", ".4f"), ("MR", "mr", ".1f"), ("H@1", "hits@1", ".4f"), ("H@3", "hits@3", ".4f"),
                ("H@10", "hits@10", ".4f")]
AUX_COLUMNS = [("in-image MRR", ("in_image", "mrr"), ".4f"), ("in-image H@1", ("in_image", "hits@1"), ".4f"),
               ("image-first", ("in_image", "image_first"), ".3f"), ("relation MRR", ("relation", "mrr"), ".4f"),
               ("relation H@1", ("relation", "hits@1"), ".4f")]


def _rr(path):
    """(keys sorted by (h, r, t), per-triple RR = (1 / rank_head + 1 / rank_tail) / 2 in that order)."""
    from compute_metrics import read_ranks
    keys, rh, rt = read_ranks(path)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return keys[order], ((1 / rh + 1 / rt) / 2)[order]


def load_cell(root, cell):
    """{record, seeds: {seed: {metrics, keys, rr[, rr_mask]}}, skipped: [seeds without test]} or None."""
    d = os.path.join(root, cell)
    if not os.path.isdir(d):
        return None
    import e3_cells
    rec_path = os.path.join(d, e3_cells.RECORD)
    rec = e3_cells.read_json(rec_path) if os.path.exists(rec_path) else None
    out = {"record": rec, "seeds": {}, "skipped": [], "gate": {}}
    seen = {}
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("metrics_") and fn.endswith(".json")) or fn.endswith("_maskall.json"):
            continue
        stem = fn[len("metrics_"):-len(".json")]
        m = e3_cells.read_json(os.path.join(d, fn))
        seed = m.get("seed")
        if rec is not None and stem != e3_cells.stem(rec["dataset"], rec["variant"], rec["options_tag"], seed):
            raise ValueError(f"{cell}: {fn} does not belong to the cell's record (dataset / variant / options tag "
                             f"{rec['dataset']} / {rec['variant']} / {rec['options_tag']})")
        if seed in seen:
            raise ValueError(f"{cell}: two metrics files for seed {seed}: {seen[seed]}, {fn}")
        seen[seed] = fn
        if m.get("expect_valid") is not None:
            out["gate"][seed] = m["expect_valid"]
        if "test" not in m:
            out["skipped"].append(seed)
            continue
        keys, rr = _rr(os.path.join(d, f"ranks_{stem}_test.csv"))
        s = {"metrics": m, "keys": keys, "rr": rr}
        mask_path = os.path.join(d, f"ranks_{stem}_test_maskall.csv")
        if os.path.exists(mask_path):
            mkeys, s["rr_mask"] = _rr(mask_path)
            if not np.array_equal(mkeys, keys):
                raise ValueError(f"{cell} seed {m['seed']}: the masked rank file has other triples")
        out["seeds"][m["seed"]] = s
    return out


def image_groups(keys, mask):
    """Number of ends of each (h, r, t) triple that have an image (0, 1, 2)."""
    mask = np.asarray(mask, dtype=bool)
    return mask[keys[:, 0]].astype(int) + mask[keys[:, 2]].astype(int)


def _ms(xs, fmt):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return format(xs[0], fmt)
    return f"{format(float(np.mean(xs)), fmt)} ± {format(float(np.std(xs, ddof=1)), fmt)}"


def _get(m, path):
    for k in path:
        m = (m or {}).get(k)
    return m


def _pair_row(label, a, b, st, note=""):
    seeds = " / ".join(f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for _, v in sorted(st["seeds"].items()))
    t = "–" if st["t"] is None else f"{st['t']:.2f} (p {st['p_t']:.3f})"
    std = "" if st["std"] is None else f" ± {st['std']:.4f}"
    return f"| `{a}` − `{b}` | {label} | {st['mean']:+.4f}{std} | {seeds} | {t} | **{st['verdict']}** | {note} |"


PAIR_HEAD = ["| Pair | Reading | Mean Δ MRR | Δ per seed [95% CI] | t-test over seeds | Verdict | Note |",
             "|---|---|---|---|---|---|---|"]


def summarize(e3_root, data_root, datasets, cells=None, seeds=None, kinds=None, n_boot=N_BOOT):
    """(markdown, data) for the datasets given; cells / seeds / kinds default to e3_cells.CELLS / SEEDS / KINDS."""
    import e3_cells
    cells = cells or e3_cells.CELLS
    seeds = seeds or e3_cells.SEEDS
    kinds = kinds or KINDS
    md, data = [], {}
    for ds in datasets:
        kind = kinds[ds]
        loaded = {c: load_cell(os.path.join(e3_root, ds), c) for c in cells[ds]}
        stopped = {c: v for c, v in loaded.items() if v is not None and not v["seeds"] and v["gate"]}
        loaded = {c: v for c, v in loaded.items() if v is not None and v["seeds"]}
        ref = None
        for c, v in loaded.items():
            for s, x in v["seeds"].items():
                if ref is None:
                    ref = (c, s, x["keys"])
                elif not np.array_equal(ref[2], x["keys"]):
                    raise ValueError(f"{ds}: {c} seed {s} was evaluated on other test triples than {ref[0]} seed "
                                     f"{ref[1]}")
        planned = tuple(seeds[ds])
        complete = {c for c, v in loaded.items() if tuple(sorted(v["seeds"])) == planned}
        lower = {c for c, v in loaded.items() if (v["record"] or {}).get("lower_bound")}
        d = {"cells": {}, "pairs": {}, "groups": {}, "masked": None}
        head = "| Cell | Seeds | " + " | ".join(c[0] for c in CELL_COLUMNS) + (
            " | " + " | ".join(c[0] for c in AUX_COLUMNS) if kind == "vrd" else "") + " | Note |"
        md += [f"## E3 `{ds}` (test; {len(planned)} seed{'s' if len(planned) > 1 else ''})", "", head,
               "|" + "---|" * (head.count("|") - 1)]
        for c in cells[ds]:
            v = loaded.get(c)
            if v is None and c in stopped:
                ev = stopped[c]["gate"].get(42) or {}
                md.append(f"| `{c}` | 0 / {len(planned)} | stopped by the seed-42 check: valid MRR "
                          f"{ev.get('got_mrr', float('nan')):.4f} vs {ev.get('mrr', float('nan')):.4f}, best epoch "
                          f"{ev.get('got_epoch')} vs {ev.get('epoch')} |" + " |" * (head.count("|") - 4))
                continue
            if v is None:
                md.append(f"| `{c}` | 0 / {len(planned)} | missing |" + " |" * (head.count("|") - 4))
                continue
            ms = [x["metrics"] for _, x in sorted(v["seeds"].items())]
            late = sum(m["best_epoch"] > LATE_FRACTION * m["epoch_budget"]["epochs"] for m in ms)
            row = {"n_seeds": len(ms), "late_seeds": late, "lower_bound": c in lower,
                   **{k: [m["test"].get(k) for m in ms] for _, k, _ in CELL_COLUMNS}}
            if kind == "vrd":
                row.update({"/".join(p): [_get(m["test"], p) for m in ms] for _, p, _ in AUX_COLUMNS})
            ev = v["gate"].get(42)
            if ev is not None:
                row["seed42_check"] = {"d_mrr": ev["got_mrr"] - ev["mrr"], "got_epoch": ev["got_epoch"],
                                       "epoch": ev["epoch"], "ok": ev["ok"]}
            d["cells"][c] = row
            note = ["lower bound (late best epoch after the doubled budget)"] if c in lower else []
            if ev is not None:
                note.append(f"seed-42 check: valid MRR {ev['got_mrr'] - ev['mrr']:+.4f} vs tune, best epoch "
                            f"{ev['got_epoch']} ({ev['epoch']} in the tune run)")
            if late:
                note.append(f"{late} seed(s) with the best epoch in the last 10%")
            if v["skipped"]:
                note.append(f"seeds without test: {v['skipped']}")
            cols = [_ms(row[k], f) for _, k, f in CELL_COLUMNS]
            if kind == "vrd":
                cols += [_ms(row["/".join(p)], f) for _, p, f in AUX_COLUMNS]
            md.append(f"| `{c}` | {len(ms)} / {len(planned)} | " + " | ".join(cols) + f" | {'; '.join(note)} |")
        md += ["", *PAIR_HEAD]
        for a, b, label in PAIRS[kind]:
            if a not in complete or b not in complete:
                if a in cells[ds] and b in cells[ds]:
                    md.append(f"| `{a}` − `{b}` | {label} | incomplete (needs seeds {list(planned)} of both) | | | | |")
                continue
            st = pair_stats({s: loaded[a]["seeds"][s]["rr"] for s in planned},
                            {s: loaded[b]["seeds"][s]["rr"] for s in planned}, n_boot)
            d["pairs"][f"{a} - {b}"] = st
            md.append(_pair_row(label, a, b, st, lower_bound_note(a, b, lower)))
        if kind == "mmkg" and {"fixed", "no_depth"} <= complete:
            import torch
            mask = torch.load(os.path.join(data_root, ds, "ent_img_mask.pt"), weights_only=True).numpy()
            g = image_groups(ref[2], mask)
            md += ["", f"### Image groups (ends of the triple with an image; no verdict below {MIN_GROUP} triples)",
                   "", *PAIR_HEAD]
            for k, name in GROUP_NAMES.items():
                sel = g == k
                if not sel.any():
                    continue
                st = pair_stats({s: loaded["fixed"]["seeds"][s]["rr"][sel] for s in planned},
                                {s: loaded["no_depth"]["seeds"][s]["rr"][sel] for s in planned}, n_boot)
                if sel.sum() < MIN_GROUP:
                    st["verdict"] = "—"
                mrr = {c: float(np.mean([loaded[c]["seeds"][s]["rr"][sel].mean() for s in planned]))
                       for c in ("fixed", "no_depth")}
                d["groups"][name] = {"n": int(sel.sum()), "mrr": mrr, "pair": st}
                md.append(_pair_row(f"group '{name}' ({int(sel.sum())} triples; MRR `fixed` {mrr['fixed']:.4f}, "
                                    f"`no_depth` {mrr['no_depth']:.4f})", "fixed", "no_depth", st))
            if all("rr_mask" in loaded[c]["seeds"][s] for c in ("fixed", "no_depth") for s in planned):
                st = pair_stats({s: loaded["fixed"]["seeds"][s]["rr_mask"] for s in planned},
                                {s: loaded["no_depth"]["seeds"][s]["rr_mask"] for s in planned}, n_boot)
                d["masked"] = st
                clean = "clean control" if st["verdict"] != "yes" and st["verdict"] != "yes (negative)" else (
                    "NOT clean: read the full-test Δdepth against this")
                md += ["", "### Every image masked at test (spec 6.4)", "", *PAIR_HEAD,
                       _pair_row(f"masked: {clean}", "fixed", "no_depth", st)]
        md.append("")
        data[ds] = d
    md += _claims(data, datasets, kinds)
    return "\n".join(md) + "\n", data


def _claims(data, datasets, kinds):
    verdict_of = lambda ds, a, b: (data.get(ds, {}).get("pairs", {}).get(f"{a} - {b}") or {}).get("verdict")  # noqa
    have = lambda ds, a, b: verdict_of(ds, a, b) is not None  # noqa: E731
    lines = []
    if any(kinds.get(d) == "mmkg" for d in datasets):
        lines.append(read_c2({d: verdict_of(d, "fixed", "no_depth") for d in MMKG_SETS
                              if have(d, "fixed", "no_depth")}))
    if any(kinds.get(d) == "vrd" for d in datasets):
        lines.append(read_c3({d: {p: verdict_of(d, *p) for p in DEPTH_STATS_PAIRS + VECTOR_PAIRS}
                              for d in VRD_SETS if all(have(d, *p) for p in DEPTH_STATS_PAIRS + VECTOR_PAIRS)}))
        lines.append(read_c4({d: verdict_of(d, "box_grid9", "box") for d in VRD_SETS
                              if have(d, "box_grid9", "box")}))
        means = {}
        for d in VRD_SETS:
            p = data.get(d, {}).get("pairs", {})
            if all(k in p for k in ("boxnorm - no_depth", "depth_stats_only - boxnorm", "depth_stats_only - no_depth")):
                gain = p["depth_stats_only - no_depth"]["mean"]
                share = float("nan") if abs(gain) < EPS else p["boxnorm - no_depth"]["mean"] / gain
                means[d] = (p["boxnorm - no_depth"]["mean"], p["depth_stats_only - boxnorm"]["mean"], share)
        lines.append(read_e2b(means))
    if not lines:
        return []
    return ["## Claims (pre-registered rules, spec 6.2)", ""] + [f"- **{s}**: {t}" for s, t in lines] + [""]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="local copy of geometry-e3/ (<root>/<dataset>/<cell>/)")
    p.add_argument("--data-root", required=True, help="folder with <dataset>/ent_img_mask.pt")
    p.add_argument("--datasets", nargs="+", default=None, help="default: every dataset folder under --root")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", default=None, help="output prefix (.md and .json)")
    a = p.parse_args(argv)
    datasets = a.datasets or [d for d in KINDS if os.path.isdir(os.path.join(a.root, d))]
    md, data = summarize(a.root, a.data_root, datasets, n_boot=a.n_boot)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    if a.out:
        with open(a.out + ".md", "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
