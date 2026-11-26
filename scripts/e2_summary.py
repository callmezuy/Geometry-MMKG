"""
E2 (RESEARCH_PLAN P6.0, C4; docs/superpowers/specs/2026-11-26-e2-box-control-design.md): box control on the scene
graphs, valid only. One markdown table per dataset with four cells:
  no_depth, depth_stats_only (grid9) : the P2.10 choice, rerun with the auxiliary metrics in E1 (geometry-e1);
  box, box_grid9                     : depth_stats_only fed the scripts/box_geometry.py tables, tuned like P2.10
                                       (geometry-e2/<dataset>/<arm>/depth_stats_only/<config>/), chosen by
                                       scripts/tune_summary.py;
then the rows box_grid9 − box (what depth adds beyond the box) and box − depth_stats_only.
With --e2b (E2b, docs/superpowers/specs/2026-11-26-e2b-shared-scale-design.md, C2 / C3): one more cell, boxnorm =
depth_stats_only fed grid9 min-max normalised per box (scripts/depth_stats_boxnorm.py,
geometry-e2b/<dataset>/boxnorm/depth_stats_only/<config>/), the rows boxnorm − no_depth and
depth_stats_only − boxnorm, and the share of the depth_stats gain that boxnorm keeps.
  python scripts/e2_summary.py --dataset Spatial-VRD --e1 results/geometry-e1/Spatial-VRD \
      --e2 results/geometry-e2/Spatial-VRD [--e2b results/geometry-e2b/Spatial-VRD] [--out e2_Spatial-VRD.md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_summary as ts  # noqa: E402
from e1_summary import E1_CONFIGS, _load, config_dir  # noqa: E402

ARMS = ("box", "box_grid9")
E2B_ARM = "boxnorm"
E1_CELLS = ("no_depth", "depth_stats_only")
VARIANT = "depth_stats_only"   # the model of both arms
# (column, getter on best_valid, format)
COLUMNS = [
    ("Valid MRR", lambda b: b.get("mrr"), ".4f"),
    ("In-image MRR", lambda b: (b.get("in_image") or {}).get("mrr"), ".4f"),
    ("In-image H@1", lambda b: (b.get("in_image") or {}).get("hits@1"), ".4f"),
    ("Image-first", lambda b: (b.get("in_image") or {}).get("image_first"), ".3f"),
    ("Relation MRR", lambda b: (b.get("relation") or {}).get("mrr"), ".4f"),
    ("Relation H@1", lambda b: (b.get("relation") or {}).get("hits@1"), ".4f"),
]


def _arm(e2_root, arm):
    """(config string, metrics dict, note) of the configuration chosen for one arm, or None."""
    root = os.path.join(e2_root, arm)
    runs = ts.load_runs(root) if os.path.isdir(root) else []
    choice = ts.choose(runs).get(VARIANT)
    if choice is None:
        return None
    with open(os.path.join(root, choice["file"]), encoding="utf-8") as f:
        m = json.load(f)
    note = [] if choice["final"] else ["edge / convergence points left"]
    if choice["late_best_epoch"]:
        note.append(choice["action"])
    cfg = os.path.basename(os.path.dirname(os.path.join(root, choice["file"])))
    return cfg, m, "; ".join(note + [f"{choice['n_configs']} runs"])


def _fmt(x, fmt):
    return "–" if x is None else format(x, fmt)


def summarize(dataset, e1_root, e2_root, e2b_root=None):
    head = "| Cell | Config | Best epoch | " + " | ".join(c[0] for c in COLUMNS) + " | Note |"
    n_cols = head.count("|") - 1
    lines = [f"### E2 `{dataset}` (valid, seed 42)", "", head, "|" + "---|" * n_cols]
    values = {}
    cells = [(v, config_dir(lr, do, ep), _load(e1_root, v, config_dir(lr, do, ep)), "E1 (P2.10 choice)")
             for v, lr, do, ep in E1_CONFIGS[dataset] if v in E1_CELLS]
    arms = [(e2_root, arm) for arm in ARMS] + ([(e2b_root, E2B_ARM)] if e2b_root else [])
    for root, arm in arms:
        got = _arm(root, arm)
        cells.append((arm, *(got if got else (None, None, None))))
    for name, cfg, m, note in cells:
        if m is None:
            lines.append(f"| `{name}` | {cfg or '–'} | missing |" + " |" * (n_cols - 3))
            continue
        b = m.get("best_valid") or {}
        values[name] = [get(b) for _, get, _ in COLUMNS]
        lines.append(f"| `{name}` | {cfg} | {m['best_epoch']} | "
                     + " | ".join(_fmt(get(b), fmt) for _, get, fmt in COLUMNS) + f" | {note} |")
    pairs = [("box_grid9", "box"), ("box", "depth_stats_only")]
    if e2b_root:
        pairs += [(E2B_ARM, "no_depth"), ("depth_stats_only", E2B_ARM)]
    for a, b in pairs:
        if a in values and b in values:
            diffs = [None if x is None or y is None else x - y for x, y in zip(values[a], values[b])]
            txt = " | ".join("–" if d is None else format(d, "+.4f") for d in diffs)
            lines.append(f"| `{a}` − `{b}` | | | {txt} | |")
    if e2b_root and all(k in values for k in ("no_depth", "depth_stats_only", E2B_ARM)):
        nd, dso, bn = (values[k][0] for k in ("no_depth", "depth_stats_only", E2B_ARM))
        share = "–" if abs(dso - nd) < 1e-12 else format((bn - nd) / (dso - nd), ".2f")
        lines += ["", f"Share of the depth_stats gain kept by `{E2B_ARM}` (valid MRR): {share}"]
    lines += ["", "Differences below about 0.003 are within the noise of one seed (P2.10); conclusions wait for E3."]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(E1_CONFIGS))
    p.add_argument("--e1", required=True, help="E1 runs: <e1>/<variant>/<config>/metrics_*.json")
    p.add_argument("--e2", required=True, help="E2 runs: <e2>/<arm>/depth_stats_only/<config>/metrics_*.json")
    p.add_argument("--e2b", default=None, help="E2b runs: <e2b>/boxnorm/depth_stats_only/<config>/metrics_*.json")
    p.add_argument("--out", default=None, help="also write the table to this markdown file")
    a = p.parse_args(argv)
    table = summarize(a.dataset, a.e1, a.e2, a.e2b)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(table)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
