"""
E1 (RESEARCH_PLAN P6.0; docs/superpowers/specs/2026-11-25-e1-mmsg-aux-metrics-design.md): the 2 x 2 depth split
rerun with the auxiliary metrics on the scene graphs, valid only.

E1_CONFIGS holds the configuration chosen for each cell in P2.10 (one --tune run each, seed 42); the notebook runs
them into <e1 root>/<variant>/<config>/. This script prints one markdown table per dataset: main valid MRR (and its
drift from the P2.10 run of the same configuration), in-image ranking and relation prediction at the best epoch.
  python scripts/e1_summary.py --dataset Spatial-VRD --root results/geometry-e1/Spatial-VRD \
      --reference results/geometry-tune/Spatial-VRD [--out e1_Spatial-VRD.md]
"""
import argparse
import glob
import json
import os
import sys

# (variant, lr, dropout, epochs; None = the dataset's epoch budget), from RESEARCH_PLAN P2.10 (25/11/2026)
E1_CONFIGS = {
    "Spatial-VRD": [("no_depth", 3e-4, 0.0, None), ("depth_map_only", 1e-3, 0.0, None),
                    ("depth_stats_only", 1e-4, 0.0, None), ("fixed", 1e-3, 0.0, None)],
    "Spatial-VRD-Trans": [("no_depth", 1e-3, 0.0, None), ("depth_map_only", 1e-3, 0.0, None),
                          ("depth_stats_only", 3e-4, 0.0, 200), ("fixed", 3e-4, 0.0, 200)],
}
MAX_DRIFT = 0.002   # rerun vs P2.10 valid MRR; beyond this (or another best epoch) the numbers are not used as is


def config_dir(lr, dropout, epochs):
    """Folder of a configuration, the same string as scripts/train.py config_tag()."""
    return f"lr{lr:g}-do{dropout:g}" + (f"-ep{epochs}" if epochs else "")


def _load(root, variant, cfg):
    if root is None:
        return None
    paths = sorted(glob.glob(os.path.join(root, variant, cfg, "metrics_*.json")))
    if not paths:
        return None
    with open(paths[0], encoding="utf-8") as f:
        return json.load(f)


def _f(x, fmt=".4f"):
    return "–" if x is None else format(x, fmt)


def summarize(dataset, e1_root, reference_root=None):
    head = ("| Variant | Config | Best epoch | Valid MRR | Δ vs P2.10 | P2.10 best epoch | In-image MRR | H@1 | H@3 | "
            "Image-first | In-image MRR w/o known | H@1 w/o known | Known beats truth | Relation MRR | H@1 | H@5 | "
            "Cand | Cand (filtered) | Random MRR |")
    n_cols = head.count("|") - 1
    lines = [f"### E1 `{dataset}` (valid, seed 42)", "", head, "|" + "---|" * n_cols]
    for variant, lr, dropout, epochs in E1_CONFIGS[dataset]:
        cfg = config_dir(lr, dropout, epochs)
        m = _load(e1_root, variant, cfg)
        if m is None:
            lines.append(f"| `{variant}` | {cfg} | missing |" + " |" * (n_cols - 3))
            continue
        b = m["best_valid"] or {}
        ii, rel = b.get("in_image") or {}, b.get("relation") or {}
        ex = ii.get("excl_known") or {}
        ref = _load(reference_root, variant, cfg)
        ref_mrr = (ref["best_valid"] or {}).get("mrr") if ref else None
        drift = None if ref_mrr is None else b["mrr"] - ref_mrr
        flag = ref is not None and (abs(drift) > MAX_DRIFT or ref["best_epoch"] != m["best_epoch"])
        lines.append(
            f"| `{variant}` | {cfg} | {m['best_epoch']} | {_f(b.get('mrr'))} | "
            f"{'–' if drift is None else format(drift, '+.4f')}{' ⚠' if flag else ''} | "
            f"{ref['best_epoch'] if ref else '–'} | {_f(ii.get('mrr'))} | {_f(ii.get('hits@1'))} | "
            f"{_f(ii.get('hits@3'))} | {_f(ii.get('image_first'), '.3f')} | {_f(ex.get('mrr'))} | "
            f"{_f(ex.get('hits@1'))} | {_f(ii.get('known_beats_truth'), '.3f')} | {_f(rel.get('mrr'))} | "
            f"{_f(rel.get('hits@1'))} | {_f(rel.get('hits@5'))} | {_f(ii.get('n_cand'), '.1f')} | "
            f"{_f(ii.get('n_cand_filtered'), '.1f')} | {_f(ii.get('mrr_random'), '.3f')} |")
    if reference_root is not None:
        lines += ["", f"⚠ = valid MRR drifts more than {MAX_DRIFT} from P2.10, or another best epoch: find the "
                      "cause before using the numbers."]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(E1_CONFIGS))
    p.add_argument("--root", required=True, help="E1 runs: <root>/<variant>/<config>/metrics_*.json")
    p.add_argument("--reference", default=None, help="P2.10 tune runs of the same dataset, same layout")
    p.add_argument("--out", default=None, help="also write the table to this markdown file")
    args = p.parse_args(argv)
    table = summarize(args.dataset, args.root, args.reference)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Δ / ⚠ on a Windows console
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
