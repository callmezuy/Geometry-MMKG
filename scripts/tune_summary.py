"""
Summary of the P2.10 grid of one dataset (RESEARCH_PLAN.md P2.10): the best configuration per variant by valid MRR,
and the convergence rule (if the best epoch of the chosen configuration lies in the last 10% of the epoch budget,
rerun it with twice the budget).

Input: a folder <root>/<variant>/<config>/metrics_*.json written by `scripts/train.py --tune` (only --tune runs
are read; test numbers never exist there).
  python scripts/tune_summary.py --root results/geometry-tune/Spatial-VRD --out tune_Spatial-VRD
writes <out>.md (table of every run, then the choice per variant) and <out>.json (the choice).
"""
import argparse
import glob
import json
import os

LATE_FRACTION = 0.9   # best epoch > 90% of the budget -> not converged


def load_runs(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "metrics_*.json"))):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if not m.get("tune") or "test" in m:
            continue
        b = m["best_valid"] or {}
        runs.append({
            "variant": m["variant"], "config": m["config"], "seed": m["seed"], "options_tag": m["options_tag"],
            "epochs": m["epoch_budget"]["epochs"], "best_epoch": m["best_epoch"],
            "valid_mrr": b.get("mrr"), "valid_h1": b.get("hits@1"), "valid_h10": b.get("hits@10"),
            "valid_mr": b.get("mr"), "code_commit": m.get("code_commit"), "file": os.path.relpath(p, root),
        })
    return runs


def choose(runs):
    """Best run per variant (highest valid MRR; ties: lower lr, then lower dropout) and its convergence flag."""
    out = {}
    for v in sorted({r["variant"] for r in runs}):
        rs = [r for r in runs if r["variant"] == v and r["valid_mrr"] is not None]
        if not rs:
            continue
        best = sorted(rs, key=lambda r: (-r["valid_mrr"], r["config"]["lr"], r["config"]["dropout"]))[0]
        late = best["best_epoch"] > LATE_FRACTION * best["epochs"]
        tags = {r["options_tag"] for r in rs}
        out[v] = {**best, "n_configs": len(rs), "late_best_epoch": late,
                  "action": f"rerun with --epochs {2 * best['epochs']}" if late else "converged",
                  "mixed_options_tags": sorted(tags) if len(tags) > 1 else None}
    return out


def to_markdown(runs, choice, root):
    lines = [f"# P2.10 grid: `{root}`", "", "| Variant | lr | dropout | best epoch / budget | valid MRR | H@1 | H@10 | MR |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["variant"], r["config"]["lr"], r["config"]["dropout"])):
        mark = " **(chosen)**" if choice.get(r["variant"], {}).get("file") == r["file"] else ""
        lines.append(f"| {r['variant']}{mark} | {r['config']['lr']:g} | {r['config']['dropout']:g} | "
                     f"{r['best_epoch']} / {r['epochs']} | {r['valid_mrr']:.4f} | {r['valid_h1']:.3f} | "
                     f"{r['valid_h10']:.3f} | {r['valid_mr']:.0f} |")
    lines += ["", "| Variant | configs | chosen lr / dropout | valid MRR | convergence |", "|---|---|---|---|---|"]
    for v, c in choice.items():
        lines.append(f"| {v} | {c['n_configs']} | {c['config']['lr']:g} / {c['config']['dropout']:g} | "
                     f"{c['valid_mrr']:.4f} | {c['action']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.md and .json are added)")
    a = ap.parse_args()
    runs = load_runs(a.root)
    choice = choose(runs)
    md = to_markdown(runs, choice, a.root)
    with open(a.out + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"root": a.root, "n_runs": len(runs), "choice": choice}, f, indent=2)
    print(md)


if __name__ == "__main__":
    main()
