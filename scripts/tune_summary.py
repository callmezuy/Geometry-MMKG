"""
Summary of the P2.10 grid of one dataset (RESEARCH_PLAN.md P2.10): the best configuration per variant by valid MRR,
the convergence rule (if the best epoch of the chosen configuration lies in the last 10% of the epoch budget,
rerun it with twice the budget) and the edge rule (decided 20/11/2026): if the chosen configuration lies at an edge
of the tested range of an axis, the point one step further along that axis (other axis at its chosen value) is added,
until the choice is inside, at most EXTRA_STEPS steps beyond the base grid in each direction.

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
LR_LADDER = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
DROPOUT_LADDER = [0.0, 0.1, 0.3, 0.5]
BASE_GRID = {"lr": [1e-4, 3e-4, 1e-3], "dropout": [0.1, 0.3]}
EXTRA_STEPS = 2


def _index(ladder, x):
    return min(range(len(ladder)), key=lambda i: abs(ladder[i] - x) / max(abs(x), 1e-12))


def edge_points(tested, chosen):
    """Points to add for one variant under the edge rule. tested: list of (lr, dropout); chosen: (lr, dropout)."""
    out = []
    for axis, ladder, k in (("lr", LR_LADDER, 0), ("dropout", DROPOUT_LADDER, 1)):
        vals = sorted({t[k] for t in tested})
        base = [_index(ladder, v) for v in BASE_GRID[axis]]
        i = _index(ladder, chosen[k])
        for edge, step, limit in ((vals[0], -1, min(base) - EXTRA_STEPS), (vals[-1], +1, max(base) + EXTRA_STEPS)):
            if abs(chosen[k] - edge) > 1e-12 * max(1.0, abs(edge)):   # chosen value not at this edge
                continue
            j = i + step
            if j < 0 or j >= len(ladder) or (step < 0 and j < limit) or (step > 0 and j > limit):
                continue
            p = (ladder[j], chosen[1]) if k == 0 else (chosen[0], ladder[j])
            if not any(abs(p[0] - t[0]) < 1e-12 and abs(p[1] - t[1]) < 1e-12 for t in tested):
                out.append({"lr": p[0], "dropout": p[1], "axis": axis, "edge": "low" if step < 0 else "high"})
    return out


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
    """Best run per variant (highest valid MRR; ties: lower lr, then lower dropout, then fewer epochs), its
    convergence flag and the points still to run: the convergence rerun (same configuration, twice the budget; once,
    as decided) and the edge-rule points. A variant is final when nothing is left."""
    out = {}
    for v in sorted({r["variant"] for r in runs}):
        rs = [r for r in runs if r["variant"] == v and r["valid_mrr"] is not None]
        if not rs:
            continue
        best = sorted(rs, key=lambda r: (-r["valid_mrr"], r["config"]["lr"], r["config"]["dropout"],
                                         r["epochs"]))[0]
        late = best["best_epoch"] > LATE_FRACTION * best["epochs"]
        same = [r for r in rs if r["config"] == best["config"]]
        budget = min(r["epochs"] for r in same)
        doubled = any(r["epochs"] >= 2 * budget for r in same)
        rerun = None
        if late and not doubled:
            rerun = {"lr": best["config"]["lr"], "dropout": best["config"]["dropout"], "epochs": 2 * best["epochs"]}
        tags = {r["options_tag"] for r in rs}
        ext = edge_points([(r["config"]["lr"], r["config"]["dropout"]) for r in rs],
                          (best["config"]["lr"], best["config"]["dropout"]))
        action = ("converged" if not late else f"rerun with --epochs {rerun['epochs']}" if rerun
                  else "late best epoch after the doubled budget (rule applied once; report it)")
        out[v] = {**best, "n_configs": len(rs), "late_best_epoch": late, "action": action, "rerun": rerun,
                  "edge_points": ext, "final": not ext and rerun is None,
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
    lines += ["", "| Variant | configs | chosen lr / dropout | valid MRR | convergence | points to add (edge rule) |",
              "|---|---|---|---|---|---|"]
    for v, c in choice.items():
        ext = ", ".join(f"lr {p['lr']:g} / do {p['dropout']:g}" for p in c["edge_points"]) or "none (final)"
        lines.append(f"| {v} | {c['n_configs']} | {c['config']['lr']:g} / {c['config']['dropout']:g} | "
                     f"{c['valid_mrr']:.4f} | {c['action']} | {ext} |")
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
