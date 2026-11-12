"""
Single source of truth for link-prediction metrics (RESEARCH_PLAN.md, section 4, P1.3).

Every model (GeometryMMKG, NativE, SNAG, MKGformer, ...) writes per-triplet ranks as CSV
    h_id,r_id,t_id,rank_head,rank_tail
(ids from the dataset's entity2id / relation2id; filtered ranks, head AND tail).
This script computes, from those files only:
  * overall MRR / MR / Hits@1,3,10 (head and tail ranks pooled, as in OpenKE) + tail / head MRR;
  * the same per relation category (relation_category.tsv: GEOMETRIC / POSSESSIVE / SEMANTIC / MISC)
    and optionally per relation;
  * with --reference NAME: paired bootstrap 95% CI of the MRR difference (model - reference)
    over the triples shared by both files (use it for delta-depth, e.g. Full vs w/o Depth).

Usage (from Code/):
  python scripts/compute_metrics.py --data data/Spatial-VRD \
      --ranks full=results/ranks_geometry_vrd_seed42_test.csv nodepth=results/ranks_geometry_nodepth_vrd_seed42_test.csv \
      --reference nodepth --by-relation --out results/vrd_metrics.md
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np


def read_ranks(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need = {"h_id", "r_id", "t_id", "rank_head", "rank_tail"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path}: expected CSV header {sorted(need)}")
    keys = np.array([(int(r["h_id"]), int(r["r_id"]), int(r["t_id"])) for r in rows], dtype=np.int64)
    rh = np.array([int(r["rank_head"]) for r in rows], dtype=np.float64)
    rt = np.array([int(r["rank_tail"]) for r in rows], dtype=np.float64)
    if (rh < 1).any() or (rt < 1).any():
        raise SystemExit(f"{path}: ranks must be >= 1")
    return keys, rh, rt


def read_relations(data_dir):
    """relation id -> (name, category or None)."""
    id2name = {}
    with open(os.path.join(data_dir, "relation2id.txt"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                id2name[int(p[1])] = p[0]
    cat = {}
    path = os.path.join(data_dir, "relation_category.tsv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                cat[int(row["id"])] = row["category"]
    return {i: (n, cat.get(i)) for i, n in id2name.items()}


def metrics(rh, rt):
    pooled = np.concatenate([rh, rt])
    if pooled.size == 0:
        return None
    return OrderedDict(
        n=len(rh), mrr=np.mean(1 / pooled), mr=np.mean(pooled),
        h1=np.mean(pooled <= 1), h3=np.mean(pooled <= 3), h10=np.mean(pooled <= 10),
        mrr_tail=np.mean(1 / rt), mrr_head=np.mean(1 / rh))


def paired_bootstrap(rr_a, rr_b, n_boot=2000, seed=0):
    """rr_*: per-triple reciprocal rank (mean of head and tail), aligned. Returns (delta, lo, hi, p)."""
    rng = np.random.default_rng(seed)
    d = rr_a - rr_b
    n = len(d)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))  # two-sided, percentile bootstrap
    return d.mean(), lo, hi, min(p, 1.0)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_m(m):
    return [m["n"], f"{m['mrr']:.4f}", f"{m['mr']:.1f}", f"{m['h1']:.4f}", f"{m['h3']:.4f}",
            f"{m['h10']:.4f}", f"{m['mrr_tail']:.4f}", f"{m['mrr_head']:.4f}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir with relation2id.txt (+ relation_category.tsv)")
    ap.add_argument("--ranks", nargs="+", required=True, help="NAME=path.csv (one per model / run)")
    ap.add_argument("--reference", default=None, help="NAME of the model the others are compared against")
    ap.add_argument("--by-relation", action="store_true", help="also report every relation")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="write the markdown report here as well")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    rel = read_relations(args.data)
    runs = OrderedDict()
    for spec in args.ranks:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--ranks expects NAME=path, got {spec!r}")
        runs[name] = read_ranks(path)
    if args.reference and args.reference not in runs:
        raise SystemExit(f"--reference {args.reference!r} is not one of {list(runs)}")

    head = ["model", "triples", "MRR", "MR", "H@1", "H@3", "H@10", "MRR tail", "MRR head"]
    parts = [f"# Link prediction metrics: `{os.path.normpath(args.data)}`\n",
             "Filtered, head + tail pooled, all entities, no type constraint.\n",
             "## Overall\n", table(head, [[n] + fmt_m(metrics(rh, rt)) for n, (_, rh, rt) in runs.items()])]

    # warn when runs were evaluated on different triple sets
    sets = {n: {tuple(k) for k in keys} for n, (keys, _, _) in runs.items()}
    first = next(iter(sets.values()))
    for n, s in sets.items():
        if s != first:
            parts.append(f"\n> ⚠️ `{n}` was evaluated on a different triple set "
                         f"({len(s)} vs {len(first)} triples); compare with care.")

    cats = sorted({c for _, c in rel.values() if c})
    if cats:
        rows = []
        for c in cats:
            ids = {i for i, (_, cc) in rel.items() if cc == c}
            for n, (keys, rh, rt) in runs.items():
                m = np.isin(keys[:, 1], list(ids))
                mm = metrics(rh[m], rt[m])
                if mm:
                    rows.append([c, n] + fmt_m(mm))
        parts += ["\n## Per relation category\n", table(["category"] + head, rows)]

    if args.by_relation:
        rows = []
        for i in sorted(rel, key=lambda i: rel[i][0]):
            for n, (keys, rh, rt) in runs.items():
                m = keys[:, 1] == i
                mm = metrics(rh[m], rt[m])
                if mm:
                    rows.append([rel[i][0], rel[i][1] or "", n] + fmt_m(mm))
        parts += ["\n## Per relation\n", table(["relation", "category"] + head, rows)]

    if args.reference:
        rk, rrh, rrt = runs[args.reference]
        ref = {tuple(k): (1 / a + 1 / b) / 2 for k, a, b in zip(rk, rrh, rrt)}
        rows = []
        groups = [("ALL", None)] + [(c, {i for i, (_, cc) in rel.items() if cc == c}) for c in cats]
        for n, (keys, rh, rt) in runs.items():
            if n == args.reference:
                continue
            mine = {tuple(k): (1 / a + 1 / b) / 2 for k, a, b in zip(keys, rh, rt)}
            shared = [k for k in mine if k in ref]
            for g, ids in groups:
                ks = [k for k in shared if ids is None or k[1] in ids]
                a = np.array([mine[k] for k in ks]); b = np.array([ref[k] for k in ks])
                d, lo, hi, p = paired_bootstrap(a, b, args.n_boot)
                rows.append([n, g, len(ks), f"{d:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]", f"{p:.3f}",
                             "yes" if lo > 0 or hi < 0 else "no"])
        parts += [f"\n## ΔMRR vs `{args.reference}` (paired bootstrap, {args.n_boot} resamples)\n",
                  "Per-triple RR = mean of head and tail reciprocal rank; only triples present in both files.\n",
                  table(["model", "group", "triples", "ΔMRR", "95% CI", "p", "CI excludes 0"], rows)]

    report = "\n".join(parts) + "\n"
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
