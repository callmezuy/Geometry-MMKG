"""
Convert the strict ranks CMR writes (evaluate.py --rank-file, baselines/patches/CMR.diff: tab-separated
head_id, relation, tail_id, rank_head, rank_tail with CMR's entity ids and relation strings) to the shared format of
scripts/compute_metrics.py (CSV h_id,r_id,t_id,rank_head,rank_tail with our ids), and check that the rows are
exactly our test (or valid) split, in order (RESEARCH_PLAN.md P3.7).

  python scripts/cmr_ranks.py --cmr <checkout> --task FB15k237_ind --data data/FB15K-237-ind \
      --ranks <checkout>/ranks_cmr_test.tsv --out results/cmr/ranks_cmr_FB15K-237-ind_test.csv [--split test]
"""
import argparse
import csv
import json
import os


def read_count_file(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if len(rows) != n:
        raise ValueError(f"{path}: header says {n} rows, read {len(rows)}")
    return rows


def convert(cmr, task, data, ranks, out, split="test"):
    ent = {e: int(i) for e, i in read_count_file(os.path.join(data, "entity2id.txt"))}
    rel = {r: int(i) for r, i in read_count_file(os.path.join(data, "relation2id.txt"))}
    with open(os.path.join(cmr, "data", task, "relations.json"), encoding="utf-8") as f:
        to_cmr = json.load(f)                                    # our raw relation -> CMR relation string
    back = {}
    for raw, s in to_cmr.items():
        if s in back and back[s] != raw:
            raise ValueError(f"two relations share the CMR string {s!r}")
        back[s] = raw
    with open(os.path.join(cmr, "data", task, f"{split}.txt.json"), encoding="utf-8") as f:
        expected = [(e["head_id"], e["relation"], e["tail_id"]) for e in json.load(f)]
    rows = []
    with open(ranks, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["head_id", "relation", "tail_id", "rank_head", "rank_tail"]:
            raise ValueError(f"{ranks}: unexpected header {header}")
        for line in f:
            h, r, t, rh, rt = line.rstrip("\n").split("\t")
            rows.append((h, r, t, int(rh), int(rt)))
    if [(h, r, t) for h, r, t, _, _ in rows] != expected:
        raise ValueError(f"{ranks}: rows are not the {split} split of data/{task} in order")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["h_id", "r_id", "t_id", "rank_head", "rank_tail"])
        for h, r, t, rh, rt in rows:
            w.writerow([ent[h], rel[back[r]], ent[t], rh, rt])
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmr", required=True)
    ap.add_argument("--task", required=True, choices=["FB15k237_ind", "SpatialVRD_ind"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    a = ap.parse_args(argv)
    n = convert(a.cmr, a.task, a.data, a.ranks, a.out, a.split)
    print(f"{n} rows -> {a.out}")


if __name__ == "__main__":
    main()
