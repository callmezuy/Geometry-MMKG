"""
Build a TRANSDUCTIVE control split for a Multi-Modal Scene Graph (MMSG) dataset.

Motivation
----------
The original Spatial-VRD / Spatial-VG splits are split by image, so 100% of
test entities never appear in train (inductive setting). To separate the effect
of "graph type" (shared viewing space) from the effect of "evaluation regime"
(inductive vs transductive), we need a transductive MMSG variant in which every
entity in valid/test also appears in train.

Procedure
---------
1. Pool all triples from train/valid/test (deduplicated).
2. Shuffle with a fixed seed and greedily move a triple (h, r, t) to the
   held-out pool only if, after removal, h, t and r each still occur in the
   remaining training triples (remaining degree >= 1).
3. Split the held-out pool into valid / test.

Guarantees: every entity and relation in valid/test occurs in train.
Entity2id / relation2id and all pre-extracted .pt features stay identical, so
the feature files of the source dataset can be reused as-is.

Output (same conventions as the source datasets):
  train.txt / valid.txt / test.txt      : count header + "s \t p \t o"
  train2id.txt / valid2id.txt / test2id.txt : count header + "h \t t \t r" (OpenKE)
  entity2id.txt, relation2id.txt, entity2text.txt (copied)
  split_stats.json

Usage (from the Code/ directory):
  python scripts/make_transductive_split.py --src data/Spatial-VRD --dst data/Spatial-VRD-Trans
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter

SPLITS = ["train", "valid", "test"]
COPY_FILES = ["entity2id.txt", "relation2id.txt", "entity2text.txt"]


def read_id_map(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = int(parts[1])
    return m


def read_triples(path):
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # count header
        return [tuple(l.rstrip("\n").split("\t")) for l in f if l.strip()]


def write_triples(path, triples):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(triples)}\n")
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def write_openke(path, triples, ent2id, rel2id):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(triples)}\n")
        for s, p, o in triples:
            f.write(f"{ent2id[s]}\t{ent2id[o]}\t{rel2id[p]}\n")


def make_split(triples, valid_ratio, test_ratio, seed):
    ent_deg = Counter()
    rel_cnt = Counter()
    for s, p, o in triples:
        ent_deg[s] += 1
        ent_deg[o] += 1
        rel_cnt[p] += 1

    target_holdout = int(round(len(triples) * (valid_ratio + test_ratio)))
    order = list(range(len(triples)))
    random.Random(seed).shuffle(order)

    holdout_idx = set()
    for i in order:
        if len(holdout_idx) >= target_holdout:
            break
        s, p, o = triples[i]
        need_s = 2 if s == o else 1  # self-loop removes two occurrences of s
        if ent_deg[s] - need_s >= 1 and (s == o or ent_deg[o] - 1 >= 1) and rel_cnt[p] - 1 >= 1:
            holdout_idx.add(i)
            ent_deg[s] -= 1
            ent_deg[o] -= 1
            rel_cnt[p] -= 1

    train = [triples[i] for i in range(len(triples)) if i not in holdout_idx]
    holdout = [triples[i] for i in order if i in holdout_idx]  # keep shuffled order
    n_valid = int(round(len(holdout) * valid_ratio / (valid_ratio + test_ratio)))
    return train, holdout[:n_valid], holdout[n_valid:]


def summarize(name, split_triples, train_ents, orig_test_set):
    ents = {e for s, _, o in split_triples for e in (s, o)}
    return {
        "triples": len(split_triples),
        "entities": len(ents),
        "relations": len({p for _, p, _ in split_triples}),
        "entities_unseen_in_train": len(ents - train_ents),
        "from_original_test": sum(1 for t in split_triples if t in orig_test_set),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Source dataset dir (e.g. data/Spatial-VRD)")
    ap.add_argument("--dst", required=True, help="Output dir (e.g. data/Spatial-VRD-Trans)")
    ap.add_argument("--valid-ratio", type=float, default=0.08)
    ap.add_argument("--test-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ent2id = read_id_map(os.path.join(args.src, "entity2id.txt"))
    rel2id = read_id_map(os.path.join(args.src, "relation2id.txt"))

    per_split = {sp: read_triples(os.path.join(args.src, f"{sp}.txt")) for sp in SPLITS}
    pooled, seen = [], set()
    n_dup = 0
    for sp in SPLITS:
        for t in per_split[sp]:
            if t in seen:
                n_dup += 1
                continue
            seen.add(t)
            pooled.append(t)
    # deterministic base order before seeded shuffle
    pooled.sort()

    train, valid, test = make_split(pooled, args.valid_ratio, args.test_ratio, args.seed)

    # ---- sanity checks ----
    train_ents = {e for s, _, o in train for e in (s, o)}
    train_rels = {p for _, p, _ in train}
    for name, sp in [("valid", valid), ("test", test)]:
        assert all(s in train_ents and o in train_ents for s, _, o in sp), f"{name}: unseen entity"
        assert all(p in train_rels for _, p, _ in sp), f"{name}: unseen relation"
    assert len(train) + len(valid) + len(test) == len(pooled)
    assert not (set(train) & set(valid)) and not (set(train) & set(test)) and not (set(valid) & set(test))

    # ---- write ----
    os.makedirs(args.dst, exist_ok=True)
    for fn in COPY_FILES:
        src_fn = os.path.join(args.src, fn)
        if os.path.exists(src_fn):
            shutil.copyfile(src_fn, os.path.join(args.dst, fn))
    for sp, data in [("train", train), ("valid", valid), ("test", test)]:
        write_triples(os.path.join(args.dst, f"{sp}.txt"), data)
        write_openke(os.path.join(args.dst, f"{sp}2id.txt"), data, ent2id, rel2id)

    # hard-link feature tensors (entity2id unchanged -> same rows); no extra disk space
    features = {}
    for fn in sorted(os.listdir(args.src)):
        if fn.endswith(".pt"):
            dst_fn = os.path.join(args.dst, fn)
            try:
                if not os.path.exists(dst_fn):
                    os.link(os.path.join(args.src, fn), dst_fn)
                features[fn] = "hardlinked"
            except OSError:
                features[fn] = "skipped (reuse from source dir)"

    orig_test = set(per_split["test"])
    ent_deg_all = Counter(e for s, _, o in pooled for e in (s, o))
    stats = {
        "source": os.path.normpath(args.src),
        "seed": args.seed,
        "target_ratios": {"valid": args.valid_ratio, "test": args.test_ratio},
        "pooled_triples": len(pooled),
        "duplicates_removed": n_dup,
        "entities_in_triples": len(ent_deg_all),
        "entities_degree_1": sum(1 for v in ent_deg_all.values() if v == 1),
        "achieved_ratios": {
            "valid": round(len(valid) / len(pooled), 4),
            "test": round(len(test) / len(pooled), 4),
        },
        "train": summarize("train", train, train_ents, orig_test),
        "valid": summarize("valid", valid, train_ents, orig_test),
        "test": summarize("test", test, train_ents, orig_test),
        "test_relation_counts": dict(Counter(p for _, p, _ in test).most_common()),
        "features": features,
    }
    with open(os.path.join(args.dst, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in stats.items() if k != "test_relation_counts"}, indent=2, ensure_ascii=False))
    print(f"Test relations covered: {len(stats['test_relation_counts'])} / {len(train_rels)} train relations")


if __name__ == "__main__":
    main()
