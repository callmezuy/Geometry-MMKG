"""Remove cross-source equivalent-relation leakage from Merged-MMKG valid/test (data v3.1).

Problem
-------
Merged-MMKG merges DB15K, MKG-W and MKG-Y. Entities are aligned across sources,
relations are not (they keep the "SRC::" prefix). So the same fact can sit in test
under one source's relation and in train under an equivalent or inverse relation of
another source, e.g. DB15K birthPlace (test) <-> MKG-W P19 (train), DB15K starring
<-> MKG-Y actedIn (inverse). A model can then answer by learning "P19 => birthPlace"
alone; this is the leakage that turned FB15k into FB15k-237 (Toutanova & Chen, 2015).

Rule
----
Drop a valid/test triple (h, r, t) if the unordered pair {h, t} is linked in TRAIN by
at least one relation of a DIFFERENT source than r. Train is never changed, so every
entity / relation that was seen stays seen; entity2id, relation2id and the .pt
features are untouched. Pairs linked by another relation of the SAME source are kept
(normal multi-relational structure inside one KG, as in the original benchmarks).

Usage (from Code/):
  python scripts/remove_cross_source_leakage.py --src data/Merged-MMKG --dst data/_build/Merged-MMKG
"""
import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

SPLITS = ["train", "valid", "test"]


def read_triples(path):
    with open(path, "r", encoding="utf-8") as f:
        n = int(f.readline())
        rows = [tuple(l.rstrip("\n").split("\t")) for l in f if l.strip()]
    assert len(rows) == n, (path, n, len(rows))
    return rows


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(rows)}\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def source(rel):
    return rel.split("::", 1)[0]


def short(rel):
    s, r = rel.split("::", 1)
    return f"{s}:{r.strip('<>').rsplit('/', 1)[-1].rsplit('#', 1)[-1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    text = {sp: read_triples(os.path.join(a.src, f"{sp}.txt")) for sp in SPLITS}  # s p o
    ids = {sp: read_triples(os.path.join(a.src, f"{sp}2id.txt")) for sp in SPLITS}  # h t r
    for sp in SPLITS:
        assert len(text[sp]) == len(ids[sp])

    pair_rels = defaultdict(set)
    for s, p, o in text["train"]:
        pair_rels[frozenset((s, o))].add(p)

    os.makedirs(a.dst, exist_ok=True)
    stats = {"source": os.path.normpath(a.src), "rule": "drop valid/test (h,r,t) if {h,t} is linked in train by a relation of another source"}
    for sp in SPLITS:
        keep, dropped, by_src, by_pair = [], [], Counter(), Counter()
        for i, (s, p, o) in enumerate(text[sp]):
            other = [q for q in pair_rels.get(frozenset((s, o)), ()) if source(q) != source(p)] if sp != "train" else []
            if other:
                dropped.append(i)
                by_src[source(p)] += 1
                for q in other:
                    by_pair[f"{short(p)} <- {short(q)}"] += 1
            else:
                keep.append(i)
        write_rows(os.path.join(a.dst, f"{sp}.txt"), [text[sp][i] for i in keep])
        write_rows(os.path.join(a.dst, f"{sp}2id.txt"), [ids[sp][i] for i in keep])
        stats[sp] = {"before": len(text[sp]), "after": len(keep), "removed": len(dropped),
                     "removed_by_source": dict(by_src), "top_relation_pairs": dict(by_pair.most_common(15))}
        print(f"{sp}: {len(text[sp]):,} -> {len(keep):,} (removed {len(dropped)}: {dict(by_src)})")

    for fn in sorted(os.listdir(a.src)):
        if fn in ("entity2id.txt", "relation2id.txt", "clean_stats.json", "split_stats.json"):
            shutil.copyfile(os.path.join(a.src, fn), os.path.join(a.dst, fn))
        elif fn.endswith(".pt") and not os.path.exists(os.path.join(a.dst, fn)):
            try:
                os.link(os.path.join(a.src, fn), os.path.join(a.dst, fn))
            except OSError:
                pass
    with open(os.path.join(a.dst, "leakage_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
