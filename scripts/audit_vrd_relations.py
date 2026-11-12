"""
Audit the relation labels of Spatial-VRD against the raw sg_dataset annotations.

Rebuilds every triple exactly as construct_spatial_mmsg.py did
(entity id = "<photo_id>_<obj_idx>"), applies the OLD alias heuristic from
clean_mmsg_aliases.py, and checks:
  1. that the rebuilt triples reproduce the v1 splits (data/_archive_v1/Spatial-VRD),
  2. which raw predicates the old substring heuristic mapped to which label,
     flagging mappings where the label's keyword only matched as a substring.
  3. where duplicate triples come from (raw JSON vs. alias merging).

Usage (from Code/):
  python scripts/audit_vrd_relations.py --data data/Spatial-VRD --out data/Spatial-VRD/relation_audit.tsv
"""
import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict


def load_old_mapper(code_dir):
    spec = importlib.util.spec_from_file_location(
        "old_alias", os.path.join(code_dir, "scripts", "legacy", "clean_mmsg_aliases.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_base_relation


def read_triples(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return [tuple(l.rstrip("\n").split("\t")) for l in f if l.strip()]


def rebuild(json_path, entity2id):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [{"filename": k, **v} for k, v in data.items()]
    out = []
    for img in data:
        fn = img.get("filename")
        pid = str(img.get("photo_id", os.path.splitext(fn)[0]))
        for rel in img.get("relationships", []):
            objs = rel.get("objects")
            if not objs or len(objs) != 2:
                continue
            s, o = f"{pid}_{objs[0]}", f"{pid}_{objs[1]}"
            if s in entity2id and o in entity2id:
                out.append((s, rel.get("relationship", "unknown"), o, " ".join(rel.get("text", [])[1:2])))
    return out


# Keywords the old heuristic tested with plain substring checks (`'x' in rel`), per label.
SUBSTRING_KEYS = {
    "in front of": ["front of", "infront", "in fron"],
    "near": ["near", "next to", "beside", "adjacent", "close to", "nex to", "next tp"],
    "wearing": ["wear", "has on"], "holding": ["hold", "carry", "grasp", "grab"],
    "looking at": ["look", "watch", "see"], "riding": ["ride", "rid", "driv"], "playing": ["play"],
    "eating": ["eat", "feed", "bite", "graze"], "moving": ["walk", "run", "move", "go"],
    "on": ["upon", "top of", "stand", "sit", "lie", "lay", "rest", "park"],
    "in": ["inside"],
}
# Directional words: if the raw predicate contains one of these but the label is
# a different direction, spatial information was lost.
DIRECTIONS = {
    "left": "left of", "right": "right of", "behind": "behind", "front": "in front of",
    "under": "under", "below": "under", "beneath": "under", "above": "above", "over": "above",
}


def flag(raw, label):
    r = raw.lower().strip()
    words = r.split()
    # (a) label reached only via a keyword hidden inside another word
    #     e.g. 'displaying' -> playing, 'opposite' -> on ('sit'), 'underneath' -> eating ('eat')
    keys = SUBSTRING_KEYS.get(label, [])
    hits = [k for k in keys if k in r]
    word_level = label == "on" and "on" in words or label == "in" and "in" in words
    if hits and not word_level and not any(
            any(w.startswith(k.split()[0]) or w.startswith(k[:3]) for w in words) for k in hits):
        return "SUBSTRING_BUG"
    # (b) a direction word in the raw predicate was overridden by another label
    for w, lab in DIRECTIONS.items():
        if w in words and label != lab and label in DIRECTIONS.values():
            return "DIRECTION_LOST"
        if w in words and label in ("on", "near", "moving", "eating", "playing", "holding"):
            return "DIRECTION_LOST"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Spatial-VRD")
    ap.add_argument("--out", default="data/Spatial-VRD/relation_audit.tsv")
    ap.add_argument("--stored", default="data/_archive_v1/Spatial-VRD",
                    help="dir with the ORIGINAL (v1) splits produced by the old heuristic")
    args = ap.parse_args()
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_map = load_old_mapper(code_dir)

    entity2id = {}
    with open(os.path.join(args.data, "entity2id.txt"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            entity2id[p[0]] = int(p[1])

    raw = rebuild(os.path.join(args.data, "sg_train_annotations.json"), entity2id) + \
          rebuild(os.path.join(args.data, "sg_test_annotations.json"), entity2id)

    # 1. reproduction check
    rebuilt = Counter((s, old_map(p), o) for s, p, o, _ in raw)
    stored = Counter(t for sp in ("train", "valid", "test")
                     for t in read_triples(os.path.join(args.stored, f"{sp}.txt")))
    print(f"raw relationships with known entities : {len(raw)}")
    print(f"stored triples (train+valid+test)    : {sum(stored.values())}")
    print(f"rebuilt == stored (as multisets)     : {rebuilt == stored}")
    if rebuilt != stored:
        print(f"  only in rebuilt: {sum((rebuilt - stored).values())}, only in stored: {sum((stored - rebuilt).values())}")

    # 2. mapping audit
    pair_cnt = Counter((p, old_map(p)) for _, p, _, _ in raw)
    examples = defaultdict(list)
    for _, p, _, txt in raw:
        if len(examples[p]) < 3 and txt and txt not in examples[p]:
            examples[p].append(txt)
    rows = []
    for (p, lab), n in pair_cnt.most_common():
        rows.append((p, lab, n, flag(p, lab), " | ".join(examples[p])))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("raw_relationship\told_label\tcount\tflag\traw_text_examples\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    sus = [r for r in rows if r[3]]
    print(f"\nraw predicates: {len(pair_cnt)} -> old labels: {len(set(l for _, l in pair_cnt))}")
    for kind in ("SUBSTRING_BUG", "DIRECTION_LOST"):
        sel = [r for r in sus if r[3] == kind]
        print(f"\n{kind}: {len(sel)} predicates, {sum(r[2] for r in sel)} triples")
        for r in sel[:30]:
            print(f"  {r[0]!r:34} -> {r[1]!r:14} x{r[2]}")
    merged = Counter()
    for (p, lab), n in pair_cnt.items():
        if p.lower().strip() != lab:
            merged[lab] += n
    print("\ntriples relabelled by the old heuristic, per label (top 12):")
    for lab, n in merged.most_common(12):
        print(f"  {lab!r:14} {n}")

    # 3. duplicate origin
    raw_dup = sum(n - 1 for n in Counter((s, p, o) for s, p, o, _ in raw).values() if n > 1)
    mapped_dup = sum(n - 1 for n in rebuilt.values() if n > 1)
    print(f"\nduplicates already in raw JSON          : {raw_dup}")
    print(f"duplicates after old alias merging      : {mapped_dup}  (+{mapped_dup - raw_dup} created by merging)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
