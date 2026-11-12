"""
Build the cleaned train/valid/test splits of an MMKG / MMSG dataset (RESEARCH_PLAN.md, P0).

Steps, in this order:
  0. (optional) Rebuild the triples from the raw source instead of the stored splits:
       --vrd-review TSV : Spatial-VRD from the sg_dataset JSON, relation labels and
                          head/tail inversion taken from relation_review.tsv
                          (built by build_vrd_relation_review.py).
       --vg-h5 / --vg-dicts : Spatial-VG from VG-SGG.h5 (VG150-curated), using the
                          official image-level splits 0/1/2 and real predicate names.
     Relations with fewer than --min-rel-count triples are dropped.
  1. (optional) --valid-group image-prefix: pool train+valid and re-split BY IMAGE
     (entity names "<image_id>_<obj_idx>") so valid is inductive like test.
  2. Remove exact duplicate triples inside every split.
  3. Remove leakage: valid triples also in train or test, test triples also in train.

entity2id.txt is never changed, so ent_*_emb.pt stay valid; they are hard-linked
into --dst (no extra disk space) when possible.

Outputs in --dst:
  train/valid/test.txt (count header + "s 	 p 	 o"), *2id.txt (OpenKE "h 	 t 	 r"),
  entity2id.txt, entity2text.txt, relation2id.txt, relation_category.tsv, clean_stats.json

--dst must differ from --src. Commands used for the current data (from Code/),
writing to data/_build/ and then moving the files into data/<name>/:
  python scripts/clean_splits.py --src data/Spatial-VRD --dst data/_build/Spatial-VRD       --vrd-review data/Spatial-VRD/relation_review.tsv --valid-group image-prefix --valid-ratio 0.1
  python scripts/clean_splits.py --src data/Spatial-VG --dst data/_build/Spatial-VG       --vg-h5 data/Spatial-VG/VG-SGG.h5 --vg-dicts data/Spatial-VG/VG-SGG-dicts.json
  python scripts/clean_splits.py --src data/_archive_v1/Merged-MMKG --dst data/_build/Merged-MMKG
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

SPLITS = ["train", "valid", "test"]

# ---------------------------------------------------------------------------
# 1. Grouping entities by image
# ---------------------------------------------------------------------------
class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def group_triples(triples, mode, ent2id):
    """Return a list of groups; each group is a list of triples sharing no entity with other groups."""
    dsu = DSU()
    if mode == "image-prefix":
        for s, _, o in triples:
            img_s, img_o = s.rsplit("_", 1)[0], o.rsplit("_", 1)[0]
            dsu.union(("img", img_s), s)
            dsu.union(("img", img_o), o)
            dsu.union(s, o)  # safety: a cross-image triple must not be split
    else:
        raise ValueError(mode)

    groups = defaultdict(list)
    for t in triples:
        groups[dsu.find(t[0])].append(t)
    return [groups[k] for k in sorted(groups, key=str)]


def regroup_valid(train, valid, mode, ratio, seed, ent2id):
    pool = train + valid
    groups = group_triples(pool, mode, ent2id)
    random.Random(seed).shuffle(groups)
    target = int(round(len(pool) * ratio))
    new_valid, new_train = [], []
    for g in groups:
        (new_valid if len(new_valid) < target else new_train).extend(g)
    return new_train, new_valid, len(groups)


# ---------------------------------------------------------------------------
# 0. Rebuilding triples from the raw sources
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vrd_from_review(json_dir, review_tsv, ent2id):
    """VRD: raw sg_dataset JSON + reviewed label table. Valid is left empty (re-grouped later)."""
    audit = _load_module("audit_vrd_relations")
    table = {}
    with open(review_tsv, encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            lab = p[ci["proposed_label"]].strip()
            table[p[ci["raw_relationship"]]] = (None if lab in ("", "<DROP>") else lab,
                                                p[ci["inverse"]].strip() == "1")
    out, missing, dropped, inverted = {}, 0, 0, 0
    for sp, fn in (("train", "sg_train_annotations.json"), ("test", "sg_test_annotations.json")):
        triples = []
        for s, raw, o, _ in audit.rebuild(os.path.join(json_dir, fn), ent2id):
            if raw not in table:
                missing += 1
                continue
            lab, inv = table[raw]
            if lab is None:
                dropped += 1
                continue
            if inv:
                s, o = o, s
                inverted += 1
            triples.append((s, lab, o))
        out[sp] = triples
    out["valid"] = []
    return out, {"source": "vrd-json+review", "review": os.path.normpath(review_tsv),
                 "dropped_by_review": dropped, "inverted": inverted, "raw_missing_in_review": missing}


def load_vg_from_h5(h5_path, dicts_path, ent2id):
    """VG: official image-level splits of VG-SGG.h5 (0 train, 1 valid, 2 test), predicate names from dicts."""
    import h5py
    with open(dicts_path, encoding="utf-8") as f:
        d = json.load(f)
    i2p, i2l = d["idx_to_predicate"], d["idx_to_label"]
    h = h5py.File(h5_path, "r")
    rel, pred, split = h["relationships"][:], h["predicates"][:, 0], h["split"][:]
    fr, lr, fb, lb = (h[k][:] for k in ("img_to_first_rel", "img_to_last_rel", "img_to_first_box", "img_to_last_box"))
    labels = h["labels"][:, 0]
    code = {0: "train", 1: "valid", 2: "test"}
    out = {sp: [] for sp in SPLITS}
    for i in range(len(split)):
        if fb[i] < 0 or fr[i] < 0 or lr[i] < 0:
            continue
        for r in range(fr[i], lr[i] + 1):
            s, o = str(rel[r, 0]), str(rel[r, 1])
            if s in ent2id and o in ent2id:
                out[code[int(split[i])]].append((s, i2p[str(pred[r])], o))
    entity_text = {str(b): i2l[str(int(labels[b]))] for b in range(len(labels))}
    return out, entity_text, {"source": "vg-h5", "h5": os.path.normpath(h5_path),
                              "dicts": os.path.normpath(dicts_path), "images": int(len(split))}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
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
        next(f)
        out = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                out.append(tuple(parts))
        return out


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


def dedupe(triples):
    seen, out = set(), []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def entity_stats(splits):
    ents = {sp: {e for s, _, o in splits[sp] for e in (s, o)} for sp in SPLITS}
    tr = ents["train"]
    return {
        sp: {
            "triples": len(splits[sp]),
            "entities": len(ents[sp]),
            "relations": len({p for _, p, _ in splits[sp]}),
            "entities_unseen_in_train_pct": round(100 * len(ents[sp] - tr) / max(1, len(ents[sp])), 2),
        }
        for sp in SPLITS
    }


def link_or_skip(src, dst):
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError:
        return "skipped (reuse from source dir)"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-rel-count", type=int, default=10,
                    help="drop relations with fewer total triples (only with --vrd-review / --vg-h5)")
    ap.add_argument("--valid-group", choices=["none", "image-prefix"], default="none")
    ap.add_argument("--valid-ratio", type=float, default=None,
                    help="share of train+valid that goes to valid (default: keep the original share)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vrd-review", default=None,
                    help="relation_review.tsv: rebuild VRD triples from the raw sg_dataset JSON with these labels")
    ap.add_argument("--vrd-json-dir", default=None, help="dir with sg_{train,test}_annotations.json (default: --src)")
    ap.add_argument("--vg-h5", default=None, help="VG-SGG.h5: rebuild VG triples with the official image-level splits")
    ap.add_argument("--vg-dicts", default=None, help="VG-SGG-dicts.json matching --vg-h5 (predicate / object names)")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--dst must differ from --src (never overwrite the original data)")

    ent2id = read_id_map(os.path.join(args.src, "entity2id.txt"))
    rel2id_src = read_id_map(os.path.join(args.src, "relation2id.txt"))
    splits = {sp: read_triples(os.path.join(args.src, f"{sp}.txt")) for sp in SPLITS}
    stats = {"source": os.path.normpath(args.src), "seed": args.seed, "before": entity_stats(splits)}
    stats["before"]["duplicates_within_split"] = {sp: len(v) - len(set(v)) for sp, v in splits.items()}

    # ---- 0. rebuild triples from the raw source (replaces the stored splits) ----
    entity_text = None
    relabelled = False
    if args.vrd_review:
        splits, info = load_vrd_from_review(args.vrd_json_dir or args.src, args.vrd_review, ent2id)
        stats["rebuilt_from"] = info
        relabelled = True
    elif args.vg_h5:
        splits, entity_text, info = load_vg_from_h5(args.vg_h5, args.vg_dicts, ent2id)
        stats["rebuilt_from"] = info
        relabelled = True
    if relabelled:
        cnt = Counter(p for sp in SPLITS for _, p, _ in splits[sp])
        keep = {r for r, n in cnt.items() if n >= args.min_rel_count}
        dropped = {sp: sum(1 for _, p, _ in splits[sp] if p not in keep) for sp in SPLITS}
        splits = {sp: [t for t in v if t[1] in keep] for sp, v in splits.items()}
        stats["min_rel_count_filter"] = {"min_rel_count": args.min_rel_count,
                                         "dropped_relations": sorted(set(cnt) - keep),
                                         "dropped_triples": dropped}

    # ---- 2. valid re-grouping by image ----
    if args.valid_group != "none":
        ratio = args.valid_ratio
        if ratio is None:
            ratio = len(splits["valid"]) / (len(splits["train"]) + len(splits["valid"])) or 0.1
        splits["train"], splits["valid"], n_groups = regroup_valid(
            splits["train"], splits["valid"], args.valid_group, ratio, args.seed, ent2id)
        stats["valid_regroup"] = {"mode": args.valid_group, "ratio": round(ratio, 4), "groups": n_groups}

    # ---- 3. dedupe within split ----
    removed_dup = {}
    for sp in SPLITS:
        before = len(splits[sp])
        splits[sp] = dedupe(splits[sp])
        removed_dup[sp] = before - len(splits[sp])
    stats["removed_duplicates"] = removed_dup

    # ---- 4. remove cross-split leakage ----
    tr, te = set(splits["train"]), set(splits["test"])
    n_v = len(splits["valid"])
    splits["valid"] = [t for t in splits["valid"] if t not in tr and t not in te]
    n_t = len(splits["test"])
    splits["test"] = [t for t in splits["test"] if t not in tr]
    stats["removed_leakage"] = {"valid_in_train_or_test": n_v - len(splits["valid"]),
                                "test_in_train": n_t - len(splits["test"])}

    # ---- relation ids ----
    rel_cnt = Counter(p for sp in SPLITS for _, p, _ in splits[sp])
    if not relabelled:
        rel2id = rel2id_src
    else:
        rel2id = {r: i for i, (r, _) in enumerate(sorted(rel_cnt.items(), key=lambda x: (-x[1], x[0])))}

    # ---- sanity checks ----
    for sp in SPLITS:
        assert len(splits[sp]) == len(set(splits[sp])), f"{sp}: duplicates left"
        assert all(s in ent2id and o in ent2id and p in rel2id for s, p, o in splits[sp]), f"{sp}: unknown id"
    sets = {sp: set(v) for sp, v in splits.items()}
    assert not (sets["train"] & sets["valid"]) and not (sets["train"] & sets["test"]) and not (sets["valid"] & sets["test"])
    train_rels = {p for _, p, _ in splits["train"]}
    stats["relations_missing_from_train"] = {
        sp: sorted({p for _, p, _ in splits[sp]} - train_rels) for sp in ("valid", "test")}

    # ---- write ----
    os.makedirs(args.dst, exist_ok=True)
    for fn in ["entity2id.txt", "entity2text.txt"]:
        if os.path.exists(os.path.join(args.src, fn)):
            shutil.copyfile(os.path.join(args.src, fn), os.path.join(args.dst, fn))
    with open(os.path.join(args.dst, "relation2id.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(rel2id)}\n")
        for r, i in sorted(rel2id.items(), key=lambda x: x[1]):
            f.write(f"{r}\t{i}\n")
    for sp in SPLITS:
        write_triples(os.path.join(args.dst, f"{sp}.txt"), splits[sp])
        write_openke(os.path.join(args.dst, f"{sp}2id.txt"), splits[sp], ent2id, rel2id)
    if entity_text is not None:
        with open(os.path.join(args.dst, "entity2text.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{len(entity_text)}\n")
            for e, t in entity_text.items():
                f.write(f"{e}\t{t}\n")
    if relabelled:
        cat = _load_module("build_vrd_relation_review").CATEGORY
        with open(os.path.join(args.dst, "relation_category.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("relation\tid\tcategory\tcount\n")
            for r, i in sorted(rel2id.items(), key=lambda x: x[1]):
                f.write(f"{r}\t{i}\t{cat.get(r, 'UNKNOWN')}\t{rel_cnt[r]}\n")
    stats["features"] = {fn: link_or_skip(os.path.join(args.src, fn), os.path.join(args.dst, fn))
                         for fn in sorted(os.listdir(args.src)) if fn.endswith(".pt")}
    stats["after"] = entity_stats(splits)
    stats["final_relations"] = len(rel2id)
    stats["relation_counts"] = dict(rel_cnt.most_common())
    with open(os.path.join(args.dst, "clean_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in stats.items() if k != "relation_counts"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
