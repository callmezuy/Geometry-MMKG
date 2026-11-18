"""
Build data/FB15K-237-ind/ (RESEARCH_PLAN.md P0.2, data v0): the inductive FB15k-237 split of BLP
(Daza et al., WWW 2021), used with images and descriptions by CMR (Zhao et al., SIGIR 2024).

Sources:
  data/raw/fb15k237_ind/blp/FB15k-237.tar.gz   BLP data release, Zenodo record 4501273
                                               (md5 c7cf03efea81b5fbef958f6ce3bfc198; the surfdrive link in the BLP
                                               README is dead). ind-{train,dev,test}.tsv + {train,dev,test}-ents.txt;
                                               byte-identical to OreOZhao/CMR data/FB15k237_ind/{train,valid,test}.txt
  data/FB15k-237-IMG/                          built first by scripts/build_fb15k237_img.py; same 14,541 entities and
                                               237 relations, so entity2id / relation2id, texts and the image index are
                                               copied unchanged and the features can be shared by both datasets

Checks (CMR Table 1): train 11,633 entities / 215,082 triples, valid 1,454 / 42,164, test 1,454 / 52,870,
237 relations; train / valid / test entity sets pairwise disjoint; the three splits together are exactly the
310,116 triples of the standard FB15k-237 split.

Evaluation note: valid / test triples mix new and known entities (a test triple may also mention valid entities).
BLP and CMR both rank test answers against all 14,541 entities, filtered with the triples of all three splits;
for valid BLP ranks against train + valid entities, CMR against all entities.

Output (same conventions as the other data/<name>/ folders):
  train.txt / valid.txt / test.txt, *2id.txt, entity2id.txt, relation2id.txt   (as FB15k-237-IMG)
  entity2text.txt, entity2textlong.txt, relation2text.txt, image_index.tsv,
  image_placeholders.tsv                                                       (copied from FB15k-237-IMG)
  train_entities.txt / valid_entities.txt / test_entities.txt                 BLP entity split (one MID per line)
  clean_stats.json

Usage (from Code/):
  python scripts/build_fb15k237_ind.py --raw data/raw/fb15k237_ind --img data/FB15k-237-IMG --dst data/FB15K-237-ind
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile

from build_fb15k237_img import read_lines, write

TAR_MD5 = "c7cf03efea81b5fbef958f6ce3bfc198"
SPLITS = [("train", "ind-train.tsv", "train-ents.txt"),
          ("valid", "ind-dev.tsv", "dev-ents.txt"),
          ("test", "ind-test.tsv", "test-ents.txt")]
TABLE1 = {"train": (11633, 215082), "valid": (1454, 42164), "test": (1454, 52870)}
COPY = ["entity2id.txt", "relation2id.txt", "entity2text.txt", "entity2textlong.txt", "relation2text.txt",
        "image_index.tsv", "image_placeholders.tsv"]


def read_tar_member(tar, name):
    return tar.extractfile(f"FB15k-237/{name}").read().decode("utf-8").splitlines()


def read_counted(path):
    rows = [l.split("\t") for l in read_lines(path)]
    assert int(rows[0][0]) == len(rows) - 1, path
    return rows[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/fb15k237_ind")
    ap.add_argument("--img", default="data/FB15k-237-IMG")
    ap.add_argument("--dst", default="data/FB15K-237-ind")
    a = ap.parse_args()
    tar_path = os.path.join(a.raw, "blp", "FB15k-237.tar.gz")
    with open(tar_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    assert md5 == TAR_MD5, f"{tar_path}: md5 {md5} != Zenodo {TAR_MD5}"
    os.makedirs(a.dst, exist_ok=True)

    e2id = {e: int(i) for e, i in read_counted(os.path.join(a.img, "entity2id.txt"))}
    r2id = {r: int(i) for r, i in read_counted(os.path.join(a.img, "relation2id.txt"))}
    std = set()
    for s in ("train", "valid", "test"):
        std |= {tuple(x) for x in read_counted(os.path.join(a.img, f"{s}.txt"))}

    triples, ents = {}, {}
    with tarfile.open(tar_path) as tar:
        for name, tfile, efile in SPLITS:
            rows = [tuple(l.split("\t")) for l in read_tar_member(tar, tfile) if l.strip()]
            assert all(len(t) == 3 for t in rows)
            ents[name] = [l.strip() for l in read_tar_member(tar, efile) if l.strip()]
            triples[name] = rows

    stats = {"source": {"split": "BLP (Daza et al., 2021), Zenodo 4501273, FB15k-237.tar.gz", "tar_md5": md5,
                        "entities_relations_text_images": a.img}}
    checks, split_stats = {}, {}
    seen = set()
    for name, _, _ in SPLITS:
        rows, es = triples[name], set(ents[name])
        assert len(es) == len(ents[name]), f"duplicate entity in {name} entity list"
        tri_ents = {x for h, _, t in rows for x in (h, t)}
        unknown = [t for t in rows if t[0] not in e2id or t[2] not in e2id or t[1] not in r2id]
        assert not unknown, (name, unknown[:3])
        checks[f"{name}_table1"] = (len(es), len(rows)) == TABLE1[name]
        split_stats[name] = {
            "entities_in_split_list": len(es),
            "triples": len(rows),
            "duplicates": len(rows) - len(set(rows)),
            "relations": len({r for _, r, _ in rows}),
            "entities_in_triples": len(tri_ents),
            "entities_in_triples_from_train": len(tri_ents & set(ents["train"])),
            "entities_in_triples_from_valid": len(tri_ents & set(ents["valid"])),
            "entities_in_triples_from_test": len(tri_ents & set(ents["test"])),
        }
        if name != "train":  # "new" = entity of this split, unseen in train (and, for test, in valid)
            split_stats[name].update({
                "triples_with_new_head_only": sum(h in es and t not in es for h, _, t in rows),
                "triples_with_new_tail_only": sum(t in es and h not in es for h, _, t in rows),
                "triples_with_both_new": sum(h in es and t in es for h, _, t in rows),
                "triples_with_no_new": sum(h not in es and t not in es for h, _, t in rows),
            })
        assert tri_ents <= seen | es, f"{name} triples mention entities of a later split"
        seen |= es
    tr, va, te = (set(ents[n]) for n in ("train", "valid", "test"))
    checks["entity_sets_disjoint"] = not (tr & va or tr & te or va & te)
    checks["entities_equal_fb15k237"] = (tr | va | te) == set(e2id)
    checks["train_triples_only_train_entities"] = split_stats["train"]["entities_in_triples"] == len(tr) and \
        split_stats["train"]["entities_in_triples_from_train"] == len(tr)
    checks["valid_test_relations_seen_in_train"] = {r for n in ("valid", "test") for _, r, _ in triples[n]} <= \
        {r for _, r, _ in triples["train"]}
    checks["relations_total"] = len({r for rows in triples.values() for _, r, _ in rows}) == len(r2id) == 237
    union = set(triples["train"]) | set(triples["valid"]) | set(triples["test"])
    checks["union_equals_standard_fb15k237"] = union == std and sum(len(v) for v in triples.values()) == len(std)
    checks["no_duplicates"] = all(v["duplicates"] == 0 for v in split_stats.values())
    failed = [k for k, v in checks.items() if not v]
    assert not failed, f"checks failed: {failed}"
    stats["checks"] = checks
    stats["splits"] = split_stats

    for name, rows in triples.items():
        write(os.path.join(a.dst, f"{name}.txt"), len(rows), rows)
        write(os.path.join(a.dst, f"{name}2id.txt"), len(rows), [(e2id[h], e2id[t], r2id[r]) for h, r, t in rows])
        write(os.path.join(a.dst, f"{name}_entities.txt"), None, [(e,) for e in ents[name]])
    for f in COPY:
        shutil.copyfile(os.path.join(a.img, f), os.path.join(a.dst, f))

    # image coverage (status == ok: placeholders count as missing, see build_fb15k237_img.py)
    has = {e for e, _, st in (l.split("\t") for l in read_lines(os.path.join(a.img, "image_index.tsv"))) if st == "ok"}
    cov = {}
    for name, rows in triples.items():
        es = set(ents[name])
        cov[name] = {
            "split_entities_with_image": len(es & has),
            "split_entities_with_image_pct": round(100 * len(es & has) / len(es), 2),
            "triples_both_with_image": sum(h in has and t in has for h, _, t in rows),
            "triples_one_with_image": sum((h in has) != (t in has) for h, _, t in rows),
            "triples_none_with_image": sum(h not in has and t not in has for h, _, t in rows),
        }
    stats["image_coverage"] = cov
    stats["evaluation_note"] = ("test answers ranked against all 14,541 entities (BLP and CMR), filtered with the "
                                "triples of train + valid + test; test triples may also mention valid entities")
    with open(os.path.join(a.dst, "clean_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
