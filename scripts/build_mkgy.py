"""
Build data/MKG-Y/ (RESEARCH_PLAN.md P0.12, data v4): MKG-Y of MMRNS (Xu et al., ACM MM 2022) with the official
images and descriptions of the MMRNS Google Drive (folder 1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj).

Sources (data/raw/mmrns_official/):
  MKG-Y/{train,valid,test}.txt, MKG-Y/ent_links   split by name (head \t relation \t tail); identical to
                                                  MKG-W-Y-raw_data/MKG-Y/ on the same Drive, and, mapped through
                                                  entity2id / relation2id, identical (same order) to
                                                  baselines/NativE/benchmarks/MKG-Y/ and baselines/SNAG/.../MKG-Y/
  MKG-Y_img.zip                                   DY_open/<entity>/<files>
  MKG-Y_ent2descreption.txt                       DBpedia URI \t "description"@en

entity2id / relation2id are taken from the NativE benchmark (the ids every baseline was published with), so author
features (baselines/SNAG/SNAG_MKGC/embeddings/MKG-Y-*.pth) keep their row order.

Image folders -> entities. Folder names were written on Windows, which drops \\ / : * ? " < > | and trailing dots
and spaces; some also lost "..." or turned "/" or "*" into "_". A folder is matched in three tiers, each accepted
only when it yields exactly one entity that is not matched yet:
  exact        folder name == entity name
  windows      equal after removing \\/:*?"<>| and trailing ". "
  alnum        equal after keeping only letters and digits
(The zip carries the UTF-8 flag, so non-ASCII names decode correctly; no U+FFFD masking is needed.)

Images: every file except Thumbs.db is decoded with PIL (RGB, first frame); undecodable files are left out.
Byte-identical copies inside one entity folder are kept once. Placeholders are flagged with the same rules as
FB15k-237-IMG (see scripts/build_fb15k237_img.py), using the reviewed list scripts/mkgy_image_placeholders.tsv.

Descriptions: entity2textlong.txt = MMRNS description (N-Triples quotes, @en and trailing " ." removed, escaped
quotes restored; characters already lost as U+FFFD in the source are kept and counted); entities without one get
their name. entity2text.txt = readable name (underscores -> spaces).

Output: train/valid/test.txt, *2id.txt, entity2id.txt, relation2id.txt, entity2text.txt, entity2textlong.txt,
image_index.tsv (entity \t zip member \t status), image_placeholders.tsv, image_folder_match.tsv, clean_stats.json

Usage (from Code/):
  python scripts/build_mkgy.py --raw data/raw/mmrns_official --bench baselines/NativE/benchmarks/MKG-Y --dst data/MKG-Y
"""
import argparse
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool

from PIL import Image, ImageStat

from build_fb15k237_img import describe, dhash, read_lines, read_placeholders, write

EXPECTED = {"train": 21310, "valid": 2665, "test": 2663}
N_ENT, N_REL = 15000, 28
PLACEHOLDER_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mkgy_image_placeholders.tsv")
PHASH_MAX, NEAR_DUP_MIN_STD, UNIFORM_STD = 2, 10.0, 3.0  # same rules as build_fb15k237_img.py
IMG_ROOT = "DY_open/"
Image.MAX_IMAGE_PIXELS = 200_000_000  # a few scans are large; above this PIL refuses (counted as unreadable)

_zip = None


def _init(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path)


def _check(member):
    import hashlib
    import io
    data = _zip.read(member)
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            rgb = im.convert("RGB")
            rgb.load()
        dh, std, err = dhash(rgb), max(ImageStat.Stat(rgb).stddev), None
    except Exception as e:  # noqa: BLE001 - any decode failure makes the file unusable
        w = h = dh = 0
        std, err = 0.0, re.sub(r" at 0x[0-9A-Fa-f]+", "", f"{type(e).__name__}: {e}")[:200]
    return member, hashlib.md5(data).hexdigest(), w, h, dh, std, err


def read_id_file(path):
    rows = [re.split(r"\s+", l.strip()) for l in read_lines(path)]
    assert int(rows[0][0]) == len(rows) - 1, path
    return rows[1:]


def win_key(n):
    return re.sub(r'[\\/:*?"<>|]', "", n).rstrip(". ")


def alnum_key(n):
    return "".join(c for c in n if c.isalnum())


def clean_desc(s):
    s = s.strip()
    if s.endswith(" ."):  # N-Triples style: "text"@en .
        s = s[:-2].rstrip()
    if s.endswith("@en"):
        s = s[:-3]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.replace('\\"', '"').replace("\t", " ").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/mmrns_official")
    ap.add_argument("--bench", default="baselines/NativE/benchmarks/MKG-Y")
    ap.add_argument("--dst", default="data/MKG-Y")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--scan-cache", help="optional .json to store / reuse the per-image decode results")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    stats = {"source": {"split": "MMRNS Google Drive MKG-Y/{train,valid,test}.txt",
                        "ids": a.bench, "images": "MKG-Y_img.zip", "descriptions": "MKG-Y_ent2descreption.txt"}}

    # ---- ids and split ----
    e2id = {e: int(i) for e, i in read_id_file(os.path.join(a.bench, "entity2id.txt"))}
    r2id = {r: int(i) for r, i in read_id_file(os.path.join(a.bench, "relation2id.txt"))}
    assert sorted(e2id.values()) == list(range(N_ENT)) and sorted(r2id.values()) == list(range(N_REL))
    entities = sorted(e2id, key=e2id.get)
    relations = sorted(r2id, key=r2id.get)
    triples, split_stats = {}, {}
    for s in ("train", "valid", "test"):
        rows = [tuple(l.split("\t")) for l in read_lines(os.path.join(a.raw, "MKG-Y", f"{s}.txt"))]
        assert all(len(t) == 3 for t in rows) and len(rows) == EXPECTED[s], (s, len(rows))
        ids = [tuple(map(int, x)) for x in read_id_file(os.path.join(a.bench, f"{s}2id.txt"))]
        assert ids == [(e2id[h], e2id[t], r2id[r]) for h, r, t in rows], f"{s}: MMRNS split != benchmark ids"
        triples[s] = rows
        write(os.path.join(a.dst, f"{s}.txt"), len(rows), rows)
        write(os.path.join(a.dst, f"{s}2id.txt"), len(rows), ids)
    write(os.path.join(a.dst, "entity2id.txt"), N_ENT, [(e, e2id[e]) for e in entities])
    write(os.path.join(a.dst, "relation2id.txt"), N_REL, [(r, r2id[r]) for r in relations])
    links = {n for _, n in (l.split("\t") for l in read_lines(os.path.join(a.raw, "MKG-Y", "ent_links")))}
    assert links == set(e2id), "ent_links names != entity2id"
    train_ents = {x for h, _, t in triples["train"] for x in (h, t)}
    for s, rows in triples.items():
        ents = {x for h, _, t in rows for x in (h, t)}
        split_stats[s] = {"triples": len(rows), "duplicates": len(rows) - len(set(rows)), "entities": len(ents),
                          "relations": len({r for _, r, _ in rows}),
                          "entities_unseen_in_train": len(ents - train_ents)}
    split_stats["valid_test_triples_in_train"] = {
        s: len(set(triples[s]) & set(triples["train"])) for s in ("valid", "test")}
    split_stats["entities_in_no_split"] = len(set(entities) - {x for rows in triples.values() for h, _, t in rows for x in (h, t)})
    stats["splits"] = split_stats

    # ---- text ----
    desc = {}
    for l in read_lines(os.path.join(a.raw, "MKG-Y_ent2descreption.txt")):
        uri, _, d = l.partition("\t")
        name = uri.rsplit("/resource/", 1)[-1]
        assert name not in desc, name
        desc[name] = clean_desc(d)
    has_desc = [e for e in entities if desc.get(e)]
    write(os.path.join(a.dst, "entity2text.txt"), None, [(e, e.replace("_", " ")) for e in entities])
    write(os.path.join(a.dst, "entity2textlong.txt"), None, [(e, desc.get(e) or e.replace("_", " ")) for e in entities])
    stats["text"] = {"description": len(has_desc), "description_filled_with_name": N_ENT - len(has_desc),
                     "descriptions_with_u_fffd_in_source": sum("�" in desc[e] for e in has_desc),
                     "descriptions_not_in_mkgy": len(set(desc) - set(e2id))}

    # ---- folders -> entities ----
    zip_path = os.path.join(a.raw, "MKG-Y_img.zip")
    zf = zipfile.ZipFile(zip_path)
    folders = defaultdict(list)
    other = Counter()
    for info in zf.infolist():
        p = info.filename.split("/")
        if info.filename.startswith(IMG_ROOT) and len(p) == 3 and p[2]:
            if p[2].lower() == "thumbs.db":
                other["thumbs_db"] += 1
            else:
                folders[p[1]].append(info.filename)
        elif not info.is_dir():
            other["outside_entity_folders"] += 1
    zf.close()
    match, tier_count = {}, Counter()
    taken = set()
    for tier, key in (("exact", lambda n: n), ("windows", win_key), ("alnum", alnum_key)):
        k2e = defaultdict(list)
        for e in entities:
            if e not in taken:
                k2e[key(e)].append(e)
        k2f = defaultdict(list)
        for f in folders:
            if f not in match:
                k2f[key(f)].append(f)
        for k, fs in k2f.items():
            es = k2e.get(k, [])
            if len(fs) == 1 and len(es) == 1:
                match[fs[0]] = es[0]
                taken.add(es[0])
                tier_count[tier] += 1
    unmatched = sorted(set(folders) - set(match))
    write(os.path.join(a.dst, "image_folder_match.tsv"), None,
          [(f, match[f], "exact" if f == match[f] else ("windows" if win_key(f) == win_key(match[f]) else "alnum"))
           for f in sorted(match)])

    # ---- decode ----
    members = sorted(m for f in match for m in folders[f])
    if a.scan_cache and os.path.exists(a.scan_cache):
        with open(a.scan_cache, encoding="utf-8") as f:
            results = [tuple(r) for r in json.load(f)]
        assert [r[0] for r in results] == members, "scan cache out of date"
    else:
        print(f"{len(members):,} files in {len(match):,} matched folders; decoding...", flush=True)
        with Pool(a.workers, initializer=_init, initargs=(zip_path,)) as pool:
            results = pool.map(_check, members, chunksize=64)
        if a.scan_cache:
            with open(a.scan_cache, "w", encoding="utf-8") as f:
                json.dump(results, f)

    # ---- placeholders and status ----
    ph = read_placeholders(PLACEHOLDER_LIST)
    ph_kind = {r["md5"]: r["kind"] for r in ph}
    by_member = {r[0]: r for r in results}
    exemplars = []
    for r in ph:
        _, md5, _, _, dh, std, err = by_member[r["example"]]
        assert md5 == r["md5"] and not err, f"placeholder list out of date: {r}"
        if std >= NEAR_DUP_MIN_STD:
            exemplars.append(dh)
    status = defaultdict(list)
    errors, small, md5_ents, kind_count, seen = [], [], defaultdict(set), Counter(), set()
    for m, md5, w, h, dh, std, err in results:
        e = match[m.split("/")[1]]
        if err:
            errors.append({"member": m, "error": err})
            continue
        if (e, md5) in seen:
            st = "duplicate_in_entity"
        elif md5 in ph_kind:
            st = "placeholder_hash"
            kind_count[ph_kind[md5]] += 1
        elif std < UNIFORM_STD:
            st = "placeholder_uniform"
        elif any(bin(dh ^ x).count("1") <= PHASH_MAX for x in exemplars):
            st = "placeholder_near_dup"
        else:
            st = "ok"
        seen.add((e, md5))
        status[e].append((m, st))
        if st != "duplicate_in_entity":
            md5_ents[md5].add(e)
        if min(w, h) < 32:
            small.append(m)
    write(os.path.join(a.dst, "image_index.tsv"), None,
          [(e, m, st) for e in entities for m, st in sorted(status.get(e, []))])
    shutil.copyfile(PLACEHOLDER_LIST, os.path.join(a.dst, "image_placeholders.tsv"))

    def coverage(with_img):
        counts = [n for e in entities if (n := len(with_img.get(e, ())))]
        has = {e for e in entities if with_img.get(e)}
        return has, {
            "entities_with_image": len(has),
            "entities_with_image_pct": round(100 * len(has) / N_ENT, 2),
            "entities_without_image": N_ENT - len(has),
            "images_per_entity_with_image": describe(counts),
            "images_per_entity_histogram": dict(sorted(Counter(counts).items())),
            "by_split": {
                n: {"entities_with_image_pct": round(100 * len(ents & has) / len(ents), 2),
                    "triples_both_with_image": sum(h in has and t in has for h, _, t in rows),
                    "triples_one_with_image": sum((h in has) != (t in has) for h, _, t in rows),
                    "triples_none_with_image": sum(h not in has and t not in has for h, _, t in rows)}
                for n, rows in triples.items() for ents in [{x for h, _, t in rows for x in (h, t)}]},
        }

    _, cov_all = coverage({e: [m for m, st in v if st != "duplicate_in_entity"] for e, v in status.items()})
    has_clean, cov_clean = coverage({e: [m for m, st in v if st == "ok"] for e, v in status.items()})
    shared = sorted(((h, es) for h, es in md5_ents.items() if len(es) > 1), key=lambda x: -len(x[1]))
    stats["images"] = {
        "zip_files_other": dict(other),
        "folders": len(folders),
        "folders_matched": len(match),
        "folders_matched_by_tier": dict(tier_count),
        "folders_unmatched": unmatched,
        "files_considered": len(members),
        "files_readable": len(members) - len(errors),
        "files_unreadable": len(errors),
        "unreadable": errors[:200],
        "unreadable_by_extension": dict(Counter(e["member"].rsplit(".", 1)[-1].lower() if "." in e["member"].split("/")[-1]
                                                else "<none>" for e in errors).most_common()),
        "images_min_side_lt_32px": len(small),
        "placeholder_rules": {"reviewed_md5": len(ph), "near_dup_exemplars": len(exemplars),
                              "phash_max_bits": PHASH_MAX, "near_dup_exemplar_min_rgb_std": NEAR_DUP_MIN_STD,
                              "uniform_max_rgb_std": UNIFORM_STD},
        "image_status": dict(sorted(Counter(st for v in status.values() for _, st in v).items())),
        "placeholder_hash_by_kind": dict(sorted(kind_count.items())),
        "coverage_all_readable": cov_all,
        "coverage_without_placeholders": cov_clean,
        "entities_only_placeholders": sorted(e for e in entities if status.get(e) and e not in has_clean
                                             and any(st != "duplicate_in_entity" for _, st in status[e])),
        "entities_without_image_ids": sorted(e for e in entities if e not in has_clean),
        "image_hashes_shared_by_several_entities": len(shared),
        "images_in_shared_hashes": sum(len(es) for _, es in shared),
        "top_shared_hashes": [{"md5": h, "n_entities": len(es), "placeholder": ph_kind.get(h), "examples": sorted(es)[:5]}
                              for h, es in shared[:20]],
    }
    with open(os.path.join(a.dst, "clean_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    drop = ("unreadable", "entities_without_image_ids", "top_shared_hashes", "folders_unmatched", "entities_only_placeholders")
    brief = {k: v for k, v in stats["images"].items() if k not in drop}
    for k in ("coverage_all_readable", "coverage_without_placeholders"):
        brief[k] = {kk: vv for kk, vv in brief[k].items() if kk != "images_per_entity_histogram"}
    brief["folders_unmatched"] = len(unmatched)
    brief["entities_only_placeholders"] = len(stats["images"]["entities_only_placeholders"])
    print(json.dumps({"splits": stats["splits"], "text": stats["text"], "images": brief}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
