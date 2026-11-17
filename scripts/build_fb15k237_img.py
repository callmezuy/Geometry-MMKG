"""
Build data/FB15k-237-IMG/ (RESEARCH_PLAN.md P0.10, data v4).

Sources (downloaded in P0.10a, under data/raw/fb15k237/):
  kgbert/{train,dev,test}.tsv        standard FB15k-237 split (Toutanova & Chen, 2015), as shipped by KG-BERT
                                     (byte-identical to baselines/MKGformer/MKG/dataset/FB15k-237/)
  kgbert/{entities,relations}.txt    entity / relation lists (their order defines entity2id / relation2id)
  kgbert/entity2text.txt             short names; entity2textlong.txt: long descriptions; relation2text.txt
  fb15k_wn18_images.zip              image store released by MKGformer (Drive id 197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5-),
                                     FB15k-images/m.xxx/{bing,google}_k.jpg; MKGformer maps /m/xxx -> m.xxx

The split is kept exactly as published (no filtering, no re-split): dev.tsv becomes valid.txt.

Images: every image is fully decoded (PIL, RGB); unreadable ones are left out of the index. The store also holds
placeholders that show nothing of the entity (blank / 1x1 images, "image no longer available" notices, broken-image
icons, default thumbnails of MTV, Vimeo, timeanddate.com, ...). They are kept in image_index.tsv but flagged, so a
placeholder-only entity counts as an entity without image. An image is a placeholder when
  placeholder_hash     its md5 is in scripts/fb15k237_image_placeholders.tsv (reviewed by eye, see its header),
  placeholder_near_dup its 64-bit dHash is within PHASH_MAX bits of a listed exemplar with RGB std >= NEAR_DUP_MIN_STD
                       (re-encoded copies; at <= 2 bits every match in the review was a placeholder, from 3 bits real
                       photos appear; sparser exemplars, e.g. an icon on white, also matched a chemical structure),
  placeholder_uniform  it is nearly one colour (max RGB-channel std < UNIFORM_STD).
Colour entities (tails of the */colors relations: Maroon, Navy Blue, ...) are exempt from the blank and uniform
rules, since a solid swatch depicts them faithfully.

Output (same conventions as the other data/<name>/ folders):
  train.txt / valid.txt / test.txt            count header + "h \t r \t t"
  train2id.txt / valid2id.txt / test2id.txt   count header + "h_id \t t_id \t r_id" (OpenKE)
  entity2id.txt, relation2id.txt              count header + "name \t id"
  entity2text.txt                             entity \t short name (KG-BERT)
  entity2textlong.txt                         entity \t long description; short name when KG-BERT has none
  relation2text.txt                           relation \t text (KG-BERT)
  image_index.tsv                             entity \t zip member \t status (ok | placeholder_*), readable images only
  image_placeholders.tsv                      copy of the reviewed list used for this build
  clean_stats.json                            counts, image coverage (all readable / without placeholders),
                                              images per entity, placeholder counts, shared hashes

Usage (from Code/):
  python scripts/build_fb15k237_img.py --raw data/raw/fb15k237 --dst data/FB15k-237-IMG
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool

from PIL import Image, ImageStat

SPLITS = [("train", "train.tsv"), ("valid", "dev.tsv"), ("test", "test.tsv")]
EXPECTED = {"train": 272115, "valid": 17535, "test": 20466}
N_ENT, N_REL = 14541, 237
IMG_PREFIX = "FB15k-images/"
PLACEHOLDER_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fb15k237_image_placeholders.tsv")
PHASH_MAX = 2
UNIFORM_STD = 3.0  # 3-5 still holds real dark or sparse images (MKG-Y album cover, map tile); reviewed by hash instead
NEAR_DUP_MIN_STD = 10.0  # low-content exemplars (icon on white) have dHash ~0 and match any sparse image

_zip = None


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\r\n") for l in f if l.strip()]


def read_text_map(path):
    m = {}
    for l in read_lines(path):
        k, _, v = l.partition("\t")
        assert k not in m, f"duplicate key {k!r} in {path}"
        m[k] = v.strip()
    return m


def write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if header is not None:
            f.write(f"{header}\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")


def _init(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path)


def dhash(im):
    """64-bit difference hash: sign of horizontal gradients on a 9x8 grayscale thumbnail."""
    p = list(im.convert("L").resize((9, 8), Image.LANCZOS).getdata())
    return sum(1 << i for i in range(64) if p[(i // 8) * 9 + i % 8] > p[(i // 8) * 9 + i % 8 + 1])


def _check(member):
    data = _zip.read(member)
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            rgb = im.convert("RGB")
            rgb.load()
        dh, std, err = dhash(rgb), max(ImageStat.Stat(rgb).stddev), None
    except Exception as e:  # noqa: BLE001 - any decode failure makes the image unusable
        w = h = dh = 0
        std, err = 0.0, re.sub(r" at 0x[0-9A-Fa-f]+", "", f"{type(e).__name__}: {e}")
    return member, hashlib.md5(data).hexdigest(), w, h, dh, std, err


def read_placeholders(path):
    rows = []
    for l in read_lines(path):
        if l.startswith("#") or l.startswith("md5\t"):
            continue
        md5, kind, _, example, note = l.split("\t")
        rows.append({"md5": md5, "kind": kind, "example": example, "note": note})
    assert len({r["md5"] for r in rows}) == len(rows), "duplicate md5 in placeholder list"
    return rows


def describe(xs):
    xs = sorted(xs)
    return {"min": xs[0], "median": statistics.median(xs), "mean": round(statistics.mean(xs), 2), "max": xs[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/fb15k237")
    ap.add_argument("--dst", default="data/FB15k-237-IMG")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    a = ap.parse_args()
    kg = os.path.join(a.raw, "kgbert")
    zip_path = os.path.join(a.raw, "fb15k_wn18_images.zip")
    os.makedirs(a.dst, exist_ok=True)
    stats = {"sources": {"split_and_text": "KG-BERT (data/raw/fb15k237/kgbert)",
                         "images": "MKGformer image store (fb15k_wn18_images.zip, FB15k-images/)"}}

    # ---- entities / relations ----
    entities = read_lines(os.path.join(kg, "entities.txt"))
    relations = read_lines(os.path.join(kg, "relations.txt"))
    assert len(entities) == len(set(entities)) == N_ENT, len(entities)
    assert len(relations) == len(set(relations)) == N_REL, len(relations)
    e2id = {e: i for i, e in enumerate(entities)}
    r2id = {r: i for i, r in enumerate(relations)}

    # ---- splits ----
    triples = {}
    for name, fname in SPLITS:
        rows = [tuple(l.split("\t")) for l in read_lines(os.path.join(kg, fname))]
        assert all(len(t) == 3 for t in rows)
        assert len(rows) == EXPECTED[name], (name, len(rows), EXPECTED[name])
        unknown = [t for t in rows if t[0] not in e2id or t[2] not in e2id or t[1] not in r2id]
        assert not unknown, (name, unknown[:3])
        triples[name] = rows
        write(os.path.join(a.dst, f"{name}.txt"), len(rows), rows)
        write(os.path.join(a.dst, f"{name}2id.txt"), len(rows), [(e2id[h], e2id[t], r2id[r]) for h, r, t in rows])
    write(os.path.join(a.dst, "entity2id.txt"), N_ENT, [(e, i) for e, i in e2id.items()])
    write(os.path.join(a.dst, "relation2id.txt"), N_REL, [(r, i) for r, i in r2id.items()])

    train_ents = {x for h, _, t in triples["train"] for x in (h, t)}
    train_rels = {r for _, r, _ in triples["train"]}
    split_stats = {}
    for name, rows in triples.items():
        ents = {x for h, _, t in rows for x in (h, t)}
        split_stats[name] = {
            "triples": len(rows),
            "duplicates": len(rows) - len(set(rows)),
            "entities": len(ents),
            "relations": len({r for _, r, _ in rows}),
            "entities_unseen_in_train": len(ents - train_ents),
            "relations_unseen_in_train": len({r for _, r, _ in rows} - train_rels),
        }
    split_stats["valid_test_triples_in_train"] = {
        n: len(set(triples[n]) & set(triples["train"])) for n in ("valid", "test")}
    split_stats["entities_in_no_split"] = len(set(entities) - {x for rows in triples.values() for h, _, t in rows for x in (h, t)})
    stats["splits"] = split_stats

    # ---- text ----
    short = read_text_map(os.path.join(kg, "entity2text.txt"))
    long_ = read_text_map(os.path.join(kg, "entity2textlong.txt"))
    rel_text = read_text_map(os.path.join(kg, "relation2text.txt"))
    assert all(e in short and short[e] for e in entities), "entity without a short name"
    assert all(r in rel_text for r in relations), "relation without text"
    long_missing = [e for e in entities if not long_.get(e)]
    write(os.path.join(a.dst, "entity2text.txt"), None, [(e, short[e]) for e in entities])
    write(os.path.join(a.dst, "entity2textlong.txt"), None, [(e, long_.get(e) or short[e]) for e in entities])
    write(os.path.join(a.dst, "relation2text.txt"), None, [(r, rel_text[r]) for r in relations])
    stats["text"] = {
        "short_name": sum(bool(short.get(e)) for e in entities),
        "long_description": N_ENT - len(long_missing),
        "long_description_filled_with_short_name": len(long_missing),
        "long_description_filled_ids": long_missing,
        "kgbert_text_entities_outside_fb15k237": {"short": len(set(short) - set(entities)),
                                                  "long": len(set(long_) - set(entities))},
    }

    # ---- images ----
    zf = zipfile.ZipFile(zip_path)
    by_dir = defaultdict(list)
    for info in zf.infolist():
        if info.filename.startswith(IMG_PREFIX) and not info.is_dir():
            d = info.filename[len(IMG_PREFIX):].split("/")[0]
            by_dir[d].append(info.filename)
    zf.close()
    ent_of_dir = {e[1:].replace("/", "."): e for e in entities}  # /m/01rng -> m.01rng (MKGformer convention)
    members = sorted(m for d, ms in by_dir.items() if d in ent_of_dir for m in ms)
    print(f"{len(members):,} images in {sum(d in ent_of_dir for d in by_dir):,} FB15k-237 entity folders; decoding...", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(zip_path,)) as pool:
        results = pool.map(_check, members, chunksize=256)

    # ---- placeholders ----
    colour_ents = {t for h, r, t in triples["train"] if r.endswith("/colors")}
    ph = read_placeholders(PLACEHOLDER_LIST)
    ph_kind = {r["md5"]: r["kind"] for r in ph}
    by_member = {r[0]: r for r in results}
    exemplars = []
    for r in ph:
        if r["example"] not in by_member:
            raise SystemExit(f"placeholder example {r['example']} is not an image of an FB15k-237 entity")
        _, md5, _, _, dh, std, err = by_member[r["example"]]
        assert md5 == r["md5"] and not err, f"placeholder list out of date: {r}"
        if std >= NEAR_DUP_MIN_STD:  # sparse / uniform images have dHash ~0: exact md5 match only
            exemplars.append(dh)

    status = defaultdict(list)  # entity -> [(member, status)]
    errors, small, md5_ents, kind_count = [], [], defaultdict(set), Counter()
    for m, md5, w, h, dh, std, err in results:
        e = ent_of_dir[m[len(IMG_PREFIX):].split("/")[0]]
        if err:
            errors.append({"member": m, "error": err})
            continue
        colour = e in colour_ents
        if md5 in ph_kind and not (colour and ph_kind[md5] == "blank"):
            st = "placeholder_hash"
            kind_count[ph_kind[md5]] += 1
        elif std < UNIFORM_STD and not colour:
            st = "placeholder_uniform"
        elif any(bin(dh ^ x).count("1") <= PHASH_MAX for x in exemplars):
            st = "placeholder_near_dup"
        else:
            st = "ok"
        status[e].append((m, st))
        md5_ents[md5].add(e)
        if min(w, h) < 32:
            small.append({"member": m, "size": [w, h]})
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

    _, cov_all = coverage({e: v for e, v in status.items()})
    has_clean, cov_clean = coverage({e: [m for m, st in v if st == "ok"] for e, v in status.items()})
    lost = sorted(e for e in entities if status.get(e) and e not in has_clean)
    shared = sorted(((h, es) for h, es in md5_ents.items() if len(es) > 1), key=lambda x: -len(x[1]))
    st_count = Counter(st for v in status.values() for _, st in v)
    stats["images"] = {
        "store_fb15k_images_total": sum(len(v) for v in by_dir.values()),
        "store_entity_folders_total": len(by_dir),
        "store_folders_not_in_fb15k237": sum(d not in ent_of_dir for d in by_dir),
        "images_considered": len(members),
        "images_readable": len(members) - len(errors),
        "images_unreadable": len(errors),
        "unreadable": errors[:100],
        "images_min_side_lt_32px": len(small),
        "placeholder_rules": {"reviewed_md5": len(ph), "near_dup_exemplars": len(exemplars),
                              "phash_max_bits": PHASH_MAX, "near_dup_exemplar_min_rgb_std": NEAR_DUP_MIN_STD,
                              "uniform_max_rgb_std": UNIFORM_STD,
                              "colour_entities_exempt": len(colour_ents)},
        "image_status": dict(sorted(st_count.items())),
        "placeholder_hash_by_kind": dict(sorted(kind_count.items())),
        "coverage_all_readable": cov_all,
        "coverage_without_placeholders": cov_clean,
        "entities_only_placeholders": len(lost),
        "entities_only_placeholders_ids": lost,
        "entities_without_image_ids": sorted(e for e in entities if e not in has_clean),
        "image_hashes_shared_by_several_entities": len(shared),
        "images_in_shared_hashes": sum(len(es) for _, es in shared),
        "top_shared_hashes": [{"md5": h, "n_entities": len(es), "placeholder": ph_kind.get(h), "examples": sorted(es)[:5]}
                              for h, es in shared[:20]],
    }
    with open(os.path.join(a.dst, "clean_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    drop = ("unreadable", "entities_without_image_ids", "entities_only_placeholders_ids", "top_shared_hashes")
    brief = {k: v for k, v in stats["images"].items() if k not in drop}
    for k in ("coverage_all_readable", "coverage_without_placeholders"):
        brief[k] = {kk: vv for kk, vv in brief[k].items() if kk != "images_per_entity_histogram"}
    print(json.dumps({"splits": stats["splits"], "text": {k: v for k, v in stats["text"].items() if k != "long_description_filled_ids"},
                      "images": brief}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
