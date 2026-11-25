"""
Write one of our v0 datasets for the CMR code (RESEARCH_PLAN.md P3.7), at the data level only. The CMR checkout comes
from scripts/setup_baseline.py cmr. Writes, under the checkout:
  data/<Task>/{train,valid,test}.txt.json  [{head_id, head, relation, tail_id, tail}]
  data/<Task>/relations.json               {raw relation: CMR relation string}
  data/<Task>/img_entities.json            [{entity_id, entity, entity_desc, entity_imgpath}]; the file
                                           triplet.EntityDict reads (the released preprocess.py writes entities.json
                                           without image paths); every entity of our entity2id, i.e. the candidate
                                           set of section 4.2
  img_data/<Task>/...                      one image per entity
  data/<Task>/gm_prepare.json              sources, counts, sha256 of the shared missing-image picture

Tasks:
  FB15k237_ind (FB15K-237-ind; --data data/FB15K-237-ind --zip <MKGformer image zip>)
    Triples and relation strings: the authors' own data/FB15k237_ind/*.txt.json, kept as released, after checking that
    they are exactly our split (same triples in the same order). Names and descriptions: as the authors' preprocess.py
    builds them (FB15k_mid2name.txt; FB15k_mid2description.txt truncated to 50 words). Image: the first image with
    status ok in our image_index.tsv (the MKGformer images every model uses).
  SpatialVRD_ind (Spatial-VRD; --data data/Spatial-VRD --zip <Kaggle VRD zip>)
    Triples from our v0 files; entity text = entity2text.txt (the text every model gets), no description; relation
    string = the relation name. Image = the entity's box cropped from its photo, with the box rules of
    scripts/extract_features.py (clamped to the image; boxes under 2 px widened; boxes wholly outside = no image).
Entities without an image (FB15K-237-ind 249; Spatial-VRD 2 boxes outside their image) all point to ONE shared noise
picture (decided 25/11/2026): the paper "pads with random values", and one constant input is the rule for every
baseline (section 3.2), so it carries no per-entity identity.

  python scripts/prepare_cmr_data.py FB15k237_ind --cmr <checkout> --data data/FB15K-237-ind \
      --zip data/raw/fb15k237/fb15k_wn18_images.zip
  python scripts/prepare_cmr_data.py SpatialVRD_ind --cmr <checkout> --data data/Spatial-VRD \
      --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip
"""
import argparse
import hashlib
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000   # as scripts/extract_features.py
MIN_BOX = 2.0                          # as scripts/extract_features.py
NOISE_SEED, NOISE_SIZE = 0, 384        # CMR resizes every image to 384 x 384 before its ViT processor
NOISE_NAME = "_missing_noise.png"


def read_count_file(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if len(rows) != n:
        raise ValueError(f"{path}: header says {n} rows, read {len(rows)}")
    return rows


def read_2id(path):
    """OpenKE {split}2id.txt: count line, then 'h t r' per line."""
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [tuple(map(int, line.split())) for line in f if line.strip()]
    if len(rows) != n:
        raise ValueError(f"{path}: header says {n} rows, read {len(rows)}")
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_noise(img_dir):
    """The shared missing-image picture: uniform RGB noise, fixed seed, PNG (lossless, so the hash is stable)."""
    path = os.path.join(img_dir, NOISE_NAME)
    rng = np.random.default_rng(NOISE_SEED)
    Image.fromarray(rng.integers(0, 256, (NOISE_SIZE, NOISE_SIZE, 3), dtype=np.uint8), "RGB").save(path)
    return path


def rel_path(cmr, path):
    """CMR opens image paths relative to its working directory (the checkout root)."""
    return "./" + os.path.relpath(path, cmr).replace(os.sep, "/")


def truncate(text, max_len):
    # CMR preprocess.py _truncate
    return " ".join(text.split()[:max_len])


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


# ------------------------------------------------------------------------------------------- FB15K-237-ind
def prepare_fb(cmr, data, zip_path, out_dir, img_dir):
    ents = [row[0] for row in read_count_file(os.path.join(data, "entity2id.txt"))]
    splits = {}
    for split in ("train", "valid", "test"):
        ex = load_json(os.path.join(out_dir, f"{split}.txt.json"))
        with open(os.path.join(data, f"{split}.txt"), encoding="utf-8") as f:
            ours = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip() and "\t" in line]
        theirs = [(e["head_id"], e["tail_id"]) for e in ex]
        if [(h, t) for h, _, t in ours] != theirs:
            raise ValueError(f"{split}: the authors' {split}.txt.json is not our split (order or triples differ)")
        splits[split] = ex
    rel_map = load_json(os.path.join(out_dir, "relations.json"))
    for split, ex in splits.items():   # relation strings: the authors' normalisation of our raw relations
        with open(os.path.join(data, f"{split}.txt"), encoding="utf-8") as f:
            raw = [line.rstrip("\n").split("\t")[1] for line in f if line.strip() and "\t" in line]
        if any(rel_map[r] != e["relation"] for r, e in zip(raw, ex)):
            raise ValueError(f"{split}: relation strings differ from relations.json")
    # names / descriptions as preprocess.py (_load_fb15k237_wikidata, _load_fb15k237_desc, dump_all_entities)
    desc = {}
    with open(os.path.join(out_dir, "FB15k_mid2description.txt"), encoding="utf-8") as f:
        for line in f:
            fs = line.strip().split("\t")
            assert len(fs) == 2, line[:80]
            desc[fs[0]] = truncate(fs[1], 50)
    name = {}
    for ex in splits.values():
        for e in ex:
            name.setdefault(e["head_id"], e["head"])
            name.setdefault(e["tail_id"], e["tail"])
    missing_name = [e for e in ents if e not in name]
    if missing_name:
        raise ValueError(f"{len(missing_name)} entities of entity2id never occur in the CMR json, e.g. {missing_name[:3]}")
    # image: the first status-ok image of image_index.tsv
    first = {}
    with open(os.path.join(data, "image_index.tsv"), encoding="utf-8") as f:
        for line in f:
            e, member, status = line.rstrip("\n").split("\t")
            if status == "ok" and e not in first:
                first[e] = member
    paths = {}
    with zipfile.ZipFile(zip_path) as z:
        for e in ents:
            if e not in first:
                continue
            member = first[e]
            dst = os.path.join(img_dir, e.strip("/").replace("/", ".") + os.path.splitext(member)[1].lower())
            with open(dst, "wb") as f:
                f.write(z.read(member))
            paths[e] = dst
    return [dict(entity_id=e, entity=name[e], entity_desc=desc.get(e, ""), img=paths.get(e)) for e in ents], \
        {"triples": {s: len(x) for s, x in splits.items()}, "split_check": "authors' json == our split",
         "n_with_description": sum(e in desc for e in ents)}


# ----------------------------------------------------------------------------------------------- Spatial-VRD
def clamp_box(box, W, H):
    """Box rules of scripts/extract_features.py; None = wholly outside the image."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0:
        return None
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
    if x2 - x1 < MIN_BOX:
        c = (x1 + x2) / 2
        x1, x2 = max(0.0, c - MIN_BOX / 2), min(float(W), c + MIN_BOX / 2)
    if y2 - y1 < MIN_BOX:
        c = (y1 + y2) / 2
        y1, y2 = max(0.0, c - MIN_BOX / 2), min(float(H), c + MIN_BOX / 2)
    return x1, y1, x2, y2


def prepare_vrd(cmr, data, zip_path, out_dir, img_dir):
    rows = read_count_file(os.path.join(data, "entity2id.txt"))
    ents = [r[0] for r in rows]
    eid = {e: int(i) for e, i in rows}
    with open(os.path.join(data, "entity2text.txt"), encoding="utf-8") as f:   # no count line
        text = dict(line.rstrip("\n").split("\t", 1) for line in f if line.strip())
    no_text = [e for e in ents if e not in text]
    if no_text:
        raise ValueError(f"{len(no_text)} entities without a line in entity2text.txt, e.g. {no_text[:3]}")
    rels = {int(i): r for r, i in read_count_file(os.path.join(data, "relation2id.txt"))}
    splits = {}
    for split in ("train", "valid", "test"):
        ex = []
        for h, t, r in read_2id(os.path.join(data, f"{split}2id.txt")):     # OpenKE order: h t r
            h, t, r = ents[h], ents[t], rels[r]
            ex.append({"head_id": h, "head": text[h], "relation": r, "tail_id": t, "tail": text[t]})
        dump_json(ex, os.path.join(out_dir, f"{split}.txt.json"))
        splits[split] = len(ex)
    dump_json({r: r for r in rels.values()}, os.path.join(out_dir, "relations.json"))
    # crops: same objects / ids / boxes as scripts/extract_features.py vrd_scenes
    paths, outside, unreadable = {}, [], []
    with zipfile.ZipFile(zip_path) as z:
        by_name = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.lower().endswith((".jpg", ".png", ".jpeg"))}
        for js in ("sg_train_annotations.json", "sg_test_annotations.json"):
            for img in load_json(os.path.join(data, js)):
                fn = img.get("filename")
                photo = str(img.get("photo_id", os.path.splitext(fn)[0]))
                boxes = []
                for k, obj in enumerate(img.get("objects", [])):
                    e, b = f"{photo}_{k}", obj.get("bbox")
                    if e in eid and b:
                        boxes.append((e, (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])))
                if not boxes or fn not in by_name:
                    continue
                try:
                    im = Image.open(io.BytesIO(z.read(by_name[fn])))
                    im.load()
                    im = im.convert("RGB")
                except Exception:
                    unreadable.extend(e for e, _ in boxes)
                    continue
                for e, box in boxes:
                    cb = clamp_box(box, *im.size)
                    if cb is None:
                        outside.append(e)
                        continue
                    dst = os.path.join(img_dir, f"{e}.jpg")
                    im.crop(cb).save(dst, quality=95)
                    paths[e] = dst
    return [dict(entity_id=e, entity=text[e], entity_desc="", img=paths.get(e)) for e in ents], \
        {"triples": splits, "boxes_outside_image": outside, "unreadable": unreadable}


TASKS = {"FB15k237_ind": prepare_fb, "SpatialVRD_ind": prepare_vrd}


def prepare(task, cmr, data, zip_path):
    cmr = os.path.abspath(cmr)
    out_dir = os.path.join(cmr, "data", task)
    img_dir = os.path.join(cmr, "img_data", task)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    entities, info = TASKS[task](cmr, data, zip_path, out_dir, img_dir)
    # the entities left without a picture must be exactly those without an image in v0 (ent_img_mask.pt), the
    # rows every feature-table baseline gets the constant for (entities are in entity2id order)
    import torch
    mask = torch.load(os.path.join(data, "ent_img_mask.pt"), map_location="cpu", weights_only=True)
    no_pic = [i for i, e in enumerate(entities) if e["img"] is None]
    if len(mask) != len(entities) or no_pic != (~mask).nonzero(as_tuple=True)[0].tolist():
        raise ValueError(f"entities without a picture ({len(no_pic)}) differ from ent_img_mask.pt ({int((~mask).sum())})")
    noise = write_noise(img_dir)
    n_missing = 0
    for e in entities:
        img = e.pop("img")
        if img is None:
            img, n_missing = noise, n_missing + 1
        e["entity_imgpath"] = rel_path(cmr, img)
    dump_json(entities, os.path.join(out_dir, "img_entities.json"))
    info.update({"task": task, "data": os.path.abspath(data), "zip": os.path.abspath(zip_path),
                 "n_entities": len(entities), "n_missing_image": n_missing,
                 "missing_image": {"file": rel_path(cmr, noise), "sha256": sha256_file(noise),
                                   "seed": NOISE_SEED, "size": NOISE_SIZE}})
    dump_json(info, os.path.join(out_dir, "gm_prepare.json"))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--cmr", required=True, help="CMR checkout (scripts/setup_baseline.py cmr)")
    ap.add_argument("--data", required=True, help="our dataset folder")
    ap.add_argument("--zip", required=True, help="image zip (FB15K-237-ind: MKGformer images; Spatial-VRD: VRD)")
    a = ap.parse_args(argv)
    info = prepare(a.task, a.cmr, a.data, a.zip)
    print(json.dumps({k: v for k, v in info.items() if k not in ("boxes_outside_image", "unreadable")}, indent=1))


if __name__ == "__main__":
    main()
