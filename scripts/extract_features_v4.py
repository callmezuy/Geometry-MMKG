"""Entity features for data v4: RGB / depth with DINOv2, text with RoBERTa (replaces extract_features.py
and extract_mmkg_features.py).

Why v4 (audit 18/11/2026, RESEARCH_PLAN.md P0.7)
-------------------------------------------------
The v1-v3 ent_depth_emb.pt carried almost no information:
  1. depth maps were saved as 16-bit PNG and re-read with Image.convert('L'), which CLIPS every value
     above 255 to 255 -> nearly all-white maps;
  2. they were then encoded by DepthEncoder (AlexNet-BN) with random, never-trained weights.
Measured: mean pairwise cosine 0.97-0.999, a linear probe on the 150 VG object classes scored below the
majority class, and depth 10-NN overlapped RGB 10-NN at chance level. RGB / text also went through a
randomly initialised fc layer.

Pipeline (per source image)
---------------------------
  RGB image -> ZipDepth (zipdepth_base, 384x384, ImageNet norm) -> relative inverse depth map
  -> min-max normalised PER IMAGE (keeps the near/far order of objects in one scene)
  -> crop each entity box with roi_align (224x224), replicate to 3 channels, ImageNet norm -> DINOv2
  RGB crop (PIL, original resolution) -> 224x224, ImageNet norm -> DINOv2
  DINOv2 = facebook/dinov2-base, pooler_output (layer-normed CLS), 768-d, frozen, no projection layer.
  Text = roberta-base, attention-masked mean of the last hidden states, 768-d, no projection layer.
Entities whose image is missing get an all-zero row and are listed in feature_stats.json (never a
silent black image). entity2id.txt is never changed: row i = entity id i, as in v1-v3.

Usage (from Code/)
------------------
  python scripts/extract_features_v4.py mmkg --data data/Merged-MMKG --out data/_feat/Merged-MMKG \
      --raw data/raw/mmkg --bench baselines/SNAG/SNAG_MKGC/benchmarks
  python scripts/extract_features_v4.py vrd --data data/Spatial-VRD --out data/_feat/Spatial-VRD \
      --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip
  python scripts/extract_features_v4.py vg --data data/Spatial-VG --out <dir> \
      --zip data/raw/vg/images.zip data/raw/vg/images2.zip --image-data data/raw/vg/image_data.json.zip
  (VG: notebooks/extract-vg-features.py runs this on a Molab GPU.)
Output: ent_rgb_emb.pt, ent_depth_emb.pt, ent_text_emb.pt (float32 [n_ent, 768]),
        for --mean-concat also ent_rgb_depth_mean_emb.pt / ent_rgb_depth_concat_emb.pt, and feature_stats.json.
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import roi_align

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEPTH_RES = 384
ENC_RES = 224
DINO_NAME = "facebook/dinov2-base"
TEXT_NAME = "roberta-base"
MIN_BOX = 2.0  # pixels; smaller boxes are expanded around their centre


def read_count_file(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    assert len(rows) == n, (path, n, len(rows))
    return rows


# --------------------------------------------------------------------------- image sources
class ZipPool:
    """Opens each zip lazily, once per process (DataLoader workers get their own handles)."""

    def __init__(self, paths):
        self.paths = list(paths)
        self._z = {}

    def read(self, zi, member):
        if zi not in self._z:
            self._z[zi] = zipfile.ZipFile(self.paths[zi])
        return self._z[zi].read(member)

    def __getstate__(self):
        return {"paths": self.paths, "_z": {}}


def mmkg_scenes(data_dir, raw_dir, bench_dir):
    """Rebuild entity -> image exactly as scripts/merge_mmkgs.py chose it (first source with a valid image)."""
    zips = {d: os.path.join(raw_dir, f"{d}.zip") for d in ["DB15K", "MKG-W", "MKG-Y"]}
    pool = ZipPool([zips["DB15K"], zips["MKG-W"], zips["MKG-Y"]])
    zi = {"DB15K": 0, "MKG-W": 1, "MKG-Y": 2}
    members, sizes = {}, {}
    for d, p in zips.items():
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if not i.is_dir():
                    members.setdefault(d, {})[i.filename.rsplit("/", 1)[-1]] = i.filename
                    sizes[(d, i.filename)] = i.file_size
    with open(os.path.join(raw_dir, "DB15K_ImageIndex.txt"), encoding="utf-8") as f:
        db_index = {a.strip("<>").strip(): b.strip() for a, b in (l.rstrip("\n").split("\t") for l in f if "\t" in l)}
    with zipfile.ZipFile(zips["MKG-W"]) as z:
        wiki = json.loads(z.read("MKG-W/wikidata_to_wiki_dict.json"))
    # MKG-Y zip lost non-ASCII characters of file names (each became U+FFFD): match with that mask
    mask = lambda s: "".join(c if c.isascii() else "�" for c in s)
    y_masked = {}
    for fn in members["MKG-Y"]:
        y_masked.setdefault(mask(fn), []).append(fn)

    def unified(d, e):
        if d == "DB15K":
            return urllib.parse.unquote(e.strip("<>").split("/")[-1])
        if d == "MKG-Y":
            return urllib.parse.unquote(e.strip("<>").replace("Yago_", ""))
        q = e.split("/")[-1].replace(">", "")
        return wiki.get(q, q)

    def image_member(d, e):
        if d == "DB15K":
            img = db_index.get(e.strip("<>").strip())
            fn = f"{img}.jpg" if img else None
            m = members[d].get(fn) if fn else None
        elif d == "MKG-W":
            m = members[d].get(e.split("/")[-1].replace(">", "") + ".jpg")
        else:
            fn = e.replace("<", "").replace(">", "").replace("/", "_").replace(":", "_") + ".jpg"
            m = members[d].get(fn)
            if m is None:
                cand = y_masked.get(mask(fn), [])
                m = members[d][cand[0]] if len(cand) == 1 else None
        if m is None or sizes[(d, m)] <= 1024:  # merge_mmkgs.py: os.path.getsize(path) > 1024
            return None
        return m

    names = [n for n, _ in read_count_file(os.path.join(data_dir, "entity2id.txt"))]
    gid = {n: i for i, n in enumerate(names)}
    chosen = {}
    for d in ["DB15K", "MKG-W", "MKG-Y"]:
        with open(os.path.join(bench_dir, d, "entity2id.txt"), encoding="utf-8") as f:  # "name id", space-separated
            bench_ents = [" ".join(l.split()[:-1]) for l in f.readlines()[1:] if l.strip()]
        for ent in bench_ents:
            u = unified(d, ent)
            if u not in gid or gid[u] in chosen:
                continue
            m = image_member(d, ent)
            if m is not None:
                chosen[gid[u]] = (zi[d], m)
    scenes = [((z, m), [(g, None)]) for g, (z, m) in sorted(chosen.items())]
    texts = [n.replace("_", " ") for n in names]
    src_count = Counter(["DB15K", "MKG-W", "MKG-Y"][z] for z, _ in chosen.values())
    return pool, scenes, texts, len(names), {"image_source": dict(src_count)}


def vrd_scenes(data_dir, zip_path):
    pool = ZipPool([zip_path])
    with zipfile.ZipFile(zip_path) as z:
        by_name = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.lower().endswith((".jpg", ".png", ".jpeg"))}
    ent2id = {n: int(i) for n, i in read_count_file(os.path.join(data_dir, "entity2id.txt"))}
    scenes = []
    for js in ["sg_train_annotations.json", "sg_test_annotations.json"]:
        with open(os.path.join(data_dir, js), encoding="utf-8") as f:
            data = json.load(f)
        for img in data:
            fn = img.get("filename")
            photo_id = str(img.get("photo_id", os.path.splitext(fn)[0]))
            boxes = []
            for k, obj in enumerate(img.get("objects", [])):  # same ids as construct_spatial_mmsg.py
                eid = ent2id.get(f"{photo_id}_{k}")
                b = obj.get("bbox")
                if eid is None or not b:
                    continue
                boxes.append((eid, (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])))
            if boxes and fn in by_name:
                scenes.append(((0, by_name[fn]), boxes))
    texts = [t for _, t in read_count_file(os.path.join(data_dir, "entity2text.txt"))]
    return pool, scenes, texts, len(ent2id), {}


def vg_scenes(data_dir, zip_paths, image_data_zip):
    import h5py

    pool = ZipPool(zip_paths)
    by_name = {}
    for zi_, p in enumerate(zip_paths):
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.lower().endswith(".jpg"):
                    by_name[n.rsplit("/", 1)[-1]] = (zi_, n)
    with zipfile.ZipFile(image_data_zip) as z:
        im = json.loads(z.read([n for n in z.namelist() if n.endswith(".json")][0]))
    im = [x for x in im if x["image_id"] not in {1592, 1722, 4616, 4617}]  # corrupted, dropped by VG-SGG.h5
    with h5py.File(os.path.join(data_dir, "VG-SGG.h5"), "r") as h:
        boxes = h["boxes_1024"][:].astype(np.float64)  # (xc, yc, w, h), longer side scaled to 1024
        fb, lb = h["img_to_first_box"][:], h["img_to_last_box"][:]
    assert len(im) == len(fb), (len(im), len(fb))
    n_ent = len(read_count_file(os.path.join(data_dir, "entity2id.txt")))
    assert n_ent == len(boxes), (n_ent, len(boxes))  # entity id = box index in VG-SGG.h5
    scenes = []
    for i, meta in enumerate(im):
        if fb[i] < 0:
            continue
        s = max(meta["width"], meta["height"]) / 1024.0
        bb = boxes[fb[i] : lb[i] + 1] * s
        x1, y1 = bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2
        lst = [(int(fb[i]) + k, (x1[k], y1[k], x1[k] + bb[k, 2], y1[k] + bb[k, 3])) for k in range(len(bb))]
        key = by_name.get(f"{meta['image_id']}.jpg")
        if key:
            scenes.append((key, lst))
    texts = [t for _, t in read_count_file(os.path.join(data_dir, "entity2text.txt"))]
    return pool, scenes, texts, n_ent, {}


# --------------------------------------------------------------------------- loading
class SceneSet(Dataset):
    def __init__(self, pool, scenes):
        self.pool, self.scenes = pool, scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        (zi, member), boxes = self.scenes[i]
        try:
            img = Image.open(io.BytesIO(self.pool.read(zi, member)))
            img.load()
            img = img.convert("RGB")
        except Exception as e:  # unreadable image: all its entities are reported missing
            return {"ok": False, "ids": [b[0] for b in boxes], "err": f"{member}: {e}"}
        W, H = img.size
        full = torch.from_numpy(np.asarray(img.resize((DEPTH_RES, DEPTH_RES), Image.BICUBIC), dtype=np.float32) / 255.0)
        full = full.permute(2, 0, 1)
        ids, crops, rois = [], [], []
        for eid, box in boxes:
            if box is None:
                box = (0, 0, W, H)
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            if x2 - x1 < MIN_BOX:
                c = (x1 + x2) / 2
                x1, x2 = max(0.0, c - MIN_BOX / 2), min(float(W), c + MIN_BOX / 2)
            if y2 - y1 < MIN_BOX:
                c = (y1 + y2) / 2
                y1, y2 = max(0.0, c - MIN_BOX / 2), min(float(H), c + MIN_BOX / 2)
            crop = img.crop((x1, y1, x2, y2)).resize((ENC_RES, ENC_RES), Image.BICUBIC)
            crops.append(torch.from_numpy(np.asarray(crop, dtype=np.uint8)).permute(2, 0, 1))
            rois.append([x1 * DEPTH_RES / W, y1 * DEPTH_RES / H, x2 * DEPTH_RES / W, y2 * DEPTH_RES / H])
            ids.append(eid)
        return {"ok": True, "ids": ids, "full": full, "crops": torch.stack(crops), "rois": torch.tensor(rois)}


def collate(batch):
    return batch


# --------------------------------------------------------------------------- models
def load_zipdepth(device):
    sys.path.insert(0, os.path.join(CODE_DIR, "ZipDepth"))
    from zipdepth.model.architecture import create_model

    m = create_model("zipdepth_base", pretrained=False)
    sd = torch.load(os.path.join(CODE_DIR, "ZipDepth", "checkpoints", "zipdepth_base.pth"), map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)  # strict: no silent random init
    return m.to(device).eval()


def norm_img(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def encode_text(texts, device, batch=512):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT_NAME)
    model = AutoModel.from_pretrained(TEXT_NAME).to(device).eval()
    uniq = sorted(set(texts))
    pos = {t: i for i, t in enumerate(uniq)}
    out = torch.empty(len(uniq), 768)
    for i in range(0, len(uniq), batch):
        enc = tok(uniq[i : i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[i : i + len(uniq[i : i + batch])] = ((h * m).sum(1) / m.sum(1)).cpu()
    idx = torch.tensor([pos[t] for t in texts])
    del model
    return out[idx], len(uniq)


# --------------------------------------------------------------------------- main loop
def extract(pool, scenes, n_ent, out_dir, device, workers, img_batch, enc_batch):
    from transformers import AutoModel

    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    zd = load_zipdepth(device)
    rgb = np.lib.format.open_memmap(os.path.join(out_dir, "_rgb.npy"), mode="w+", dtype=np.float32, shape=(n_ent, 768))
    dep = np.lib.format.open_memmap(os.path.join(out_dir, "_depth.npy"), mode="w+", dtype=np.float32, shape=(n_ent, 768))
    done = np.zeros(n_ent, dtype=bool)
    errors = []
    loader = DataLoader(SceneSet(pool, scenes), batch_size=img_batch, num_workers=workers, collate_fn=collate,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    q_ids, q_rgb, q_dep = [], [], []

    def flush(final=False):
        nonlocal q_ids, q_rgb, q_dep
        while q_ids and (final or len(q_ids) >= enc_batch):
            ids, r, d = q_ids[:enc_batch], torch.cat(q_rgb)[:enc_batch], torch.cat(q_dep)[:enc_batch]
            rest_r, rest_d = torch.cat(q_rgb)[enc_batch:], torch.cat(q_dep)[enc_batch:]
            q_ids, q_rgb, q_dep = q_ids[enc_batch:], [rest_r] if len(rest_r) else [], [rest_d] if len(rest_d) else []
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                x = torch.cat([norm_img(r.to(device, non_blocking=True).float() / 255.0), norm_img(d.to(device))])
                f = dino(pixel_values=x).pooler_output.float().cpu().numpy()
            ids = np.array(ids)
            rgb[ids], dep[ids] = f[: len(ids)], f[len(ids) :]
            done[ids] = True

    t0 = time.time()
    for bi, batch in enumerate(loader):
        good = [b for b in batch if b["ok"]]
        errors += [b["err"] for b in batch if not b["ok"]]
        if good:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                depth = zd(norm_img(torch.stack([b["full"] for b in good]).to(device))).float()  # [B,1,384,384]
            if depth.shape[-1] != DEPTH_RES:
                depth = F.interpolate(depth, size=(DEPTH_RES, DEPTH_RES), mode="bilinear", align_corners=False)
            lo = depth.amin(dim=(1, 2, 3), keepdim=True)
            hi = depth.amax(dim=(1, 2, 3), keepdim=True)
            depth = (depth - lo) / (hi - lo).clamp_min(1e-6)  # per-image min-max
            rois = torch.cat([torch.cat([torch.full((len(b["rois"]), 1), float(k)), b["rois"]], 1) for k, b in enumerate(good)]).to(device)
            dcrops = roi_align(depth, rois, output_size=(ENC_RES, ENC_RES), spatial_scale=1.0, sampling_ratio=2, aligned=True)
            q_dep.append(dcrops.repeat(1, 3, 1, 1).half().cpu() if device.type == "cuda" else dcrops.repeat(1, 3, 1, 1).cpu())
            q_rgb.append(torch.cat([b["crops"] for b in good]))
            for b in good:
                q_ids += b["ids"]
            flush()
        if bi % 50 == 0:
            n = int(done.sum())
            print(f"[{time.time() - t0:7.0f}s] images {min((bi + 1) * img_batch, len(scenes)):,}/{len(scenes):,}  entities {n:,}/{n_ent:,}", flush=True)
    flush(final=True)
    rgb.flush(), dep.flush()
    return rgb, dep, done, errors


@torch.no_grad()
def quality(X, sample=2000, seed=0):
    rng = np.random.default_rng(seed)
    nz = np.flatnonzero(np.abs(X).sum(1) > 0)
    idx = np.sort(rng.choice(nz, min(sample, len(nz)), replace=False))
    A = torch.from_numpy(np.asarray(X[idx], dtype=np.float32))
    An = F.normalize(A, dim=1)
    C = An @ An.T
    cos = float(C[torch.triu(torch.ones_like(C, dtype=torch.bool), 1)].mean())
    ev = torch.linalg.svdvals(A - A.mean(0)) ** 2
    return {"mean_cos": round(cos, 4), "pc1_var": round(float(ev[0] / ev.sum()), 4), "pc1_5_var": round(float(ev[:5].sum() / ev.sum()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["mmkg", "vrd", "vg"])
    ap.add_argument("--data", required=True, help="dataset dir with entity2id.txt (and annotations / VG-SGG.h5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", help="mmkg: dir with DB15K.zip, MKG-W.zip, MKG-Y.zip, DB15K_ImageIndex.txt")
    ap.add_argument("--bench", default=os.path.join(CODE_DIR, "baselines", "SNAG", "SNAG_MKGC", "benchmarks"))
    ap.add_argument("--zip", nargs="+", help="vrd: Kaggle VRD zip; vg: images.zip images2.zip")
    ap.add_argument("--image-data", help="vg: image_data.json.zip")
    ap.add_argument("--mean-concat", action="store_true", help="also write ent_rgb_depth_{mean,concat}_emb.pt")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--img-batch", type=int, default=16, help="images per ZipDepth batch")
    ap.add_argument("--enc-batch", type=int, default=256, help="crops per DINOv2 batch (x2: rgb + depth)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    if a.dataset == "mmkg":
        pool, scenes, texts, n_ent, extra = mmkg_scenes(a.data, a.raw, a.bench)
    elif a.dataset == "vrd":
        pool, scenes, texts, n_ent, extra = vrd_scenes(a.data, a.zip[0])
    else:
        pool, scenes, texts, n_ent, extra = vg_scenes(a.data, a.zip, a.image_data)
    if a.limit:
        scenes = scenes[: a.limit]
    n_box = sum(len(b) for _, b in scenes)
    print(f"{len(scenes):,} images, {n_box:,} entity crops, {n_ent:,} entities", flush=True)
    assert len(texts) == n_ent

    t0 = time.time()
    txt, n_uniq = encode_text(texts, device)
    torch.save(txt, os.path.join(a.out, "ent_text_emb.pt"))
    print(f"text done ({n_uniq:,} unique strings, {time.time() - t0:.0f}s)", flush=True)

    rgb, dep, done, errors = extract(pool, scenes, n_ent, a.out, device, a.workers, a.img_batch, a.enc_batch)
    rgb[~done] = 0.0
    dep[~done] = 0.0
    R, D = torch.from_numpy(np.asarray(rgb)), torch.from_numpy(np.asarray(dep))
    torch.save(R, os.path.join(a.out, "ent_rgb_emb.pt"))
    torch.save(D, os.path.join(a.out, "ent_depth_emb.pt"))
    if a.mean_concat:
        torch.save((R + D) / 2.0, os.path.join(a.out, "ent_rgb_depth_mean_emb.pt"))
        torch.save(torch.cat([R, D], dim=-1), os.path.join(a.out, "ent_rgb_depth_concat_emb.pt"))
    missing = np.flatnonzero(~done)
    stats = {
        "dataset": a.dataset, "n_entities": n_ent, "images": len(scenes), "entity_crops": n_box,
        "entities_with_image": int(done.sum()), "entities_missing_image": int(len(missing)),
        "missing_ids_first_100": missing[:100].tolist(), "unreadable_images": errors[:100], "n_unreadable": len(errors),
        "encoders": {"rgb": DINO_NAME + " pooler_output", "depth": f"ZipDepth zipdepth_base ({DEPTH_RES}px, per-image min-max) -> {DINO_NAME} pooler_output",
                     "text": f"{TEXT_NAME} masked mean, {n_uniq} unique strings"},
        "quality": {"rgb": quality(rgb), "depth": quality(dep), "text": quality(txt.numpy())},
        "seconds": round(time.time() - t0), **extra,
    }
    with open(os.path.join(a.out, "feature_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    del rgb, dep, R, D
    for fn in ["_rgb.npy", "_depth.npy"]:
        os.remove(os.path.join(a.out, fn))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("missing_ids_first_100", "unreadable_images")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
