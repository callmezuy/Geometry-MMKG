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
  Text = roberta-base, attention-masked mean of the last hidden states, 768-d, no projection layer;
  entity2textlong.txt (descriptions) when the dataset has one, else entity2text.txt; --text-max-len tokens.
Knowledge graphs (mode kg: FB15k-237-IMG, FB15K-237-ind, MKG-Y) have several whole images per entity
(image_index.tsv, status == ok only): each image is one "box" covering the whole image, and the entity
feature is the mean over its images, RGB and depth separately. Scene graphs (vrd, vg): one box per entity.

Missing images (RESEARCH_PLAN.md 3.2). ent_img_mask.pt (bool [n_ent]) = entity has at least one encoded
image. In ent_rgb_emb.pt / ent_depth_emb.pt every entity without image gets ONE constant vector, the mean
feature of the TRAIN entities that have an image (entities of train.txt), computed separately for RGB and
depth (ent_fill.pt). Baselines read these tables as they are; GeometryMMKG reads the same table plus the
mask and replaces the masked rows by its learned "missing" vector, so a separate raw table is not needed.
The constant depends on the train split: `fill` re-fills the tables for another dataset with the same
entity2id (FB15K-237-ind reuses the FB15k-237-IMG features). entity2id.txt is never changed: row i = id i.

Usage (from Code/)
------------------
  python scripts/extract_features_v4.py kg --data data/FB15k-237-IMG --out data/_feat/FB15k-237-IMG
      --zip data/raw/fb15k237/fb15k_wn18_images.zip --mean-concat
  python scripts/extract_features_v4.py fill --feat data/_feat/FB15k-237-IMG --data data/FB15K-237-ind
      --out data/_feat/FB15K-237-ind --mean-concat
  python scripts/extract_features_v4.py kg --data data/MKG-Y --out data/_feat/MKG-Y
      --zip data/raw/mmrns_official/MKG-Y_img.zip --mean-concat
  python scripts/extract_features_v4.py vrd --data data/Spatial-VRD --out data/_feat/Spatial-VRD
      --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip --mean-concat
  python scripts/extract_features_v4.py vg --data data/Spatial-VG --out <dir>
      --zip data/raw/vg/images.zip data/raw/vg/images2.zip --image-data data/raw/vg/image_data.json.zip
Output: ent_rgb_emb.pt, ent_depth_emb.pt, ent_text_emb.pt (float32 [n_ent, 768]), ent_img_mask.pt,
        ent_fill.pt, for --mean-concat also ent_rgb_depth_mean_emb.pt / ent_rgb_depth_concat_emb.pt,
        and feature_stats.json.
"""
import argparse
import gc
import io
import json
import os
import shutil
import sys
import time
import zipfile

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
Image.MAX_IMAGE_PIXELS = 200_000_000  # same limit as the dataset builders (scripts/build_*.py)


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


def kg_scenes(data_dir, zip_path):
    """One whole-image 'box' per image of image_index.tsv (entity, zip member, status), status ok only."""
    pool = ZipPool([zip_path])
    ent2id = {n: int(i) for n, i in read_count_file(os.path.join(data_dir, "entity2id.txt"))}
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    scenes, status = [], {}
    with open(os.path.join(data_dir, "image_index.tsv"), encoding="utf-8") as f:
        for l in f:
            e, m, st = l.rstrip("\n").split("\t")
            status[st] = status.get(st, 0) + 1
            if st == "ok":
                assert m in names, f"{m} not in {zip_path}"
                scenes.append(((0, m), [(ent2id[e], None)]))
    return pool, scenes, len(ent2id), {"image_index_status": status}


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
    return pool, scenes, len(ent2id), {}


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
    return pool, scenes, n_ent, {}


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
            return {"ok": False, "ids": [b[0] for b in boxes], "err": f"{member}: {e}", "outside": []}
        W, H = img.size
        full = torch.from_numpy(np.asarray(img.resize((DEPTH_RES, DEPTH_RES), Image.BICUBIC), dtype=np.float32) / 255.0)
        full = full.permute(2, 0, 1)
        ids, crops, rois, outside = [], [], [], []
        for eid, box in boxes:
            if box is None:
                box = (0, 0, W, H)
            x1, y1, x2, y2 = (float(v) for v in box)
            if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0:
                # box entirely outside the image (annotation error; Spatial-VRD: 2 boxes): no visual region,
                # so the entity is treated as having no image (mask 0, filled), never a strip at the border
                outside.append(eid)
                continue
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            if x2 - x1 < MIN_BOX:
                c = (x1 + x2) / 2
                x1, x2 = max(0.0, c - MIN_BOX / 2), min(float(W), c + MIN_BOX / 2)
            if y2 - y1 < MIN_BOX:
                c = (y1 + y2) / 2
                y1, y2 = max(0.0, c - MIN_BOX / 2), min(float(H), c + MIN_BOX / 2)
            crop = img.crop((x1, y1, x2, y2)).resize((ENC_RES, ENC_RES), Image.BICUBIC)
            crops.append(torch.from_numpy(np.array(crop, dtype=np.uint8)).permute(2, 0, 1))  # copy: writable
            rois.append([x1 * DEPTH_RES / W, y1 * DEPTH_RES / H, x2 * DEPTH_RES / W, y2 * DEPTH_RES / H])
            ids.append(eid)
        if not ids:
            return {"ok": False, "ids": [], "err": None, "outside": outside}
        return {"ok": True, "ids": ids, "full": full, "crops": torch.stack(crops), "rois": torch.tensor(rois),
                "outside": outside}


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
def encode_text(texts, device, max_len, batch=32):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT_NAME)
    model = AutoModel.from_pretrained(TEXT_NAME).to(device).eval()
    uniq = sorted(set(texts), key=len)  # length-sorted batches waste little padding
    pos = {t: i for i, t in enumerate(uniq)}
    out = torch.empty(len(uniq), 768)
    truncated = 0
    for i in range(0, len(uniq), batch):
        chunk = uniq[i : i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        truncated += int((enc["attention_mask"].sum(1) == max_len).sum())
        enc = enc.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[i : i + len(chunk)] = ((h * m).sum(1) / m.sum(1)).cpu()
    idx = torch.tensor([pos[t] for t in texts])
    del model, h, enc
    if device.type == "cuda":
        # RoBERTa on 512-token descriptions reserves ~2 GB; without releasing it the image pass spills into
        # shared memory on a 4 GB GPU (FB15k-237: 16 instead of ~80 images/s)
        torch.cuda.empty_cache()
    return out[idx], {"unique_strings": len(uniq), "max_len": max_len, "truncated_strings": truncated}


# --------------------------------------------------------------------------- main loop
def extract(pool, scenes, n_ent, out_dir, device, workers, img_batch, enc_batch):
    """Sum DINOv2 features of every crop into its entity row; returns (rgb_sum, depth_sum, count, errors, outside)."""
    from transformers import AutoModel

    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    zd = load_zipdepth(device)
    rgb = np.lib.format.open_memmap(os.path.join(out_dir, "_rgb.npy"), mode="w+", dtype=np.float32, shape=(n_ent, 768))
    dep = np.lib.format.open_memmap(os.path.join(out_dir, "_depth.npy"), mode="w+", dtype=np.float32, shape=(n_ent, 768))
    cnt = np.zeros(n_ent, dtype=np.int32)
    errors, outside = [], []
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
            np.add.at(rgb, ids, f[: len(ids)])  # an entity can have several images in one batch
            np.add.at(dep, ids, f[len(ids) :])
            np.add.at(cnt, ids, 1)

    t0 = time.time()
    for bi, batch in enumerate(loader):
        good = [b for b in batch if b["ok"]]
        errors += [b["err"] for b in batch if not b["ok"] and b["err"]]
        outside += [e for b in batch for e in b["outside"]]
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
            done = min((bi + 1) * img_batch, len(scenes))
            el = time.time() - t0
            eta = el / done * (len(scenes) - done) if done else 0
            print(f"[{el:7.0f}s] images {done:,}/{len(scenes):,}  entities {int((cnt > 0).sum()):,}/{n_ent:,}  eta {eta / 60:.0f} min", flush=True)
    flush(final=True)
    rgb.flush(), dep.flush()
    return rgb, dep, cnt, errors, sorted(int(e) for e in outside)


@torch.no_grad()
def quality(X, rows, sample=2000, seed=0):
    """Mean pairwise cosine and share of variance on the first PCs, over a sample of the given rows."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rows, min(sample, len(rows)), replace=False))
    A = torch.as_tensor(np.asarray(X[idx], dtype=np.float32))
    An = F.normalize(A, dim=1)
    C = An @ An.T
    cos = float(C[torch.triu(torch.ones_like(C, dtype=torch.bool), 1)].mean())
    ev = torch.linalg.svdvals(A - A.mean(0)) ** 2
    return {"mean_cos": round(cos, 4), "pc1_var": round(float(ev[0] / ev.sum()), 4), "pc1_5_var": round(float(ev[:5].sum() / ev.sum()), 4)}


def train_entity_ids(data_dir, n_ent):
    ids = np.zeros(n_ent, dtype=bool)
    for h, t, _ in read_count_file(os.path.join(data_dir, "train2id.txt")):
        ids[int(h)] = ids[int(t)] = True
    return ids


def fill_and_save(R, D, mask, data_dir, out_dir, mean_concat, chunk=200_000):
    """Write ent_{rgb,depth}_emb.pt with every unmasked row = mean of the train entities that have an image.
    R, D: [n_ent, 768] arrays (rows with mask False are overwritten); returns the fill statistics."""
    n_ent = len(mask)
    train = train_entity_ids(data_dir, n_ent)
    src = np.flatnonzero(train & mask)
    assert len(src), "no train entity has an image"
    fill = {}
    for name, X in (("rgb", R), ("depth", D)):
        acc = np.zeros(768, dtype=np.float64)
        for i in range(0, len(src), chunk):
            acc += np.asarray(X[src[i : i + chunk]], dtype=np.float64).sum(0)
        fill[name] = torch.from_numpy((acc / len(src)).astype(np.float32))
        miss = np.flatnonzero(~mask)
        if len(miss):
            X[miss] = fill[name].numpy()
    torch.save({"rgb": fill["rgb"], "depth": fill["depth"], "n_train_entities_with_image": int(len(src))},
               os.path.join(out_dir, "ent_fill.pt"))
    torch.save(torch.from_numpy(mask.copy()), os.path.join(out_dir, "ent_img_mask.pt"))
    Rt, Dt = torch.from_numpy(np.asarray(R)), torch.from_numpy(np.asarray(D))
    torch.save(Rt, os.path.join(out_dir, "ent_rgb_emb.pt"))
    torch.save(Dt, os.path.join(out_dir, "ent_depth_emb.pt"))
    if mean_concat:
        torch.save((Rt + Dt) / 2.0, os.path.join(out_dir, "ent_rgb_depth_mean_emb.pt"))
        torch.save(torch.cat([Rt, Dt], dim=-1), os.path.join(out_dir, "ent_rgb_depth_concat_emb.pt"))
    return {"train_entities": int(train.sum()), "train_entities_with_image": int(len(src)),
            "filled_entities": int((~mask).sum()), "filled_train_entities": int((train & ~mask).sum())}


def run_fill(a):
    """Re-fill the tables of --feat for dataset --data (same entity2id, different train split)."""
    same = open(os.path.join(a.feat_data, "entity2id.txt"), "rb").read() == open(os.path.join(a.data, "entity2id.txt"), "rb").read()
    assert same, "fill: entity2id.txt of the feature source and of --data differ"
    os.makedirs(a.out, exist_ok=True)
    mask = torch.load(os.path.join(a.feat, "ent_img_mask.pt")).numpy()
    R = torch.load(os.path.join(a.feat, "ent_rgb_emb.pt")).numpy().copy()
    D = torch.load(os.path.join(a.feat, "ent_depth_emb.pt")).numpy().copy()
    fstats = fill_and_save(R, D, mask, a.data, a.out, a.mean_concat)
    shutil.copyfile(os.path.join(a.feat, "ent_text_emb.pt"), os.path.join(a.out, "ent_text_emb.pt"))
    with open(os.path.join(a.feat, "feature_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    stats.update({"refilled_from": a.feat, "data": a.data, "fill": fstats})
    with open(os.path.join(a.out, "feature_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(fstats, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["kg", "vrd", "vg", "fill"])
    ap.add_argument("--data", required=True, help="dataset dir with entity2id.txt, train2id.txt (+ image_index.tsv / annotations / VG-SGG.h5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", nargs="+", help="kg: image zip; vrd: Kaggle VRD zip; vg: images.zip images2.zip")
    ap.add_argument("--image-data", help="vg: image_data.json.zip")
    ap.add_argument("--feat", help="fill: feature dir to re-fill")
    ap.add_argument("--feat-data", help="fill: dataset the features were extracted for (default: from feature_stats.json)")
    ap.add_argument("--mean-concat", action="store_true", help="also write ent_rgb_depth_{mean,concat}_emb.pt")
    ap.add_argument("--text-max-len", type=int, default=512, help="RoBERTa tokens per entity text (512 = model maximum)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--img-batch", type=int, default=16, help="images per ZipDepth batch")
    ap.add_argument("--enc-batch", type=int, default=64,
                    help="crops per DINOv2 batch (x2: rgb + depth). RTX 3050 4 GB: 64 runs at ~175 crops/s, "
                         "256 (512 images, 4.6 GB) spills into shared memory and drops to ~21 crops/s")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    a = ap.parse_args()

    if a.dataset == "fill":
        if not a.feat_data:
            with open(os.path.join(a.feat, "feature_stats.json"), encoding="utf-8") as f:
                a.feat_data = json.load(f)["data"]
        return run_fill(a)

    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    if a.dataset == "kg":
        pool, scenes, n_ent, extra = kg_scenes(a.data, a.zip[0])
    elif a.dataset == "vrd":
        pool, scenes, n_ent, extra = vrd_scenes(a.data, a.zip[0])
    else:
        pool, scenes, n_ent, extra = vg_scenes(a.data, a.zip, a.image_data)
    if a.limit:
        scenes = scenes[: a.limit]
    n_box = sum(len(b) for _, b in scenes)
    print(f"{len(scenes):,} images, {n_box:,} entity crops, {n_ent:,} entities", flush=True)
    text_file = "entity2textlong.txt" if os.path.exists(os.path.join(a.data, "entity2textlong.txt")) else "entity2text.txt"
    rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(a.data, text_file), encoding="utf-8") if l.strip()]
    if len(rows[0]) == 1 and rows[0][0].isdigit():  # Spatial-VG has a count header, the others do not
        rows = rows[1:]
    ent2id = {n: int(i) for n, i in read_count_file(os.path.join(a.data, "entity2id.txt"))}
    assert len(ent2id) == n_ent
    texts = [None] * n_ent
    for e, t in rows:
        if e in ent2id:
            texts[ent2id[e]] = t
    no_text = sum(t is None for t in texts)
    assert no_text == 0, f"{no_text} entities without a line in {text_file}"
    extra["text_lines_not_in_entity2id"] = sum(e not in ent2id for e, _ in rows)  # Spatial-VRD: 23 dropped entities

    t0 = time.time()
    txt, tstats = encode_text(texts, device, a.text_max_len)
    torch.save(txt, os.path.join(a.out, "ent_text_emb.pt"))
    print(f"text done ({tstats}, {time.time() - t0:.0f}s)", flush=True)

    rgb, dep, cnt, errors, outside = extract(pool, scenes, n_ent, a.out, device, a.workers, a.img_batch, a.enc_batch)
    mask = cnt > 0
    for X in (rgb, dep):  # sum -> mean over the entity's images, in chunks (VG: 1.4M rows)
        for i in range(0, n_ent, 200_000):
            c = cnt[i : i + 200_000]
            X[i : i + 200_000] = np.where(c[:, None] > 0, X[i : i + 200_000] / np.maximum(c, 1)[:, None], 0.0)
    rows_img = np.flatnonzero(mask)
    q = {"rgb": quality(rgb, rows_img), "depth": quality(dep, rows_img), "text": quality(txt.numpy(), np.arange(n_ent))}
    fstats = fill_and_save(rgb, dep, mask, a.data, a.out, a.mean_concat)
    missing = np.flatnonzero(~mask)
    per_ent = cnt[mask]
    stats = {
        "dataset": a.dataset, "data": a.data, "n_entities": n_ent, "images": len(scenes), "entity_crops": n_box,
        "entities_with_image": int(mask.sum()), "entities_missing_image": int(len(missing)),
        "crops_per_entity_with_image": {"min": int(per_ent.min()), "median": float(np.median(per_ent)),
                                        "mean": round(float(per_ent.mean()), 2), "max": int(per_ent.max())} if len(per_ent) else {},
        "missing_ids_first_100": missing[:100].tolist(), "unreadable_images": errors[:100], "n_unreadable": len(errors),
        "boxes_outside_image": len(outside), "boxes_outside_image_ids": outside[:100],
        "encoders": {"rgb": DINO_NAME + " pooler_output",
                     "depth": f"ZipDepth zipdepth_base ({DEPTH_RES}px, per-image min-max) -> {DINO_NAME} pooler_output",
                     "text": f"{TEXT_NAME} masked mean of {text_file}", "text_stats": tstats},
        "fill": fstats, "quality_rows_with_image": q, "limit": a.limit,
        "seconds": round(time.time() - t0), **extra,
    }
    with open(os.path.join(a.out, "feature_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    del rgb, dep, X  # Windows cannot delete a memory-mapped file while any reference is alive
    gc.collect()
    for fn in ["_rgb.npy", "_depth.npy"]:
        os.remove(os.path.join(a.out, fn))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("missing_ids_first_100", "unreadable_images")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
