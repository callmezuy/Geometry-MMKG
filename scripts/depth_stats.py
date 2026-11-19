"""
Per-entity depth statistics table of the scene-graph datasets (data v0, RESEARCH_PLAN.md P0.7 / P2.9).

ent_depth_stats.pt = float32 [n_ent, 9]: means of a 3x3 grid of sub-regions of the entity box on the ZipDepth depth
map, min-max normalised per image (1 = nearest point of the image, 0 = farthest). Same depth map and same boxes as
ent_depth_emb.pt (scripts/extract_features.py: zipdepth_base, 384x384, box clipping, MIN_BOX); only the box region
is summarised instead of being encoded by DINOv2, so objects of one image stay comparable (P2.9 A2/A3, round 5b).
Box coordinates are not part of the table.
Entities without an image (ent_img_mask.pt False) hold the mean of the TRAIN entities that have one, like the other
tables; the fill vector is written to depth_stats.json.

Only for scene graphs (one box per entity). Knowledge graphs have one whole image per entity and no shared scene.

  python scripts/depth_stats.py vrd --data data/Spatial-VRD --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip --out data/_feat/Spatial-VRD
  python scripts/depth_stats.py vg --data data/Spatial-VG --zip images.zip images2.zip --image-data image_data.json.zip --out OUT
  python scripts/depth_stats.py fill --src data/Spatial-VRD --data data/Spatial-VRD-Trans --out data/_feat/Spatial-VRD-Trans
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as ef  # noqa: E402

GRID = 32       # roi_align samples per side, as scripts/vrd_depth_stats.py
CELLS = 3       # 3x3 grid -> 9 numbers
TABLE = "ent_depth_stats.pt"
INFO = "depth_stats.json"


class DepthSceneSet(ef.SceneSet):
    """ef.SceneSet without the RGB crops: the full image for ZipDepth and the box rois, with exactly the same box
    clipping (boxes entirely outside the image are dropped, small boxes widened to MIN_BOX)."""

    def __getitem__(self, i):
        (zi, member), boxes = self.scenes[i]
        try:
            img = Image.open(io.BytesIO(self.pool.read(zi, member)))
            img.load()
            img = img.convert("RGB")
        except Exception as e:
            return {"ok": False, "ids": [b[0] for b in boxes], "err": f"{member}: {e}", "outside": []}
        W, H = img.size
        full = torch.from_numpy(np.asarray(img.resize((ef.DEPTH_RES, ef.DEPTH_RES), Image.BICUBIC),
                                           dtype=np.float32) / 255.0).permute(2, 0, 1)
        ids, rois, outside = [], [], []
        for eid, box in boxes:
            if box is None:
                box = (0, 0, W, H)
            x1, y1, x2, y2 = (float(v) for v in box)
            if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0:
                outside.append(eid)
                continue
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            if x2 - x1 < ef.MIN_BOX:
                c = (x1 + x2) / 2
                x1, x2 = max(0.0, c - ef.MIN_BOX / 2), min(float(W), c + ef.MIN_BOX / 2)
            if y2 - y1 < ef.MIN_BOX:
                c = (y1 + y2) / 2
                y1, y2 = max(0.0, c - ef.MIN_BOX / 2), min(float(H), c + ef.MIN_BOX / 2)
            rois.append([x1 * ef.DEPTH_RES / W, y1 * ef.DEPTH_RES / H, x2 * ef.DEPTH_RES / W, y2 * ef.DEPTH_RES / H])
            ids.append(eid)
        if not ids:
            return {"ok": False, "ids": [], "err": None, "outside": outside}
        return {"ok": True, "ids": ids, "full": full, "rois": torch.tensor(rois), "outside": outside}


def grid_means(depth, rois):
    """depth [B,1,R,R] normalised per image; rois [n,5] (batch index, x1, y1, x2, y2) -> [n, 9]."""
    g = roi_align(depth, rois, output_size=(GRID, GRID), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return F.adaptive_avg_pool2d(g, CELLS).flatten(1)


@torch.no_grad()
def compute(pool, scenes, n_ent, device, img_batch, workers):
    """Raw table [n_ent, 9] (NaN where no box was encoded) and run statistics."""
    loader = DataLoader(DepthSceneSet(pool, scenes), batch_size=img_batch, num_workers=workers,
                        collate_fn=ef.collate)
    zd = ef.load_zipdepth(device)
    out = np.full((n_ent, CELLS * CELLS), np.nan, dtype=np.float32)
    n_img, n_box, errors, outside, t0 = 0, 0, [], [], time.time()
    for bi, batch in enumerate(loader):
        good = [b for b in batch if b["ok"]]
        errors += [b["err"] for b in batch if not b["ok"] and b["err"]]
        outside += [e for b in batch for e in b["outside"]]
        if not good:
            continue
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            depth = zd(ef.norm_img(torch.stack([b["full"] for b in good]).to(device))).float()
        if depth.shape[-1] != ef.DEPTH_RES or depth.shape[-2] != ef.DEPTH_RES:
            depth = F.interpolate(depth, size=(ef.DEPTH_RES, ef.DEPTH_RES), mode="bilinear", align_corners=False)
        lo = depth.amin(dim=(1, 2, 3), keepdim=True)
        hi = depth.amax(dim=(1, 2, 3), keepdim=True)
        depth = (depth - lo) / (hi - lo).clamp_min(1e-6)
        rois = torch.cat([torch.cat([torch.full((len(b["rois"]), 1), float(i)), b["rois"]], 1)
                          for i, b in enumerate(good)]).to(device)
        ids = np.array([e for b in good for e in b["ids"]])
        out[ids] = grid_means(depth, rois).cpu().numpy()
        n_img += len(good)
        n_box += len(ids)
        if bi % 200 == 0:
            print(f"{n_img:,} images, {n_box:,} boxes, {time.time() - t0:.0f}s", flush=True)
    return out, {"n_images": n_img, "n_boxes": n_box, "n_unreadable": len(errors), "errors": errors[:20],
                 "n_outside": len(outside), "sec": round(time.time() - t0, 1)}


def fill_table(raw, data_dir):
    """Fill the rows of entities without an image (ent_img_mask.pt of data_dir) with the mean of the train entities
    that have one; check that exactly those rows lack statistics."""
    mask = torch.load(os.path.join(data_dir, "ent_img_mask.pt"), weights_only=True).numpy().astype(bool)
    assert len(mask) == len(raw), (len(mask), len(raw))
    has = np.isfinite(raw).all(1)
    if not np.array_equal(has, mask):
        raise ValueError(f"entities with depth statistics ({int(has.sum())}) != entities with an image "
                         f"({int(mask.sum())}); {int((has & ~mask).sum())} extra, {int((~has & mask).sum())} missing")
    train = ef.train_entity_ids(data_dir, len(mask))
    src = np.flatnonzero(train & mask)
    assert len(src), "no train entity has an image"
    fill = raw[src].astype(np.float64).mean(0).astype(np.float32)
    table = raw.copy()
    table[~mask] = fill
    return table, {"fill": [round(float(x), 6) for x in fill], "train_entities_with_image": int(len(src)),
                   "filled_entities": int((~mask).sum())}


def code_commit():
    try:
        return subprocess.run(["git", "-C", ef.CODE_DIR, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def save(table, info, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    assert table.dtype == np.float32 and np.isfinite(table).all()
    torch.save(torch.from_numpy(table), os.path.join(out_dir, TABLE))
    with open(os.path.join(out_dir, INFO), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.join(out_dir, TABLE)} {tuple(table.shape)} and {INFO}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["vrd", "vg", "fill"])
    ap.add_argument("--data", required=True, help="dataset dir (entity2id.txt, train2id.txt, ent_img_mask.pt, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", nargs="+", help="vrd: Kaggle VRD zip; vg: images.zip images2.zip")
    ap.add_argument("--image-data", help="vg: image_data.json.zip")
    ap.add_argument("--src", help="fill: dataset dir whose ent_depth_stats.pt is re-filled for --data")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--img-batch", type=int, default=16, help="images per ZipDepth batch")
    a = ap.parse_args()
    base = {"table": TABLE, "content": f"{CELLS}x{CELLS} grid means of the per-image min-max normalised ZipDepth "
                                       "depth map inside the entity box (1 = nearest)",
            "depth_model": f"ZipDepth zipdepth_base ({ef.DEPTH_RES}px, per-image min-max)", "grid_samples": GRID,
            "data": a.data, "code_commit": code_commit()}

    if a.dataset == "fill":
        same = (open(os.path.join(a.src, "entity2id.txt"), "rb").read()
                == open(os.path.join(a.data, "entity2id.txt"), "rb").read())
        assert same, "fill: entity2id.txt of --src and --data differ"
        src_table = torch.load(os.path.join(a.src, TABLE), weights_only=True).numpy()
        src_mask = torch.load(os.path.join(a.src, "ent_img_mask.pt"), weights_only=True).numpy().astype(bool)
        raw = src_table.copy()
        raw[~src_mask] = np.nan
        table, finfo = fill_table(raw, a.data)
        return save(table, {**base, "refilled_from": a.src, **finfo}, a.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.dataset == "vrd":
        pool, scenes, n_ent, _ = ef.vrd_scenes(a.data, a.zip[0])
    else:
        pool, scenes, n_ent, _ = ef.vg_scenes(a.data, a.zip, a.image_data)
    print(f"{a.dataset}: {len(scenes):,} images, {n_ent:,} entities, device {device}", flush=True)
    raw, run = compute(pool, scenes, n_ent, device, a.img_batch, a.workers)
    table, finfo = fill_table(raw, a.data)
    save(table, {**base, **run, **finfo}, a.out)


if __name__ == "__main__":
    main()
