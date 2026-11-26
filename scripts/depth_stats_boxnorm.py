"""
E2b (RESEARCH_PLAN P6.0, C2 / C3; docs/superpowers/specs/2026-11-26-e2b-shared-scale-design.md): does the gain of
depth_stats need the depth scale shared by the objects of one image? Same grid9 as scripts/depth_stats.py (ZipDepth,
384 px, same boxes, 32 x 32 roi_align samples, 3 x 3 cell means) with one change: the samples of each box are min-max
normalised by that box (1 = nearest point of the box, 0 = farthest) instead of by the image, so every object gets its
own scale, as an entity of a knowledge graph with its own image. scripts/depth_stats.py (data v0) is not modified.

One ZipDepth pass computes both normalisations; the per-image one must reproduce the dataset's ent_depth_stats.pt
(max abs difference <= CHECK_TOL on the entities with an image), otherwise nothing is written.
Boxes whose depth is constant (max - min < CONST_EPS) get 9 zeros and are counted.

Output <out>/depth_stats_boxnorm.pt = {"grid9": [N, 9] per-box, "grid9_image": [N, 9] per-image (the check), "model",
"norm", "source"}, float32, entities without an image filled like the data tables (scripts/depth_stats.fill_table);
read by `train.py --variant depth_stats_only --depth-stats <out>/depth_stats_boxnorm.pt:grid9`.
<out>/depth_stats_boxnorm.json: counts, the check, sha256 of the .pt.

  python scripts/depth_stats_boxnorm.py vrd --data data/Spatial-VRD \
      --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip --out OUT/Spatial-VRD
  python scripts/depth_stats_boxnorm.py fill --src OUT/Spatial-VRD --src-data data/Spatial-VRD \
      --data data/Spatial-VRD-Trans --out OUT/Spatial-VRD-Trans
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_stats as ds  # noqa: E402
import extract_features as ef  # noqa: E402

TABLE = "depth_stats_boxnorm.pt"
INFO = "depth_stats_boxnorm.json"
CHECK_TOL = 1e-3     # per-image table vs ent_depth_stats.pt (fp16 ZipDepth on GPU)
CONST_EPS = 1e-6     # box depth range below this -> constant box -> zeros


def box_minmax_grid(depth, rois):
    """depth [B,1,R,R]; rois [n,5] (batch index, x1, y1, x2, y2) -> ([n, 9] 3x3 means of the per-box min-max
    normalised samples, [n] bool: constant box, given zeros)."""
    g = roi_align(depth, rois, output_size=(ds.GRID, ds.GRID), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    lo = g.amin(dim=(1, 2, 3), keepdim=True)
    rng = g.amax(dim=(1, 2, 3), keepdim=True) - lo
    const = rng.flatten() < CONST_EPS
    g = (g - lo) / rng.clamp_min(CONST_EPS)
    g[const] = 0.0
    return F.adaptive_avg_pool2d(g, ds.CELLS).flatten(1), const


def check_image_table(table, reference, mask, tol=CHECK_TOL):
    """Max abs difference between the recomputed per-image table and the dataset's, on the entities with an image;
    ValueError when it exceeds tol."""
    table, reference, mask = np.asarray(table), np.asarray(reference), np.asarray(mask, dtype=bool)
    if table.shape != reference.shape:
        raise ValueError(f"shape {table.shape} != reference {reference.shape}")
    diff = float(np.abs(table[mask].astype(np.float64) - reference[mask].astype(np.float64)).max())
    if not diff <= tol:
        raise ValueError(f"per-image table differs from ent_depth_stats.pt by {diff:.3g} (> {tol:g})")
    return diff


@torch.no_grad()
def compute(pool, scenes, n_ent, device, img_batch, workers):
    """Raw tables (per-image, per-box) [n_ent, 9], NaN where no box was encoded, and run statistics."""
    loader = DataLoader(ds.DepthSceneSet(pool, scenes), batch_size=img_batch, num_workers=workers,
                        collate_fn=ef.collate)
    zd = ef.load_zipdepth(device)
    out_img = np.full((n_ent, ds.CELLS * ds.CELLS), np.nan, dtype=np.float32)
    out_box = out_img.copy()
    n_img, n_box, n_const, errors, outside, t0 = 0, 0, 0, [], [], time.time()
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
        depth = (depth - lo) / (hi - lo).clamp_min(1e-6)            # as scripts/depth_stats.compute
        rois = torch.cat([torch.cat([torch.full((len(b["rois"]), 1), float(i)), b["rois"]], 1)
                          for i, b in enumerate(good)]).to(device)
        ids = np.array([e for b in good for e in b["ids"]])
        out_img[ids] = ds.grid_means(depth, rois).cpu().numpy()
        box, const = box_minmax_grid(depth, rois)
        out_box[ids] = box.cpu().numpy()
        n_const += int(const.sum())
        n_img += len(good)
        n_box += len(ids)
        if bi % 200 == 0:
            print(f"{n_img:,} images, {n_box:,} boxes, {time.time() - t0:.0f}s", flush=True)
    return out_img, out_box, {"n_images": n_img, "n_boxes": n_box, "n_constant_boxes": n_const,
                              "n_unreadable": len(errors), "errors": errors[:20], "n_outside": len(outside),
                              "sec": round(time.time() - t0, 1)}


def finish(raw_img, raw_box, data_dir, out_dir, info):
    """Fill, check the per-image table against data_dir/ent_depth_stats.pt, then write the .pt and the .json."""
    img, fill_img = ds.fill_table(raw_img, data_dir)
    box, fill_box = ds.fill_table(raw_box, data_dir)
    mask = torch.load(os.path.join(data_dir, "ent_img_mask.pt"), weights_only=True).numpy().astype(bool)
    ref = torch.load(os.path.join(data_dir, ds.TABLE), weights_only=True).numpy()
    diff = check_image_table(img, ref, mask)
    print(f"check: per-image table matches {ds.TABLE} (max abs diff {diff:.2e})", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, TABLE)
    torch.save({"grid9": torch.from_numpy(box), "grid9_image": torch.from_numpy(img), "model": "zipdepth",
                "norm": "box", "source": info["source"]}, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    info = {**info, "check_max_abs_diff": diff, "fill_box": fill_box["fill"], "fill_image": fill_img["fill"],
            "train_entities_with_image": fill_box["train_entities_with_image"],
            "filled_entities": fill_box["filled_entities"], "sha256": sha}
    with open(os.path.join(out_dir, INFO), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"wrote {path} (sha256 {sha}) and {INFO}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["vrd", "fill"])
    ap.add_argument("--data", required=True, help="dataset dir (entity2id.txt, train2id.txt, ent_img_mask.pt, "
                                                  "ent_depth_stats.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", help="vrd: Kaggle VRD zip")
    ap.add_argument("--src", help="fill: output dir of the vrd build, refilled for --data (same entity2id.txt)")
    ap.add_argument("--src-data", help="fill: the dataset dir the vrd build was made from (its ent_img_mask.pt)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--img-batch", type=int, default=16, help="images per ZipDepth batch")
    a = ap.parse_args(argv)
    info = {"table": TABLE, "content": f"{ds.CELLS}x{ds.CELLS} grid means of the ZipDepth depth map inside the "
                                       "entity box, min-max normalised per box (grid9) and per image (grid9_image)",
            "data": os.path.basename(os.path.normpath(a.data)), "code_commit": ds.code_commit()}
    try:
        if a.dataset == "fill":
            if not (a.src and a.src_data):
                ap.error("fill needs --src and --src-data")
            with open(os.path.join(a.data, "entity2id.txt"), "rb") as f1, \
                    open(os.path.join(a.src_data, "entity2id.txt"), "rb") as f2:
                if f1.read() != f2.read():
                    raise ValueError("fill: entity2id.txt of --src-data and --data differ")
            with open(os.path.join(a.src, INFO), encoding="utf-8") as f:
                src_info = json.load(f)
            if src_info["data"] != os.path.basename(os.path.normpath(a.src_data)):
                raise ValueError(f"fill: --src was built from {src_info['data']}, not from --src-data")
            src = torch.load(os.path.join(a.src, TABLE), weights_only=False)
            src_mask = torch.load(os.path.join(a.src_data, "ent_img_mask.pt"),
                                  weights_only=True).numpy().astype(bool)
            raw_img, raw_box = src["grid9_image"].numpy().copy(), src["grid9"].numpy().copy()
            raw_img[~src_mask] = np.nan
            raw_box[~src_mask] = np.nan
            info["source"] = {**src["source"], "refilled_from": src_info["data"]}
            info.update({k: src_info[k] for k in ("n_images", "n_boxes", "n_constant_boxes", "n_outside")})
            return finish(raw_img, raw_box, a.data, a.out, info)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pool, scenes, n_ent, _ = ef.vrd_scenes(a.data, a.zip)
        print(f"vrd: {len(scenes):,} images, {n_ent:,} entities, device {device}", flush=True)
        raw_img, raw_box, run = compute(pool, scenes, n_ent, device, a.img_batch, a.workers)
        if not a.zip:
            ap.error("vrd needs --zip")
        info["source"] = {"zip": os.path.basename(a.zip), "data": info["data"],
                          "depth_model": f"ZipDepth zipdepth_base ({ef.DEPTH_RES}px)", "grid_samples": ds.GRID}
        info.update(run)
        return finish(raw_img, raw_box, a.data, a.out, info)
    except ValueError as e:
        print(f"check FAILED: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
