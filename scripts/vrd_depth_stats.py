"""
Per-entity depth statistics for Spatial-VRD (RESEARCH_PLAN.md P2.9 A2/A3): a few numbers that keep the shared
per-image depth scale, to compare with the DINOv2 embedding of the depth crop.

Depth map (--model):
  zipdepth : same as scripts/extract_features.py (zipdepth_base, 384x384, ImageNet norm)
  dav2     : Depth Anything V2 base (depth-anything/Depth-Anything-V2-Base-hf), 518x518, ImageNet norm
Both give relative inverse depth; it is resized to 384x384 and min-max normalised per image
(1 = nearest point of the image, 0 = farthest). Each entity box (same boxes as the feature extraction) is sampled
on a 32x32 grid (roi_align) and summarised three ways:
  stats5 : mean, median, 10th percentile, 90th percentile, std
  robust : median of the samples inside [P10, P90] (as DM-KG, arXiv 2607.12319)
  grid9  : means of a 3x3 grid of sub-regions (close to ADVRD's regional depth, Gan & He 2022)
Box coordinates themselves are NOT part of the output (decision 19/11/2026: no box coordinates as features).

Output: <out> = {"stats5": [n,5], "robust": [n,1], "grid9": [n,9], ...}, NaN for entities without an encoded box.
Run from Code/:  python scripts/vrd_depth_stats.py --model zipdepth --out results/probe/vrd_depth_stats_zipdepth.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as ef  # noqa: E402

GRID = 32
DAV2_NAME = "depth-anything/Depth-Anything-V2-Base-hf"
DAV2_RES = 518


def load_model(name, device):
    if name == "zipdepth":
        zd = ef.load_zipdepth(device)
        return lambda x: zd(ef.norm_img(x))
    from transformers import AutoModelForDepthEstimation

    m = AutoModelForDepthEstimation.from_pretrained(DAV2_NAME).to(device).eval()

    def run(x):
        x = F.interpolate(x, size=(DAV2_RES, DAV2_RES), mode="bicubic", align_corners=False).clamp(0, 1)
        return m(pixel_values=ef.norm_img(x)).predicted_depth.unsqueeze(1)
    return run


def summarise(g):
    """g: [boxes, GRID, GRID] normalised depth samples -> stats5 [b,5], robust [b,1], grid9 [b,9]."""
    flat = g.flatten(1)
    q = torch.quantile(flat, torch.tensor([0.5, 0.1, 0.9], device=g.device), dim=1).T  # [b,3]
    stats5 = torch.stack([flat.mean(1), q[:, 0], q[:, 1], q[:, 2], flat.std(1)], 1)
    inside = (flat >= q[:, 1:2]) & (flat <= q[:, 2:3])
    robust = torch.nanmedian(flat.masked_fill(~inside, float("nan")), dim=1).values[:, None]
    grid9 = F.adaptive_avg_pool2d(g[:, None], 3).flatten(1)
    return stats5, robust, grid9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Spatial-VRD")
    ap.add_argument("--zip", default="data/raw/vrd/visual-relationship-detection-vrd-dataset.zip")
    ap.add_argument("--model", choices=["zipdepth", "dav2"], default="zipdepth")
    ap.add_argument("--out", default=None, help="default results/probe/vrd_depth_stats_<model>.pt")
    ap.add_argument("--img-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_path = a.out or f"results/probe/vrd_depth_stats_{a.model}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    pool, scenes, n_ent, _ = ef.vrd_scenes(a.data, a.zip)
    loader = DataLoader(ef.SceneSet(pool, scenes), batch_size=a.img_batch, num_workers=a.workers,
                        collate_fn=ef.collate)
    model = load_model(a.model, device)
    out = {k: np.full((n_ent, d), np.nan, dtype=np.float32) for k, d in (("stats5", 5), ("robust", 1), ("grid9", 9))}
    n_img, n_box, errors = 0, 0, []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            good = [b for b in batch if b["ok"]]
            errors += [b["err"] for b in batch if not b["ok"] and b["err"]]
            if not good:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                depth = model(torch.stack([b["full"] for b in good]).to(device)).float()
            if depth.shape[-1] != ef.DEPTH_RES or depth.shape[-2] != ef.DEPTH_RES:
                depth = F.interpolate(depth, size=(ef.DEPTH_RES, ef.DEPTH_RES), mode="bilinear", align_corners=False)
            lo = depth.amin(dim=(1, 2, 3), keepdim=True)
            hi = depth.amax(dim=(1, 2, 3), keepdim=True)
            depth = (depth - lo) / (hi - lo).clamp_min(1e-6)
            rois = torch.cat([torch.cat([torch.full((len(b["rois"]), 1), float(i)), b["rois"]], 1)
                              for i, b in enumerate(good)]).to(device)
            g = roi_align(depth, rois, output_size=(GRID, GRID), spatial_scale=1.0, sampling_ratio=2,
                          aligned=True)[:, 0]  # [boxes, GRID, GRID]
            ids = np.array([e for b in good for e in b["ids"]])
            for k, v in zip(("stats5", "robust", "grid9"), summarise(g)):
                out[k][ids] = v.cpu().numpy()
            n_img += len(good)
            n_box += len(ids)
            if bi % 100 == 0:
                print(f"{n_img} images, {n_box} boxes, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({**{k: torch.from_numpy(v) for k, v in out.items()}, "model": a.model, "n_images": n_img,
                "n_boxes": n_box, "errors": errors}, out_path)
    print(f"wrote {out_path}: {n_img} images, {n_box} boxes, {int(np.isnan(out['stats5'][:, 0]).sum())} entities "
          f"without stats, {len(errors)} unreadable images, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
