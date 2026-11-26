"""
E2 box control (RESEARCH_PLAN P6.0, C4; docs/superpowers/specs/2026-11-26-e2-box-control-design.md): per-entity
box-geometry tables for Spatial-VRD and Spatial-VRD-Trans, read by `train.py --depth-stats <file>:<key>` with the
variant depth_stats_only, so the box enters exactly like the depth statistics (zero-initialised linear term, same
standardisation, mask and depth dropout). A control only: box coordinates are not a feature of the main model
(principle of 19/11/2026).

Output <out> = {"box": [N, 9], "box_grid9": [N, 18], "model": "box", "source": ...}, float32, NaN without a box:
  box       : x1, y1, x2, y2, w, h, centre x, centre y, log(w h), divided by the image size (the P2.9 probe set);
  box_grid9 : box, then the dataset's ent_depth_stats.pt (grid9) unchanged.
Boxes are cut as in scripts/extract_features.py: fully outside the image -> NaN; otherwise clamped to the image and
widened to MIN_BOX pixels around the centre. Entity '<photo_id>_<k>' = object k of the image (vrd_scenes).
Image sizes come from the annotation except IMAGE_SIZE_FIXES (one image whose annotated size is wrong).
Spatial-VRD-Trans has no annotation files but the same entity names: pass the Spatial-VRD annotations.
<out>.json holds the counts and the float64 column sums (NaN skipped); --check compares them with EXPECTED, the
numbers of the local build (counts exact, sums to relative 1e-6: log may differ by one ulp between CPUs).

  python scripts/box_geometry.py --data data/Spatial-VRD --annotations data/Spatial-VRD \
      --out results/geometry-e2/Spatial-VRD/box_geometry.pt --check
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

MIN_BOX = 2.0   # pixels, as scripts/extract_features.py
ANNOTATIONS = ("sg_train_annotations.json", "sg_test_annotations.json")
KEYS = ("box", "box_grid9")
REL_TOL = 1e-6
# Annotation image sizes that differ from the image file (checked on all 5,000 Spatial-VRD images, 26/11/2026);
# scripts/extract_features.py uses the size of the file, so the boxes are normalised by it too. photo_id -> (W, H).
IMAGE_SIZE_FIXES = {"9210906893": (683, 1024)}   # annotation says 500 x 374; its boxes reach y = 981

# Local build of 26/11/2026 (data v0); filled in from `--data ... --out ...` on both datasets.
EXPECTED = {
    "Spatial-VRD": {"n_entities": 93809, "n_with_box": 93807, "n_nan": 2,
        "col_sums_box": [33958.380678, 37197.432388, 57856.227609, 59747.504082, 23897.846935, 22550.071668,
            45907.304141, 48472.468213, -355002.472671],
        "col_sums_box_grid9": [33958.380678, 37197.432388, 57856.227609, 59747.504082, 23897.846935, 22550.071668,
            45907.304141, 48472.468213, -355002.472671, 44425.092439, 44135.121721, 43301.281033,
            48179.363419, 47929.470805, 47243.6579, 52046.348327, 51884.439029, 51382.443282],
    },
    "Spatial-VRD-Trans": {"n_entities": 93809, "n_with_box": 93807, "n_nan": 2,
        "col_sums_box": [33958.380678, 37197.432388, 57856.227609, 59747.504082, 23897.846935, 22550.071668,
            45907.304141, 48472.468213, -355002.472671],
        "col_sums_box_grid9": [33958.380678, 37197.432388, 57856.227609, 59747.504082, 23897.846935, 22550.071668,
            45907.304141, 48472.468213, -355002.472671, 44425.089853, 44135.119553, 43301.27891,
            48179.361018, 47929.468471, 47243.655995, 52046.346036, 51884.43699, 51382.441468],
    },
}


def read_entity2id(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if len(rows) != n:
        raise ValueError(f"{path}: header says {n} entities, read {len(rows)}")
    return {name: int(i) for name, i in rows}


def box_table(entity2id, annotation_files):
    """float64 [N, 9] box geometry of every entity (row = entity id), NaN without a usable box."""
    out = np.full((len(entity2id), 9), np.nan, dtype=np.float64)
    for path in annotation_files:
        with open(path, encoding="utf-8") as f:
            images = json.load(f)
        for img in images:
            pid = str(img.get("photo_id", os.path.splitext(img.get("filename", ""))[0]))
            W, H = (float(v) for v in IMAGE_SIZE_FIXES.get(pid, (img["width"], img["height"])))
            for k, obj in enumerate(img.get("objects", [])):
                eid = entity2id.get(f"{pid}_{k}")
                b = obj.get("bbox")
                if eid is None or not b:
                    continue
                x1, y1, x2, y2 = float(b["x"]), float(b["y"]), float(b["x"] + b["w"]), float(b["y"] + b["h"])
                if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0:
                    continue
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < MIN_BOX:
                    c = (x1 + x2) / 2
                    x1, x2 = max(0.0, c - MIN_BOX / 2), min(W, c + MIN_BOX / 2)
                if y2 - y1 < MIN_BOX:
                    c = (y1 + y2) / 2
                    y1, y2 = max(0.0, c - MIN_BOX / 2), min(H, c + MIN_BOX / 2)
                x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
                w, h = x2 - x1, y2 - y1
                out[eid] = (x1, y1, x2, y2, w, h, (x1 + x2) / 2, (y1 + y2) / 2, math.log(w * h))
    return out


def _col_sums(x):
    return [float(v) for v in np.nansum(x.astype(np.float64), axis=0)]


def build(data_dir, annotation_dir):
    """The two tables of one dataset (see the module docstring) plus a summary for check()."""
    entity2id = read_entity2id(os.path.join(data_dir, "entity2id.txt"))
    box = box_table(entity2id, [os.path.join(annotation_dir, fn) for fn in ANNOTATIONS])
    grid9 = torch.load(os.path.join(data_dir, "ent_depth_stats.pt"), map_location="cpu", weights_only=True).float()
    if tuple(grid9.shape) != (len(entity2id), 9):
        raise ValueError(f"{data_dir}/ent_depth_stats.pt: expected [{len(entity2id)}, 9], got {tuple(grid9.shape)}")
    box_t = torch.from_numpy(box).float()
    both = torch.cat([box_t, grid9], dim=1)
    n_nan = int(np.isnan(box).any(axis=1).sum())
    summary = {"dataset": os.path.basename(os.path.normpath(data_dir)), "n_entities": len(entity2id),
               "n_with_box": len(entity2id) - n_nan, "n_nan": n_nan,
               "col_sums_box": _col_sums(box_t.numpy()), "col_sums_box_grid9": _col_sums(both.numpy())}
    return {"box": box_t, "box_grid9": both, "model": "box",
            "source": {"annotations": [os.path.basename(annotation_dir.rstrip("/\\"))] + list(ANNOTATIONS),
                       "grid9": "ent_depth_stats.pt"},
            "summary": summary}


def check(summary, dataset, expected=None):
    """ValueError unless the counts equal and the column sums match the local build within REL_TOL."""
    expected = EXPECTED if expected is None else expected
    if dataset not in expected:
        raise ValueError(f"no expected numbers for {dataset}")
    exp = expected[dataset]
    for k in ("n_entities", "n_with_box", "n_nan"):
        if summary[k] != exp[k]:
            raise ValueError(f"{dataset}: {k} = {summary[k]}, expected {exp[k]}")
    for k in ("col_sums_box", "col_sums_box_grid9"):
        for j, (a, b) in enumerate(zip(summary[k], exp[k])):
            if abs(a - b) > REL_TOL * max(1.0, abs(b)):
                raise ValueError(f"{dataset}: {k}[{j}] = {a!r}, expected {b!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset folder (entity2id.txt, ent_depth_stats.pt)")
    p.add_argument("--annotations", required=True, help="folder with the Spatial-VRD sg_*_annotations.json")
    p.add_argument("--out", required=True, help="output .pt (a .json summary is written next to it)")
    p.add_argument("--check", action="store_true", help="compare with EXPECTED (exit 1 when different)")
    a = p.parse_args(argv)
    out = build(a.data, a.annotations)
    s = out.pop("summary")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print(f"{s['dataset']}: {s['n_with_box']:,} / {s['n_entities']:,} entities with a box ({s['n_nan']} without) "
          f"-> {a.out}", flush=True)
    if a.check:
        try:
            check(s, s["dataset"])
        except ValueError as e:
            print(f"check FAILED: {e}", flush=True)
            sys.exit(1)
        print("check: matches the local build", flush=True)
    return s


if __name__ == "__main__":
    main()
