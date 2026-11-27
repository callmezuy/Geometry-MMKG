"""
E3 cells (RESEARCH_PLAN P6.0 E3; docs/superpowers/specs/2026-11-27-e3-official-runs-design.md 5.1, 5.3-5.5): the
official test runs, everything the notebook cell "E3" needs that has no user interface.

* CELLS / SEEDS: the cells of every dataset and their seeds (section 4.8: 3 seeds, MKG-Y and Spatial-VG 1 seed).
* resolve(): the configuration of one cell = the tune run chosen by scripts/tune_summary.py `choose` in the cell's
  source folder (P2.10 tune, E2 box arms, E2b boxnorm), with the valid MRR / best epoch the seed-42 official run must
  repeat (train.py --expect-valid). A cell whose choice is not final is refused.
* cell_record() / check_cell_record(): e3_cell.json, written once per cell; every later seed must find the same
  configuration and the same table.
* table_digest(): sha256 of the tensors of a table (not of the file: box_geometry.pt also stores a `source` that
  depends on the machine).
* train_args(), stem(), seed_state(): the train.py command of one seed, its file stem, and whether it is done.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_summary as ts  # noqa: E402

MMKG_CELLS = ("no_depth", "fixed")
VRD_CELLS = ("no_depth", "depth_stats_only", "depth_map_only", "fixed", "box", "box_grid9", "boxnorm")
CELLS = {
    "FB15k-237-IMG": MMKG_CELLS,
    "FB15K-237-ind": MMKG_CELLS,
    "MKG-Y": MMKG_CELLS,
    "Spatial-VRD": VRD_CELLS,
    "Spatial-VRD-Trans": VRD_CELLS,
    "Spatial-VG": ("no_depth", "fixed", "depth_stats_only"),
}
SEEDS = {ds: (42,) if ds in ("MKG-Y", "Spatial-VG") else (42, 43, 44) for ds in CELLS}
BOX_CELLS = ("box", "box_grid9")
BOXNORM_CELL = "boxnorm"
TABLE_VARIANT = "depth_stats_only"   # the model of the box and boxnorm cells (E2, E2b)
RECORD = "e3_cell.json"


def cell_variant(cell):
    return TABLE_VARIANT if cell in BOX_CELLS or cell == BOXNORM_CELL else cell


def source_dir(dataset, cell):
    """Repo-relative folder that tune_summary.load_runs reads for this cell (<root>/<variant>/<config>/metrics)."""
    if cell in BOX_CELLS:
        return f"geometry-e2/{dataset}/{cell}"
    if cell == BOXNORM_CELL:
        return f"geometry-e2b/{dataset}/{BOXNORM_CELL}"
    return f"geometry-tune/{dataset}"


def resolve(results_dir, dataset, cell):
    """Configuration of one cell from a local copy of the results repo (results_dir/<source_dir>/...)."""
    variant = cell_variant(cell)
    src = source_dir(dataset, cell)
    root = os.path.join(results_dir, *src.split("/"))
    runs = ts.load_runs(root) if os.path.isdir(root) else []
    c = ts.choose(runs).get(variant)
    if c is None:
        raise ValueError(f"{dataset} {cell}: no tune run of {variant} in {src}")
    if not c["final"]:
        raise ValueError(f"{dataset} {cell}: the tune choice is not final ({c['action']}; edge points "
                         f"{c['edge_points']})")
    return {
        "dataset": dataset, "cell": cell, "variant": variant, "lr": c["config"]["lr"],
        "dropout": c["config"]["dropout"], "epochs": c["epochs"], "source_dir": src,
        "source_file": c["file"].replace(os.sep, "/"), "tune_code_commit": c["code_commit"],
        "options_tag": c["options_tag"], "n_configs": c["n_configs"],
        "expected": {"valid_mrr": c["valid_mrr"], "best_epoch": c["best_epoch"]},
        "lower_bound": bool(c["late_best_epoch"]),
    }


def table_digest(path, keys):
    """sha256 over the float32 bytes of blob[k] for k in keys, in that order (other entries ignored)."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode())
        h.update(np.ascontiguousarray(blob[k].to(torch.float32).numpy()).tobytes())
    return h.hexdigest()


def cell_record(resolved, depth_stats, table_sha256, e3_code_commit):
    """e3_cell.json content; depth_stats is reduced to '<file name>[:<key>]' (no machine path)."""
    if depth_stats:
        path, sep, key = depth_stats.rpartition(":") if ":" in os.path.basename(depth_stats) else (depth_stats, "", "")
        depth_stats = os.path.basename(path) + (f":{key}" if sep else "")
    return {**resolved, "depth_stats": depth_stats or None, "table_sha256": table_sha256,
            "e3_code_commit": e3_code_commit}


def check_cell_record(existing, new):
    """ValueError when two records of one cell differ in anything but the E3 code commit."""
    keys = (set(existing) | set(new)) - {"e3_code_commit"}
    diff = sorted(k for k in keys if existing.get(k) != new.get(k))
    if diff:
        raise ValueError(f"{RECORD} differs from the one on record in: "
                         + ", ".join(f"{k} ({existing.get(k)!r} -> {new.get(k)!r})" for k in diff))


def stem(dataset, variant, options_tag, seed):
    return f"geometry_{dataset}_{variant}_{options_tag}_seed{seed}"


def train_args(resolved, data_dir, seed, out, depth_stats_spec=None):
    """train.py arguments of one official seed; seed 42 carries the --expect-valid gate (exact float repr)."""
    a = ["--data", data_dir, "--variant", resolved["variant"], "--seed", str(seed), "--lr", repr(resolved["lr"]),
         "--dropout", repr(resolved["dropout"]), "--epochs", str(resolved["epochs"]), "--out", out]
    if depth_stats_spec:
        a += ["--depth-stats", depth_stats_spec]
    if seed == 42:
        e = resolved["expected"]
        a += ["--expect-valid", f"{e['valid_mrr']!r}:{e['best_epoch']}"]
    return a


def seed_state(metrics):
    """'todo' (no metrics), 'done' (test evaluated) or 'mismatch' (the seed-42 gate stopped it)."""
    if metrics is None:
        return "todo"
    if "test" in metrics:
        return "done"
    if (metrics.get("expect_valid") or {}).get("ok") is False:
        return "mismatch"
    raise ValueError("metrics without test and without a failed expect_valid: not an official E3 run")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
