# E2 box control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-entity box-geometry tables fed through the existing `--depth-stats PATH:KEY` path, a notebook cell that
tunes the two E2 arms on Molab, and a summary table against the E1 runs.

**Architecture:** Data level only (spec section 4, approach A). `scripts/box_geometry.py` builds one file
`box_geometry.pt` = {"box": [N, 9], "box_grid9": [N, 18]} from the Spatial-VRD annotations; `train.py --variant
depth_stats_only --depth-stats <file>:<arm>` trains it unchanged. `scripts/e2_summary.py` tabulates.

**Tech Stack:** Python, numpy, torch, marimo notebook. Tests run as `python tests/<file>.py` (no pytest installed).

**Spec:** `docs/superpowers/specs/2026-11-26-e2-box-control-design.md`

## Global Constraints

- No change to `core/`, `scripts/train.py`, `baselines/`, or data v0 on HF.
- Every run is `--tune` (valid only), seed 42; results under `geometry-e2/<dataset>/<arm>/depth_stats_only/<config>/`.
- Arms: `box` (9 numbers: x1, y1, x2, y2, w, h, cx, cy, log(w·h), normalised by image size), `box_grid9` (box + the
  dataset's `ent_depth_stats.pt`, 18 numbers).
- Box clipping = `extract_features.py`: fully outside → NaN; clamp to [0, W] × [0, H]; `MIN_BOX` = 2 px widened around the
  centre.
- Check of the Molab-built table: counts exact, float64 column sums (NaN skipped) within relative 1e-6 of `EXPECTED`.
- Code, comments and commit messages in English; commit / push only when the user says so.

## Review Focus

- Annotation box with `w` or `h` = 0 → widened to `MIN_BOX`, log finite (test in Task 1).
- Entity in `entity2id` with no annotation object → NaN row, not a crash (Task 1).
- `box_grid9` built for VRD-Trans must use VRD-Trans's own `ent_depth_stats.pt`, not VRD's (Task 1: `--data` decides).
- Notebook rerun after a finished arm must skip existing points (same `remote_files` check as the tune cell, Task 3).
- Summary when an arm has no runs yet → "missing" row, no exception (Task 2).

---

### Task 1: box table builder

**Files:** Create `scripts/box_geometry.py`, `tests/test_box_geometry.py`.

**Interfaces — Produces:**
- `box_table(entity2id: dict[str, int], annotation_files: list[str]) -> np.ndarray` float64 [N, 9], NaN rows without box.
- `build(data_dir: str, annotation_dir: str) -> dict` with keys `box`, `box_grid9` (float32 torch tensors), `model`
  = "box", `source`, `summary` (`n_entities`, `n_with_box`, `n_nan`, `col_sums_box`, `col_sums_box_grid9`).
- `check(summary: dict, dataset: str) -> None`, raises `ValueError` when it does not match `EXPECTED[dataset]`.
- CLI: `python scripts/box_geometry.py --data DIR --annotations DIR --out FILE [--check]` writes FILE and FILE's
  `.json` summary.

- [ ] Step 1: tests — hand-built 2 images (one 100×50 image with a normal box, one box crossing the right edge, one zero-width
  box; one image whose only box lies fully outside), expected rows computed by hand; unknown name skipped; entity without
  object → NaN; `box_grid9[:, 9:]` equals the given grid9 table; `check` passes on its own summary and fails when a sum is
  changed by 1e-3; tiny train.py run with `--variant depth_stats_only --depth-stats <file>:box` on a fake `Spatial-VRD`
  gives `model_kwargs.depth_stats_dim` 9 (18 for `box_grid9`) and options tag ending `dstat_box_geometry_box`.
- [ ] Step 2: run, expect failure (module missing).
- [ ] Step 3: implement; build the real tables locally for both datasets and paste their summaries into `EXPECTED`.
- [ ] Step 4: run the tests and `--check` on both datasets, expect pass.

### Task 2: E2 summary

**Files:** Create `scripts/e2_summary.py`, `tests/test_e2_summary.py`.

**Interfaces — Consumes:** `tune_summary.load_runs(root)`, `tune_summary.choose(runs)`, `e1_summary.E1_CONFIGS`,
`e1_summary.config_dir`. **Produces:** `ARMS = ("box", "box_grid9")`, `summarize(dataset, e1_root, e2_root) -> str`.

- [ ] Step 1: tests — fake `geometry-e1` (4 variants) and `geometry-e2/<arm>/depth_stats_only/<cfg>` trees; table has the
  rows `no_depth`, `depth_stats_only`, `box`, `box_grid9` and the difference rows `box_grid9 − box`,
  `box − depth_stats_only`; missing arm → "missing".
- [ ] Step 2: run, expect failure. Step 3: implement. Step 4: run, expect pass.

### Task 3: notebook E2 cells

**Files:** Modify `notebooks/geometry.py` (append after the E1 cells; intro paragraph).

- [ ] Step 1: button cell + arm multiselect (`box`, `box_grid9`, both by default).
- [ ] Step 2: run cell: only Spatial-VRD / Spatial-VRD-Trans; download `sg_*_annotations.json` from `callmezuy/spatial-vrd`
  (revision `DATA_REVISION`), build + `check`; base grid (tune_summary `BASE_GRID`) then edge / convergence rounds (max 6)
  per arm with `tune_summary.choose`, skip points present on HF, `run_resumable` + `push_results` as the tune cell.
- [ ] Step 3: summary cell: download `geometry-e1/<ds>` and `geometry-e2/<ds>` metrics, print `e2_summary.summarize`.
- [ ] Step 4: `python -c "import ast; ast.parse(open('notebooks/geometry.py', encoding='utf-8').read())"` and
  `marimo check` if available, expect no error.

### Task 4: docs

- [ ] RESEARCH_PLAN P6.0 E2 line: spec path, arms, folders, expected column sums. Spec status → approved 26/11/2026.
- [ ] Run every `tests/test_*.py`, expect exit 0. Commit only when the user says so.
