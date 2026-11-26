# E2b Shared-Scale Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a per-box min-max normalised grid9 table for Spatial-VRD / VRD-Trans, tuned through the existing `depth_stats_only` variant, summarised next to E1 / E2.

**Architecture:** data level only. A new script computes both normalisations in one ZipDepth pass and checks the per-image one against the dataset's `ent_depth_stats.pt`; `train.py --depth-stats <file>:grid9` reads it unchanged; the notebook E2 tuning loop becomes a shared function used by E2 and E2b; `e2_summary.py` gains `--e2b`.

**Tech Stack:** Python, PyTorch, torchvision `roi_align`, marimo, huggingface_hub.

**Spec:** `docs/superpowers/specs/2026-11-26-e2b-shared-scale-design.md`

## Global Constraints

- Do not modify `scripts/depth_stats.py`, the model, `load_depth_stats`, `baselines/`, or v0 data on HF.
- Valid only (`--tune`), seed 42; results under `geometry-e2b/<dataset>/boxnorm/depth_stats_only/<cfg>/`.
- Reproduction check: per-image table vs `ent_depth_stats.pt`, max abs diff ≤ 1e-3 on entities with an image, else exit 1 and write nothing.
- Constant box: max − min < 1e-6 → 9 zeros; counted in the json.
- Tag: `inorm-ddrop0.3-tproj-dstat_boxnorm_grid9` (file name `depth_stats_boxnorm.pt`, key `grid9`).
- Code, comments, commit messages in English; commit / push / upload only when the user says so.

## Review Focus

- VRD-Trans has no images of its own: the raw VRD table is refilled with VRD-Trans's mask / train (`fill_table`), and checked against VRD-Trans's own `ent_depth_stats.pt` (test: fill path in Task 1).
- The 2 boxes outside their image stay without statistics; `fill_table` asserts that rows with statistics == `ent_img_mask` (covered by `fill_table` itself; the build prints counts).
- The notebook must refuse a downloaded table whose sha256 differs from the constant (Task 3, manual check of the code path).
- `e2_summary` without `--e2b` must give exactly the old table (Task 2 test).
- E2 cell behaviour must not change after the refactor: same stems / folders (Task 3: compare generated stem for one E2 point with an existing HF metrics name).

---

### Task 1: `scripts/depth_stats_boxnorm.py`

**Files:** Create `scripts/depth_stats_boxnorm.py`, `tests/test_depth_stats_boxnorm.py`.

**Interfaces — Produces:**
- `box_minmax_grid(depth: Tensor[B,1,R,R], rois: Tensor[n,5]) -> (Tensor[n,9], Tensor[n] bool constant)`
- `check_image_table(table: ndarray[N,9], reference: ndarray[N,9], mask: ndarray[N] bool, tol=1e-3) -> float` (max abs diff; ValueError when > tol)
- `compute(pool, scenes, n_ent, device, img_batch, workers) -> (raw_image [N,9], raw_box [N,9], run dict)`
- CLI: `vrd --data D --zip Z --out DIR` and `fill --src VRD_OUT --src-data VRD_DATA --data D --out DIR` (fill sets the rows of entities without an image in the VRD build to NaN using VRD's `ent_img_mask.pt`, then refills for `--data`, as `depth_stats.py fill`).
- `finish(raw_img, raw_box, data_dir, out_dir, info) -> dict`: fill, check, write.
- Output `<out>/depth_stats_boxnorm.pt` = `{"grid9", "grid9_image", "model": "zipdepth", "norm": "box", "source"}` (no absolute paths) + `<out>/depth_stats_boxnorm.json` (counts, n_constant_boxes, check max diff, sha256 of the .pt, code commit).

- [ ] Tests: affine a·d + b inside one box leaves that box's 9 numbers unchanged (atol 1e-5) and the other box's too; constant box → zeros and flagged; per-image branch equals `depth_stats.grid_means`; `check_image_table` raises at 2e-3, passes at 1e-4, ignores rows with mask False; `train.py --depth-stats <file>:grid9` 1-epoch smoke on the tiny scene-graph dataset: tag ends `dstat_boxnorm_grid9`, `depth_stats_dim` 9, `in_image` present, no `test`.
- [ ] Implement; run `python tests/test_depth_stats_boxnorm.py` → OK.

### Task 2: `scripts/e2_summary.py --e2b`

**Files:** Modify `scripts/e2_summary.py`; test `tests/test_e2_summary.py`.

- `summarize(dataset, e1_root, e2_root, e2b_root=None)`; with `e2b_root`, add row `boxnorm` (chosen by `tune_summary.choose` on `<e2b_root>/boxnorm`), rows `boxnorm` − `no_depth`, `depth_stats_only` − `boxnorm`, and a line "share of the depth_stats gain kept by boxnorm: x.xx (valid MRR)". Missing arm → "missing".
- [ ] Tests: old table unchanged without `--e2b` (string equality with a call that omits it); with `--e2b`, the three rows exist and the share line is right on fake numbers.

### Task 3: notebook

**Files:** Modify `notebooks/geometry.py`.

- New cell exporting `tune_depth_stats_arm(exp, arm, table_spec, tag, status)` (the current E2 loop: base grid, then up to 6 rounds of edge / convergence rules, skip finished points, push results); the E2 run cell calls it with `exp="geometry-e2"`, `tag=f"inorm-ddrop0.3-tproj-dstat_box_geometry_{arm}"`.
- New E2b cells: button; download `geometry-e2b/<DATASET>/depth_stats_boxnorm.pt` from the results repo, compare sha256 with `E2B_SHA256[DATASET]`, stop on mismatch; call the function with `exp="geometry-e2b"`, `arm="boxnorm"`, `table_spec=f"{path}:grid9"`, `tag="inorm-ddrop0.3-tproj-dstat_boxnorm_grid9"`; summary cell passes `e2b_root`.
- Intro paragraph gains an E2b line.
- [ ] `python -c "import ast; ast.parse(open('notebooks/geometry.py', encoding='utf-8').read())"` and `marimo check` if available.

### Task 4: build, record, plan text

- [ ] Build locally: `python scripts/depth_stats_boxnorm.py vrd --data data/Spatial-VRD --zip data/raw/vrd/visual-relationship-detection-vrd-dataset.zip --out <scratch>/e2b/Spatial-VRD`, then `fill --src <scratch>/e2b/Spatial-VRD --data data/Spatial-VRD-Trans --out <scratch>/e2b/Spatial-VRD-Trans`. Both checks pass.
- [ ] Put both sha256 into `E2B_SHA256` in the notebook; RESEARCH_PLAN P6.0: E2b design line (spec path, sha256, check numbers).
- [ ] Ask the user before committing, pushing and uploading the two tables to `geometry-mmkg-results/geometry-e2b/<dataset>/`.
