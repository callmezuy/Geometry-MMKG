# E3 Official Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the official 3-seed test runs of RESEARCH_PLAN P6.0 E3 (59 runs, 7 cells × 6 datasets as in spec 5.1), a seed-42 replication gate that blocks test evaluation, the pre-registered summary / tests (E4), and the image-masking control on the MMKG checkpoints.

**Architecture:** `train.py` gains `--expect-valid` (check before test). A new module `scripts/e3_cells.py` holds everything the notebook needs that has no UI (cell list, config resolution through `tune_summary.choose`, `e3_cell.json`, train arguments, seed state). A new notebook cell "E3" loops cells × seeds with the existing `run_resumable` / `push_results`. `scripts/e3_summary.py` reads `geometry-e3/` and applies the rules of spec section 6; `scripts/e3_mask_eval.py` re-evaluates MMKG checkpoints with every image masked.

**Tech Stack:** Python, PyTorch, NumPy, SciPy (`ttest_rel`), marimo, huggingface_hub.

**Spec:** `docs/superpowers/specs/2026-11-27-e3-official-runs-design.md`

## Global Constraints

- Do not modify the model (`core/models/*` architecture), `baselines/`, the tuning grid / rules, checkpoint selection, or v0 data.
- Never overwrite `geometry/`, `geometry-tune/` (except the new `geometry-tune/Spatial-VG/depth_stats_only/`), `geometry-e1/`, `geometry-e2/`, `geometry-e2b/`. Official results go to `geometry-e3/<dataset>/<cell>/`.
- Seed-42 gate: same best epoch and |best valid MRR − tune MRR| ≤ 0.0005, checked before test; on mismatch no test evaluation.
- Epochs = budget of the chosen tune run, passed explicitly; `--valid-every` left at the dataset default. No convergence rerun for seeds 43 / 44.
- Verdict threshold 0.003; "yes" needs all 3 seeds the same sign and every per-seed bootstrap 95% CI excluding 0. Bootstrap 10,000 resamples. Image groups below 500 triples: no verdict.
- Code, comments, commit messages in English. Commit / push / upload only when the user says so, listing the changes first. `RESEARCH_PLAN.md` currently holds another session's uncommitted baseline edits (MoMoK / CMR, sections P3): never stage them; commit only this plan's hunks (Task 6).
- Tests are plain scripts (`python tests/test_x.py`; pytest is not installed locally), each with the `if __name__ == "__main__"` runner used by the other test files.

## Review Focus

- A Molab session dies between seed 42 and seed 43: the next click must skip seed 42 (metrics with `test` on HF), re-check `e3_cell.json`, and continue; a resumed seed-42 run must still be gated (Task 2 test: `seed_state` of a finished / mismatched / missing metrics; `--expect-valid` survives resume because it is ignored in the resume key, Task 1 test).
- The box table rebuilt in a later session must give the same digest, although `box_geometry.pt` stores a `source` with machine paths: digest the tensors, not the file (Task 2 test: two saves with different `source` → same digest).
- A tune float such as 0.33021 must reach `--expect-valid` without rounding (Task 2 test: `train_args` round-trips the exact float).
- Rank files of two cells (or seeds) over different triple sets must stop the summary, not silently pair a subset (Task 4 test).
- A cell whose `choose` result is not `final` (VG `depth_stats_only` before its tune is done) must be reported and skipped, not run with a partial choice (Task 2 test).

---

### Task 1: `train.py --expect-valid`

**Files:** Modify `scripts/train.py` (parse_args, `RESUME_IGNORED_ARGS`, the block after the training loop, docstring). Test: `tests/test_train_cli.py`.

**Interfaces — Produces:**
- `EXPECT_VALID_TOL = 5e-4`; `EXPECT_MISMATCH_EXIT = 3`.
- `parse_expect_valid(s: str) -> tuple[float, int]` (argparse `type`; `"0.33021:170"` → `(0.33021, 170)`; malformed → `argparse.ArgumentTypeError`).
- `--expect-valid MRR:EPOCH` (default None; refused together with `--dev` / `--tune`).
- metrics key `expect_valid`: `{"mrr", "epoch", "got_mrr", "got_epoch", "tol", "ok"}` when the option is given, absent otherwise.
- On mismatch: write `ckpt_<stem>.pt` (unless `--no-ckpt`) and `metrics_<stem>.json` without `test`, no rank files, remove the resume file, `raise SystemExit(EXPECT_MISMATCH_EXIT)`.

- [ ] **Step 1: Write the failing test** `test_expect_valid_gates_the_test_evaluation` (tiny MMKG dataset from `test_feature_transe._tiny_dataset`, directory name `Tiny`; `--epochs 2 --valid-every 1 --batch-size 8 --no-resume`):
  1. run with `--dev` → read `best_valid["mrr"]`, `best_epoch`;
  2. official run with `--expect-valid f"{mrr!r}:{epoch}"` → `"test" in m`, `m["expect_valid"]["ok"] is True`, a `ranks_*_test.csv` and a `ckpt_*.pt` exist;
  3. official run with `f"{mrr + 0.001!r}:{epoch}"` (other `--out`) → `SystemExit` with code 3, `"test" not in m`, `m["expect_valid"]["ok"] is False`, `m["expect_valid"]["got_mrr"] == mrr`, no `ranks_*` file, `ckpt_*.pt` exists, no `resume_*.pt`;
  4. same with the epoch off by one → code 3;
  5. `f"{mrr + 0.0004!r}:{epoch}"` → passes (inside the tolerance);
  6. official run without the option → `"expect_valid" not in m`.
  Plus `test_parse_expect_valid` (`"0.3:7"`, malformed `"0.3"`, `"x:1"`) and an assertion that `"expect_valid" in train.RESUME_IGNORED_ARGS`.
- [ ] **Step 2:** `python tests/test_train_cli.py` → FAIL (unknown option).
- [ ] **Step 3: Implement.** The comparison sits between the training loop and `model.load_state_dict(best["state"])`; it uses `best["mrr"]` / `best["epoch"]` as they are (no rounding). `--dev` / `--tune` with `--expect-valid` → `p.error`.
- [ ] **Step 4:** `python tests/test_train_cli.py` and `python tests/test_depth_stats_boxnorm.py` → all pass (the second runs `train.main` end to end).
- [ ] **Step 5: Commit** `scripts/train.py tests/test_train_cli.py` — "train.py: --expect-valid gates the test evaluation on the seed-42 replication".

### Task 2: `scripts/e3_cells.py`

**Files:** Create `scripts/e3_cells.py`, `tests/test_e3_cells.py`.

**Interfaces — Consumes:** `tune_summary.load_runs(root)`, `tune_summary.choose(runs)` (choice keys `config`, `epochs`, `best_epoch`, `valid_mrr`, `final`, `late_best_epoch`, `n_configs`, `file`, `options_tag`, `code_commit`); `train.EXPECT_VALID_TOL` is not needed here.

**Interfaces — Produces:**
- `SEEDS: dict[str, tuple[int, ...]]` — 3 seeds for FB15k-237-IMG, FB15K-237-ind, Spatial-VRD, Spatial-VRD-Trans; `(42,)` for MKG-Y, Spatial-VG.
- `CELLS: dict[str, tuple[str, ...]]` — MMKG: `("no_depth", "fixed")`; the two VRD sets: `("no_depth", "depth_stats_only", "depth_map_only", "fixed", "box", "box_grid9", "boxnorm")`; Spatial-VG: `("no_depth", "fixed", "depth_stats_only")`.
- `BOX_CELLS = ("box", "box_grid9")`, `BOXNORM_CELL = "boxnorm"`; `cell_variant(cell) -> str` (`depth_stats_only` for those three, else the cell name).
- `source_dir(dataset, cell) -> str` — repo-relative: `geometry-tune/<ds>` (root holding `<variant>/<cfg>/`) for plain cells, `geometry-e2/<ds>/<cell>`, `geometry-e2b/<ds>/boxnorm`.
- `resolve(results_dir: str, dataset: str, cell: str) -> dict` — keys `dataset, cell, variant, lr, dropout, epochs, source_dir, source_file, tune_code_commit, options_tag, n_configs, expected: {"valid_mrr", "best_epoch"}, lower_bound`; `lower_bound = choice["late_best_epoch"]`; raises `ValueError` when there is no choice or it is not `final`.
- `table_digest(path: str, keys: tuple[str, ...]) -> str` — sha256 over the float32 bytes of `blob[k]` for k in `keys` (in that order), ignoring every other entry.
- `cell_record(resolved: dict, depth_stats: str | None, table_sha256: str | None, e3_code_commit: str) -> dict` — the `e3_cell.json` content (spec 5.5); `depth_stats` is a file name plus key (`box_geometry.pt:box`), never a path.
- `check_cell_record(existing: dict, new: dict) -> None` — `ValueError` naming the differing keys, ignoring `e3_code_commit`.
- `stem(dataset, variant, options_tag, seed) -> str` = `f"geometry_{dataset}_{variant}_{options_tag}_seed{seed}"`.
- `train_args(resolved, data_dir, seed, out, depth_stats_spec=None) -> list[str]` — `--data --variant --seed --lr --dropout --epochs --out` (+ `--depth-stats spec`), plus `--expect-valid f"{mrr!r}:{epoch}"` only when `seed == 42`.
- `seed_state(metrics: dict | None) -> str` — `"todo"` (None), `"done"` (has `test`), `"mismatch"` (`expect_valid.ok` false, no `test`); a metrics without `test` and without `expect_valid` → `ValueError`.

- [ ] **Step 1: Write the failing tests** with fake tune folders (metrics json with the fields `load_runs` reads: `tune`, `variant`, `config`, `seed`, `options_tag`, `epoch_budget.epochs`, `best_epoch`, `best_valid{mrr,hits@1,hits@10,mr}`, `code_commit`):
  - `test_cells_and_seeds`: 59 runs in total (`sum(len(CELLS[d]) * len(SEEDS[d]))`), `cell_variant("boxnorm") == "depth_stats_only"`.
  - `test_resolve_picks_the_choice`: two configs, the better one with `epochs` 200 and best epoch 200 after a 100-epoch run of the same config → `epochs == 200`, `lower_bound is True`, `expected` equals its MRR / epoch; a single non-final variant (best at the lr edge) → `ValueError`; missing folder → `ValueError`.
  - `test_box_source_dirs`: `source_dir("Spatial-VRD", "box_grid9") == "geometry-e2/Spatial-VRD/box_grid9"`, boxnorm → `geometry-e2b/Spatial-VRD/boxnorm`, `fixed` → `geometry-tune/Spatial-VRD`.
  - `test_table_digest_ignores_source`: same tensors saved twice with different `source` strings → equal digests; one changed value → different.
  - `test_cell_record_check`: equal except `e3_code_commit` → no error; different `lr` or `table_sha256` → `ValueError`; record has no absolute path (`depth_stats` is a basename).
  - `test_train_args`: seed 42 carries `--expect-valid` with `float(arg.split(":")[0]) == expected` exactly (use 0.33021234); seed 43 carries none; `--epochs` always present.
  - `test_seed_state`: the four cases above.
- [ ] **Step 2:** `python tests/test_e3_cells.py` → FAIL (no module).
- [ ] **Step 3: Implement** `scripts/e3_cells.py` (module docstring cites the spec; imports `tune_summary` like `e2_summary.py`).
- [ ] **Step 4:** `python tests/test_e3_cells.py` → pass. Then, against the real HF metrics already in the scratchpad (`snapshot_download` of `geometry-tune/`, `geometry-e2/`, `geometry-e2b/` metrics), print `resolve` for every cell of spec 5.1 and compare with the table there (Spatial-VG `depth_stats_only` must raise).
- [ ] **Step 5: Commit** `scripts/e3_cells.py tests/test_e3_cells.py` — "E3: cell list, config resolution and cell records".

### Task 3: `scripts/e3_summary.py` — statistics and rules

**Files:** Create `scripts/e3_summary.py` (pure functions part), `tests/test_e3_summary.py`.

**Interfaces — Consumes:** `compute_metrics.paired_bootstrap(rr_a, rr_b, n_boot, seed) -> (delta, lo, hi, p)`.

**Interfaces — Produces:**
- `THRESHOLD = 0.003`, `N_BOOT = 10000`, `BOOT_SEED = 0`, `MIN_GROUP = 500`.
- `verdict(deltas: list[float], cis: list[tuple[float, float]], thr=THRESHOLD) -> str` ∈ `{"yes", "yes (negative)", "equal", "unclear"}`. Several seeds: `equal` if |mean| ≤ thr; `yes` if mean > thr, every delta > 0 and every lo > 0; mirrored for negative; else `unclear`. One seed: `yes` / `yes (negative)` if |Δ| > thr and the CI excludes 0, `equal` if |Δ| ≤ thr, else `unclear`.
- `pair_stats(rr_a: dict[int, ndarray], rr_b: dict[int, ndarray], n_boot=N_BOOT) -> dict` — per seed `{delta, lo, hi, p}`, `mean`, `std` (ddof 1, None for one seed), `t`, `p_t` (`scipy.stats.ttest_rel`, None for one seed), `verdict`. Seeds must match between a and b (else `ValueError`).
- `PAIRS: dict[str, list[tuple[str, str, str]]]` keyed by dataset kind `"mmkg" | "vrd" | "vg"`: (a, b, label). MMKG: `("fixed", "no_depth", "C2")`. VRD: `("depth_stats_only", "no_depth", "depth")`, C3: `("depth_stats_only", "no_depth")`, `("fixed", "depth_map_only")`, `("depth_map_only", "no_depth")`, `("fixed", "depth_stats_only")`; C4: `("box_grid9", "box")`, `("box", "depth_stats_only")`; E2b: `("boxnorm", "no_depth")`, `("depth_stats_only", "boxnorm")`. VG: `("depth_stats_only", "no_depth")`, `("fixed", "no_depth")`, `("fixed", "depth_stats_only")`.
- Claim readers over the per-dataset verdicts / means (spec 6.2), each returning a short English sentence plus a status code:
  - `read_c2(verdicts: dict[ds, str]) -> (status, text)` — `holds` when all three MMKG are `equal`; `rewrite` naming datasets with `yes*`; else `report as is`; `incomplete` when a dataset is missing.
  - `read_c3(v: dict[ds, dict[pair, str]])` — `holds` when both depth_stats pairs are `yes` on both VRD sets and no vector pair is `yes`; else `per verdicts`.
  - `read_c4(v)` — `yes` both → `C4 yes`; `equal` both → `C4 no`; else `inconsistent`.
  - `read_e2b(means: dict[ds, (bn_minus_nd, dso_minus_bn, share)])` — per dataset `needed` (bn − nd ≤ thr), `not needed` (dso − bn ≤ thr), `partial` (both > thr), checked in this order; overall = the common branch, else `inconsistent`.
- `lower_bound_note(a, b, lower_bound_cells: set[str]) -> str` — "" or e.g. "`boxnorm` is a lower bound: the difference would shrink if it were higher".

- [ ] **Step 1: Write the failing tests:** `test_verdict_rules` (mean 0.0031 all positive CIs → yes; mean 0.0030 → equal; one seed negative → unclear; one CI containing 0 → unclear; mirrored negative; one seed 0.004 with CI (0.001, 0.007) → yes, CI (−0.001, 0.009) → unclear), `test_pair_stats` (synthetic RR arrays, a = b + 0.01 → mean 0.01, every CI excludes 0, verdict yes; mismatched seeds → ValueError; one seed → `std` and `p_t` None), `test_claim_readers` (one case per branch of each reader, including `incomplete`), `test_lower_bound_note` (cell on side a vs side b gives the right direction).
- [ ] **Step 2:** `python tests/test_e3_summary.py` → FAIL.
- [ ] **Step 3: Implement** the functions above.
- [ ] **Step 4:** `python tests/test_e3_summary.py` → pass.
- [ ] **Step 5: Commit** `scripts/e3_summary.py tests/test_e3_summary.py` — "E3 summary: verdicts, paired tests and the pre-registered claim rules".

### Task 4: `scripts/e3_summary.py` — loading, tables, CLI

**Files:** Modify `scripts/e3_summary.py`; test: `tests/test_e3_summary.py`.

**Interfaces — Consumes:** Task 2 `CELLS`, `SEEDS`, `stem`; Task 3 functions; `compute_metrics.read_ranks(path) -> (keys, rank_head, rank_tail)`.

**Interfaces — Produces:**
- `load_cell(root: str, cell: str) -> dict | None` — reads `<root>/<cell>/e3_cell.json` and per seed `metrics_*.json` + `ranks_*_test.csv` (+ `ranks_*_test_maskall.csv` when present); returns `{record, seeds: {seed: {metrics, keys, rr, rr_mask?}}}`; RR per triple = (1/rank_head + 1/rank_tail) / 2; seeds without `test` are left out and listed.
- `image_groups(keys: ndarray[n, 3], mask: ndarray[N] bool) -> ndarray[n]` — number of ends with an image (0, 1, 2), keys in `(h, r, t)` order as `read_ranks` returns.
- `summarize(e3_root: str, data_root: str, datasets: list[str], cells=None, seeds=None, kinds=None) -> (markdown: str, data: dict)` — `cells` / `seeds` default to Task 2 `CELLS` / `SEEDS`, `kinds` (dataset → `mmkg` / `vrd` / `vg`) to the module map; per dataset: cell table (MRR, MR, H@1/3/10 mean ± std, n seeds, lower-bound flag, seeds with best epoch > 0.9 × epochs; VRD sets add test in-image MRR / H@1, image_first, relation MRR / H@1 from `metrics["test"]`); pair table (Δ per seed, Δ̄, CIs, t-test, verdict, lower-bound note); MMKG: image-group table (groups < `MIN_GROUP` triples: numbers, verdict "—") and the masked row when every seed of both cells has `rr_mask`; then a "Claims" section from the readers (only for the datasets given). Raises `ValueError` when two rank files of one dataset have different triple sets.
- CLI: `python scripts/e3_summary.py --root <geometry-e3> --data-root data [--datasets ...] [--out e3.md]` writes `.md` and `.json` (json: every number above).

- [ ] **Step 1: Write the failing tests:** `test_summarize_tiny` — build a fake `geometry-e3/Tiny-MMKG/{no_depth,fixed}/` (record json, 3 seeds of metrics with `test` and rank CSVs over 600 triples; mask with ~20% missing via a fake `data/<ds>/ent_img_mask.pt` and `relation2id.txt`), register the fake dataset name in a test-local `CELLS`/`SEEDS`/kind map (`summarize` takes them as optional arguments `cells=None, seeds=None, kinds=None`, defaulting to the module constants); assert the markdown contains the cell rows, a `C2` pair row with a verdict, the three image groups with counts summing to 600, and no masked row; add `ranks_*_test_maskall.csv` for every seed → masked row present. `test_triple_sets_must_match` — one seed's CSV missing a triple → `ValueError`. `test_image_groups` — hand-made keys / mask.
- [ ] **Step 2:** `python tests/test_e3_summary.py` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `python tests/test_e3_summary.py` → pass; `python scripts/e3_summary.py --help` runs.
- [ ] **Step 5: Commit** — "E3 summary: tables, image groups and CLI".

### Task 5: `scripts/e3_mask_eval.py`

**Files:** Create `scripts/e3_mask_eval.py`, `tests/test_e3_mask_eval.py`.

**Interfaces — Consumes:** a `ckpt_<stem>.pt` written by `train.py` (`state_dict`, `variant`, `model_kwargs`, `options`, `args`, `epoch`); `load_entity_features`, `load_triples`, `get_all_true_triplets`, `read_count`, `TripleBatches`, `evaluate_link_prediction`, `save_ranks`, `format_metrics`, `GeometryMMKGModel`.

**Interfaces — Produces:**
- `MMKG = ("FB15k-237-IMG", "FB15K-237-ind", "MKG-Y")`, `MASKED_VARIANTS = ("no_depth", "fixed")`.
- `load_model(ckpt: dict, n_rel: int, embed_dim: int, device) -> GeometryMMKGModel` — `GeometryMMKGModel(n_rel, embed_dim=embed_dim, dropout=ckpt["args"]["dropout"], **ckpt["model_kwargs"], **ckpt["options"])`, `load_state_dict(strict=True)`, `eval()`.
- `masked_eval(model, feats, test, all_true, device, eval_batch_size=512) -> dict` — `evaluate_link_prediction(..., all_mask=torch.zeros_like(feats.mask), return_ranks=True)`.
- CLI: `--cell-dir <geometry-e3>/<ds>/<cell> --data data/<ds>`: every `ckpt_*.pt` there → `ranks_<stem>_test_maskall.csv` and `metrics_<stem>_maskall.json` (`{"masked": "all entities", "test": ..., "ckpt": basename, "code_commit"}`); refuses datasets outside `MMKG` and variants outside `MASKED_VARIANTS` (`SystemExit` with a message).

- [ ] **Step 1: Write the failing tests:** tiny MMKG dataset named `FB15k-237-IMG` (so `MMKG` accepts it), train `fixed` 2 epochs officially (`train.main`) to get a real checkpoint; run the CLI → the two files exist, ranks over the test triples; `test_masked_eval_ignores_image_features` — `masked_eval` with the RGB / depth tables replaced by random tensors gives identical ranks; a `Spatial-VRD` folder name → `SystemExit`. (No-depth vs fixed equality with shared parameters is already pinned by `tests/test_missing_modality.py`; run it in Step 4.)
- [ ] **Step 2:** `python tests/test_e3_mask_eval.py` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `python tests/test_e3_mask_eval.py` and `python tests/test_missing_modality.py` → pass.
- [ ] **Step 5: Commit** — "E3: image-masking control on the MMKG checkpoints".

### Task 6: notebook cells, RESEARCH_PLAN

**Files:** Modify `notebooks/geometry.py`, `RESEARCH_PLAN.md`.

**Interfaces — Consumes:** Task 2 (`CELLS`, `SEEDS`, `resolve`, `cell_record`, `check_cell_record`, `table_digest`, `train_args`, `stem`, `seed_state`, `BOX_CELLS`, `BOXNORM_CELL`), Task 1 (`EXPECT_MISMATCH_EXIT`), Task 4 (`summarize`); notebook `run_resumable`, `push_results`, `remote_files`, `snapshot_download`, `E2B_SHA256`, `box_geometry.EXPECTED`.

- [ ] **Step 1: Shared table helpers.** One cell returns `build_box_table(token) -> str` (the E2 code: annotations at `DATA_REVISION`, `box_geometry.py --check`, raises `RuntimeError` with the script output on mismatch) and `fetch_boxnorm_table(token) -> str` (the E2b code: download, sha256 against `E2B_SHA256[DATASET]`, raises on mismatch). Move `E2B_SHA256` into that cell. The E2 / E2b cells call them and `mo.stop` on the exception with the same messages as now. Stems and folders of E2 / E2b do not change.
- [ ] **Step 2: E3 cell (UI).** Markdown explaining E3 (Vietnamese, like the other cells), multiselect of `CELLS[DATASET]` (default all), multiselect of `SEEDS[DATASET]` (default all), ▶ Chạy E3. Datasets outside `CELLS` → message.
- [ ] **Step 3: E3 cell (run).** Download `geometry-tune/<ds>/*/*/metrics_*.json`, `geometry-e2/<ds>/*/*/*/metrics_*.json`, `geometry-e2b/<ds>/*/*/*/metrics_*.json` and `geometry-e3/<ds>/*/{e3_cell.json,metrics_*.json}`. Per selected cell: `resolve` (ValueError → status "chưa tune xong", next cell); table (box: `build_box_table` once per click, digest over `box_geometry.KEYS[...]` of that cell; boxnorm: `fetch_boxnorm_table`, digest = file sha256; failure → status, next cell); record vs HF `e3_cell.json` (`check_cell_record`; new → write locally and push with the first seed); per seed in ascending order: state from HF metrics of `stem(...)` → `done` skip, `mismatch` → status "lệch, dừng (MRR …, epoch …)" and stop the cell; else `run_resumable(["python", train.py] + train_args(...))` into `WORK_DIR/results/geometry-e3/<ds>/<cell>/`, then read the local metrics: `options_tag` must equal `resolved["options_tag"]` (else status "sai tag" and stop the cell), exit `EXPECT_MISMATCH_EXIT` → "lệch, dừng", other non-zero → "lỗi"; `push_results` after each seed (also on mismatch). Seeds 43 / 44 run only after seed 42 is `done`. Final markdown lists every cell / seed status.
- [ ] **Step 4: E3 summary cell.** Download `geometry-e3/<ds>/` metrics, rank CSVs and records; show `e3_summary.summarize(root, os.path.dirname(DATA_DIR), [DATASET])` markdown.
- [ ] **Step 5: Tune cell comment** for Spatial-VG: `depth_stats_only` is tuned there for E3 (spec 5.2); default selection on VG includes it.
- [ ] **Step 6:** `marimo check notebooks/geometry.py` clean; run `python tests/test_e3_cells.py` again. Dry check without GPU: print the `train_args` the cell would build for Spatial-VRD `box_grid9` seed 42 and compare the stem with the tune stem minus the config part.
- [ ] **Step 7: RESEARCH_PLAN** (spec section 8): P2.10 edge results of FB15K-237-ind and Spatial-VG with the numbers of spec section 8 and the updated summary table; P2.7 line on `geometry/`; 4.10 decision (mask all entities, run on the E3 checkpoints, rule of spec 6.4); P6.0 E3 line (design link, 59 runs, `--expect-valid`, `geometry-e3/`, summary and mask scripts) and E4 "merged into E3". Stage only these hunks: write the intended diff to a patch file and `git apply --cached` it, then `git diff --cached` to confirm no P3 baseline hunk is staged.
- [ ] **Step 8: Commit** `notebooks/geometry.py` and the staged RESEARCH_PLAN hunks — "E3: notebook cells; RESEARCH_PLAN edge results, E3 / E4, 4.10 decision".

---

## After the code (user runs on Molab)

1. Spatial-VRD E3 (21 runs) → check the seed-42 gate on 7 cells, `e3_summary` table.
2. Spatial-VRD-Trans, FB15k-237-IMG, FB15K-237-ind, MKG-Y E3.
3. Spatial-VG: Tune cell `depth_stats_only` (base grid, then edge / convergence), then E3.
4. `e3_mask_eval.py` on the six MMKG cells; `e3_summary.py` over all datasets; results and verdicts into RESEARCH_PLAN P6.0 (C2, C3, C4, E2b).
