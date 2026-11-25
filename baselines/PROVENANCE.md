# Baseline provenance and changes (RESEARCH_PLAN.md P3.0d)

Rule (P3, 18/11/2026): every baseline runs its authors' architecture. Only changes outside the architecture are
allowed, and each one is listed here. `patches/<baseline>.diff` holds the complete diff against the original code
(source files only: `.py .sh .h .cpp`). It is generated with `diff -u --strip-trailing-cr` against the sources below.
No model file (`mmkgc/module/model/*`) appears in either diff.

## NativE (`baselines/NativE/`)

- Source: GitHub `zjukg/NativE`, commit `c72291c` (2024-08-12). The repo copy has the same file set. The one file
  missing here is `mmkgc/release/Base.so`, which is built on the target machine with `mmkgc/make.sh`.
- Data files (`benchmarks/`) are identical to upstream. The author embeddings (DB15K, for P3.0b) come from the
  Google Drive folder in the NativE README. `notebooks/native-db15k-sanity.py` downloads them.
- Changes (`patches/NativE.diff`):

| File | Change | Category |
|---|---|---|
| `mmkgc/config/ValidSelector.py` (new) | validation-MRR checkpoint selection, Python filtered ranks | checkpoint selection |
| `WCGTrainerGP.py`, `WCGTrainerDB15KGP.py`, `WCGTrainerKuai16KGP.py` | call ValidSelector when it is set, restore the best weights | checkpoint selection |
| `mmkgc/config/Tester.py` | `rank_file=`: also write per-triple filtered ranks (strict rule, no type constraint) | evaluation |
| `args.py` | `-rank_file`, `-select_on {valid,original}`, `-valid_steps`, `-patience`, `-valid_max` | evaluation / selection |
| `run_adv_wgan_gp{,_3modal,_4modal}.py` | attach ValidSelector; pass `rank_file` | evaluation / selection |
| `run_adv_wgan_gp.py`, `run_adv_wgan_gp_3modal.py` | `TestDataLoader(..., "link", False)`: existing flag, so `type_constrain.txt` is not needed (test already uses `type_constrain=False`) | existing flag |
| `run_*.py`, `mmkgc/module/BaseModule.py` | `torch.load(..., weights_only=False)` | library compatibility (PyTorch >= 2.6) |

- `-select_on original` gives the upstream behaviour: fixed epoch budget, test on the final weights.
- Restored in P3.0d: `AdvRelRotatEDB15K.py` is byte-identical to upstream again. Commit `ed293a1` had removed its
  hard-coded pooling (4096/768 -> 256), and the `-orig_pool` flag is gone with it. So the DB15K 3-modal model only
  takes the authors' features (P3.0b). Our datasets use `AdvRelRotatE` (`run_adv_wgan_gp.py`), which reads the
  feature size from the tensors.
- Note: upstream `scripts/run_db15k.sh` passes `-adv_num=$ADV` with `ADV` undefined, so argparse fails when the
  script is run verbatim. No code reads `adv_num`, and our runs pass the default `-adv_num=1`.

## SNAG, MKGC part (`baselines/SNAG/SNAG_MKGC/`)

- Source: the GitHub repo `zjukg/SNAG` (commit `62c7ea5`) does **not** contain the MKGC code. Its
  `SNAG_MKGC/readme.md` points to a "Dataset and Code" archive, so the reference is that archive:
  - `SNAG_MKGC.zip`, Google Drive id `1jIYo7JWXNNi7LZqBIsl103FL8tHcoW-q` (also linked as Baidu and Ufile in the README).
  - 208,868,276 bytes, sha256 `b25d6823868bf56788c7851c9c45294c44c7a4622e2c2d678978b3e1da8991c3`,
    md5 `1767f200e7892215be769eb97ab833b5` (downloaded 18/11/2026).
- `benchmarks/`, `embeddings/missing.json` and the author embeddings are identical to the archive.
  The `.pth` embeddings are not in git: `notebooks/snag-mkgy-sanity.py` (P3.0b) downloads the archive, checks
  its sha256 and extracts only `MKG-Y-{visual,textual}.pth` and `missing.json`, each checked by sha256.
- `SNAG_MMEA/`, `README.md` and `LICENSE` are the GitHub repo's files.
- Removed in P3.0d (present since commit `634d7be`, now back to the archive version):
  - a depth branch in `AdvMixRotatE.py`: `depth_proj`, `depth_embeddings`, a 4-way `comb_proj` and fusion weights;
  - loading of `<dataset>-depth-visual.pth` in `run_mmkgc.py`;
  - training resume in `Trainer.py`. This was not an architecture change, but it is not needed, and in
    `select_on=valid` mode it never wrote a checkpoint.
- Changes (`patches/SNAG_MKGC.diff`):

| File | Change | Category |
|---|---|---|
| `mmkgc/config/ValidSelector.py` (new) | same file as NativE's (checked by `tests/test_valid_selector.py`) | checkpoint selection |
| `mmkgc/config/Trainer.py` | with ValidSelector set, skip the upstream every-100-epoch **test** evaluation (P3.0a leak), select on valid, store the best weights where `run_mmkgc.py` expects them | checkpoint selection |
| `mmkgc/config/Tester.py` | `rank_file=` (same code as NativE) | evaluation |
| `args.py` | `-rank_file`, `-select_on`, `-valid_steps`, `-patience`, `-valid_max` | evaluation / selection |
| `run_mmkgc.py`, `run_mmkgc_test.py` | attach ValidSelector; pass `rank_file` | evaluation / selection |

- In `valid` mode the `-MRR<x>` suffix of the checkpoint name is the **validation** MRR.

### SNAG on our data (data level only)

`scripts/prepare_snag_data.py` writes the following under a dataset name of our own (never an author name):

- `benchmarks/<name>/`: OpenKE files, plus `type_constrain.txt` built with the authors' `n-n.py` logic. On the
  authors' MKG-Y split it reproduces their file.
- `embeddings/<name>-{visual,textual}.pth`: the visual table is RGB (Control), RGB+depth mean or RGB+depth concat.

The script also merges two keys into `embeddings/missing.json`, keeping the authors' keys. The original code reads
these keys only when the dataset name contains "MKG", and it then fills the listed rows with
`mean + std * randn` of the other rows:

| Key | Meaning | Ids on our MKG-Y |
|---|---|---|
| `<name>-visual.pth` | entities without an encoded image | 519 (all within the authors' 756) |
| `<name>-textual.pth` | entities without a description | 2,684 (identical to the authors' list; same `entity2id`) |

For names without "MKG" (FB15k-237-IMG, VRD), run with `-use_pool 0`. This is an upstream flag: `use_pool=1` would
apply the hard-coded VGG-4096 pooling.

## MoMoK and CMR: not kept in this repository (decided 25/11/2026)

Neither repository publishes a licence (GitHub API, 25/11/2026: `zjukg/MoMoK` and `OreOZhao/CMR` have none), so this
public repository does not redistribute their code. It keeps only our diff; `scripts/setup_baseline.py <name> --dest
DIR` clones the authors' repository at the pinned commit and applies it (`--no-patch` gives the code as released).
Every change below is outside the architecture (P3 rule) and carries a `[Geometry-MMKG]` comment in the patched code.

### MoMoK (`baselines/patches/MoMoK.diff`)

- Source: GitHub `zjukg/MoMoK`, commit `99c2df114d48c79708ea0608644b685183d81bd9` (2025-06-12).
- Data at the data level only: `scripts/prepare_momok_data.py`. MoMoK selects its feature reshape by a substring of the
  dataset name (`models/RelMoE.py`); our 768-d image + 768-d text tables fit only the authors' `FB15K-237` branch, so
  the dataset names contain `FB15K-237` and not `MKG`. +D concat (1536-d) fits no branch and is not run.

| File | Change | Category |
|---|---|---|
| `utils/data_loader.py` `get_batch` | KvsAll label matrix built with torch on the GPU instead of numpy on the CPU (72% / 86% of a step on FB15k-237 / Spatial-VRD). Bit-identical tensor; the original is kept as `get_batch_numpy` and compared in `tests/test_momok_patch.py` | data loading |
| `utils/data_loader.py` `get_strict_metrics` (new) | filtered head + tail ranks with the strict tie rule on the model's own score (mean of the four predictions), optional rank CSV | evaluation |
| `train.py` | `--select_on valid` (default): validation every `eval_freq` epochs and after the last, best weights kept, test once; `--select_on original`: the upstream behaviour (test every `eval_freq` epochs, best value of each metric) | checkpoint selection |
| `train.py` | `get_validation_pred` returns `(metrics, attention)`; the released `train.py` indexed it as a dict and failed at its first evaluation | bug fix outside the architecture |
| `train.py` | `random.seed(args.seed)` (MIEstimator draws experts with `random.sample`) | seed |
| `train.py` | `--rank_file` | evaluation |

### CMR (`baselines/patches/CMR.diff`)

- Source: GitHub `OreOZhao/CMR`, commit `bcd23a66fb5787f5968b7615a836c8b48361b5ec` (2025-02-20).

| File | Change | Category |
|---|---|---|
| `config.py` | declare `--mem` and `--adapter-model`, read by `trainer.py` / `knowledge_store.py` / `predict.py` but never declared (the released code fails at the first training step) | bug fix outside the architecture |
| `config.py`, `evaluate.py` | `--seed` is honoured (the code overwrote it with its default 202303) | seed |
| `config.py` | task `SpatialVRD_ind` added to the allowed names | data paths |
| `config.py`, `models.py` | `--freeze-vit`: calls the code's own `_freeze_module` on the ViT, as the paper describes (sec. 5.1.3, "we fix the parameter of the visual encoder"); the released code never calls it. Measured 25/11/2026: with a trainable ViT, batch 768 does not fit 96 GB; frozen it takes 48.1 GB, the authors' A6000. Our runs pass it (decided 25/11/2026) | training setting of the paper |
| `trainer.py` | `torch.optim.AdamW(eps=1e-6)` instead of `transformers.AdamW` (removed in transformers 5; 1e-6 is its default eps) | library compatibility |
| `evaluate.py` | strict filtered ranks next to the original ones; `--rank-file`, `--rank-tensor {mm,img}`, `--rank-lambda` (the kNN weight, chosen on valid). The original metrics and their logs are unchanged | evaluation |

- Data at the data level only: `scripts/prepare_cmr_data.py` writes the split json files, `relations.json`,
  `img_entities.json` (the file `triplet.EntityDict` reads; the released `preprocess.py` writes `entities.json` without
  image paths) and one image per entity; entities without an image share one noise picture (seed 0, sha256 recorded).
  On FB15K-237-ind the authors' own split json files are used after checking they are exactly our split.
- Not changed, reported as is: at test time on a `*_ind` task `knowledge_store.py` also puts the test triples into the
  kNN memory (only neighbours with the query's head and relation are skipped). We report CMR without kNN
  (`--knn_topk 0`) and with kNN as released.
