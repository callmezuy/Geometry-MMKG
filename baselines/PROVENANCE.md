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
