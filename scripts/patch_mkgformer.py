"""
Patch the MKGformer submodule (baselines/MKGformer, upstream zjunlp/MKGformer) at run time so
its link-prediction test writes per-triplet ranks for scripts/compute_metrics.py (P1.5).

The submodule points to the upstream repo, so we cannot commit changes inside it; the notebook
runs this script before Phase 2. The patch is idempotent.

What it changes in MKG/lit_models/transformer.py:
  * _eval: additionally computes a STRICT filtered rank (1 + #entities scoring strictly higher
    than the true one, other true answers excluded), the same tie rule as OpenKE / GeometryMMKG.
    MKGformer's own rank (via argsort) breaks ties arbitrarily; its logged metrics are unchanged.
  * test_epoch_end: if env MKG_RANK_FILE is set, writes CSV h_id,r_id,t_id,rank_head,rank_tail.
    MKGformer builds 2 examples per test line, in file order: [predict head, predict tail]; every
    pair is verified against the gold labels before writing, so misalignment raises an error.
  * main.py: torch.load(..., weights_only=False) for PyTorch >= 2.6 (was patched in the notebook).

Ids: the notebook writes entities.txt / relations.txt in entity2id / relation2id order, so MKGformer's
internal ids equal the dataset ids.

Usage: python scripts/patch_mkgformer.py baselines/MKGformer/MKG
"""
import os
import sys

MARK = "[Geometry-MMKG patch]"

EVAL_OLD = "        logits += labels * -100 # mask entity\n"
EVAL_NEW = (
    f"        # {MARK} strict filtered rank (ties are not broken in the model's favour)\n"
    "        _gm_true = logits[torch.arange(bsz), label]\n"
    "        _gm_strict = (((logits > _gm_true[:, None]) & (labels == 0)).sum(dim=1) + 1).detach().cpu().numpy()\n"
    + EVAL_OLD
)
RET_OLD = "        return dict(ranks = np.array(ranks))\n"
RET_NEW = "        return dict(ranks = np.array(ranks), strict_ranks=_gm_strict, pred=label.detach().cpu().numpy())\n"
TEST_OLD = '        self.log("Test/mrr", (1. / ranks).mean())\n'
TEST_NEW = TEST_OLD + (
    f"        # {MARK} per-triplet ranks for Geometry-MMKG scripts/compute_metrics.py\n"
    '        if os.environ.get("MKG_RANK_FILE") and "strict_ranks" in outputs[0]:\n'
    '            _gm_dump_ranks(outputs, self.args.get("data_dir"), os.environ["MKG_RANK_FILE"])\n'
)
DUMP_FN = f'''

def _gm_dump_ranks(outputs, data_dir, path):
    """{MARK} Pair MKGformer's (head, tail) examples back into triples and write CSV ranks."""
    import csv
    pred = np.concatenate([o["pred"] for o in outputs])
    strict = np.concatenate([o["strict_ranks"] for o in outputs])
    def _first_col(fn):
        with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
            return [l.rstrip("\\n").split("\\t")[0].strip() for l in f if l.strip()]
    ent2id = {{e: i for i, e in enumerate(_first_col("entity2text.txt"))}}
    rel2id = {{r: i for i, r in enumerate(_first_col("relations.txt"))}}
    with open(os.path.join(data_dir, "test.tsv"), encoding="utf-8") as f:
        lines = [l.rstrip("\\n").split("\\t") for l in f if l.strip()]
    if len(pred) != 2 * len(lines):
        raise RuntimeError(f"MKG_RANK_FILE: {{len(pred)}} test examples for {{len(lines)}} test lines (expected 2 per line)")
    rows = []
    for i, (h, r, t) in enumerate(lines):
        hi, ri, ti = ent2id[h], rel2id[r], ent2id[t]
        if pred[2 * i] != hi or pred[2 * i + 1] != ti:
            raise RuntimeError(f"MKG_RANK_FILE: example order does not match test.tsv at line {{i}}")
        rows.append((hi, ri, ti, int(strict[2 * i]), int(strict[2 * i + 1])))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h_id", "r_id", "t_id", "rank_head", "rank_tail"])
        w.writerows(rows)
    rk = np.array([[x[3], x[4]] for x in rows], dtype=np.float64).reshape(-1)
    print(f"Saved {{len(rows)}} per-triplet ranks to {{path}} (strict ranks: MRR {{np.mean(1 / rk):.4f}}, "
          f"MR {{rk.mean():.1f}}, H@1 {{np.mean(rk <= 1):.4f}}, H@10 {{np.mean(rk <= 10):.4f}})")
'''


def patch_file(path, edits, append=None):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if MARK in s:
        print(f"already patched: {path}")
        return
    for old, new in edits:
        if s.count(old) != 1:
            raise SystemExit(f"{path}: expected exactly one occurrence of {old.strip()!r}; upstream changed?")
        s = s.replace(old, new)
    if append:
        s += append
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"patched: {path}")


def main(mkg_dir):
    tr = os.path.join(mkg_dir, "lit_models", "transformer.py")
    with open(tr, encoding="utf-8") as f:
        head = f.read()
    edits = [(EVAL_OLD, EVAL_NEW), (RET_OLD, RET_NEW), (TEST_OLD, TEST_NEW)]
    if "\nimport os\n" not in head:
        edits.append(("import numpy as np\n", "import numpy as np\nimport os\n"))
    patch_file(tr, edits, append=DUMP_FN)

    mp = os.path.join(mkg_dir, "main.py")
    with open(mp, encoding="utf-8") as f:
        s = f.read()
    s2 = s.replace('torch.load(args.checkpoint, map_location="cpu")[', 'torch.load(args.checkpoint, map_location="cpu", weights_only=False)[')
    s2 = s2.replace('torch.load(path)["state_dict"]', 'torch.load(path, weights_only=False)["state_dict"]')
    if s2 != s:
        with open(mp, "w", encoding="utf-8") as f:
            f.write(s2)
        print(f"patched: {mp}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("baselines", "MKGformer", "MKG"))
