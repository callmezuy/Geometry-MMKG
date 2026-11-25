"""
The MoMoK changes of baselines/patches/MoMoK.diff (RESEARCH_PLAN.md P3.6), tested on a patched checkout:
  python scripts/setup_baseline.py momok --dest <dir>
  MOMOK_DIR=<dir> python tests/test_momok_patch.py          (pytest also works; skipped without MOMOK_DIR)
1. get_batch (label matrix built with torch) is bit-identical to the original numpy get_batch (get_batch_numpy).
2. get_strict_metrics gives the strict filtered ranks of a brute-force count, with ties, and writes them in file order.
"""
import os
import sys
import tempfile
from argparse import Namespace

import numpy as np
import torch

MOMOK = os.environ.get("MOMOK_DIR")


def _skip():
    if not MOMOK or not os.path.isdir(MOMOK):
        try:
            import pytest
            pytest.skip("set MOMOK_DIR to a checkout made by scripts/setup_baseline.py momok")
        except ImportError:
            print("skipped: set MOMOK_DIR")
            sys.exit(0)


def _corpus(n=9, n_rel=2, device="cpu", batch_size=4):
    if MOMOK not in sys.path:
        sys.path.insert(0, MOMOK)
    from utils.data_loader import ConvECorpus
    rng = np.random.default_rng(0)
    tri = sorted({(int(h), int(r), int(t)) for h, r, t in zip(rng.integers(0, n, 60), rng.integers(0, n_rel, 60),
                                                              rng.integers(0, n, 60))})
    split = lambda xs: (xs, ([t for _, _, t in xs], [h for h, _, _ in xs], [r for _, r, _ in xs]), set())  # noqa
    train, valid, test = tri[:30], tri[30:40], tri[40:]
    args = Namespace(device=device, batch_size=batch_size)
    return ConvECorpus(args, split(train), split(valid), split(test), {f"e{i}": i for i in range(n)},
                       {f"r{i}": i for i in range(n_rel)}), valid, test, n, n_rel


def test_gpu_labels_are_bit_identical_to_the_original():
    _skip()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus, *_ = _corpus(device=dev)
    corpus.shuffle()
    for b in range(corpus.max_batch_num):
        i1, v1 = corpus.get_batch(b)
        i0, v0 = corpus.get_batch_numpy(b)
        assert torch.equal(i0, i1)
        assert v1.dtype == v0.dtype == torch.float32
        assert torch.equal(v0.to(v1.device), v1), b


class _Scores(torch.nn.Module):
    """Stub MoMoK: four identical predictions from a fixed table per (query entity, relation), with ties."""

    def __init__(self, n, n_rel2):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.table = (torch.randint(0, 4, (n, n_rel2, n), generator=g).float() / 4)   # many exact ties

    def forward(self, q):
        p = self.table[q[:, 0], q[:, 1]]
        return [p, p, p, p], None


def _brute(table, queries, true_of):
    out = []
    for e, rel, target in queries:
        s = table[e, rel]
        known = true_of[(e, rel)] - {target}
        out.append(1 + sum(1 for c in range(len(s)) if c not in known and s[c] > s[target]))
    return out


def test_strict_ranks_match_a_brute_force_count():
    _skip()
    corpus, valid, test, n, n_rel = _corpus()
    model = _Scores(n, 2 * n_rel)
    true_of = {}
    for h, r, t in corpus.train_triples + corpus.val_triples + corpus.test_triples:
        true_of.setdefault((h, r), set()).add(t)
        true_of.setdefault((t, r + n_rel), set()).add(h)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ranks.csv")
        m = corpus.get_strict_metrics(model, "test", rank_file=path)
        rows = open(path).read().splitlines()
    assert rows[0] == "h_id,r_id,t_id,rank_head,rank_tail"
    tail = _brute(model.table, [(h, r, t) for h, r, t in test], true_of)
    head = _brute(model.table, [(t, r + n_rel, h) for h, r, t in test], true_of)
    got = [tuple(map(int, x.split(","))) for x in rows[1:]]
    assert got == [(h, r, t, rh, rt) for (h, r, t), rh, rt in zip(test, head, tail)]
    allr = np.array(head + tail, dtype=float)
    assert abs(m["MRR"] - np.mean(1 / allr)) < 1e-12 and abs(m["MR"] - allr.mean()) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
