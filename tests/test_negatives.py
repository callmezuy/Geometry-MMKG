"""
In-batch two-sided negatives of GeometryMMKG (core/models/kg_reasoning.py; design decision (a), 18/11/2026).
Run from Code/:  python tests/test_negatives.py      (pytest also works)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_engineering.fast_dataset import EntityFeatures  # noqa: E402
from core.models.end_to_end import GeometryMMKGModel  # noqa: E402
import core.models.kg_reasoning as kgr  # noqa: E402
from core.models.kg_reasoning import TripleIndex, in_batch_loss, l1_cdist, negative_masks  # noqa: E402

N, R = 8, 2
TRAIN = torch.tensor([[0, 0, 1], [0, 0, 2], [3, 1, 1], [4, 0, 5], [6, 1, 7]])  # (h, r, t)


def test_l1_cdist_matches_cdist_and_gradcheck():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(7, 5, dtype=torch.double, generator=g, requires_grad=True)
    e = torch.randn(9, 5, dtype=torch.double, generator=g, requires_grad=True)
    assert torch.allclose(l1_cdist(q, e), torch.cdist(q, e, p=1))
    old = kgr.L1_BACKWARD_BUDGET
    try:
        kgr.L1_BACKWARD_BUDGET = 9 * 5 * 8 * 2   # 2 rows per chunk: exercises the chunked backward
        assert torch.autograd.gradcheck(l1_cdist, (q, e))
    finally:
        kgr.L1_BACKWARD_BUDGET = old


def test_triple_index_contains():
    idx = TripleIndex(TRAIN, N, R)
    h = torch.tensor([0, 0, 0, 3, 1])
    r = torch.tensor([0, 0, 1, 1, 0])
    t = torch.tensor([1, 2, 1, 1, 0])
    assert idx.contains(h, r, t).tolist() == [True, True, False, True, False]


def test_negative_masks_filter_train_and_duplicates():
    idx = TripleIndex(TRAIN, N, R)
    # batch with entity 1 twice (as tail of two positives) and entity 0 twice (as head)
    h = torch.tensor([0, 3, 0])
    r = torch.tensor([0, 1, 0])
    t = torch.tensor([1, 1, 2])
    cand = torch.unique(torch.cat([h, t]))          # [0, 1, 2, 3]
    assert cand.tolist() == [0, 1, 2, 3]
    tail_ok, head_ok = negative_masks(h, r, t, cand, idx)
    # (0, r0, ?): 1 and 2 are true tails in train -> not negatives; 0 and 3 are
    assert tail_ok[0].tolist() == [True, False, False, True]
    assert tail_ok[2].tolist() == [True, False, False, True]
    # (3, r1, ?): 1 is the true tail
    assert tail_ok[1].tolist() == [True, False, True, True]
    # (?, r0, 1): 0 is the true head; (?, r1, 1): 3 is the true head
    assert head_ok[0].tolist() == [False, True, True, True]
    assert head_ok[1].tolist() == [True, True, True, False]


def test_negative_masks_use_only_the_given_index():
    # a triple that is true only in valid/test must stay a negative during training
    idx = TripleIndex(TRAIN, N, R)
    h, r, t = torch.tensor([4]), torch.tensor([0]), torch.tensor([5])
    cand = torch.tensor([4, 5, 7])  # suppose (4, r0, 7) is a valid/test triple: not in TRAIN
    tail_ok, _ = negative_masks(h, r, t, cand, idx)
    assert tail_ok[0].tolist() == [True, False, True]


def test_in_batch_loss_both_sides():
    torch.manual_seed(0)
    D = 16
    feats = EntityFeatures(torch.randn(N, D), torch.randn(N, D), torch.randn(N, D),
                           torch.tensor([1, 1, 0, 1, 1, 1, 0, 1], dtype=torch.bool), torch.device("cpu"))
    model = GeometryMMKGModel(R, embed_dim=D).train()
    idx = TripleIndex(TRAIN, N, R)
    loss, stats = in_batch_loss(model, feats, TRAIN[:, 0], TRAIN[:, 1], TRAIN[:, 2], idx, margin=5.0, adv_temp=1.0)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.relation_embeddings.weight.grad is not None
    for side in ("tail", "head"):
        assert stats[f"loss_{side}"] > 0
        assert 1.0 <= stats[f"eff_neg_{side}"] <= stats[f"n_neg_{side}"]


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all negative-sampling tests passed")
