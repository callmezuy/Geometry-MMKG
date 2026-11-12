"""
Brute-force check of core/models/evaluation.py (head + tail filtered ranks).
Run from Code/:  python tests/test_evaluation.py      (pytest also works if installed)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models.evaluation import evaluate_link_prediction  # noqa: E402


class ToyModel(nn.Module):
    def __init__(self, num_rel, dim):
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Linear(3 * dim, dim)
        self.relation_embeddings = nn.Embedding(num_rel, dim)

    def forward_fusion(self, t, r, d):
        return self.proj(torch.cat([t, r, d], dim=-1))


def brute_force(model, text, rgb, depth, triples, true_tails, true_heads):
    with torch.no_grad():
        E = model.forward_fusion(text, rgb, depth)
        R = model.relation_embeddings.weight
    rt, rh = [], []
    for h, r, t in triples:
        # Each direction uses one expression for candidates and true answer
        # (mathematically equal to -||h + r - t||_1; avoids float-rounding self-ties).
        q_tail, q_head = E[h] + R[r], E[t] - R[r]
        s_tail = -(q_tail - E).abs().sum(1)            # candidate tails
        s_head = -(q_head - E).abs().sum(1)            # candidate heads
        ok_t = torch.ones(len(E), dtype=torch.bool)
        ok_t[list(true_tails[(h, r)] - {t})] = False
        ok_h = torch.ones(len(E), dtype=torch.bool)
        ok_h[list(true_heads[(r, t)] - {h})] = False
        rt.append(int(((s_tail > s_tail[t]) & ok_t).sum()) + 1)
        rh.append(int(((s_head > s_head[h]) & ok_h).sum()) + 1)
    return np.array(rh), np.array(rt)


def test_ranks_match_brute_force():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, R, D = 300, 4, 8
    text, rgb, depth = (torch.randn(N, D) for _ in range(3))
    model = ToyModel(R, D)
    all_triples = {(int(rng.integers(N)), int(rng.integers(R)), int(rng.integers(N))) for _ in range(900)}
    # many shared (h, r) and (r, t) so filtering matters
    all_triples |= {(5, 1, int(t)) for t in rng.integers(N, size=20)} | {(int(h), 2, 7) for h in rng.integers(N, size=20)}
    all_triples = sorted(all_triples)
    test = all_triples[::3]
    true_tails, true_heads = {}, {}
    for h, r, t in all_triples:
        true_tails.setdefault((h, r), set()).add(t)
        true_heads.setdefault((r, t), set()).add(h)

    loader = [{"h_id": torch.tensor([x[0] for x in test[i:i + 64]]),
               "r_id": torch.tensor([x[1] for x in test[i:i + 64]]),
               "t_id": torch.tensor([x[2] for x in test[i:i + 64]])} for i in range(0, len(test), 64)]
    res = evaluate_link_prediction(model, text, rgb, depth, loader, true_tails, torch.device("cpu"),
                                   entity_chunk_size=37, return_ranks=True, desc="test")
    exp_h, exp_t = brute_force(model, text, rgb, depth, test, true_tails, true_heads)
    assert np.array_equal(res["ranks"]["rank_tail"], exp_t), "tail ranks differ"
    assert np.array_equal(res["ranks"]["rank_head"], exp_h), "head ranks differ"
    pooled = np.concatenate([exp_h, exp_t])
    assert abs(res["mrr"] - np.mean(1 / pooled)) < 1e-12
    assert abs(res["mr"] - pooled.mean()) < 1e-9
    print(f"OK: {len(test)} triples, MRR {res['mrr']:.4f}, tail {res['tail']['mrr']:.4f}, head {res['head']['mrr']:.4f}")


if __name__ == "__main__":
    test_ranks_match_brute_force()
