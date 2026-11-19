"""
Training objective of GeometryMMKG (design decision (a), 18/11/2026).

In-batch negatives on BOTH sides (the evaluation ranks heads and tails):
  * candidates = the distinct entities of the batch (heads and tails of training triples), fused once;
    every candidate replaces the tail of each positive, and every candidate replaces its head;
  * a candidate that forms a TRAIN triple with the query is not a negative (the true answer included);
    valid / test triples are never used during training;
  * only training entities ever appear, so nothing about valid / test entities leaks in the inductive setting.

Loss per query (self-adversarial weighted margin ranking, weights as in RotatE / NativE, detached):
  L = sum_j p_j * max(0, margin + d_pos - d_neg_j),   p = softmax(adv_temp * (-d_neg)) over the valid negatives
with d = ||h + r - t||_1 (TransE L1). The final loss is the mean of the tail-side and head-side losses.
"""
import torch

L1_BACKWARD_BUDGET = 256 * 2 ** 20  # bytes of the [rows, C, d] sign block in L1CDist.backward


class L1CDist(torch.autograd.Function):
    """d[i, c] = ||q_i - e_c||_1 with a memory-bounded backward.
    torch.cdist(p=1) materialises a [B, C, d] tensor in backward (about 6 GB for B=1024, C=2048, d=768);
    here the sign block is built for a few rows of q at a time."""

    @staticmethod
    def forward(ctx, q, e):
        ctx.save_for_backward(q, e)
        return torch.cdist(q, e, p=1)

    @staticmethod
    def backward(ctx, grad):
        q, e = ctx.saved_tensors
        grad_q, grad_e = torch.zeros_like(q), torch.zeros_like(e)
        rows = max(1, L1_BACKWARD_BUDGET // max(1, e.numel() * e.element_size()))
        for i in range(0, len(q), rows):
            s = torch.sign(q[i:i + rows, None, :] - e[None, :, :])  # [rows, C, d]
            g = grad[i:i + rows]
            grad_q[i:i + rows] = torch.einsum("ic,icd->id", g, s)
            grad_e -= torch.einsum("ic,icd->cd", g, s)
        return grad_q, grad_e


def l1_cdist(q, e):
    return L1CDist.apply(q, e)


class TripleIndex:
    """Membership test for a set of (h, r, t) triples, vectorised (sorted int64 keys + searchsorted)."""

    def __init__(self, triples, num_entities, num_relations, device=None):
        self.n, self.r = num_entities, num_relations
        keys = self._key(triples[:, 0], triples[:, 1], triples[:, 2])
        self.keys = torch.unique(keys).to(device or triples.device)  # unique() also sorts

    def _key(self, h, r, t):
        return (h.long() * self.r + r.long()) * self.n + t.long()

    def contains(self, h, r, t):
        """Broadcasting h, r, t (long tensors) -> bool tensor."""
        q = self._key(*torch.broadcast_tensors(h, r, t)).to(self.keys.device)
        pos = torch.searchsorted(self.keys, q).clamp_(max=len(self.keys) - 1)
        return self.keys[pos] == q


def negative_masks(h, r, t, cand, index):
    """tail_ok[i, c]: (h_i, r_i, cand_c) is not a train triple; head_ok[i, c]: (cand_c, r_i, t_i) is not. [B, C]"""
    tail_ok = ~index.contains(h[:, None], r[:, None], cand[None, :])
    head_ok = ~index.contains(cand[None, :], r[:, None], t[:, None])
    return tail_ok, head_ok


def _side_loss(d_all, pos_col, ok, margin, adv_temp):
    """d_all [B, C] distances of each query to every candidate; pos_col [B] column of the true answer.
    Returns per-query (loss, effective number of negatives, number of negatives) for the queries that
    have at least one negative."""
    d_pos = d_all.gather(1, pos_col[:, None])                       # [B, 1], same expression as negatives
    with torch.no_grad():
        logits = (-adv_temp * d_all).masked_fill(~ok, float("-inf"))
        has_neg = ok.any(dim=1)
        p = torch.softmax(logits[has_neg], dim=1)
        ent = -(p * torch.log(p.clamp_min(1e-30))).sum(dim=1)
    hinge = torch.relu(margin + d_pos - d_all)[has_neg]
    return (p * hinge).sum(dim=1), ent.exp(), ok[has_neg].sum(dim=1).float()


def in_batch_loss(model, feats, h, r, t, index, margin, adv_temp):
    """h, r, t: LongTensor [B] (train triples). Returns (loss, stats dict of floats).
    A relation-conditioned model (rel_gate) fuses the candidates once per distinct relation of the batch,
    and each query is scored against the candidates fused under its own relation."""
    device = feats.device
    cand = torch.unique(torch.cat([h, t]))                          # sorted distinct entities
    cand_feats = feats.gather(cand)
    cand_geo = feats.gather_geo(cand)                               # None without depth_stats
    h_col = torch.searchsorted(cand, h.contiguous()).to(device)
    t_col = torch.searchsorted(cand, t.contiguous()).to(device)
    r_dev = r.to(device)
    rel = model.relation_embeddings(r_dev)
    tail_ok, head_ok = negative_masks(h, r, t, cand, index)
    tail_ok, head_ok = tail_ok.to(device), head_ok.to(device)

    if model.relation_conditioned:
        groups = [(r_dev == rr).nonzero(as_tuple=True)[0] for rr in torch.unique(r_dev)]
    else:
        groups = [None]
    parts = {"tail": [], "head": []}
    for rows in groups:
        if rows is None:
            e = model.forward_fusion(*cand_feats, geo=cand_geo)     # [C, d]
            rows = torch.arange(len(h_col), device=device)
        else:
            e = model.forward_fusion(*cand_feats, r_dev[rows[0]].expand(len(cand)), geo=cand_geo)
        rr = rel[rows]
        d_tail = l1_cdist(e[h_col[rows]] + rr, e)                   # ||h + r - c||_1
        d_head = l1_cdist(e[t_col[rows]] - rr, e)                   # ||c + r - t||_1 = ||(t - r) - c||_1
        parts["tail"].append(_side_loss(d_tail, t_col[rows], tail_ok[rows], margin, adv_temp))
        parts["head"].append(_side_loss(d_head, h_col[rows], head_ok[rows], margin, adv_temp))

    stats, losses = {}, []
    for side, ps in parts.items():
        l_rows, eff, nneg = (torch.cat(x) for x in zip(*ps))
        losses.append(l_rows.mean())
        stats[f"loss_{side}"] = losses[-1].item()
        stats[f"eff_neg_{side}"] = float(eff.mean())
        stats[f"n_neg_{side}"] = float(nneg.mean())
    return (losses[0] + losses[1]) / 2, stats
