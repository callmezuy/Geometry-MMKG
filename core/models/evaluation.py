"""
Filtered link prediction for GeometryMMKG (RESEARCH_PLAN.md, section 4 and P1.1 / P1.6).

Protocol (shared with every baseline, see scripts/compute_metrics.py):
  * rank BOTH directions: tail (h, r, ?) and head (?, r, t), over ALL N entities;
  * filtered setting: other true answers (train + valid + test) are not counted;
  * no type constraint;
  * rank = 1 + #{candidates scoring strictly better than the true answer}
    (same tie rule as the OpenKE C++ tester used by NativE / SNAG).

Scoring is TransE with L1 distance on fused entity representations:
  score(h, r, t) = -|| f(h) + r - f(t) ||_1
  tail query: distance from (f(h) + r) to every f(e)
  head query: distance from (f(t) - r) to every f(e)   (same norm, rearranged)
"""
import csv
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def _metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size == 0:
        return {"mrr": 0.0, "mr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0}
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "mr": float(np.mean(ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "n": int(ranks.size),
    }


def _head_filter(all_true_triplets):
    """Invert {(h, r): {t}} into {(r, t): {h}}."""
    heads = defaultdict(set)
    for (h, r), tails in all_true_triplets.items():
        for t in tails:
            heads[(r, t)].add(h)
    return heads


def _pick_store_device(num_entities, embed_dim, device, keep_on_gpu):
    """P1.6: keep the fused entity table on the GPU whenever it fits (avoids a full
    host->device copy of the table for every evaluation batch)."""
    if device.type != "cuda":
        return device
    if keep_on_gpu is not None:
        return device if keep_on_gpu else torch.device("cpu")
    need = num_entities * embed_dim * 4
    free, _ = torch.cuda.mem_get_info(device)
    return device if need < 0.5 * free else torch.device("cpu")


@torch.no_grad()
def fused_entity_table(model, all_text, all_rgb, all_depth, device, store_device, batch_size=2048, all_mask=None,
                       rel=None):
    """all_mask: optional bool [N] image mask, passed to forward_fusion when given (P2.6).
    rel: relation id (int) for a relation-conditioned model: every entity is fused under that relation."""
    out = []
    n = len(all_text)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        args = [all_text[i:j].to(device), all_rgb[i:j].to(device), all_depth[i:j].to(device)]
        if all_mask is not None or rel is not None:
            args.append(None if all_mask is None else all_mask[i:j].to(device))
        if rel is not None:
            args.append(torch.full((j - i,), int(rel), dtype=torch.long, device=device))
        fused = model.forward_fusion(*args)
        out.append(fused.to(store_device))
    return torch.cat(out, dim=0)


def _excluded_pairs(true_ids, filters, device):
    """(row, entity) pairs that are never counted as 'better': the true answer itself and the other true
    answers of the query (filtered setting). Returns two long tensors (rows, entities), sorted by entity."""
    rows = list(range(len(true_ids))) + [i for i, f in enumerate(filters) for _ in f]
    ents = list(true_ids) + [e for f in filters for e in f]
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    ents = torch.tensor(ents, dtype=torch.long, device=device)
    order = torch.argsort(ents)
    return rows[order], ents[order]


def _count_better(queries, true_scores, table, excl_rows, excl_ents, device, entity_chunk_size):
    """#{candidates e not excluded : -||q - f(e)||_1 > true score}, per query.
    Excluded pairs are removed from the comparison itself (not subtracted afterwards with a separately
    computed score), so float differences between cdist and an explicit sum cannot make the true answer or a
    filtered answer count as better than the true answer."""
    better = torch.zeros(len(queries), dtype=torch.long, device=device)
    for c in range(0, len(table), entity_chunk_size):
        ent = table[c:c + entity_chunk_size].to(device, non_blocking=True)
        cmp = -torch.cdist(queries, ent, p=1) > true_scores[:, None]
        lo = torch.searchsorted(excl_ents, c)
        hi = torch.searchsorted(excl_ents, c + len(ent))
        if hi > lo:
            cmp[excl_rows[lo:hi], excl_ents[lo:hi] - c] = False
        better += cmp.sum(dim=1)
    return better


def _rank_batch(model, table, h_ids, r_ids, t_ids, all_true_triplets, head_filter, device, entity_chunk_size):
    """Filtered (head rank, tail rank) numpy arrays of one batch of triples against the fused table."""
    h_emb = table[h_ids.to(table.device)].to(device)
    t_emb = table[t_ids.to(table.device)].to(device)
    r_emb = model.relation_embeddings(r_ids.to(device))

    q_tail = h_emb + r_emb            # compare with every f(e) as tail
    q_head = t_emb - r_emb            # compare with every f(e) as head
    true_t = -(q_tail - t_emb).abs().sum(dim=1)
    true_h = -(q_head - h_emb).abs().sum(dim=1)

    h_l, r_l, t_l = h_ids.tolist(), r_ids.tolist(), t_ids.tolist()
    f_tail = [list(all_true_triplets.get((h, r), set()) - {t}) for h, r, t in zip(h_l, r_l, t_l)]
    f_head = [list(head_filter.get((r, t), set()) - {h}) for h, r, t in zip(h_l, r_l, t_l)]
    better_t = _count_better(q_tail, true_t, table, *_excluded_pairs(t_l, f_tail, device), device,
                             entity_chunk_size)
    better_h = _count_better(q_head, true_h, table, *_excluded_pairs(h_l, f_head, device), device,
                             entity_chunk_size)
    return (better_h + 1).cpu().numpy(), (better_t + 1).cpu().numpy()


@torch.no_grad()
def evaluate_link_prediction(model, all_text, all_rgb, all_depth, dataloader, all_true_triplets, device,
                             entity_chunk_size=32768, keep_on_gpu=None, return_ranks=False, desc="Evaluating",
                             all_mask=None):
    """
    Args:
        all_mask: optional bool [N] image mask (ent_img_mask.pt) for models that take it (P2.6).
        all_true_triplets: {(h, r): set(t)} over train+valid+test (get_all_true_triplets).
        keep_on_gpu: None = automatic (P1.6), True / False to force.
        return_ranks: also return per-triplet ranks (for save_ranks / compute_metrics).
    A model with `relation_conditioned = True` (entity representations depend on the query relation, P2.8)
    is evaluated relation by relation: the entity table is fused once per relation of the evaluated triples,
    and each triple is ranked against the table of its own relation. Ranks keep the input order.
    Returns:
        dict with "mrr", "mr", "hits@1", "hits@3", "hits@10", "n" over BOTH directions
        (head and tail ranks pooled, as in OpenKE), plus "tail" and "head" sub-dicts,
        and "ranks" = {"h", "r", "t", "rank_head", "rank_tail"} numpy arrays if return_ranks.
    """
    model.eval()
    num_entities = len(all_text)
    store_device = _pick_store_device(num_entities, model.embed_dim, device, keep_on_gpu)
    head_filter = _head_filter(all_true_triplets)
    batches = list(dataloader)
    h_all = torch.cat([b["h_id"] for b in batches]) if batches else torch.zeros(0, dtype=torch.long)
    r_all = torch.cat([b["r_id"] for b in batches]) if batches else torch.zeros(0, dtype=torch.long)
    t_all = torch.cat([b["t_id"] for b in batches]) if batches else torch.zeros(0, dtype=torch.long)
    rank_h = np.zeros(len(h_all), dtype=np.int64)
    rank_t = np.zeros(len(h_all), dtype=np.int64)
    batch_size = max(1, len(batches[0]["h_id"])) if batches else 1

    if getattr(model, "relation_conditioned", False):
        rels = torch.unique(r_all).tolist()
        print(f"Relation-conditioned model: fusing all {num_entities:,} entities once per relation "
              f"({len(rels)} relations, stored on {store_device})...")
        jobs = []
        for rel in rels:
            idx = (r_all == rel).nonzero(as_tuple=True)[0]
            jobs += [(rel, idx[i:i + batch_size]) for i in range(0, len(idx), batch_size)]
        table, table_rel = None, None
        for rel, idx in tqdm(jobs, desc=desc):
            if rel != table_rel:
                del table
                table = fused_entity_table(model, all_text, all_rgb, all_depth, device, store_device,
                                           all_mask=all_mask, rel=rel)
                table_rel = rel
            rh, rt = _rank_batch(model, table, h_all[idx], r_all[idx], t_all[idx], all_true_triplets,
                                 head_filter, device, entity_chunk_size)
            rank_h[idx.numpy()], rank_t[idx.numpy()] = rh, rt
    else:
        print(f"Pre-computing fused representations for all {num_entities:,} entities (stored on {store_device})...")
        table = fused_entity_table(model, all_text, all_rgb, all_depth, device, store_device, all_mask=all_mask)
        pos = 0
        for batch in tqdm(batches, desc=desc):
            n = len(batch["h_id"])
            rh, rt = _rank_batch(model, table, batch["h_id"], batch["r_id"], batch["t_id"], all_true_triplets,
                                 head_filter, device, entity_chunk_size)
            rank_h[pos:pos + n], rank_t[pos:pos + n] = rh, rt
            pos += n
    del table

    result = _metrics(np.concatenate([rank_h, rank_t]))
    result["tail"] = _metrics(rank_t)
    result["head"] = _metrics(rank_h)
    if return_ranks:
        result["ranks"] = {"h": h_all.numpy(), "r": r_all.numpy(), "t": t_all.numpy(),
                           "rank_head": rank_h, "rank_tail": rank_t}
    return result


def format_metrics(m):
    return (f"MRR {m['mrr']:.4f} | MR {m['mr']:.1f} | Hits@1 {m['hits@1']:.4f} | "
            f"Hits@3 {m['hits@3']:.4f} | Hits@10 {m['hits@10']:.4f} "
            f"(tail MRR {m['tail']['mrr']:.4f}, head MRR {m['head']['mrr']:.4f})")


def save_ranks(ranks, path):
    """Write per-triplet ranks in the shared format read by scripts/compute_metrics.py:
    CSV header h_id,r_id,t_id,rank_head,rank_tail (ids from entity2id / relation2id)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["h_id", "r_id", "t_id", "rank_head", "rank_tail"])
        for row in zip(ranks["h"], ranks["r"], ranks["t"], ranks["rank_head"], ranks["rank_tail"]):
            w.writerow([int(x) for x in row])
    print(f"Saved {len(ranks['h']):,} per-triplet ranks to {path}")
