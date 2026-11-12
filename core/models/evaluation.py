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
def fused_entity_table(model, all_text, all_rgb, all_depth, device, store_device, batch_size=2048):
    out = []
    n = len(all_text)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        fused = model.forward_fusion(all_text[i:j].to(device), all_rgb[i:j].to(device), all_depth[i:j].to(device))
        out.append(fused.to(store_device))
    return torch.cat(out, dim=0)


def _filtered_correction(queries, true_scores, table, filters, device):
    """Number of OTHER true answers that score strictly better than the true one, per query.
    filters: list (len B) of lists of entity ids to filter."""
    rows = [i for i, f in enumerate(filters) for _ in f]
    if not rows:
        return torch.zeros(len(filters), dtype=torch.long, device=device)
    ents = torch.tensor([e for f in filters for e in f], dtype=torch.long)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    emb = table[ents.to(table.device)].to(device)
    scores = -(queries[rows] - emb).abs().sum(dim=1)
    better = (scores > true_scores[rows]).long()
    out = torch.zeros(len(filters), dtype=torch.long, device=device)
    out.index_add_(0, rows, better)
    return out


@torch.no_grad()
def evaluate_link_prediction(model, all_text, all_rgb, all_depth, dataloader, all_true_triplets, device,
                             entity_chunk_size=32768, keep_on_gpu=None, return_ranks=False, desc="Evaluating"):
    """
    Args:
        all_true_triplets: {(h, r): set(t)} over train+valid+test (get_all_true_triplets).
        keep_on_gpu: None = automatic (P1.6), True / False to force.
        return_ranks: also return per-triplet ranks (for save_ranks / compute_metrics).
    Returns:
        dict with "mrr", "mr", "hits@1", "hits@3", "hits@10", "n" over BOTH directions
        (head and tail ranks pooled, as in OpenKE), plus "tail" and "head" sub-dicts,
        and "ranks" = {"h", "r", "t", "rank_head", "rank_tail"} numpy arrays if return_ranks.
    """
    model.eval()
    num_entities = len(all_text)
    store_device = _pick_store_device(num_entities, model.embed_dim, device, keep_on_gpu)
    print(f"Pre-computing fused representations for all {num_entities:,} entities (stored on {store_device})...")
    table = fused_entity_table(model, all_text, all_rgb, all_depth, device, store_device)
    head_filter = _head_filter(all_true_triplets)

    hs, rs, ts, rank_t, rank_h = [], [], [], [], []
    for batch in tqdm(dataloader, desc=desc):
        h_ids, r_ids, t_ids = batch["h_id"], batch["r_id"], batch["t_id"]
        h_emb = table[h_ids.to(table.device)].to(device)
        t_emb = table[t_ids.to(table.device)].to(device)
        r_emb = model.relation_embeddings(r_ids.to(device))

        q_tail = h_emb + r_emb            # compare with every f(e) as tail
        q_head = t_emb - r_emb            # compare with every f(e) as head
        # True-answer scores are computed with the SAME expression as the candidates of each
        # direction; otherwise float rounding can make the true entity "beat" itself.
        true_t = -(q_tail - t_emb).abs().sum(dim=1)
        true_h = -(q_head - h_emb).abs().sum(dim=1)

        better_t = torch.zeros(len(h_ids), dtype=torch.long, device=device)
        better_h = torch.zeros(len(h_ids), dtype=torch.long, device=device)
        for c in range(0, num_entities, entity_chunk_size):
            ent = table[c:c + entity_chunk_size].to(device, non_blocking=True)
            better_t += (-torch.cdist(q_tail, ent, p=1) > true_t[:, None]).sum(dim=1)
            better_h += (-torch.cdist(q_head, ent, p=1) > true_h[:, None]).sum(dim=1)

        h_l, r_l, t_l = h_ids.tolist(), r_ids.tolist(), t_ids.tolist()
        f_tail = [list(all_true_triplets.get((h, r), set()) - {t}) for h, r, t in zip(h_l, r_l, t_l)]
        f_head = [list(head_filter.get((r, t), set()) - {h}) for h, r, t in zip(h_l, r_l, t_l)]
        better_t -= _filtered_correction(q_tail, true_t, table, f_tail, device)
        better_h -= _filtered_correction(q_head, true_h, table, f_head, device)

        rank_t.append((better_t + 1).cpu().numpy())
        rank_h.append((better_h + 1).cpu().numpy())
        hs.append(np.asarray(h_l)); rs.append(np.asarray(r_l)); ts.append(np.asarray(t_l))

    rank_t = np.concatenate(rank_t) if rank_t else np.zeros(0, dtype=np.int64)
    rank_h = np.concatenate(rank_h) if rank_h else np.zeros(0, dtype=np.int64)
    result = _metrics(np.concatenate([rank_h, rank_t]))
    result["tail"] = _metrics(rank_t)
    result["head"] = _metrics(rank_h)
    if return_ranks:
        result["ranks"] = {"h": np.concatenate(hs), "r": np.concatenate(rs), "t": np.concatenate(ts),
                           "rank_head": rank_h, "rank_tail": rank_t}
    del table
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
