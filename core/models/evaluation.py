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

Auxiliary metrics on the scene graphs (E1, docs/superpowers/specs/2026-11-25-e1-mmsg-aux-metrics-design.md),
computed when `entity_image` (image id of every entity) is given; reported only, never used to select a checkpoint:
  * in-image ranking: the candidates of a query are the entities of the image of the known entity (every true
    answer of a scene graph lies in that image); same filter and tie rule as the main metric;
  * relation prediction (h, ?, t): rank the true relation among all R relations, filtered with {(h, t): {r}}.
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
                       rel=None, all_geo=None):
    """all_mask: optional bool [N] image mask, passed to forward_fusion when given (P2.6).
    rel: relation id (int) for a relation-conditioned model: every entity is fused under that relation.
    all_geo: optional [N, k] depth statistics, passed as geo= when given (P2.9)."""
    out = []
    n = len(all_text)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        args = [all_text[i:j].to(device), all_rgb[i:j].to(device), all_depth[i:j].to(device)]
        if all_mask is not None or rel is not None:
            args.append(None if all_mask is None else all_mask[i:j].to(device))
        if rel is not None:
            args.append(torch.full((j - i,), int(rel), dtype=torch.long, device=device))
        kw = {} if all_geo is None else {"geo": all_geo[i:j].to(device)}
        fused = model.forward_fusion(*args, **kw)
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


def _relation_filter(all_true_triplets):
    """{(h, r): {t}} -> {(h, t): {r}}, the filter of relation prediction."""
    rels = defaultdict(set)
    for (h, r), tails in all_true_triplets.items():
        for t in tails:
            rels[(h, t)].add(r)
    return rels


def _image_members(entity_image):
    """[n_img, M] entity ids of every image (M = largest image), padded with -1."""
    img = entity_image.long().cpu()
    n_img = int(img.max()) + 1
    counts = torch.bincount(img, minlength=n_img)
    order = torch.argsort(img, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    members = torch.full((n_img, int(counts.max())), -1, dtype=torch.long)
    members[img[order], torch.arange(len(img)) - starts[img[order]]] = order
    return members


def _rank_in_image(queries, true_scores, true_ids, filters, cand, table, device, known_ids):
    """In-image filtered rank of each query against its own candidates cand [B, M] (entity ids, -1 = padding).
    known_ids: the known entity of each query (h for a tail query, t for a head query); it is a candidate but never
    a true answer (no self-loops), so whether it beats the truth is also returned, to report the ranking without it.
    Returns (rank, n_cand, n_cand_filtered, known_better) long tensors [B]; the counts include the true answer."""
    cand = cand.to(device)
    valid = cand >= 0
    emb = table[cand.clamp(min=0).to(table.device)].to(device)               # [B, M, d]
    scores = -torch.cdist(queries[:, None, :], emb, p=1)[:, 0, :]             # [B, M]
    rows, ents = _excluded_pairs(true_ids, filters, device)
    excluded = torch.zeros(cand.shape, dtype=torch.long, device=device)
    excluded.index_add_(0, rows, (cand[rows] == ents[:, None]).long())
    keep = valid & (excluded == 0)
    better = (scores > true_scores[:, None]) & keep
    is_known = cand == torch.as_tensor(known_ids, dtype=torch.long, device=device)[:, None]
    known_better = (better & is_known).any(dim=1).long()
    return better.sum(dim=1) + 1, valid.sum(dim=1), keep.sum(dim=1) + 1, known_better


def _rank_relation(diff, true_scores, r_l, filters, rel_weight, device):
    """Filtered rank of the true relation among all R: score(r') = -||diff + r'||_1 with diff = f(h) - f(t)."""
    scores = -torch.cdist(diff, -rel_weight, p=1)                             # [B, R]
    rows, rels = _excluded_pairs(r_l, filters, device)
    keep = torch.ones(scores.shape, dtype=torch.bool, device=device)
    keep[rows, rels] = False
    return ((scores > true_scores[:, None]) & keep).sum(dim=1) + 1


def _rank_batch(model, table, h_ids, r_ids, t_ids, all_true_triplets, head_filter, device, entity_chunk_size,
                aux=None):
    """Filtered (head rank, tail rank) numpy arrays of one batch of triples against the fused table.
    aux: {"members", "entity_image", "rel_filter"} to also return a dict of the E1 auxiliary ranks and counts."""
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
    ranks = (better_h + 1).cpu().numpy(), (better_t + 1).cpu().numpy()
    if aux is None:
        return ranks

    members, img = aux["members"], aux["entity_image"]
    # h and t share their image (checked by the caller), so the candidates of both directions are that image
    cand = members[img[h_ids.cpu()]]
    rt, nt, ft, kt = _rank_in_image(q_tail, true_t, t_l, f_tail, cand, table, device, known_ids=h_l)
    rh, nh, fh, kh = _rank_in_image(q_head, true_h, h_l, f_head, cand, table, device, known_ids=t_l)
    f_rel = [list(aux["rel_filter"].get((h, t), set()) - {r}) for h, r, t in zip(h_l, r_l, t_l)]
    rr = _rank_relation(h_emb - t_emb, true_t, r_l, f_rel, model.relation_embeddings.weight.to(device), device)
    out = {"rank_head_img": rh, "rank_tail_img": rt, "n_cand_head": nh, "n_cand_tail": nt,
           "n_filt_head": fh, "n_filt_tail": ft, "known_head": kh, "known_tail": kt, "rank_rel": rr}
    return ranks + ({k: v.cpu().numpy() for k, v in out.items()},)


def _in_image_metrics(aux, rank_h, rank_t):
    pooled = np.concatenate([aux["rank_head_img"], aux["rank_tail_img"]])
    n_filt = np.concatenate([aux["n_filt_head"], aux["n_filt_tail"]])
    gap = np.concatenate([rank_h, rank_t]) - pooled   # entities of other images that beat the true answer
    harmonic = np.cumsum(1.0 / np.arange(1, int(n_filt.max()) + 1)) if n_filt.size else np.zeros(0)
    known = np.concatenate([aux["known_head"], aux["known_tail"]])
    # the same ranking with the known entity removed from the candidates (it never is a true answer)
    excl = _metrics(pooled - known)
    excl["mrr_random"] = float(np.mean(harmonic[n_filt - 2] / (n_filt - 1))) if n_filt.size else 0.0
    m = _metrics(pooled)
    m.update({
        "n_cand": float(np.concatenate([aux["n_cand_head"], aux["n_cand_tail"]]).mean()) if n_filt.size else 0.0,
        "n_cand_filtered": float(n_filt.mean()) if n_filt.size else 0.0,
        "mrr_random": float(np.mean(harmonic[n_filt - 1] / n_filt)) if n_filt.size else 0.0,
        "image_first": float(np.mean(gap <= 0)) if n_filt.size else 0.0,
        "n_negative_gap": int((gap < 0).sum()),
        "known_beats_truth": float(known.mean()) if n_filt.size else 0.0,
        "excl_known": excl,
        "head": _metrics(aux["rank_head_img"]),
        "tail": _metrics(aux["rank_tail_img"]),
    })
    return m


def _relation_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size == 0:
        return {"mrr": 0.0, "mr": 0.0, "hits@1": 0.0, "hits@5": 0.0, "n": 0}
    return {"mrr": float(np.mean(1.0 / ranks)), "mr": float(np.mean(ranks)), "hits@1": float(np.mean(ranks <= 1)),
            "hits@5": float(np.mean(ranks <= 5)), "n": int(ranks.size)}


@torch.no_grad()
def evaluate_link_prediction(model, all_text, all_rgb, all_depth, dataloader, all_true_triplets, device,
                             entity_chunk_size=32768, keep_on_gpu=None, return_ranks=False, desc="Evaluating",
                             all_mask=None, all_geo=None, entity_image=None):
    """
    Args:
        all_mask: optional bool [N] image mask (ent_img_mask.pt) for models that take it (P2.6).
        all_geo: optional [N, k] depth statistics for models with depth_stats (P2.9).
        entity_image: optional LongTensor [N] image id of every entity (scene graphs, entity_image_ids). When given,
            also compute the E1 auxiliary metrics (module docstring). Every evaluated triple must have h and t in
            the same image, and the model must not be relation-conditioned (ValueError otherwise).
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
        With entity_image: also "in_image" (in-image ranking: MRR / MR / Hits pooled over both directions, "head"
        and "tail" sub-dicts, n_cand, n_cand_filtered, mrr_random, image_first, n_negative_gap) and "relation"
        (mrr, mr, hits@1, hits@5, n), plus "rank_head_img", "rank_tail_img", "rank_rel" in "ranks".
    """
    model.eval()
    num_entities = len(all_text)
    batches = list(dataloader)
    h_all = torch.cat([b["h_id"] for b in batches]) if batches else torch.zeros(0, dtype=torch.long)
    r_all = torch.cat([b["r_id"] for b in batches]) if batches else torch.zeros(0, dtype=torch.long)
    t_all = torch.cat([b["t_id"] for b in batches]) if batches else torch.zeros(0, dtype=torch.long)
    aux = None
    if entity_image is not None:
        if getattr(model, "relation_conditioned", False):
            raise ValueError("auxiliary metrics (entity_image) are not supported for relation-conditioned models")
        entity_image = entity_image.long().cpu()
        cross = int((entity_image[h_all] != entity_image[t_all]).sum())
        if cross:
            raise ValueError(f"{cross} evaluated triples join two images; in-image ranking assumes none does")
        aux = {"members": _image_members(entity_image), "entity_image": entity_image,
               "rel_filter": _relation_filter(all_true_triplets)}
    store_device = _pick_store_device(num_entities, model.embed_dim, device, keep_on_gpu)
    head_filter = _head_filter(all_true_triplets)
    rank_h = np.zeros(len(h_all), dtype=np.int64)
    rank_t = np.zeros(len(h_all), dtype=np.int64)
    aux_ranks = {k: np.zeros(len(h_all), dtype=np.int64) for k in (
        "rank_head_img", "rank_tail_img", "n_cand_head", "n_cand_tail", "n_filt_head", "n_filt_tail",
        "known_head", "known_tail", "rank_rel")}
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
                                           all_mask=all_mask, rel=rel, all_geo=all_geo)
                table_rel = rel
            rh, rt = _rank_batch(model, table, h_all[idx], r_all[idx], t_all[idx], all_true_triplets,
                                 head_filter, device, entity_chunk_size)
            rank_h[idx.numpy()], rank_t[idx.numpy()] = rh, rt
    else:
        print(f"Pre-computing fused representations for all {num_entities:,} entities (stored on {store_device})...")
        table = fused_entity_table(model, all_text, all_rgb, all_depth, device, store_device, all_mask=all_mask,
                                   all_geo=all_geo)
        pos = 0
        for batch in tqdm(batches, desc=desc):
            n = len(batch["h_id"])
            out = _rank_batch(model, table, batch["h_id"], batch["r_id"], batch["t_id"], all_true_triplets,
                              head_filter, device, entity_chunk_size, aux=aux)
            rank_h[pos:pos + n], rank_t[pos:pos + n] = out[0], out[1]
            if aux is not None:
                for k, v in out[2].items():
                    aux_ranks[k][pos:pos + n] = v
            pos += n
    del table

    result = _metrics(np.concatenate([rank_h, rank_t]))
    result["tail"] = _metrics(rank_t)
    result["head"] = _metrics(rank_h)
    if aux is not None:
        result["in_image"] = _in_image_metrics(aux_ranks, rank_h, rank_t)
        result["relation"] = _relation_metrics(aux_ranks["rank_rel"])
    if return_ranks:
        result["ranks"] = {"h": h_all.numpy(), "r": r_all.numpy(), "t": t_all.numpy(),
                           "rank_head": rank_h, "rank_tail": rank_t}
        if aux is not None:
            result["ranks"].update({k: aux_ranks[k] for k in ("rank_head_img", "rank_tail_img", "rank_rel")})
    return result


def format_metrics(m):
    return (f"MRR {m['mrr']:.4f} | MR {m['mr']:.1f} | Hits@1 {m['hits@1']:.4f} | "
            f"Hits@3 {m['hits@3']:.4f} | Hits@10 {m['hits@10']:.4f} "
            f"(tail MRR {m['tail']['mrr']:.4f}, head MRR {m['head']['mrr']:.4f})")


def format_aux_metrics(m):
    """One line of the E1 auxiliary metrics, '' when the evaluation did not compute them."""
    if not m.get("in_image"):
        return ""
    i, r = m["in_image"], m["relation"]
    return (f"in-image MRR {i['mrr']:.4f} | H@1 {i['hits@1']:.4f} | H@3 {i['hits@3']:.4f} | "
            f"image-first {i['image_first']:.3f} | cand {i['n_cand']:.1f} ({i['n_cand_filtered']:.1f} filtered, "
            f"random MRR {i['mrr_random']:.3f}) | without the known entity: MRR {i['excl_known']['mrr']:.4f}, "
            f"H@1 {i['excl_known']['hits@1']:.4f} (it beats the truth in {i['known_beats_truth']:.3f}) || "
            f"relation MRR {r['mrr']:.4f} | H@1 {r['hits@1']:.4f} | H@5 {r['hits@5']:.4f}")


def save_ranks(ranks, path):
    """Write per-triplet ranks in the shared format read by scripts/compute_metrics.py:
    CSV header h_id,r_id,t_id,rank_head,rank_tail (ids from entity2id / relation2id)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["h_id", "r_id", "t_id", "rank_head", "rank_tail"])
        for row in zip(ranks["h"], ranks["r"], ranks["t"], ranks["rank_head"], ranks["rank_tail"]):
            w.writerow([int(x) for x in row])
    print(f"Saved {len(ranks['h']):,} per-triplet ranks to {path}")


def save_aux_ranks(ranks, path):
    """Per-triplet E1 auxiliary ranks (in-image head / tail, relation), next to the shared ranks file:
    CSV header h_id,r_id,t_id,rank_head_img,rank_tail_img,rank_rel."""
    cols = ("h", "r", "t", "rank_head_img", "rank_tail_img", "rank_rel")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["h_id", "r_id", "t_id", "rank_head_img", "rank_tail_img", "rank_rel"])
        for row in zip(*(ranks[c] for c in cols)):
            w.writerow([int(x) for x in row])
    print(f"Saved {len(ranks['h']):,} per-triplet auxiliary ranks to {path}")
