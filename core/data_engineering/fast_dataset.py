"""
Data access for GeometryMMKG on the v0 datasets (data/README.md).

Triples come from the OpenKE files `{split}2id.txt` (first line = count, then `head_id tail_id relation_id`),
the same files every baseline reads; they are returned as LongTensor [M, 3] in (h, r, t) order.
Entity features are the frozen tables `ent_{text,rgb,depth}_emb.pt` [n, 768] plus `ent_img_mask.pt` [n].
"""
import os
from collections import defaultdict

import torch


def read_count(path):
    with open(path, encoding="utf-8") as f:
        return int(f.readline().strip())


def load_triples(data_dir, split):
    """(h, r, t) LongTensor [M, 3] from `{split}2id.txt`."""
    rows = []
    with open(os.path.join(data_dir, f"{split}2id.txt"), encoding="utf-8") as f:
        n = int(f.readline().strip())
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                h, t, r = map(int, parts)
                rows.append((h, r, t))
    if len(rows) != n:
        raise ValueError(f"{split}2id.txt: header says {n} rows, read {len(rows)}")
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 3)


def get_all_true_triplets(data_dir, splits=("train", "valid", "test")):
    """{(h, r): set(t)} over train + valid + test, the filter set of the evaluation (RESEARCH_PLAN.md 4.3)."""
    all_true = defaultdict(set)
    for split in splits:
        for h, r, t in load_triples(data_dir, split).tolist():
            all_true[(h, r)].add(t)
    return all_true


class TripleBatches:
    """Iterable of {"h_id", "r_id", "t_id"} batches over a [M, 3] triple tensor (fixed order), for evaluation."""

    def __init__(self, triples, batch_size):
        self.triples, self.batch_size = triples, batch_size

    def __len__(self):
        return (len(self.triples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.triples), self.batch_size):
            b = self.triples[i:i + self.batch_size]
            yield {"h_id": b[:, 0], "r_id": b[:, 1], "t_id": b[:, 2]}


class EntityFeatures:
    """Frozen feature tables and image mask; `gather(ids)` returns them on the compute device.
    `store` is where the tables live (GPU when they fit, else CPU; RESEARCH_PLAN.md P1.6)."""

    def __init__(self, text, rgb, depth, mask, device, store=None):
        store = store or device
        self.text, self.rgb, self.depth = text.to(store), rgb.to(store), depth.to(store)
        self.mask = mask.to(store)
        self.device, self.store = device, store

    def __len__(self):
        return len(self.text)

    def gather(self, ids):
        ids = ids.to(self.store)
        return tuple(x[ids].to(self.device, non_blocking=True)
                     for x in (self.text, self.rgb, self.depth, self.mask))


def load_entity_features(data_dir, device, keep_on_gpu=None):
    """Load the v0 feature tables and check them (shape, dtype, finite, mask length).
    keep_on_gpu: None = automatic (tables on the GPU if they take < 50% of the free memory)."""
    n = read_count(os.path.join(data_dir, "entity2id.txt"))
    load = lambda fn: torch.load(os.path.join(data_dir, fn), map_location="cpu", weights_only=True)  # noqa: E731
    tables = {k: load(f"ent_{k}_emb.pt") for k in ("text", "rgb", "depth")}
    mask = load("ent_img_mask.pt")
    for k, x in tables.items():
        if x.dim() != 2 or x.shape[0] != n or x.dtype != torch.float32:
            raise ValueError(f"ent_{k}_emb.pt: expected float32 [{n}, d], got {x.dtype} {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError(f"ent_{k}_emb.pt has non-finite values")
    if mask.dtype != torch.bool or mask.shape != (n,):
        raise ValueError(f"ent_img_mask.pt: expected bool [{n}], got {mask.dtype} {tuple(mask.shape)}")
    store = device
    if device.type == "cuda":
        need = sum(x.numel() * 4 for x in tables.values())
        free, _ = torch.cuda.mem_get_info(device)
        if keep_on_gpu is False or (keep_on_gpu is None and need >= 0.5 * free):
            store = torch.device("cpu")
    return EntityFeatures(tables["text"], tables["rgb"], tables["depth"], mask, device, store)
