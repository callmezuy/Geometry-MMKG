"""Thống kê các bộ dữ liệu trong data/ (v0) và in ra bảng Markdown.

Chạy từ Code/:
    python scripts/dataset_stats.py                         # cả 6 bộ, có kiểm tra embedding
    python scripts/dataset_stats.py --no-emb                # bỏ qua embedding (nhanh)
    python scripts/dataset_stats.py --out experiment_results/dataset_stats.md

Nội dung:
  1. Kích thước: số thực thể / quan hệ / bộ ba theo split, mật độ đồ thị.
  2. Thiết lập: tỉ lệ thực thể valid/test không có trong train (transductive vs inductive).
  3. Phân bố quan hệ: độ lệch (top-k), quan hệ hiếm, quan hệ test không có trong train.
  4. Loại ánh xạ quan hệ 1-1 / 1-N / N-1 / N-N (Bordes et al. 2013, ngưỡng 1.5) và quan hệ đối xứng.
  5. MMSG: nhóm quan hệ (GEOMETRIC/POSSESSIVE/SEMANTIC/MISC), số ảnh, thực thể/ảnh.
  6. Embedding: số chiều, hàng toàn 0 hoặc NaN.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

DATASETS = ["FB15k-237-IMG", "FB15K-237-ind", "MKG-Y", "Spatial-VRD", "Spatial-VRD-Trans", "Spatial-VG"]
SPLITS = ["train", "valid", "test"]


def read_count_file(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    assert len(rows) == n, (path, n, len(rows))
    return rows


def load_dataset(d):
    ent = read_count_file(os.path.join(d, "entity2id.txt"))
    rel = read_count_file(os.path.join(d, "relation2id.txt"))
    trip = {}
    for s in SPLITS:
        rows = read_count_file(os.path.join(d, f"{s}2id.txt"))
        trip[s] = np.array([[int(h), int(r), int(t)] for h, t, r in rows], dtype=np.int64)  # OpenKE: h t r
    return [e[0] for e in ent], [r[0] for r in rel], trip


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def mapping_types(train, n_rel):
    """Phân loại quan hệ theo tph/hpt trên train (Bordes 2013): <1.5 là 'một', >=1.5 là 'nhiều'."""
    tails_per_head = defaultdict(set)
    heads_per_tail = defaultdict(set)
    for h, r, t in train:
        tails_per_head[(r, h)].add(t)
        heads_per_tail[(r, t)].add(h)
    tph = defaultdict(list)
    hpt = defaultdict(list)
    for (r, _), v in tails_per_head.items():
        tph[r].append(len(v))
    for (r, _), v in heads_per_tail.items():
        hpt[r].append(len(v))
    cat = {}
    for r in range(n_rel):
        if r not in tph:
            continue
        a = np.mean(tph[r]) >= 1.5
        b = np.mean(hpt[r]) >= 1.5
        cat[r] = {(False, False): "1-1", (True, False): "1-N", (False, True): "N-1", (True, True): "N-N"}[(a, b)]
    return cat


def symmetric_rels(train, min_support=10, thr=0.5):
    """Quan hệ có >= thr bộ ba (h,r,t) mà (t,r,h) cũng có trong train."""
    s = set(map(tuple, train.tolist()))
    cnt = Counter(train[:, 1].tolist())
    rev = Counter(r for h, r, t in s if h != t and (t, r, h) in s)
    return {r for r, c in cnt.items() if c >= min_support and rev[r] / c >= thr}


def emb_check(d, n_ent):
    import torch

    out = []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("ent_") and f.endswith(".pt")):
            continue
        t = torch.load(os.path.join(d, f), map_location="cpu", weights_only=True, mmap=True)
        if not torch.is_tensor(t) or t.dim() != 2:  # ent_fill.pt (dict), ent_img_mask.pt (1-D bool)
            continue
        zero = nan = 0
        for i in range(0, t.shape[0], 200_000):
            c = t[i : i + 200_000].float()
            nan += int(torch.isnan(c).any(1).sum())
            zero += int((c.abs().sum(1) == 0).sum())
        out.append((f, tuple(t.shape), str(t.dtype).replace("torch.", ""), zero, nan, t.shape[0] == n_ent))
        del t
    return out


def stats(name, d, do_emb):
    ents, rels, trip = load_dataset(d)
    n_ent, n_rel = len(ents), len(rels)
    tr = trip["train"]
    L = [f"## {name}", ""]

    # 1-2. Kích thước và thiết lập
    train_ents = set(tr[:, 0].tolist()) | set(tr[:, 2].tolist())
    all_used = set(train_ents)
    L += ["| split | triples | entities | relations | entity unseen in train (%) | triples with unseen h or t (%) |",
          "|---|---:|---:|---:|---:|---:|"]
    for s in SPLITS:
        x = trip[s]
        es = set(x[:, 0].tolist()) | set(x[:, 2].tolist())
        all_used |= es
        unseen = es - train_ents
        tu = sum(1 for h, _, t in x if h not in train_ents or t not in train_ents)
        L.append(f"| {s} | {len(x):,} | {len(es):,} | {len(set(x[:, 1].tolist()))} | {pct(len(unseen), len(es)):.1f} | {pct(tu, len(x)):.1f} |")
    deg = Counter(tr[:, 0].tolist()) + Counter(tr[:, 2].tolist())
    degs = np.array(list(deg.values()))
    total = sum(len(trip[s]) for s in SPLITS)
    L += ["",
          f"- Thực thể trong entity2id: **{n_ent:,}** (xuất hiện trong ít nhất một split: {len(all_used):,}); quan hệ: **{n_rel}**; tổng bộ ba: **{total:,}**.",
          f"- Bậc trong train: trung bình {degs.mean():.2f}, trung vị {np.median(degs):.0f}, max {degs.max():,}; thực thể bậc 1: {pct((degs == 1).sum(), len(degs)):.1f}%.",
          f"- Mật độ train (bộ ba / thực thể): {len(tr) / len(train_ents):.2f}.",
          ""]

    # 3. Phân bố quan hệ
    rc = Counter(tr[:, 1].tolist())
    freq = sorted(rc.values(), reverse=True)
    test_rel = Counter(trip["test"][:, 1].tolist())
    miss = [r for r in test_rel if r not in rc]
    L += [f"- Quan hệ: top-1 chiếm {pct(freq[0], len(tr)):.1f}% train, top-5 {pct(sum(freq[:5]), len(tr)):.1f}%, "
          f"top-10 {pct(sum(freq[:10]), len(tr)):.1f}%; quan hệ có < 10 bộ ba train: {sum(1 for c in freq if c < 10)}/{len(rc)}; "
          f"quan hệ không có trong train: {n_rel - len(rc)}.",
          f"- Quan hệ có trong test nhưng không có trong train: {len(miss)} ({sum(test_rel[r] for r in miss)} bộ ba test).",
          f"- Top-5 quan hệ train: " + ", ".join(f"`{rels[r]}` {c:,}" for r, c in rc.most_common(5)) + ".",
          ""]

    # 4. Loại ánh xạ + đối xứng
    mt = mapping_types(tr, n_rel)
    sym = symmetric_rels(tr)
    te = trip["test"]
    tcat = Counter(mt.get(r, "unseen") for r in te[:, 1].tolist())
    rcat = Counter(mt.values())
    L += ["| mapping | relations | test triples (%) |", "|---|---:|---:|"]
    for k in ["1-1", "1-N", "N-1", "N-N", "unseen"]:
        if rcat.get(k) or tcat.get(k):
            L.append(f"| {k} | {rcat.get(k, 0)} | {pct(tcat.get(k, 0), len(te)):.1f} |")
    L += ["",
          f"- Quan hệ đối xứng (≥ 50% bộ ba có chiều ngược trong train, ≥ 10 bộ ba): {len(sym)}"
          + (" (" + ", ".join(f"`{rels[r]}`" for r in sorted(sym)[:10]) + ")" if sym else "") + ".",
          ""]

    # 5. Đặc thù từng loại
    cat_file = os.path.join(d, "relation_category.tsv")
    if os.path.exists(cat_file):
        with open(cat_file, encoding="utf-8") as f:
            next(f)
            r2c = {int(p[1]): p[2] for p in (l.rstrip("\n").split("\t") for l in f) if len(p) >= 3}
        L += ["| category | relations | train (%) | test (%) |", "|---|---:|---:|---:|"]
        crel = Counter(r2c.values())
        ctr = Counter(r2c.get(r, "?") for r in tr[:, 1].tolist())
        cte = Counter(r2c.get(r, "?") for r in te[:, 1].tolist())
        for c in ["GEOMETRIC", "POSSESSIVE", "SEMANTIC", "MISC"]:
            if crel.get(c):
                L.append(f"| {c} | {crel[c]} | {pct(ctr[c], len(tr)):.1f} | {pct(cte[c], len(te)):.1f} |")
        L.append("")
    if "_" in ents[0] and ents[0].rsplit("_", 1)[1].isdigit():  # VRD: tên thực thể = <image>_<object>
        img_of = lambda e: ents[e].rsplit("_", 1)[0]
        L += ["| split | images | entities / image | triples / image |", "|---|---:|---:|---:|"]
        imgs_by = {}
        for s in SPLITS:
            x = trip[s]
            ti = Counter(img_of(h) for h in x[:, 0].tolist())
            ei = defaultdict(set)
            for h, _, t in x:
                ei[img_of(h)].update((h, t))
            imgs_by[s] = set(ti)
            L.append(f"| {s} | {len(ti):,} | {np.mean([len(v) for v in ei.values()]):.2f} | {np.mean(list(ti.values())):.2f} |")
        ov = len(imgs_by["test"] & imgs_by["train"])
        L += ["", f"- Ảnh test cũng có trong train: {ov:,}/{len(imgs_by['test']):,} ({pct(ov, len(imgs_by['test'])):.1f}%).", ""]

    # 6. Embedding
    if do_emb:
        L += ["| embedding | shape | dtype | zero rows | NaN rows | rows = n_ent |", "|---|---|---|---:|---:|---|"]
        for f, shp, dt, z, nn, ok in emb_check(d, n_ent):
            L.append(f"| `{f}` | {shp[0]:,} × {shp[1]} | {dt} | {z:,} | {nn:,} | {'✓' if ok else '✗'} |")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--no-emb", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = ["# Thống kê dữ liệu", "", f"Sinh bởi `scripts/dataset_stats.py` từ `{a.data_root}/`.", ""]
    for n in a.datasets:
        L += stats(n, os.path.join(a.data_root, n), not a.no_emb)
    text = "\n".join(L)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
