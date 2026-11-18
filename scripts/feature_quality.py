"""
Quality checks of the v4 entity features (RESEARCH_PLAN.md P0.7 / P0.13), per modality (rgb, depth, text):
  mean_cos      mean pairwise cosine over a sample (v1-v3 depth: 0.97-0.999, i.e. all rows nearly identical)
  pc1_var       share of variance on the first principal component
  probe_acc     linear probe (multinomial logistic regression on standardised features, 80/20 stratified split)
                predicting an entity label; compared with the majority class (v1-v3 VG depth: 7.7% < 8.4%)
  knn10_overlap mean overlap of the 10 nearest neighbours (cosine) of depth and of RGB, with the chance level
Only entities with an image (ent_img_mask.pt) are used for rgb / depth, so filled rows do not inflate the numbers.

Labels:
  vrd       object class (objects[k].names[0] of sg_{train,test}_annotations.json; entity "<photo_id>_<k>")
  vg        VG150 object class (VG-SGG.h5 labels; entity id = box index)
  fb15k237  BLP top-50 Freebase type (data/raw/fb15k237_ind/blp/FB15k-237/top50-entity2type.txt)
  none      no probe (MKG-Y)

Usage (from Code/):
  python scripts/feature_quality.py --data data/FB15k-237-IMG --labels fb15k237
  python scripts/feature_quality.py --data data/Spatial-VRD --labels vrd
Writes <feat>/feature_quality.json.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

BLP_TYPES = "data/raw/fb15k237_ind/blp/FB15k-237/top50-entity2type.txt"


def read_count_file(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    assert len(rows) == n, path
    return rows


def labels_for(kind, data_dir, n_ent):
    y = np.full(n_ent, -1, dtype=np.int64)
    names = {}
    if kind == "none":
        return y, names
    if kind == "vg":
        import h5py
        with h5py.File(os.path.join(data_dir, "VG-SGG.h5"), "r") as h:
            y[:] = h["labels"][:, 0]
        return y, names
    ent2id = {n: int(i) for n, i in read_count_file(os.path.join(data_dir, "entity2id.txt"))}
    raw = {}
    if kind == "vrd":
        for js in ["sg_train_annotations.json", "sg_test_annotations.json"]:
            with open(os.path.join(data_dir, js), encoding="utf-8") as f:
                for img in json.load(f):
                    pid = str(img.get("photo_id", os.path.splitext(img["filename"])[0]))
                    for k, obj in enumerate(img.get("objects", [])):
                        e = f"{pid}_{k}"
                        if e in ent2id and obj.get("names"):
                            raw[ent2id[e]] = obj["names"][0]
    elif kind == "fb15k237":
        with open(BLP_TYPES, encoding="utf-8") as f:
            for l in f:
                e, t = l.rstrip("\n").split("\t")
                if e in ent2id:
                    raw[ent2id[e]] = t
    classes = sorted(set(raw.values()))
    cid = {c: i for i, c in enumerate(classes)}
    for i, c in raw.items():
        y[i] = cid[c]
    return y, dict(enumerate(classes))


@torch.no_grad()
def geometry(X, rows, rng, sample=2000):
    idx = np.sort(rng.choice(rows, min(sample, len(rows)), replace=False))
    A = torch.as_tensor(X[idx], dtype=torch.float32)
    An = F.normalize(A, dim=1)
    C = An @ An.T
    cos = float(C[torch.triu(torch.ones_like(C, dtype=torch.bool), 1)].mean())
    ev = torch.linalg.svdvals(A - A.mean(0)) ** 2
    return {"mean_cos": round(cos, 4), "pc1_var": round(float(ev[0] / ev.sum()), 4)}


def probe(X, y, rows, rng, max_n, min_per_class=20):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rows = rows[y[rows] >= 0]
    keep = {c for c, n in Counter(y[rows].tolist()).items() if n >= min_per_class}
    rows = rows[np.isin(y[rows], list(keep))]
    if len(rows) > max_n:
        rows = np.sort(rng.choice(rows, max_n, replace=False))
    tr, te = train_test_split(rows, test_size=0.2, random_state=0, stratify=y[rows])
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    acc = float((clf.predict(sc.transform(X[te])) == y[te]).mean())
    maj = Counter(y[tr].tolist()).most_common(1)[0][0]
    return {"probe_acc": round(acc, 4), "majority_acc": round(float((y[te] == maj).mean()), 4),
            "probe_classes": len(keep), "probe_entities": int(len(rows))}


@torch.no_grad()
def knn_overlap(A, B, rows, rng, k=10, sample=2000):
    q = np.sort(rng.choice(rows, min(sample, len(rows)), replace=False))
    pool = rows if len(rows) <= 200_000 else np.sort(rng.choice(rows, 200_000, replace=False))
    q = np.intersect1d(q, pool)
    def nn(X):
        P = F.normalize(torch.as_tensor(X[pool], dtype=torch.float32), dim=1)
        Q = F.normalize(torch.as_tensor(X[q], dtype=torch.float32), dim=1)
        s = Q @ P.T
        s[torch.arange(len(q)), torch.as_tensor(np.searchsorted(pool, q))] = -2  # drop self
        return s.topk(k, dim=1).indices
    a, b = nn(A), nn(B)
    ov = np.mean([len(set(x.tolist()) & set(z.tolist())) / k for x, z in zip(a, b)])
    return {"knn10_overlap_depth_rgb": round(float(ov), 4), "chance": round(k / (len(pool) - 1), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir (entity2id.txt, labels)")
    ap.add_argument("--feat", help="feature dir (default: --data)")
    ap.add_argument("--labels", required=True, choices=["vrd", "vg", "fb15k237", "none"])
    ap.add_argument("--max-probe", type=int, default=30000, help="entities per probe (subsampled)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    a.feat = a.feat or a.data
    mask = torch.load(os.path.join(a.feat, "ent_img_mask.pt")).numpy()
    n_ent = len(mask)
    y, classes = labels_for(a.labels, a.data, n_ent)
    out = {"data": a.data, "labels": a.labels, "entities": n_ent, "entities_with_image": int(mask.sum()),
           "entities_with_label": int((y >= 0).sum())}
    feats = {}
    for name in ("rgb", "depth", "text"):
        X = torch.load(os.path.join(a.feat, f"ent_{name}_emb.pt"), mmap=True).numpy()
        assert X.shape[0] == n_ent
        rows = np.flatnonzero(mask) if name != "text" else np.arange(n_ent)
        r = geometry(X, rows, rng)
        if a.labels != "none":
            # text is probed on the same entities as the images, so the three numbers are comparable
            r.update(probe(X, y, np.flatnonzero(mask), rng, a.max_probe))
        out[name] = r
        feats[name] = X
        print(name, r, flush=True)
    out.update(knn_overlap(feats["depth"], feats["rgb"], np.flatnonzero(mask), rng))
    with open(os.path.join(a.feat, "feature_quality.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
