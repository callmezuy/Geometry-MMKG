"""
Depth probe (RESEARCH_PLAN.md P2.9 A): do the frozen depth features carry information about spatial relations
that the RGB / text features and the 2D box geometry do not?

Pairs (head, tail) of Spatial-VRD objects in the same image, labelled with their relation(s). Small classifiers on
frozen per-entity features, trained on train images (10% of the train images held out for early stopping) and
reported on the valid split. The test split is never read.

Tasks
  rel52     : which of the 52 relations (multi-label pairs: top-1 counts as correct if it is any true relation)
  front     : "in front of" vs "behind"   (depth order: the case where depth should help)
  vertical  : "above" / "over" vs "under" (2D control: the box should solve it, depth should add little)
  horizontal: "left of" vs "right of"     (2D control)
Feature sets: concatenations of [f(head), f(tail)] for text / rgb / depth, and `box` = normalised boxes of both
objects plus pair geometry (centre offsets, log size ratios, IoU, overlap fractions).
Every input dimension is standardised with train statistics.

Run from Code/:  python scripts/depth_probe.py --out experiment_results/depth_probe
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_engineering.fast_dataset import load_triples  # noqa: E402

FEATURE_SETS = ["text", "rgb", "depth", "box", "rgb+depth", "text+rgb", "text+rgb+depth", "text+rgb+box",
                "text+rgb+box+depth", "box+rgb", "box+depth", "box+rgb+depth"]
BINARY_TASKS = {"front": (["in front of"], ["behind"]),
                "vertical": (["above", "over"], ["under"]),
                "horizontal": (["left of"], ["right of"])}
HOLDOUT_FRAC = 0.1
HOLDOUT_SEED = 0


def read_names(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        out = {}
        for _ in range(n):
            name, idx = f.readline().rstrip("\n").split("\t")
            out[name] = int(idx)
    return out


def load_boxes(data_dir, entity2id):
    """[n_ent, 4] (x1, y1, x2, y2) / image size, NaN where unknown."""
    boxes = np.full((len(entity2id), 4), np.nan, dtype=np.float64)
    for fn in ("sg_train_annotations.json", "sg_test_annotations.json"):
        with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
            data = json.load(f)
        for img in data:
            pid = str(img.get("photo_id", os.path.splitext(img["filename"])[0]))
            W, H = float(img["width"]), float(img["height"])
            for k, obj in enumerate(img.get("objects", [])):
                eid = entity2id.get(f"{pid}_{k}")
                b = obj.get("bbox")
                if eid is None or not b:
                    continue
                boxes[eid] = (b["x"] / W, b["y"] / H, (b["x"] + b["w"]) / W, (b["y"] + b["h"]) / H)
    return boxes


def box_pair_features(bh, bt):
    """bh, bt: [P, 4] normalised boxes -> [P, 24]."""
    def single(b):
        w = np.clip(b[:, 2] - b[:, 0], 1e-4, None)
        h = np.clip(b[:, 3] - b[:, 1], 1e-4, None)
        return np.stack([b[:, 0], b[:, 1], b[:, 2], b[:, 3], w, h, (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                         np.log(w * h)], 1), w, h
    fh, wh, hh = single(bh)
    ft, wt, ht = single(bt)
    ix = np.clip(np.minimum(bh[:, 2], bt[:, 2]) - np.maximum(bh[:, 0], bt[:, 0]), 0, None)
    iy = np.clip(np.minimum(bh[:, 3], bt[:, 3]) - np.maximum(bh[:, 1], bt[:, 1]), 0, None)
    inter = ix * iy
    ah, at = wh * hh, wt * ht
    pair = np.stack([fh[:, 6] - ft[:, 6], fh[:, 7] - ft[:, 7], np.log(wh / wt), np.log(hh / ht),
                     inter / (ah + at - inter), inter / ah], 1)
    return np.concatenate([fh, ft, pair[:, :4], pair[:, 4:]], 1).astype(np.float32)


def build_pairs(triples, keep_rel=None):
    """triples [N, 3] (h, r, t) -> heads, tails, list of relation-id sets (pairs in first-seen order)."""
    rels = defaultdict(set)
    order = []
    for h, r, t in triples.tolist():
        if keep_rel is not None and r not in keep_rel:
            continue
        if (h, t) not in rels:
            order.append((h, t))
        rels[(h, t)].add(r)
    heads = np.array([p[0] for p in order], dtype=np.int64)
    tails = np.array([p[1] for p in order], dtype=np.int64)
    return heads, tails, [rels[p] for p in order]


def image_of(entity_names, ids):
    return np.array([entity_names[i].rsplit("_", 1)[0] for i in ids])


class Probe(nn.Module):
    def __init__(self, d_in, n_out, hidden):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_out))
        else:
            self.net = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.net(x)


def assemble(fs, feats, boxfeat, heads, tails):
    parts = []
    for name in fs.split("+"):
        if name == "box":
            parts.append(boxfeat)
        else:
            parts.append(np.concatenate([feats[name][heads], feats[name][tails]], 1))
    return np.concatenate(parts, 1)


def train_eval(X_tr, Y_tr, X_ho, Y_ho, X_va, Y_va, seed, hidden, device, max_epochs=200, patience=10, bs=1024):
    """Y_*: [P, C] float targets (rows sum to 1). Early stopping on held-out top-1 accuracy. Returns valid logits."""
    torch.manual_seed(seed)
    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
    to = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(device)  # noqa: E731
    Xtr, Xho, Xva = to(X_tr), to(X_ho), to(X_va)
    Ytr = torch.from_numpy(Y_tr).to(device)
    m = Probe(Xtr.shape[1], Ytr.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = -1.0, None, 0
    for _ep in range(max_epochs):
        m.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(bs):
            idx = idx.to(device)
            loss = -(Ytr[idx] * torch.log_softmax(m(Xtr[idx]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            acc = top1_correct(m(Xho).cpu().numpy(), Y_ho).mean()
        if acc > best + 1e-4:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return m(Xva).cpu().numpy()


def top1_correct(logits, Y):
    return Y[np.arange(len(Y)), logits.argmax(1)] > 0


def targets(rel_sets, classes):
    col = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(rel_sets), len(classes)), dtype=np.float32)
    for i, s in enumerate(rel_sets):
        for r in s:
            if r in col:
                Y[i, col[r]] = 1.0
        Y[i] /= max(Y[i].sum(), 1.0)
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Spatial-VRD")
    ap.add_argument("--out", default="experiment_results/depth_probe")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=512, help="0 = linear probe")
    ap.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    ap.add_argument("--depth-stats", default=None,
                    help="scripts/vrd_depth_stats.py output; adds the feature `dstat` (its 5 basic statistics)")
    ap.add_argument("--extra", nargs="+", default=[], metavar="NAME=PATH:KEY",
                    help="extra per-entity features from scripts/vrd_depth_stats.py outputs, e.g. "
                         "dz5=results/probe/vrd_depth_stats_zipdepth.pt:stats5")
    ap.add_argument("--pca", type=int, default=0,
                    help="project text / rgb / depth to K dims (PCA fitted on train entities); 0 = off")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    e2id = read_names(os.path.join(a.data, "entity2id.txt"))
    r2id = read_names(os.path.join(a.data, "relation2id.txt"))
    id2e = {v: k for k, v in e2id.items()}
    id2r = {v: k for k, v in r2id.items()}
    category = {}
    with open(os.path.join(a.data, "relation_category.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            name, _rid, cat, _cnt = line.rstrip("\n").split("\t")
            category[r2id[name]] = cat
    feats = {k: torch.load(os.path.join(a.data, f"ent_{k}_emb.pt"), weights_only=True).numpy() for k in ("text", "rgb", "depth")}
    mask = torch.load(os.path.join(a.data, "ent_img_mask.pt"), weights_only=True).numpy().astype(bool)
    boxes = load_boxes(a.data, e2id)
    has = mask & ~np.isnan(boxes).any(1)
    train = load_triples(a.data, "train").numpy()
    valid = load_triples(a.data, "valid").numpy()
    extra = [f"dstat={a.depth_stats}:" + ("stats" if "stats" in torch.load(a.depth_stats, weights_only=False)
                                           else "stats5")] if a.depth_stats else []
    for spec in extra + a.extra:
        name, rest = spec.split("=", 1)
        path, key = rest.rsplit(":", 1)
        feats[name] = torch.load(path, weights_only=False)[key].numpy()
        has &= ~np.isnan(feats[name]).any(1)
    if a.pca:
        tr_ents = np.unique(train[:, [0, 2]])
        for k in ("text", "rgb", "depth"):
            X = feats[k]
            mu = X[tr_ents].mean(0, keepdims=True)
            _u, _s, vt = np.linalg.svd(X[tr_ents] - mu, full_matrices=False)
            feats[k] = ((X - mu) @ vt[: a.pca].T).astype(np.float32)

    def split(triples, keep_rel=None):
        h, t, rs = build_pairs(triples, keep_rel)
        ok = has[h] & has[t]
        return h[ok], t[ok], [rs[i] for i in np.flatnonzero(ok)]

    tasks = {"rel52": None}
    for name, (pos, neg) in BINARY_TASKS.items():
        tasks[name] = ({r2id[x] for x in pos}, {r2id[x] for x in neg})

    results = {"args": vars(a), "device": str(device), "tasks": {}}
    for task, spec in tasks.items():
        if spec is None:
            classes = sorted(id2r)
            h_tr, t_tr, rs_tr = split(train)
            h_va, t_va, rs_va = split(valid)
            Ytr_all, Yva = targets(rs_tr, classes), targets(rs_va, classes)
        else:
            pos, neg = spec
            keep = pos | neg

            def binary(triples):
                h, t, rs = split(triples, keep)
                lab = np.array([int(bool(s & pos)) - int(bool(s & neg)) for s in rs])  # +1 pos, -1 neg, 0 both
                ok = lab != 0
                Y = np.zeros((ok.sum(), 2), dtype=np.float32)
                Y[np.arange(ok.sum()), (lab[ok] > 0).astype(int)] = 1.0
                return h[ok], t[ok], Y, [rs[i] for i in np.flatnonzero(ok)]
            h_tr, t_tr, Ytr_all, rs_tr = binary(train)
            h_va, t_va, Yva, rs_va = binary(valid)
        # hold out 10% of the train IMAGES for early stopping
        imgs = image_of(id2e, h_tr)
        uniq = np.unique(imgs)
        rng = np.random.default_rng(HOLDOUT_SEED)
        ho_imgs = set(rng.choice(uniq, size=max(1, int(len(uniq) * HOLDOUT_FRAC)), replace=False).tolist())
        ho = np.array([i in ho_imgs for i in imgs])
        box_tr = box_pair_features(boxes[h_tr], boxes[t_tr])
        box_va = box_pair_features(boxes[h_va], boxes[t_va])
        majority = top1_correct(np.tile(Ytr_all[~ho].sum(0), (len(Yva), 1)), Yva).mean()
        info = {"n_train": int((~ho).sum()), "n_holdout": int(ho.sum()), "n_valid": int(len(Yva)),
                "majority_acc": float(majority), "feature_sets": {}}
        if spec is None:
            groups = {"all": np.ones(len(rs_va), bool)}
            for cat in ("GEOMETRIC", "POSSESSIVE", "SEMANTIC"):
                groups[cat] = np.array([any(category[r] == cat for r in s) for s in rs_va])
            for gname, names in (("front/behind", ["in front of", "behind"]),
                                 ("above/under/over", ["above", "under", "over"]),
                                 ("left/right", ["left of", "right of"])):
                ids = {r2id[x] for x in names}
                groups[gname] = np.array([bool(s & ids) for s in rs_va])
            info["groups_n"] = {g: int(v.sum()) for g, v in groups.items()}
        print(f"[{task}] train {info['n_train']}  holdout {info['n_holdout']}  valid {info['n_valid']}  "
              f"majority {majority:.4f}", flush=True)
        for fs in a.feature_sets:
            X_all = assemble(fs, feats, box_tr, h_tr, t_tr)
            X_va = assemble(fs, feats, box_va, h_va, t_va)
            per_seed = []
            for s in a.seeds:
                logits = train_eval(X_all[~ho], Ytr_all[~ho], X_all[ho], Ytr_all[ho], X_va, Yva, s, a.hidden, device)
                corr = top1_correct(logits, Yva)
                if spec is None:
                    per_seed.append({g: float(corr[m].mean()) for g, m in groups.items()})
                else:
                    per_seed.append({"acc": float(corr.mean())})
            keys = per_seed[0].keys()
            summ = {k: {"mean": float(np.mean([p[k] for p in per_seed])), "std": float(np.std([p[k] for p in per_seed])),
                        "per_seed": [p[k] for p in per_seed]} for k in keys}
            info["feature_sets"][fs] = summ
            print(f"  {fs:<20} " + "  ".join(f"{k} {v['mean']:.4f}+-{v['std']:.4f}" for k, v in summ.items()), flush=True)
        results["tasks"][task] = info
    results["total_sec"] = time.time() - t0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tag = ("mlp" if a.hidden else "linear") + (f"_pca{a.pca}" if a.pca else "")
    with open(f"{a.out}_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}_{tag}.json  ({results['total_sec'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
