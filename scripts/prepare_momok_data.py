"""
Write one of our v0 datasets in the format of the MoMoK code (RESEARCH_PLAN.md P3.6), at the data level only:
  <momok>/datasets/<name>/{entity2id,relation2id,train,valid,test}.txt   ids as tokens e<id> / r<id>, so MoMoK's
                                                                         internal ids are our ids
  <momok>/datasets/<name>/img_features.pth   image slot: ent_rgb_emb.pt (Control) or ent_rgb_depth_mean_emb.pt (+D mean)
  <momok>/datasets/<name>/text_features.pth  ent_text_emb.pt

MoMoK chooses its feature reshape by a substring of the dataset name (models/RelMoE.py). Our tables are 768-d image
+ 768-d text, which only the authors' "FB15K-237" branch takes (12 x 64 -> AdaptiveAvgPool2d (4, 64) -> 256 for
both); the "MKG" branch, tested first, expects 384-d text. So <name> must contain "FB15K-237" and must not contain
"MKG". +D concat (1536-d) fits no branch (the reshape would silently give 2N rows) and is refused.

  python scripts/prepare_momok_data.py --data data/FB15k-237-IMG --momok <checkout> --visual rgb
  python scripts/prepare_momok_data.py --data data/Spatial-VRD --momok <checkout> --visual rgb_depth_mean
"""
import argparse
import os

import torch

VISUAL = {"rgb": "ent_rgb_emb.pt", "rgb_depth_mean": "ent_rgb_depth_mean_emb.pt"}
DIM = 768  # 12 x 64, the only size the FB15K-237 branch reshapes correctly


def read_openke(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        rows = [line.split() for line in f if line.strip()]
    if len(rows) != n:
        raise ValueError(f"{path}: header says {n} rows, read {len(rows)}")
    return rows


def default_name(dataset, visual):
    tag = "" if visual == "rgb" else "-Dmean"
    return f"FB15K-237-branch-{dataset.replace('MKG', 'mkg')}-v0{tag}"


def prepare(data, momok, visual, name=None):
    dataset = os.path.basename(os.path.normpath(data))
    name = name or default_name(dataset, visual)
    if "FB15K-237" not in name or "MKG" in name:
        raise ValueError(f"--name {name!r}: must contain 'FB15K-237' and not 'MKG' (see the module docstring)")
    dst = os.path.join(momok, "datasets", name)
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(momok, "checkpoint", name), exist_ok=True)   # train.py saves there with --save 1
    ents = read_openke(os.path.join(data, "entity2id.txt"))
    rels = read_openke(os.path.join(data, "relation2id.txt"))
    n = len(ents)
    ids = sorted(int(row[-1]) for row in ents)          # names may contain spaces; the id is the last field
    if ids != list(range(n)):
        raise ValueError("entity2id ids must be 0..N-1")
    with open(os.path.join(dst, "entity2id.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"e{i} {i}\n" for i in range(n))
    with open(os.path.join(dst, "relation2id.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"r{i} {i}\n" for i in range(len(rels)))
    counts = {}
    for split in ("train", "valid", "test"):
        rows = read_openke(os.path.join(data, f"{split}2id.txt"))   # OpenKE order: h t r
        with open(os.path.join(dst, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"e{h} r{r} e{t}\n" for h, t, r in rows)
        counts[split] = len(rows)
    for ours, theirs in ((VISUAL[visual], "img_features.pth"), ("ent_text_emb.pt", "text_features.pth")):
        x = torch.load(os.path.join(data, ours), map_location="cpu", weights_only=True)
        if x.shape != (n, DIM) or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError(f"{ours}: expected finite float32 [{n}, {DIM}], got {x.dtype} {tuple(x.shape)}")
        torch.save(x, os.path.join(dst, theirs))
    return name, n, len(rels), counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="our dataset folder, e.g. data/FB15k-237-IMG")
    ap.add_argument("--momok", required=True, help="MoMoK checkout (scripts/setup_baseline.py momok)")
    ap.add_argument("--visual", required=True, choices=sorted(VISUAL))
    ap.add_argument("--name", default=None, help="MoMoK --dataset name (default FB15K-237-branch-<dataset>-v0[-Dmean])")
    a = ap.parse_args(argv)
    name, n, r, counts = prepare(a.data, a.momok, a.visual, a.name)
    print(f"{name}: {n} entities, {r} relations, {counts} -> {os.path.join(a.momok, 'datasets', name)}")


if __name__ == "__main__":
    main()
