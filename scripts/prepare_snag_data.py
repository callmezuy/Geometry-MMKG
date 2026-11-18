"""
Prepare one of our datasets for the original SNAG MKGC code (RESEARCH_PLAN.md P3.0d), at the data level only:
the model code is the authors' (baselines/SNAG/SNAG_MKGC, see baselines/PROVENANCE.md) and is not touched.

SNAG reads, for -dataset=<name>:
  benchmarks/<name>/{entity2id,relation2id,train2id,valid2id,test2id}.txt   OpenKE files (copied from --data)
  benchmarks/<name>/type_constrain.txt   read by TestDataLoader (type_constrain=True by default; the C++ reader
                                         crashes when it is missing). Written with the logic of the authors'
                                         benchmarks/*/n-n.py. Evaluation itself runs with type_constrain=False.
  embeddings/<name>-visual.pth           image slot: ent_rgb_emb.pt (Control), ent_rgb_depth_mean_emb.pt (+D mean)
                                         or ent_rgb_depth_concat_emb.pt (+D concat, 1536-d); AdvMixRotatE takes
                                         the feature size from the tensor (no pooling when the name contains "MKG",
                                         otherwise run with -use_pool 0)
  embeddings/<name>-textual.pth          ent_text_emb.pt
  embeddings/missing.json                only used when <name> contains "MKG": AdvMixRotatE then replaces the
                                         listed rows by mean + std * randn of the other rows. Keys
                                         "<name>-visual.pth" / "<name>-textual.pth" -> entity ids, the same format
                                         as the authors' file. The authors' keys are kept, so the author features
                                         (paper reproduction) still work from the same folder.
                                         visual  = entities without an encoded image (~ent_img_mask.pt); depth comes
                                                   from the same images, so the list is the same for +D tables.
                                         textual = entities without a description (entity2textlong.txt is only the
                                                   name from entity2text.txt). On MKG-Y this is exactly the authors'
                                                   MKG-Y-textual.pth list (2,684 ids; same entity2id).

Usage (from Code/):
  python scripts/prepare_snag_data.py --data data/MKG-Y --snag baselines/SNAG/SNAG_MKGC --name MKG-Y-v0 --visual rgb
  python scripts/prepare_snag_data.py --data data/MKG-Y --snag baselines/SNAG/SNAG_MKGC --name MKG-Y-v0-Dmean \
      --visual rgb_depth_mean
  python scripts/prepare_snag_data.py --data data/FB15k-237-IMG --snag baselines/SNAG/SNAG_MKGC --name FB15k-237-IMG \
      --visual rgb
"""
import argparse
import json
import os
import shutil

import torch

OPENKE_FILES = ["entity2id.txt", "relation2id.txt", "train2id.txt", "valid2id.txt", "test2id.txt"]
VISUAL = {"rgb": "ent_rgb_emb.pt", "rgb_depth_mean": "ent_rgb_depth_mean_emb.pt",
          "rgb_depth_concat": "ent_rgb_depth_concat_emb.pt"}
AUTHOR_DATASETS = {"DB15K", "MKG-W", "MKG-Y", "FBIMG"}


def read_triples(path):
    with open(path) as f:
        n = int(f.readline())
        rows = [line.split() for line in f if line.strip()]
    assert len(rows) == n, f"{path}: header says {n} triples, found {len(rows)}"
    return rows


def type_constrain(bench):
    """benchmarks/*/n-n.py of the authors: for every relation, the heads and tails seen in train, valid, test
    (first-seen order), written as 'r<TAB>n<TAB>e1...' (heads line, then tails line)."""
    heads, tails = {}, {}
    for fn in ("train2id.txt", "valid2id.txt", "test2id.txt"):
        for h, t, r in read_triples(os.path.join(bench, fn)):
            heads.setdefault(r, {})[h] = 1
            tails.setdefault(r, {})[t] = 1
    with open(os.path.join(bench, "relation2id.txt")) as f:
        n_rel = int(f.readline())
    # the C++ reader loops over all relationTotal relations; a relation without triples would misalign it
    assert len(heads) == n_rel, f"{len(heads)} relations have triples, relation2id.txt has {n_rel}"
    lines = ["%d" % len(heads)]
    for r in heads:
        lines.append("\t".join([r, str(len(heads[r]))] + list(heads[r])))
        lines.append("\t".join([r, str(len(tails[r]))] + list(tails[r])))
    with open(os.path.join(bench, "type_constrain.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return n_rel


def read_map(path):
    with open(path, encoding="utf-8") as f:
        return dict(line.rstrip("\n").split("\t", 1) for line in f if line.strip())


def entity_ids(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        ids = {}
        for line in f:
            if line.strip():
                name, i = line.rstrip("\n").rsplit("\t", 1)
                ids[name] = int(i)
    assert len(ids) == n and sorted(ids.values()) == list(range(n)), f"{path}: ids are not 0..{n - 1}"
    return ids


def missing_lists(data, n_ent):
    mask = torch.load(os.path.join(data, "ent_img_mask.pt"), weights_only=True)
    assert mask.dtype == torch.bool and len(mask) == n_ent
    visual = torch.nonzero(~mask).flatten().tolist()
    textual = []
    long_path = os.path.join(data, "entity2textlong.txt")
    if os.path.exists(long_path):
        ids = entity_ids(os.path.join(data, "entity2id.txt"))
        name, desc = read_map(os.path.join(data, "entity2text.txt")), read_map(long_path)
        assert set(desc) == set(ids) == set(name), "entity2text / entity2textlong / entity2id cover different entities"
        textual = sorted(ids[e] for e in ids if desc[e] == name[e])
    return visual, textual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="our dataset folder, e.g. data/MKG-Y")
    ap.add_argument("--snag", required=True, help="SNAG_MKGC folder (benchmarks/ and embeddings/ are written here)")
    ap.add_argument("--name", required=True, help="-dataset name for run_mmkgc.py")
    ap.add_argument("--visual", required=True, choices=sorted(VISUAL))
    a = ap.parse_args()
    assert a.name not in AUTHOR_DATASETS, f"{a.name} is an author dataset; pick another name"
    assert not any(c.isspace() or c == "/" for c in a.name), "the name becomes a folder and a file prefix"

    bench = os.path.join(a.snag, "benchmarks", a.name)
    emb = os.path.join(a.snag, "embeddings")
    os.makedirs(bench, exist_ok=True)
    os.makedirs(emb, exist_ok=True)
    for fn in OPENKE_FILES:
        shutil.copyfile(os.path.join(a.data, fn), os.path.join(bench, fn))
    n_rel = type_constrain(bench)

    n_ent = len(entity_ids(os.path.join(a.data, "entity2id.txt")))
    shapes = {}
    for slot, src in (("visual", VISUAL[a.visual]), ("textual", "ent_text_emb.pt")):
        x = torch.load(os.path.join(a.data, src), weights_only=True)
        assert x.dim() == 2 and x.shape[0] == n_ent and x.dtype == torch.float32, (src, x.shape, x.dtype)
        assert torch.isfinite(x).all(), f"{src} has non-finite values"
        torch.save(x.contiguous().clone(), os.path.join(emb, f"{a.name}-{slot}.pth"))
        shapes[slot] = list(x.shape)

    report = {"name": a.name, "data": a.data, "visual_table": VISUAL[a.visual], "entities": n_ent,
              "relations": n_rel, "shapes": shapes}
    if "MKG" in a.name:
        visual, textual = missing_lists(a.data, n_ent)
        mj_path = os.path.join(emb, "missing.json")
        mj = {}
        if os.path.exists(mj_path):
            with open(mj_path) as f:
                mj = json.load(f)
        mj[f"{a.name}-visual.pth"] = visual
        mj[f"{a.name}-textual.pth"] = textual
        with open(mj_path, "w") as f:
            json.dump(mj, f)
        report["missing"] = {"visual": len(visual), "textual": len(textual), "keys_in_file": sorted(mj)}
    else:
        report["note"] = "name has no 'MKG': no missing.json entry; run run_mmkgc.py with -use_pool 0"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
