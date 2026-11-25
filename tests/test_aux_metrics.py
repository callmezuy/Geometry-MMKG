"""
E1 auxiliary metrics on the scene graphs (docs/superpowers/specs/2026-11-25-e1-mmsg-aux-metrics-design.md):
image ids of the entities, in-image ranking and relation prediction.
Run from Code/:  python tests/test_aux_metrics.py
"""
import csv
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_engineering.fast_dataset import entity_image_ids  # noqa: E402
from core.models.evaluation import evaluate_link_prediction, save_aux_ranks  # noqa: E402
from core.models.feature_transe import FeatureTransE  # noqa: E402

CPU = torch.device("cpu")


def _write_entities(root, name, names):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "entity2id.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(names)}\n" + "".join(f"{n}\t{i}\n" for i, n in enumerate(names)))
    return d


def test_entity_image_ids():
    with tempfile.TemporaryDirectory() as root:
        d = _write_entities(root, "Spatial-VRD", ["5073960523_0", "5073960523_1", "77_a_b_0", "5073960523_2"])
        assert entity_image_ids(d).tolist() == [0, 0, 1, 0]          # last '_' splits; "77_a_b" is one image
        assert entity_image_ids(d + os.sep).tolist() == [0, 0, 1, 0]  # trailing separator
        assert entity_image_ids(_write_entities(root, "FB15k-237-IMG", ["/m/0_x"])) is None
        bad = _write_entities(root, "Spatial-VRD-Trans", ["5073960523"])
        try:
            entity_image_ids(bad)
            raise AssertionError("expected ValueError for a name without '_'")
        except ValueError:
            pass


# ---------------------------------------------------------------------------------------------------------------
# Hand-built scene graph: the fused table is the text feature itself, so every L1 distance is a small integer.
#   image 0: e0 (0,0), e1 (1,0), e2 (1,2), e6 (3,0)   (e6 has no triple: still a candidate)
#   image 1: e3 (5,5), e4 (1,1), e5 (6,5)
#   relations r0 (1,0), r1 (0,2), r2 (0,1)
#   train (0,r0,1); test (0,r0,2), (0,r1,2), (3,r0,5), (4,r2,3)
# L1 distances to the query point (truth marked *):
#   (0,r0,2) tail q=(1,0): e0 1, e1 0 (filtered: train), e2 2*, e6 2 (tie, not better) | e3 9, e4 1, e5 10
#            -> in image: e0 better -> 2; global: e0, e4 -> 3
#            head q=e2-r0=(0,2): e0 2*, e1 3, e2 1, e6 5 | e3 8, e4 2 (tie), e5 9 -> 2 and 2
#   (0,r1,2) tail q=(0,2): e0 2, e1 3, e2 1*, e6 5 | e4 2 ... -> 1 and 1
#            head q=e2-r1=(1,0): e0 1*, e1 0, e2 2, e6 2 | e3 9, e4 1 (tie), e5 10 -> 2 and 2
#   (3,r0,5) tail q=(6,5): e3 1, e4 9, e5 0* -> 1 and 1;  head q=(5,5): e3 0*, e4 8, e5 1 -> 1 and 1
#   (4,r2,3) tail q=(1,2): e3 7*, e4 1, e5 8 | e0 3, e1 2, e2 0, e6 4 -> 2 and 6
#            head q=e3-r2=(5,4): e3 1, e4 7*, e5 2 | e0 9, e1 8, e2 6, e6 6 -> 3 and 5
# Relation prediction, distance |f(h) + r - f(t)|_1:
#   (0,?,2) d=(-1,-2): r0 2, r1 1, r2 2. Truth r0: r1 better but (0,r1,2) is true -> filtered -> 1; truth r1 -> 1
#   (3,?,5) d=(-1,0): r0 0*, r1 3, r2 2 -> 1
#   (4,?,3) d=(-4,-4): r0 7 (tie), r1 6 (better), r2 7* -> 2
# ---------------------------------------------------------------------------------------------------------------
E = torch.tensor([(0, 0), (1, 0), (1, 2), (5, 5), (1, 1), (6, 5), (3, 0)], dtype=torch.float32)
IMG = torch.tensor([0, 0, 0, 1, 1, 1, 0])
REL = torch.tensor([(1, 0), (0, 2), (0, 1)], dtype=torch.float32)
TRAIN, TEST = [(0, 0, 1)], [(0, 0, 2), (0, 1, 2), (3, 0, 5), (4, 2, 3)]


class IdentityModel(nn.Module):
    def __init__(self, rel):
        super().__init__()
        self.embed_dim = rel.shape[1]
        self.relation_embeddings = nn.Embedding.from_pretrained(rel, freeze=True)

    def forward_fusion(self, text, rgb, depth):
        return text


class RelConditionedModel(IdentityModel):
    relation_conditioned = True

    def forward_fusion(self, text, rgb, depth, mask=None, rel=None):
        return text


def _filter(triples):
    true = {}
    for h, r, t in triples:
        true.setdefault((h, r), set()).add(t)
    return true


def _loader(triples, bs):
    return [{"h_id": torch.tensor([x[0] for x in triples[i:i + bs]]),
             "r_id": torch.tensor([x[1] for x in triples[i:i + bs]]),
             "t_id": torch.tensor([x[2] for x in triples[i:i + bs]])} for i in range(0, len(triples), bs)]


def _hand_eval(model=None, img=IMG, **kw):
    model = model or IdentityModel(REL)
    return evaluate_link_prediction(model, E, E, E, _loader(TEST, 3), _filter(TRAIN + TEST), CPU,
                                    entity_chunk_size=4, return_ranks=True, desc="hand", entity_image=img, **kw)


def test_hand_built_ranks():
    res = _hand_eval(keep_on_gpu=False)
    ranks = res["ranks"]
    assert ranks["rank_tail"].tolist() == [3, 1, 1, 6]      # main metric unchanged
    assert ranks["rank_head"].tolist() == [2, 2, 1, 5]
    assert ranks["rank_tail_img"].tolist() == [2, 1, 1, 2]
    assert ranks["rank_head_img"].tolist() == [2, 2, 1, 3]
    assert ranks["rank_rel"].tolist() == [1, 1, 1, 2]
    ii, rel = res["in_image"], res["relation"]
    assert abs(ii["mrr"] - 2 / 3) < 1e-12 and ii["hits@1"] == 3 / 8 and ii["hits@3"] == 1.0 and ii["n"] == 8
    assert ii["tail"]["mrr"] == 0.75 and abs(ii["head"]["mrr"] - (0.5 + 0.5 + 1 + 1 / 3) / 4) < 1e-12
    assert ii["n_cand"] == 3.5 and ii["n_cand_filtered"] == 27 / 8
    assert abs(ii["image_first"] - 5 / 8) < 1e-12 and ii["n_negative_gap"] == 0
    H = lambda n: sum(1 / k for k in range(1, n + 1))  # noqa: E731
    # n_cand_filtered per query, head then tail: (0,r0,2) tail drops e1 (4 -> 3)
    assert abs(ii["mrr_random"] - np.mean([H(n) / n for n in [4, 4, 3, 3, 3, 4, 3, 3]])) < 1e-12
    assert abs(rel["mrr"] - 0.875) < 1e-12 and rel["hits@1"] == 0.75 and rel["hits@5"] == 1.0
    assert rel["mr"] == 1.25 and rel["n"] == 4
    json.dumps({k: v for k, v in res.items() if k != "ranks"})  # goes straight into metrics_*.json


def test_known_entity_variant():
    # The known entity (h for a tail query, t for a head query) is a candidate but never a true answer.
    # It beats the truth in: (0,r0,2) tail (e0 at 1 < 2) and head (e2 at 1 < 2); (4,r2,3) tail (e4 at 1 < 7) and
    # head (e3 at 1 < 7). Without it the in-image ranks are head [1, 2, 1, 2], tail [1, 1, 1, 1].
    ii = _hand_eval()["in_image"]
    assert ii["known_beats_truth"] == 0.5
    ex = ii["excl_known"]
    assert ex["mrr"] == 0.875 and ex["hits@1"] == 0.75 and ex["hits@3"] == 1.0 and ex["n"] == 8
    H = lambda n: sum(1 / k for k in range(1, n + 1))  # noqa: E731
    assert abs(ex["mrr_random"] - np.mean([H(n - 1) / (n - 1) for n in [4, 4, 3, 3, 3, 4, 3, 3]])) < 1e-12


def _random_scene_graph(seed, n_img=20, per_img=10, R=5, D=6, n_triples=600, integer=False):
    rng = np.random.default_rng(seed)
    N = n_img * per_img
    img = torch.tensor(rng.permutation(np.repeat(np.arange(n_img), per_img)))
    members = [np.flatnonzero(img.numpy() == i) for i in range(n_img)]
    if integer:
        text = torch.tensor(rng.integers(-5, 6, size=(N, D)), dtype=torch.float32)
        rel = torch.tensor(rng.integers(-3, 4, size=(R, D)), dtype=torch.float32)
    else:
        text = torch.tensor(rng.standard_normal((N, D)), dtype=torch.float32)
        rel = torch.tensor(rng.standard_normal((R, D)), dtype=torch.float32)
    triples = set()
    for _ in range(n_triples):
        h, t = rng.choice(members[rng.integers(n_img)], size=2, replace=False)
        triples.add((int(h), int(rng.integers(R)), int(t)))
    # shared (h, r), (r, t) and (h, t) so that every filter matters
    h0, t0 = members[0][:2]
    triples |= {(int(h0), r, int(t0)) for r in range(min(R, 3))}
    triples |= {(int(h0), 1, int(t)) for t in members[0][2:6]} | {(int(h), 2, int(t0)) for h in members[0][2:6]}
    triples = sorted(triples)
    return text, rel, img, triples


def test_single_image_matches_main_ranks():
    text, rel, _, triples = _random_scene_graph(0, integer=True)
    test = triples[::3]
    res = evaluate_link_prediction(IdentityModel(rel), text, text, text, _loader(test, 64), _filter(triples), CPU,
                                   entity_chunk_size=37, return_ranks=True, desc="one image",
                                   entity_image=torch.zeros(len(text), dtype=torch.long))
    r = res["ranks"]
    assert np.array_equal(r["rank_tail_img"], r["rank_tail"]), "tail: one image must equal the main ranks"
    assert np.array_equal(r["rank_head_img"], r["rank_head"]), "head: one image must equal the main ranks"
    assert res["in_image"]["image_first"] == 1.0 and res["in_image"]["n_cand"] == len(text)


def test_relation_ranks_match_brute_force():
    text, rel, img, triples = _random_scene_graph(1)
    test = triples[::2]
    true_rel = {}
    for h, r, t in triples:
        true_rel.setdefault((h, t), set()).add(r)
    assert max(len(v) for v in true_rel.values()) >= 3
    res = evaluate_link_prediction(IdentityModel(rel), text, text, text, _loader(test, 50), _filter(triples), CPU,
                                   entity_chunk_size=37, return_ranks=True, desc="relation", entity_image=img)
    exp = []
    for h, r, t in test:
        s = -(text[h][None] + rel - text[t][None]).abs().sum(1)   # one expression for candidates and truth
        ok = torch.ones(len(rel), dtype=torch.bool)
        ok[list(true_rel[(h, t)])] = False
        exp.append(int(((s > s[r]) & ok).sum()) + 1)
    assert res["ranks"]["rank_rel"].tolist() == exp


def test_main_metrics_unchanged():
    text, rel, img, triples = _random_scene_graph(2)
    test = triples[::3]
    args = (IdentityModel(rel), text, text, text, _loader(test, 64), _filter(triples), CPU)
    a = evaluate_link_prediction(*args, entity_chunk_size=37, return_ranks=True, desc="plain")
    b = evaluate_link_prediction(*args, entity_chunk_size=37, return_ranks=True, desc="aux", entity_image=img)
    assert "in_image" not in a and "relation" not in a and "rank_rel" not in a["ranks"]
    for k in ("rank_head", "rank_tail"):
        assert np.array_equal(a["ranks"][k], b["ranks"][k])
    assert a["mrr"] == b["mrr"] and a["tail"] == b["tail"] and a["head"] == b["head"]


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_cross_image_triple_raises():
    img = IMG.clone()
    img[2] = 1   # (0, r0, 2) now joins two images
    _raises(lambda: _hand_eval(img=img))


def test_relation_conditioned_raises():
    _raises(lambda: _hand_eval(model=RelConditionedModel(REL)))


def test_feature_transe_runs():
    torch.manual_seed(0)
    model = FeatureTransE(3, feature_dim=2, embed_dim=2, hidden_dim=4, use_depth=False)
    res = _hand_eval(model=model)
    assert res["in_image"]["n"] == 8 and res["relation"]["n"] == 4


def test_save_aux_ranks():
    ranks = _hand_eval()["ranks"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ranks_aux.csv")
        save_aux_ranks(ranks, path)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
    assert rows[0] == ["h_id", "r_id", "t_id", "rank_head_img", "rank_tail_img", "rank_rel"]
    assert rows[1:] == [["0", "0", "2", "2", "2", "1"], ["0", "1", "2", "2", "1", "1"],
                        ["3", "0", "5", "1", "1", "1"], ["4", "2", "3", "3", "2", "2"]]


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("OK")
