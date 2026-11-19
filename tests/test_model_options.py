"""
Development options of GeometryMMKG (RESEARCH_PLAN.md P2.8, P2.9): relation-conditioned gate, input normalisation,
depth modality dropout, text projection, depth statistics.
Run from Code/:  python tests/test_model_options.py      (pytest also works)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_engineering.fast_dataset import EntityFeatures  # noqa: E402
from core.data_engineering.fast_dataset import load_depth_stats  # noqa: E402
from core.models.end_to_end import FUSION_MODES, VARIANTS, GeometryMMKGModel, effective_options  # noqa: E402
from core.models.evaluation import evaluate_link_prediction  # noqa: E402
from core.models.kg_reasoning import TripleIndex, _side_loss, in_batch_loss, negative_masks  # noqa: E402

D, N, R = 16, 10, 3
ALL = dict(rel_gate=True, input_norm=True, depth_dropout=0.3, text_proj=True)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    text, rgb, depth = (torch.randn(N, D, generator=g) for _ in range(3))
    mask = torch.ones(N, dtype=torch.bool)
    mask[[1, 4, 7]] = False
    return text, rgb, depth, mask


def _model(seed=0, dropout=0.0, **kw):
    torch.manual_seed(seed)
    return GeometryMMKGModel(num_relations=R, embed_dim=D, dropout=dropout, **kw).eval()


def _randomize(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def test_text_proj_and_rel_gate_start_as_the_plain_model():
    # W_t = I and zero relation gate biases at init: the options change nothing until trained
    text, rgb, depth, mask = _inputs()
    plain = _model()
    opt = _model(rel_gate=True, text_proj=True)
    opt.load_state_dict(plain.state_dict(), strict=False)
    rel = torch.full((N,), 2, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(plain.forward_fusion(text, rgb, depth, mask),
                              opt.forward_fusion(text, rgb, depth, mask, rel), atol=1e-6)


def test_rel_gate_depends_on_relation_except_for_missing_images():
    text, rgb, depth, mask = _inputs()
    m = _model(rel_gate=True)
    _randomize(m)
    with torch.no_grad():
        a = m.forward_fusion(text, rgb, depth, mask, torch.zeros(N, dtype=torch.long))
        b = m.forward_fusion(text, rgb, depth, mask, torch.ones(N, dtype=torch.long))
    assert not torch.allclose(a[mask], b[mask])
    assert torch.allclose(a[~mask], b[~mask], atol=1e-6)
    assert m.relation_conditioned
    try:
        m.forward_fusion(text, rgb, depth, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("a relation-conditioned model needs the relation")


def test_rel_gate_only_for_gate_mode():
    for mode in FUSION_MODES:
        if mode == "gate":
            continue
        try:
            _model(fusion_mode=mode, rel_gate=True)
        except ValueError:
            continue
        raise AssertionError(mode)
    assert not _model(use_depth=False).relation_conditioned


def test_input_norm_makes_fusion_scale_invariant():
    text, rgb, depth, mask = _inputs()
    m = _model(input_norm=True)
    _randomize(m)
    with torch.no_grad():
        a = m.forward_fusion(text, rgb, depth, mask)
        b = m.forward_fusion(text, rgb * 10.0, depth * 0.1, mask)
    assert torch.allclose(a, b, atol=1e-4)


def test_depth_dropout_only_in_training_and_p1_equals_no_depth():
    text, rgb, depth, mask = _inputs()
    full = _model(depth_dropout=1.0)
    nod = _model(use_depth=False)
    nod.load_state_dict(full.state_dict(), strict=False)
    with torch.no_grad():
        full.train()
        a = full.forward_fusion(text, rgb, depth, mask)
        b = nod.forward_fusion(text, rgb, depth, mask)
        assert torch.allclose(a, b, atol=1e-6)              # p = 1: depth always dropped
        full.eval()
        c = full.forward_fusion(text, rgb, depth, mask)
        assert not torch.allclose(c[mask], b[mask])       # no dropout at evaluation


def test_missing_rows_same_output_with_all_options():
    text, rgb, depth, mask = _inputs()
    ref = _model(use_depth=False, input_norm=True, text_proj=True)
    _randomize(ref)
    m = _model(**ALL)
    _randomize(m, seed=5)
    m.load_state_dict(ref.state_dict(), strict=False)
    rel = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    with torch.no_grad():
        a = m.forward_fusion(text, rgb, depth, mask, rel)
        b = ref.forward_fusion(text, rgb, depth, mask)
    assert torch.allclose(a[~mask], b[~mask], atol=1e-6)


def test_relation_conditioned_loss_matches_per_query_computation():
    torch.manual_seed(0)
    n, dim = 12, 8
    feats = EntityFeatures(torch.randn(n, dim), torch.randn(n, dim), torch.randn(n, dim),
                           torch.tensor([1] * 10 + [0] * 2, dtype=torch.bool), torch.device("cpu"))
    m = GeometryMMKGModel(R, embed_dim=dim, dropout=0.0, rel_gate=True)
    _randomize(m)
    train = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 0, 5], [6, 2, 7], [8, 1, 9], [10, 0, 11], [0, 1, 3]])
    idx = TripleIndex(train, n, R)
    h, r, t = train[:, 0], train[:, 1], train[:, 2]
    loss, _ = in_batch_loss(m, feats, h, r, t, idx, margin=1.0, adv_temp=1.0)

    cand = torch.unique(torch.cat([h, t]))
    tail_ok, head_ok = negative_masks(h, r, t, cand, idx)
    lt, lh = [], []
    for i in range(len(h)):
        e = m.forward_fusion(*feats.gather(cand), r[i].expand(len(cand)))
        hc, tc = (cand == h[i]).nonzero()[0], (cand == t[i]).nonzero()[0]
        rel = m.relation_embeddings(r[i:i + 1])
        dt = torch.cdist(e[hc] + rel, e, p=1)
        dh = torch.cdist(e[tc] - rel, e, p=1)
        lt.append(_side_loss(dt, tc, tail_ok[i:i + 1], 1.0, 1.0)[0])
        lh.append(_side_loss(dh, hc, head_ok[i:i + 1], 1.0, 1.0)[0])
    expected = (torch.cat(lt).mean() + torch.cat(lh).mean()) / 2
    assert torch.allclose(loss, expected, atol=1e-5), (float(loss), float(expected))


K = 9  # depth statistics per entity (grid9)


def _geo(seed=3):
    return torch.randn(N, K, generator=torch.Generator().manual_seed(seed))


def test_depth_stats_zero_at_init_and_ignored_without_image():
    text, rgb, depth, mask = _inputs()
    plain = _model(input_norm=True, text_proj=True)
    opt = _model(input_norm=True, text_proj=True, depth_stats_dim=K)
    opt.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(plain.forward_fusion(text, rgb, depth, mask),
                              opt.forward_fusion(text, rgb, depth, mask, geo=_geo()), atol=1e-6)
        _randomize(opt)
        a = opt.forward_fusion(text, rgb, depth, mask, geo=_geo(3))
        b = opt.forward_fusion(text, rgb, depth, mask, geo=_geo(4))
    assert not torch.allclose(a[mask], b[mask])                  # used for entities with an image
    assert torch.allclose(a[~mask], b[~mask], atol=1e-6)         # never for entities without one
    try:
        opt.forward_fusion(text, rgb, depth, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("a model with depth_stats needs geo")


def test_depth_stats_is_linear_in_the_depth_difference():
    # h_a - h_b changes by exactly W_s (s_a - s_b): the TransE translation sees the depth difference
    text, rgb, depth, mask = _inputs()
    m = _model(depth_stats_dim=K)
    _randomize(m)
    g = _geo()
    with torch.no_grad():
        h0 = m.forward_fusion(text, rgb, depth, mask, geo=torch.zeros_like(g))
        h1 = m.forward_fusion(text, rgb, depth, mask, geo=g)
        W = m.fusion_module.depth_stats_proj.weight
    assert torch.allclose((h1 - h0)[mask], g[mask] @ W.T, atol=1e-5)


def test_depth_stats_only_with_depth_and_dropped_with_depth():
    assert effective_options(VARIANTS["no_depth"], depth_stats_dim=K)["depth_stats_dim"] == 0
    assert effective_options(VARIANTS["full"], depth_stats_dim=K)["depth_stats_dim"] == K
    try:
        _model(use_depth=False, depth_stats_dim=K)
    except ValueError:
        pass
    else:
        raise AssertionError("depth_stats without depth must fail")
    text, rgb, depth, mask = _inputs()
    full = _model(depth_dropout=1.0, depth_stats_dim=K)
    _randomize(full)
    nod = _model(use_depth=False)
    nod.load_state_dict(full.state_dict(), strict=False)
    full.train()
    with torch.no_grad():
        a = full.forward_fusion(text, rgb, depth, mask, geo=_geo())
        b = nod.forward_fusion(text, rgb, depth, mask)
    assert torch.allclose(a, b, atol=1e-6)                       # p = 1: depth and its statistics dropped


def test_depth_stats_in_loss_and_evaluation():
    torch.manual_seed(0)
    n, dim = 12, 8
    mask = torch.tensor([1] * 10 + [0] * 2, dtype=torch.bool)
    geo = torch.randn(n, K)
    feats = EntityFeatures(torch.randn(n, dim), torch.randn(n, dim), torch.randn(n, dim), mask,
                           torch.device("cpu"), geo=geo)
    m = GeometryMMKGModel(R, embed_dim=dim, dropout=0.0, depth_stats_dim=K)
    _randomize(m)
    train = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 0, 5], [6, 2, 7], [8, 1, 9], [10, 0, 11]])
    idx = TripleIndex(train, n, R)
    loss, _ = in_batch_loss(m, feats, train[:, 0], train[:, 1], train[:, 2], idx, margin=1.0, adv_temp=1.0)
    loss.backward()
    assert m.fusion_module.depth_stats_proj.weight.grad.abs().sum() > 0
    feats.geo = torch.randn(n, K)
    loss2, _ = in_batch_loss(m, feats, train[:, 0], train[:, 1], train[:, 2], idx, margin=1.0, adv_temp=1.0)
    assert not torch.allclose(loss, loss2)
    # evaluation passes the same statistics: ranks equal a brute force on forward_fusion(..., geo=...)
    true_tails = {}
    for h, r, t in train.tolist():
        true_tails.setdefault((h, r), set()).add(t)
    loader = [{"h_id": train[:, 0], "r_id": train[:, 1], "t_id": train[:, 2]}]
    res = evaluate_link_prediction(m, feats.text, feats.rgb, feats.depth, loader, true_tails, torch.device("cpu"),
                                   return_ranks=True, desc="t", all_mask=mask, all_geo=feats.geo)
    with torch.no_grad():
        E = m.forward_fusion(feats.text, feats.rgb, feats.depth, mask, geo=feats.geo)
        Rw = m.relation_embeddings.weight
    for k, (h, r, t) in enumerate(train.tolist()):
        s = -(E[h] + Rw[r] - E).abs().sum(1)
        ok = torch.ones(n, dtype=torch.bool)
        ok[list(true_tails[(h, r)] - {t})] = False
        assert res["ranks"]["rank_tail"][k] == int(((s > s[t]) & ok).sum()) + 1


def test_load_depth_stats_standardises_on_train_entities(tmp_path=None):
    import tempfile
    n = 6
    x = torch.tensor([[1.0], [3.0], [100.0], [float("nan")], [5.0], [7.0]])
    mask = torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool)
    train = torch.tensor([[0, 0, 1], [3, 0, 4]])               # train entities 0, 1, 3, 4
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vrd_depth_stats_toy.pt")
        torch.save({"grid9": x, "model": "toy"}, p)
        z, info = load_depth_stats(p, n, mask, train)
    # reference = train entities with an image and finite stats: 0 and 1 (3 is NaN, 4 has no image)
    assert info["n_train_ref"] == 2
    mu, sd = 2.0, torch.tensor([1.0, 3.0]).std().item()
    assert abs(z[2, 0].item() - (100.0 - mu) / sd) < 1e-4    # non-train entity uses train statistics
    assert z[3, 0] == 0 and z[4, 0] == 0                      # no stats / no image -> 0


def test_load_depth_stats_reads_the_data_table():
    # ent_depth_stats.pt of scripts/depth_stats.py is a plain tensor (filled rows for entities without an image)
    import tempfile
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0], [9.0, 9.0], [0.5, 0.5]])
    mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    train = torch.tensor([[0, 0, 1], [1, 0, 3]])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ent_depth_stats.pt")
        torch.save(x, p)
        z, info = load_depth_stats(p, 4, mask, train)
    assert info["dim"] == 2 and info["key"] is None and info["n_train_ref"] == 2
    assert torch.allclose(z[0], -z[1]) and z[3].abs().sum() == 0  # standardised on entities 0, 1; filled row -> 0


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all model-option tests passed")
