"""
Development options of GeometryMMKG (RESEARCH_PLAN.md P2.8): relation-conditioned gate, input normalisation,
depth modality dropout, text projection.
Run from Code/:  python tests/test_model_options.py      (pytest also works)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_engineering.fast_dataset import EntityFeatures  # noqa: E402
from core.models.end_to_end import FUSION_MODES, GeometryMMKGModel  # noqa: E402
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


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all model-option tests passed")
