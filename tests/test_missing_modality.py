"""
Missing-image handling of GeometryMMKG (RESEARCH_PLAN.md P2.6, section 3.2 item 4).
Run from Code/:  python tests/test_missing_modality.py      (pytest also works)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models.end_to_end import FUSION_MODES, VARIANTS, GeometryMMKGModel  # noqa: E402

D, N, R = 16, 10, 3


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    text, rgb, depth = (torch.randn(N, D, generator=g) for _ in range(3))
    mask = torch.ones(N, dtype=torch.bool)
    mask[[1, 4, 7]] = False
    return text, rgb, depth, mask


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = GeometryMMKGModel(num_relations=R, embed_dim=D, **kw)
    return m.eval()


def test_missing_rows_same_output_with_and_without_depth():
    text, rgb, depth, mask = _inputs()
    ref = _model(use_depth=False)
    for mode in FUSION_MODES:
        m = _model(seed=1, use_depth=True, fusion_mode=mode)
        # share every parameter the no-depth model has (missing vector, text-vision gate, LN, relations)
        missing = m.load_state_dict(ref.state_dict(), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        with torch.no_grad():
            a = m.forward_fusion(text, rgb, depth, mask)
            b = ref.forward_fusion(text, rgb, depth, mask)
        assert torch.allclose(a[~mask], b[~mask], atol=1e-6), mode
        assert not torch.allclose(a[mask], b[mask]), mode  # depth matters where an image exists


def test_missing_rows_ignore_visual_values():
    text, rgb, depth, mask = _inputs()
    for kw in ({"use_depth": False},) + tuple({"fusion_mode": f} for f in FUSION_MODES):
        m = _model(**kw)
        rgb2, depth2 = rgb.clone(), depth.clone()
        rgb2[~mask] += 5.0
        depth2[~mask] -= 3.0
        with torch.no_grad():
            a = m.forward_fusion(text, rgb, depth, mask)
            b = m.forward_fusion(text, rgb2, depth2, mask)
        assert torch.allclose(a, b, atol=1e-6), kw


def test_constant_mode_ignores_mask():
    text, rgb, depth, mask = _inputs()
    m = _model(missing_mode="constant")
    rgb2 = rgb.clone()
    rgb2[~mask] += 5.0
    with torch.no_grad():
        a = m.forward_fusion(text, rgb, depth, mask)
        b = m.forward_fusion(text, rgb2, depth, mask)
        c = m.forward_fusion(text, rgb, depth, None)
    assert not torch.allclose(a[~mask], b[~mask])   # the filled table is read as is
    assert torch.allclose(a, c, atol=1e-6)            # mask has no effect


def test_missing_only_batch_gives_no_gradient_to_rgb_depth_fusion():
    text, rgb, depth, mask = _inputs()
    for mode in FUSION_MODES:
        m = _model(fusion_mode=mode).train()
        out = m.forward_fusion(text[~mask], rgb[~mask], depth[~mask], mask[~mask])
        # (sum of a LayerNorm output is constant, so project on a random direction instead)
        (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(0))).sum().backward()
        for name, p in m.vision_fusion.named_parameters():
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, (mode, name)
        assert m.missing_vision.grad is not None and torch.count_nonzero(m.missing_vision.grad) > 0


def test_learned_mask_requires_mask():
    text, rgb, depth, _ = _inputs()
    m = _model()
    try:
        m.forward_fusion(text, rgb, depth, None)
    except ValueError:
        return
    raise AssertionError("learned_mask without a mask must raise")


def test_entity_norm_is_fixed():
    # fixed scale as in TransE: every entity has L2 norm 1, whatever the parameters (no learnable LN gain)
    text, rgb, depth, mask = _inputs()
    for kw in VARIANTS.values():
        m = _model(**kw)
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(3.0)
            out = m.forward_fusion(text, rgb, depth, mask)
        assert torch.allclose(out.norm(dim=1), torch.ones(N), atol=1e-3), kw


def test_variants_build():
    for name, kw in VARIANTS.items():
        m = _model(**kw)
        text, rgb, depth, mask = _inputs()
        with torch.no_grad():
            out = m.forward_fusion(text, rgb, depth, mask)
        assert out.shape == (N, D) and torch.isfinite(out).all(), name
    assert set(VARIANTS) == {"full", "no_depth", "concat", "mean", "fixed", "constant", "constant_fixed",
                             "depth_map_only", "depth_stats_only"}
    # each constant-fill variant differs from a learned-mask variant only in the missing-image handling
    for const, masked in (("constant", "full"), ("constant_fixed", "fixed")):
        assert {**VARIANTS[const], "missing_mode": "learned_mask"} == VARIANTS[masked], const


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all missing-modality tests passed")
