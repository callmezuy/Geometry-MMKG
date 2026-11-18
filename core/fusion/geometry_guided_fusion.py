"""
Geometry-guided fusion of GeometryMMKG (RESEARCH_PLAN.md sections 3.1, 3.2 item 4; P2.1, P2.3, P2.6, P2.8).

Inputs are frozen per-entity features: text t, RGB x, depth z (all [B, d]) and an image mask m [B]
(m = 0: the entity has no usable image, so RGB and depth are both missing).

0. (option input_norm) x <- LN_x(x), z <- LN_z(z): separate learnable LayerNorms, so both tables enter the
   fusion on the same scale.
1. RGB-depth fusion -> vision v (`fusion_mode`, or v = x when use_depth=False):
     gate   : G = sigmoid(W_g [x; z] + b_r),  v = G * x + (1 - G) * z  (per-dimension, input-dependent;
              b_r is a per-relation gate bias with option rel_gate, P2.2, zero at init)
     concat : v = W_c [x; z]                                          (Linear 2d -> d)
     mean   : v = (x + z) / 2
     fixed  : a = sigmoid(alpha) (one learned scalar),  v = a * x + (1 - a) * z
   (option depth_dropout p) during training, each entity independently uses v = x with probability p,
   i.e. the depth input is dropped and the entity goes through the no-depth path.
2. Missing images (`missing_mode`):
     learned_mask : v <- v_miss (one learned vector) where m = 0. RGB and depth come from the same image and
                    are missing together, so the replacement happens once, at the vision level: an entity
                    without an image goes through exactly the same computation in every variant.
     constant     : the mask is ignored and the (constant-filled) feature rows are fused as they are,
                    like the baselines read them.
3. Text-vision gate: u = W_v v,  g = sigmoid(W_tg [t'; u]),  h = LN(t' + Dropout(g * u)) / sqrt(d),
   with t' = t, or t' = W_t t with option text_proj (W_t = identity at init).
   Text decides, per dimension, how much visual information enters the entity representation.
   (This replaces a cross-attention over a length-1 sequence, whose softmax is always 1.)
   LN has no learnable gain / bias and its output is divided by sqrt(d), so every entity has L2 norm 1
   (up to LN's eps), as in TransE (Bordes et al., 2013). With a learnable gain the model could shrink or
   inflate the whole space during training, which changes what the margin and the self-adversarial
   temperature mean (measured on Spatial-VRD: true-answer distance 435 at init, 88 after one epoch).

With rel_gate the entity representation depends on the relation of the query (like TransR's relation-specific
projections), so training and evaluation fuse the candidates once per relation.
"""
import math

import torch
import torch.nn as nn

FUSION_MODES = ("gate", "concat", "mean", "fixed")
MISSING_MODES = ("learned_mask", "constant")


class VisionFusion(nn.Module):
    """RGB-depth fusion (steps 0-1)."""

    def __init__(self, dim, use_depth=True, fusion_mode="gate", num_relations=None, rel_gate=False,
                 input_norm=False, depth_dropout=0.0):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}")
        if rel_gate and not (use_depth and fusion_mode == "gate"):
            raise ValueError("rel_gate needs use_depth=True and fusion_mode='gate'")
        if not 0.0 <= depth_dropout <= 1.0:
            raise ValueError(f"depth_dropout must be in [0, 1], got {depth_dropout}")
        self.use_depth = use_depth
        self.fusion_mode = fusion_mode
        self.depth_dropout = depth_dropout if use_depth else 0.0
        self.norm_rgb = nn.LayerNorm(dim) if input_norm else None
        self.norm_depth = nn.LayerNorm(dim) if input_norm and use_depth else None
        self.gate_rel = None
        if use_depth and fusion_mode == "gate":
            self.W_g = nn.Linear(2 * dim, dim)
            if rel_gate:
                if not num_relations:
                    raise ValueError("rel_gate needs num_relations")
                self.gate_rel = nn.Embedding(num_relations, dim)
                nn.init.zeros_(self.gate_rel.weight)
        elif use_depth and fusion_mode == "concat":
            self.W_c = nn.Linear(2 * dim, dim)
        elif use_depth and fusion_mode == "fixed":
            self.alpha = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5 at start

    @property
    def relation_conditioned(self):
        return self.gate_rel is not None

    def depth_share(self, rgb, depth, rel=None):
        """1 - G (gate) or 1 - a (fixed): the weight on depth, [B, d] or scalar; None for other modes."""
        if not self.use_depth or self.fusion_mode not in ("gate", "fixed"):
            return None
        rgb, depth = self._norm(rgb, depth)
        if self.fusion_mode == "fixed":
            return 1 - torch.sigmoid(self.alpha)
        return 1 - self._gate(rgb, depth, rel)

    def _norm(self, rgb, depth):
        if self.norm_rgb is not None:
            rgb = self.norm_rgb(rgb)
        if self.norm_depth is not None:
            depth = self.norm_depth(depth)
        return rgb, depth

    def _gate(self, rgb, depth, rel):
        logits = self.W_g(torch.cat([rgb, depth], dim=-1))
        if self.gate_rel is not None:
            if rel is None:
                raise ValueError("a relation-conditioned gate needs the relation ids")
            logits = logits + self.gate_rel(rel)
        return torch.sigmoid(logits)

    def forward(self, rgb, depth, rel=None):
        rgb, depth = self._norm(rgb, depth)
        if not self.use_depth:
            return rgb
        if self.fusion_mode == "gate":
            G = self._gate(rgb, depth, rel)
            v = G * rgb + (1 - G) * depth
        elif self.fusion_mode == "concat":
            v = self.W_c(torch.cat([rgb, depth], dim=-1))
        elif self.fusion_mode == "mean":
            v = (rgb + depth) / 2
        else:  # fixed
            a = torch.sigmoid(self.alpha)
            v = a * rgb + (1 - a) * depth
        if self.training and self.depth_dropout > 0:
            drop = torch.rand(v.shape[0], device=v.device) < self.depth_dropout
            v = torch.where(drop.unsqueeze(-1), rgb, v)
        return v


class TextVisionGate(nn.Module):
    """Step 3: h = LN(t' + Dropout(g * u)) / sqrt(d), u = W_v v, g = sigmoid(W_tg [t'; u]); ||h||_2 = 1."""

    def __init__(self, dim, dropout=0.1, text_proj=False):
        super().__init__()
        self.W_t = None
        if text_proj:
            self.W_t = nn.Linear(dim, dim)
            with torch.no_grad():
                self.W_t.weight.copy_(torch.eye(dim))
                self.W_t.bias.zero_()
        self.W_v = nn.Linear(dim, dim)
        self.W_tg = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, text, vision):
        if self.W_t is not None:
            text = self.W_t(text)
        u = self.W_v(vision)
        g = torch.sigmoid(self.W_tg(torch.cat([text, u], dim=-1)))
        return self.layer_norm(text + self.dropout(g * u)) * self.scale


class GeometryGuidedFusion(nn.Module):
    def __init__(self, dim=768, use_depth=True, fusion_mode="gate", missing_mode="learned_mask", dropout=0.1,
                 num_relations=None, rel_gate=False, input_norm=False, depth_dropout=0.0, text_proj=False):
        super().__init__()
        if missing_mode not in MISSING_MODES:
            raise ValueError(f"missing_mode must be one of {MISSING_MODES}, got {missing_mode!r}")
        self.missing_mode = missing_mode
        self.vision_fusion = VisionFusion(dim, use_depth, fusion_mode, num_relations, rel_gate, input_norm,
                                          depth_dropout)
        if missing_mode == "learned_mask":
            self.missing_vision = nn.Parameter(torch.randn(dim) * 0.02)
        else:
            self.missing_vision = None
        self.text_vision = TextVisionGate(dim, dropout, text_proj)

    def forward(self, text, rgb, depth, mask=None, rel=None):
        """text, rgb, depth: [B, d]; mask: bool [B] (True = has an image); rel: long [B] (rel_gate only)."""
        v = self.vision_fusion(rgb, depth, rel)
        if self.missing_mode == "learned_mask":
            if mask is None:
                raise ValueError("missing_mode='learned_mask' needs the image mask (ent_img_mask.pt)")
            v = torch.where(mask.to(v.device).unsqueeze(-1), v, self.missing_vision.expand_as(v))
        return self.text_vision(text, v)
