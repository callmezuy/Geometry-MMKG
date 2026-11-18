"""
Geometry-guided fusion of GeometryMMKG (RESEARCH_PLAN.md sections 3.1, 3.2 item 4; P2.1, P2.3, P2.6).

Inputs are frozen per-entity features: text t, RGB x, depth z (all [B, d]) and an image mask m [B]
(m = 0: the entity has no usable image, so RGB and depth are both missing).

1. RGB-depth fusion -> vision v (`fusion_mode`, or v = x when use_depth=False):
     gate   : G = sigmoid(W_g [x; z]),  v = G * x + (1 - G) * z      (per-dimension, input-dependent)
     concat : v = W_c [x; z]                                          (Linear 2d -> d)
     mean   : v = (x + z) / 2
     fixed  : a = sigmoid(alpha) (one learned scalar),  v = a * x + (1 - a) * z
2. Missing images (`missing_mode`):
     learned_mask : v <- v_miss (one learned vector) where m = 0. RGB and depth come from the same image and
                    are missing together, so the replacement happens once, at the vision level: an entity
                    without an image goes through exactly the same computation in every variant.
     constant     : the mask is ignored and the (constant-filled) feature rows are fused as they are,
                    like the baselines read them.
3. Text-vision gate: u = W_v v,  g = sigmoid(W_tg [t; u]),  h = LN(t + Dropout(g * u)) / sqrt(d).
   Text decides, per dimension, how much visual information enters the entity representation.
   (This replaces a cross-attention over a length-1 sequence, whose softmax is always 1.)
   LN has no learnable gain / bias and its output is divided by sqrt(d), so every entity has L2 norm 1
   (up to LN's eps), as in TransE (Bordes et al., 2013). With a learnable gain the model could shrink or
   inflate the whole space during training, which changes what the margin and the self-adversarial
   temperature mean (measured on Spatial-VRD: true-answer distance 435 at init, 88 after one epoch).
"""
import math

import torch
import torch.nn as nn

FUSION_MODES = ("gate", "concat", "mean", "fixed")
MISSING_MODES = ("learned_mask", "constant")


class VisionFusion(nn.Module):
    """RGB-depth fusion (step 1). Has no parameters for use_depth=False and fusion_mode='mean'."""

    def __init__(self, dim, use_depth=True, fusion_mode="gate"):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}")
        self.use_depth = use_depth
        self.fusion_mode = fusion_mode
        if use_depth and fusion_mode == "gate":
            self.W_g = nn.Linear(2 * dim, dim)
        elif use_depth and fusion_mode == "concat":
            self.W_c = nn.Linear(2 * dim, dim)
        elif use_depth and fusion_mode == "fixed":
            self.alpha = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5 at start

    def forward(self, rgb, depth, return_gate=False):
        gate = None
        if not self.use_depth:
            v = rgb
        elif self.fusion_mode == "gate":
            gate = torch.sigmoid(self.W_g(torch.cat([rgb, depth], dim=-1)))
            v = gate * rgb + (1 - gate) * depth
        elif self.fusion_mode == "concat":
            v = self.W_c(torch.cat([rgb, depth], dim=-1))
        elif self.fusion_mode == "mean":
            v = (rgb + depth) / 2
        else:  # fixed
            gate = torch.sigmoid(self.alpha)
            v = gate * rgb + (1 - gate) * depth
        return (v, gate) if return_gate else v


class TextVisionGate(nn.Module):
    """Step 3: h = LN(t + Dropout(g * u)) / sqrt(d), u = W_v v, g = sigmoid(W_tg [t; u]); ||h||_2 = 1."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.W_v = nn.Linear(dim, dim)
        self.W_tg = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, text, vision):
        u = self.W_v(vision)
        g = torch.sigmoid(self.W_tg(torch.cat([text, u], dim=-1)))
        return self.layer_norm(text + self.dropout(g * u)) * self.scale


class GeometryGuidedFusion(nn.Module):
    def __init__(self, dim=768, use_depth=True, fusion_mode="gate", missing_mode="learned_mask", dropout=0.1):
        super().__init__()
        if missing_mode not in MISSING_MODES:
            raise ValueError(f"missing_mode must be one of {MISSING_MODES}, got {missing_mode!r}")
        self.missing_mode = missing_mode
        self.vision_fusion = VisionFusion(dim, use_depth, fusion_mode)
        if missing_mode == "learned_mask":
            self.missing_vision = nn.Parameter(torch.randn(dim) * 0.02)
        else:
            self.missing_vision = None
        self.text_vision = TextVisionGate(dim, dropout)

    def forward(self, text, rgb, depth, mask=None):
        """text, rgb, depth: [B, d]; mask: bool [B] (True = has an image). Returns [B, d]."""
        v = self.vision_fusion(rgb, depth)
        if self.missing_mode == "learned_mask":
            if mask is None:
                raise ValueError("missing_mode='learned_mask' needs the image mask (ent_img_mask.pt)")
            v = torch.where(mask.to(v.device).unsqueeze(-1), v, self.missing_vision.expand_as(v))
        return self.text_vision(text, v)
