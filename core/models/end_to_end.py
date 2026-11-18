"""
GeometryMMKG: frozen per-entity features (text, RGB, depth) -> geometry-guided fusion -> TransE (L1).

There are no per-entity ID embeddings: an entity is fully described by its features, so the same model runs
in the transductive and the inductive setting (RESEARCH_PLAN.md section 2.2). Relations are learned embeddings.
"""
import torch.nn as nn

from core.fusion.geometry_guided_fusion import FUSION_MODES, MISSING_MODES, GeometryGuidedFusion  # noqa: F401

# Ablation variants of RESEARCH_PLAN.md section 3.1 (P2.1) -> model keyword arguments.
VARIANTS = {
    "full": dict(use_depth=True, fusion_mode="gate", missing_mode="learned_mask"),
    "no_depth": dict(use_depth=False, missing_mode="learned_mask"),
    "concat": dict(use_depth=True, fusion_mode="concat", missing_mode="learned_mask"),
    "mean": dict(use_depth=True, fusion_mode="mean", missing_mode="learned_mask"),
    "fixed": dict(use_depth=True, fusion_mode="fixed", missing_mode="learned_mask"),
    "constant": dict(use_depth=True, fusion_mode="gate", missing_mode="constant"),
}


class GeometryMMKGModel(nn.Module):
    def __init__(self, num_relations, embed_dim=768, use_depth=True, fusion_mode="gate",
                 missing_mode="learned_mask", dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_module = GeometryGuidedFusion(embed_dim, use_depth, fusion_mode, missing_mode, dropout)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    @property
    def vision_fusion(self):
        return self.fusion_module.vision_fusion

    @property
    def missing_vision(self):
        return self.fusion_module.missing_vision

    def forward_fusion(self, h_text, h_rgb, h_depth, mask=None):
        """Entity representations [B, d] from features [B, d] and the image mask [B] (bool)."""
        return self.fusion_module(h_text, h_rgb, h_depth, mask)

    def score(self, h, r_ids, t):
        """TransE L1: -||h + r - t||_1 (higher = more plausible)."""
        return -(h + self.relation_embeddings(r_ids) - t).abs().sum(dim=-1)
