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
    # P2.10 (20/11/2026): the constant-fill ablation keeps the fusion chosen for `full` on the other datasets (fixed).
    "constant_fixed": dict(use_depth=True, fusion_mode="fixed", missing_mode="constant"),
    # P2.10 (24/11/2026): 2 x 2 split of the depth input on the scene graphs (depth map x depth statistics);
    # `fixed` has both, `no_depth` neither.
    "depth_map_only": dict(use_depth=True, fusion_mode="fixed", missing_mode="learned_mask"),
    "depth_stats_only": dict(use_depth=False, missing_mode="learned_mask"),
}

# Variants that decide the depth statistics themselves; every other variant uses them exactly when it uses the
# depth map. (Not a model keyword: the statistics enter as the depth_stats_dim option.)
VARIANT_DEPTH_STATS = {"depth_map_only": False, "depth_stats_only": True}

# Development options (P2.8), all off by default.
OPTIONS = ("rel_gate", "input_norm", "depth_dropout", "text_proj", "depth_stats_dim")


def effective_options(variant_kwargs, rel_gate=False, input_norm=False, depth_dropout=0.0, text_proj=False,
                      depth_stats_dim=0, depth_stats=None):
    """Options that apply to a variant: rel_gate only for the gate; depth_stats with the depth map unless the
    variant says otherwise (`depth_stats`, from VARIANT_DEPTH_STATS); depth_dropout whenever some depth input
    (map or statistics) is used. Lets one set of requested options be passed to every variant of a comparison."""
    uses_depth = variant_kwargs.get("use_depth", True)
    uses_stats = bool(depth_stats_dim) and (uses_depth if depth_stats is None else depth_stats)
    is_gate = uses_depth and variant_kwargs.get("fusion_mode", "gate") == "gate"
    return dict(rel_gate=bool(rel_gate and is_gate), input_norm=bool(input_norm),
                depth_dropout=float(depth_dropout) if uses_depth or uses_stats else 0.0, text_proj=bool(text_proj),
                depth_stats_dim=int(depth_stats_dim) if uses_stats else 0)


class GeometryMMKGModel(nn.Module):
    def __init__(self, num_relations, embed_dim=768, use_depth=True, fusion_mode="gate",
                 missing_mode="learned_mask", dropout=0.1, rel_gate=False, input_norm=False, depth_dropout=0.0,
                 text_proj=False, depth_stats_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_module = GeometryGuidedFusion(embed_dim, use_depth, fusion_mode, missing_mode, dropout,
                                                  num_relations, rel_gate, input_norm, depth_dropout, text_proj,
                                                  depth_stats_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    @property
    def vision_fusion(self):
        return self.fusion_module.vision_fusion

    @property
    def missing_vision(self):
        return self.fusion_module.missing_vision

    @property
    def uses_depth_stats(self):
        return self.fusion_module.depth_stats_proj is not None

    @property
    def relation_conditioned(self):
        """True when entity representations depend on the query relation (rel_gate)."""
        return self.vision_fusion.relation_conditioned

    def forward_fusion(self, h_text, h_rgb, h_depth, mask=None, rel=None, geo=None):
        """Entity representations [B, d] from features [B, d], the image mask [B] (bool), for a
        relation-conditioned model the relation ids [B], and for depth_stats the depth statistics [B, k]."""
        return self.fusion_module(h_text, h_rgb, h_depth, mask, rel, geo)

    def score(self, h, r_ids, t):
        """TransE L1: -||h + r - t||_1 (higher = more plausible)."""
        return -(h + self.relation_embeddings(r_ids) - t).abs().sum(dim=-1)
