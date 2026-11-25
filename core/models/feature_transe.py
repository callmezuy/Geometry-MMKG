"""Feature-TransE, the project's simple feature baseline (RESEARCH_PLAN.md P3.1).

Frozen Text/RGB[/Depth] -> concatenate -> two-layer MLP -> unit-L2 entity vector.
Only the MLP and relation embeddings are learned; there are no entity-ID parameters,
gates, attention, or learned missing-image vectors. This is a project baseline,
not a reproduction of a separately published architecture.

Provenance: written by a collaborator; the six FB15k-237-IMG v0 runs of 19/11/2026 used the version with sha256
2b200bc2b5c713f706799c76cd3cdeec49b65a1d3bf0ffbba03c4085ab7d4b0a. Changed since:
- 25/11/2026: forward_fusion accepts `geo` (the depth-statistics argument that core/models/kg_reasoning.in_batch_loss
  passes to every model since P2.9).
- 25/11/2026 (decision): on the scene graphs the depth variant also reads the per-region depth statistics
  (depth_stats_dim = 9, `ent_depth_stats.pt`, standardised on the train entities with an image; rows without an image
  are 0, i.e. one constant like the other missing-image inputs), concatenated to the input: the same depth
  information as GeometryMMKG. With depth_stats_dim = 0 (knowledge graphs, and every run of 19/11) the model is the
  original one, parameter for parameter.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureTransE(nn.Module):
    relation_conditioned = False

    def __init__(self, num_relations, feature_dim=768, embed_dim=768, hidden_dim=768,
                 use_depth=False, dropout=0.1, depth_stats_dim=0):
        super().__init__()
        if min(num_relations, feature_dim, embed_dim, hidden_dim) < 1:
            raise ValueError("Relation count and all dimensions must be positive")
        if depth_stats_dim < 0 or (depth_stats_dim and not use_depth):
            raise ValueError("depth statistics are depth information: only the depth variant takes them")
        self.embed_dim = embed_dim
        self.use_depth = use_depth
        self.depth_stats_dim = depth_stats_dim
        self.uses_depth_stats = depth_stats_dim > 0
        self.encoder = nn.Sequential(
            nn.Linear((3 if use_depth else 2) * feature_dim + depth_stats_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim),
        )
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward_fusion(self, text, rgb, depth, mask=None, geo=None):
        # Read the common constant-filled features as they are. The mask is only
        # used by the loader to audit the data, never as a model input.
        parts = [text, rgb, depth] if self.use_depth else [text, rgb]
        if self.uses_depth_stats:
            if geo is None or geo.shape[-1] != self.depth_stats_dim:
                raise ValueError(f"this model reads {self.depth_stats_dim} depth statistics per entity (geo=)")
            parts.append(geo.to(text.dtype))
        return F.normalize(self.encoder(torch.cat(parts, dim=-1)), p=2, dim=-1)

    def score(self, head, relation_ids, tail):
        return -(head + self.relation_embeddings(relation_ids) - tail).abs().sum(-1)
