import torch
import torch.nn as nn
from core.features.encoders import TextEncoder, RGBEncoder, DepthEncoder
from core.fusion.geometry_guided_fusion import GeometryGuidedFusion
from core.models.kg_reasoning import GraphConvolution, TransEScore

class GeometryMMKGModel(nn.Module):
    """
    End-to-end model combining independent feature encoders,
    geometry-guided cross-attention fusion, GCN neighborhood aggregation,
    and Knowledge Graph scoring.
    """
    def __init__(self, num_relations, embed_dim=768, gcn_dropout=0.1, use_encoders=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_encoders = use_encoders
        
        # 1. Encoders (only instantiated when raw end-to-end extraction is needed;
        # skipped in FastMMSGDataset mode where pre-extracted .pt embeddings are used)
        if use_encoders:
            self.text_encoder = TextEncoder(out_dim=embed_dim)
            self.rgb_encoder = RGBEncoder(out_dim=embed_dim)
            self.depth_encoder = DepthEncoder(out_dim=embed_dim)
        else:
            self.text_encoder = None
            self.rgb_encoder = None
            self.depth_encoder = None
        
        # 2. Fusion Module
        self.fusion_module = GeometryGuidedFusion(feature_dim=embed_dim)
        
        # 3. Graph Reasoning
        self.gcn = GraphConvolution(in_features=embed_dim, out_features=embed_dim, dropout=gcn_dropout)
        
        # 4. Relation Embeddings & Scoring
        # Entities are derived dynamically from encoders + fusion + GCN
        # Relations are learned as standard embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        self.scorer = TransEScore(p_norm=1)

    def forward_encoders(self, text_input_ids, text_mask, rgb_imgs, depth_imgs):
        """
        Extract independent features.
        """
        h_text = self.text_encoder(text_input_ids, text_mask)
        h_rgb = self.rgb_encoder(rgb_imgs)
        h_depth = self.depth_encoder(depth_imgs)
        return h_text, h_rgb, h_depth

    def forward_fusion(self, h_text, h_rgb, h_depth):
        """
        Fuse features into initial entity representation.
        """
        h_entity_initial = self.fusion_module(h_text, h_rgb, h_depth)
        return h_entity_initial

    def forward_gcn(self, h_entity_initial, adj_matrix):
        """
        Update entity representations using structural graph neighborhood.
        """
        h_entity_final = self.gcn(h_entity_initial, adj_matrix)
        return h_entity_final

    def forward_score(self, h_entity_final, relation_ids, tails_entity_final=None):
        """
        Score triplets (h, r, t).
        If tail is provided, compute standard score.
        For training, this requires head and tail from h_entity_final.
        """
        r_embed = self.relation_embeddings(relation_ids)
        # Note: Depending on the batch setup, extracting h and t from h_entity_final 
        # happens outside this function or by passing specific indices.
        pass # To be fully wired in the training loop
