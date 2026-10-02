import torch
import torch.nn as nn

class GeometryGuidedFusion(nn.Module):
    """
    Implements the Geometry-Guided Modality Fusion module.
    Replaces naive concatenation by intelligently merging Depth (geometry) 
    with RGB (vision) via a learned gating mechanism, and then semantically 
    aligning it with Text via Cross-Attention.
    
    Notations follow the ZJU KG-MM-Survey (2023) standardization.
    """
    def __init__(self, feature_dim=768, num_attention_heads=8, dropout_prob=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Visual-Spatial Gating (W_g)
        # Input is [h_rgb || h_depth], so dimension is feature_dim * 2
        self.W_g = nn.Linear(feature_dim * 2, feature_dim)
        
        # 2. Semantic Alignment (Multi-Head Cross Attention)
        # Q = Text, K = Vision, V = Vision
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_attention_heads, 
            dropout=dropout_prob,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, h_text, h_rgb, h_depth):
        """
        Args:
            h_text: Text embedding, shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
            h_rgb: RGB visual embedding, shape (batch_size, feature_dim)
            h_depth: Depth spatial embedding, shape (batch_size, feature_dim)
            
        Returns:
            h_entity: Fused entity representation, shape (batch_size, feature_dim)
        """
        # Ensure dimensions match for cross-attention compatibility
        # If inputs are just 2D (batch_size, feature_dim), add a sequence dimension
        has_seq_len = len(h_text.shape) == 3
        if not has_seq_len:
            h_text = h_text.unsqueeze(1)
            
        h_rgb_unsq = h_rgb.unsqueeze(1) if len(h_rgb.shape) == 2 else h_rgb
        h_depth_unsq = h_depth.unsqueeze(1) if len(h_depth.shape) == 2 else h_depth

        # ---------------------------------------------------------
        # STEP 1: Visual-Spatial Gating
        # ---------------------------------------------------------
        # Concatenate RGB and Depth along the feature dimension
        concat_vis_spat = torch.cat([h_rgb_unsq, h_depth_unsq], dim=-1)
        
        # Calculate gating matrix G
        # G = \sigma(W_g [h_{rgb} || h_{depth}] + b_g)
        G = torch.sigmoid(self.W_g(concat_vis_spat))
        
        # Compute adjusted visual feature h_vision
        # h_{vision} = G \odot h_{rgb} + (1 - G) \odot h_{depth}
        h_vision = G * h_rgb_unsq + (1 - G) * h_depth_unsq
        
        # ---------------------------------------------------------
        # STEP 2: Semantic Alignment
        # ---------------------------------------------------------
        # Cross Attention: Q = h_{text}, K = h_{vision}, V = h_{vision}
        attn_output, attn_weights = self.cross_attention(
            query=h_text, 
            key=h_vision, 
            value=h_vision
        )
        
        # Residual connection + Layer Normalization
        h_entity = self.layer_norm(h_text + self.dropout(attn_output))
        
        # Revert sequence dimension if it was artificially added
        if not has_seq_len:
            h_entity = h_entity.squeeze(1)
            
        return h_entity
