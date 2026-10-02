import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer to propagate entity embeddings over the graph structure.
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(dropout)
        # Initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        x: Node features (num_nodes, in_features)
        adj: Sparse adjacency matrix (num_nodes, num_nodes)
        """
        x = self.dropout(x)
        support = torch.mm(x, self.weight)
        # Sparse matrix multiplication for efficiency
        output = torch.sparse.mm(adj, support)
        return F.relu(output)


class TransEScore(nn.Module):
    """
    TransE scoring function: f(h, r, t) = -||h + r - t||_1 (or L2 norm)
    """
    def __init__(self, p_norm=1):
        super().__init__()
        self.p_norm = p_norm

    def forward(self, h, r, t):
        # h, r, t shape: (batch_size, embed_dim)
        score = torch.norm(h + r - t, p=self.p_norm, dim=1)
        # Return negative distance so higher score means more plausible
        return -score


class JointLoss(nn.Module):
    """
    Computes the joint loss: L_total = L_link + lambda * L_align
    """
    def __init__(self, margin=5.0, lambda_align=0.1, temperature=0.07):
        super().__init__()
        # L_link: Margin Ranking Loss for link prediction
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.lambda_align = lambda_align
        self.temperature = temperature

    def forward(self, pos_score, neg_score, h_text, h_rgb, h_depth):
        """
        pos_score: Scores of positive triplets
        neg_score: Scores of negative (corrupted) triplets
        h_text, h_rgb, h_depth: Extracted features for contrastive alignment
        """
        # 1. Link Prediction Loss (Margin Ranking Loss)
        # Target is 1 because we want pos_score > neg_score by at least margin
        target = torch.ones_like(pos_score)
        l_link = self.margin_loss(pos_score, neg_score, target)

        # 2. Contrastive Alignment Loss (InfoNCE style or MSE)
        # We enforce representations of the SAME entity across different modalities to be similar
        # Normalize features
        h_text = F.normalize(h_text, p=2, dim=1)
        h_rgb = F.normalize(h_rgb, p=2, dim=1)
        h_depth = F.normalize(h_depth, p=2, dim=1)
        
        # Simple contrastive: mean squared error between modalities of the same entity
        # (A more complex InfoNCE could be used across the batch)
        l_align_rgb = F.mse_loss(h_text, h_rgb)
        l_align_depth = F.mse_loss(h_text, h_depth)
        l_align_spatial = F.mse_loss(h_rgb, h_depth)
        
        l_align = (l_align_rgb + l_align_depth + l_align_spatial) / 3.0

        # 3. Total Loss
        l_total = l_link + self.lambda_align * l_align
        return l_total, l_link, l_align
