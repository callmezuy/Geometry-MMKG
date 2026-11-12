# LEGACY: tail-only evaluation with its own loop. Superseded by core/models/evaluation.py
# (head + tail ranks, save_ranks) + scripts/compute_metrics.py (per-category metrics, bootstrap CI).
import torch
import os
from torch.utils.data import DataLoader
from core.data_engineering.fast_dataset import FastMMSGDataset, get_all_true_triplets
from core.models.end_to_end import GeometryMMKGModel
from tqdm import tqdm

def evaluate_by_relation_type():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(CODE_DIR, 'data', 'Spatial-VRD')
    
    # 1. Load Data
    print("Loading features and dataset...")
    all_text = torch.load(os.path.join(DATA_DIR, 'ent_text_emb.pt'), map_location='cpu', weights_only=False)
    all_rgb = torch.load(os.path.join(DATA_DIR, 'ent_rgb_emb.pt'), map_location='cpu', weights_only=False)
    all_depth = torch.load(os.path.join(DATA_DIR, 'ent_depth_emb.pt'), map_location='cpu', weights_only=False)
    
    test_dataset = FastMMSGDataset(DATA_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    all_true_triplets = get_all_true_triplets(DATA_DIR)
    
    # 2. Define Spatial vs Semantic Relations
    spatial_keywords = ['behind', 'front', 'under', 'above', 'top', 'left', 'right', 'near', 'next to', 'inside', 'outside']
    
    id2rel = {v: k for k, v in test_dataset.relation2id.items()}
    spatial_r_ids = set()
    semantic_r_ids = set()
    
    for r_id, r_name in id2rel.items():
        is_spatial = any(kw in r_name.lower() for kw in spatial_keywords)
        if is_spatial:
            spatial_r_ids.add(r_id)
        else:
            semantic_r_ids.add(r_id)
            
    print(f"Found {len(spatial_r_ids)} Spatial relations and {len(semantic_r_ids)} Semantic relations.")
    
    # 3. Load Model
    print("Initializing Model...")
    model = GeometryMMKGModel(num_relations=len(id2rel), embed_dim=768).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # 4. Evaluation Loop
    spatial_ranks = []
    semantic_ranks = []
    
    print("Pre-computing representations...")
    with torch.no_grad():
        all_h_entity = []
        batch_size = 1024
        for i in range(0, len(all_text), batch_size):
            end = i + batch_size
            t = all_text[i:end].to(device)
            r = all_rgb[i:end].to(device)
            d = all_depth[i:end].to(device)
            fused = model.forward_fusion(t, r, d)
            all_h_entity.append(fused)
        all_h_entity = torch.cat(all_h_entity, dim=0)
        
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            h_ids = batch['h_id'].to(device)
            r_ids = batch['r_id'].to(device)
            t_ids = batch['t_id'].to(device)
            
            h_emb = all_h_entity[h_ids]
            r_emb = model.relation_embeddings(r_ids)
            
            hr_emb = h_emb + r_emb
            
            # Filter true triplets
            for i in range(h_ids.size(0)):
                h = h_ids[i].item()
                r = r_ids[i].item()
                t = t_ids[i].item()
                
                hr = hr_emb[i]
                diff = hr.unsqueeze(0) - all_h_entity
                score_i = -torch.norm(diff, p=1, dim=1)
                
                true_tails = all_true_triplets.get((h, r), set())
                filter_mask = list(true_tails - {t})
                
                if filter_mask:
                    score_i[filter_mask] = -1e9
                    
                target_score = score_i[t].item()
                rank = (score_i > target_score).sum().item() + 1
                
                if r in spatial_r_ids:
                    spatial_ranks.append(rank)
                else:
                    semantic_ranks.append(rank)
                    
    def compute_metrics(ranks):
        if not ranks: return 0, 0, 0, 0
        import numpy as np
        ranks = np.array(ranks)
        mrr = np.mean(1.0 / ranks)
        h1 = np.mean(ranks <= 1)
        h3 = np.mean(ranks <= 3)
        h10 = np.mean(ranks <= 10)
        return mrr, h1, h3, h10

    s_mrr, s_h1, s_h3, s_h10 = compute_metrics(spatial_ranks)
    sem_mrr, sem_h1, sem_h3, sem_h10 = compute_metrics(semantic_ranks)
    
    print("\n--- SPATIAL RELATIONS ---")
    print(f"MRR: {s_mrr:.4f} | Hits@1: {s_h1:.4f} | Hits@3: {s_h3:.4f} | Hits@10: {s_h10:.4f}")
    
    print("\n--- SEMANTIC RELATIONS ---")
    print(f"MRR: {sem_mrr:.4f} | Hits@1: {sem_h1:.4f} | Hits@3: {sem_h3:.4f} | Hits@10: {sem_h10:.4f}")

if __name__ == '__main__':
    evaluate_by_relation_type()
