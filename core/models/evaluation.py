import torch
import numpy as np
from tqdm import tqdm

def evaluate_link_prediction(model, all_text, all_rgb, all_depth, dataloader, all_true_triplets, device, entity_chunk_size=32768):
    """
    Evaluates the model using the standard Filtered Setting for Link Prediction.
    Uses vectorized chunked torch.cdist to support massive graphs (e.g., 1.44M entities in Spatial-VG)
    without GPU Out-Of-Memory (OOM) errors.
    Returns: MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    ranks = []
    num_entities = len(all_text)
    
    # Keep all_h_entity on CPU if graph is massive (>250k entities) to preserve GPU VRAM
    store_on_cpu = num_entities > 250000
    target_store_device = torch.device('cpu') if store_on_cpu else device
    
    # 1. Pre-compute fused entity embeddings for ALL entities in batches
    print(f"Pre-computing fused representations for all {num_entities:,} entities...")
    with torch.no_grad():
        all_h_entity = []
        batch_size = 2048
        for i in range(0, num_entities, batch_size):
            end = min(i + batch_size, num_entities)
            t = all_text[i:end].to(device)
            r = all_rgb[i:end].to(device)
            d = all_depth[i:end].to(device)
            fused = model.forward_fusion(t, r, d)
            all_h_entity.append(fused.to(target_store_device))
        all_h_entity = torch.cat(all_h_entity, dim=0)  # (num_entities, embed_dim)
        
    # 2. Vectorized & Chunked Score and Rank
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            h_ids = batch['h_id']
            r_ids = batch['r_id']
            t_ids = batch['t_id']
            
            # Fetch head, tail, and relation embeddings on device
            h_emb = all_h_entity[h_ids].to(device)
            t_emb = all_h_entity[t_ids].to(device)
            r_emb = model.relation_embeddings(r_ids.to(device))
            
            # TransE scoring: h + r
            hr_emb = h_emb + r_emb  # (B, embed_dim)
            
            # True tail scores: -|| (h + r) - t ||_1
            true_scores = -torch.norm(hr_emb - t_emb, p=1, dim=1)  # (B,)
            
            # Count how many entities in the entire graph have strictly higher score than true_scores
            greater_counts = torch.zeros(len(h_ids), dtype=torch.long, device=device)
            for c_start in range(0, num_entities, entity_chunk_size):
                c_end = min(c_start + entity_chunk_size, num_entities)
                ent_chunk = all_h_entity[c_start:c_end].to(device)
                # L1 distance matrix via fast C++/CUDA kernel: (B, chunk_size)
                chunk_scores = -torch.cdist(hr_emb, ent_chunk, p=1)
                greater_counts += (chunk_scores > true_scores.unsqueeze(1)).sum(dim=1)
                
            greater_counts = greater_counts.cpu().numpy()
            h_list = h_ids.tolist()
            r_list = r_ids.tolist()
            t_list = t_ids.tolist()
            
            # Filtered Setting: subtract other valid true tails that scored higher than true_t
            for i in range(len(h_list)):
                h = h_list[i]
                r = r_list[i]
                true_t = t_list[i]
                
                valid_tails = all_true_triplets.get((h, r), set())
                other_valid = list(valid_tails - {true_t})
                
                filtered_out = 0
                if other_valid:
                    other_emb = all_h_entity[other_valid].to(device)
                    other_scores = -torch.norm(hr_emb[i].unsqueeze(0) - other_emb, p=1, dim=1)
                    filtered_out = (other_scores > true_scores[i]).sum().item()
                    
                rank = int(greater_counts[i]) - filtered_out + 1
                ranks.append(max(1, rank))
                
    # 3. Compute Metrics
    ranks = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks))
    hits1 = float(np.mean(ranks <= 1))
    hits3 = float(np.mean(ranks <= 3))
    hits10 = float(np.mean(ranks <= 10))
    
    return mrr, hits1, hits3, hits10
