import torch
import numpy as np
from tqdm import tqdm

def evaluate_link_prediction(model, all_text, all_rgb, all_depth, dataloader, all_true_triplets, device):
    """
    Evaluates the model using the standard Filtered Setting for Link Prediction.
    Returns: MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    ranks = []
    
    # 1. Pre-compute fused entity embeddings for ALL entities to save time
    print("Pre-computing fused representations for all 93k entities...")
    with torch.no_grad():
        # To avoid OOM during pre-computing, we can do it in batches
        all_h_entity = []
        batch_size = 1024
        for i in range(0, len(all_text), batch_size):
            end = i + batch_size
            t = all_text[i:end].to(device)
            r = all_rgb[i:end].to(device)
            d = all_depth[i:end].to(device)
            fused = model.forward_fusion(t, r, d)
            all_h_entity.append(fused)
        all_h_entity = torch.cat(all_h_entity, dim=0) # (num_entities, embed_dim)
        
    # 2. Score and Rank test triplets
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            h_ids = batch['h_id'].to(device)
            r_ids = batch['r_id'].to(device)
            t_ids = batch['t_id'].to(device)
            
            # Fetch head and relation embeddings
            h_emb = all_h_entity[h_ids]
            r_emb = model.relation_embeddings(r_ids)
            
            # TransE scoring: h + r
            hr_emb = h_emb + r_emb
            
            # Iterate through the mini-batch to compute distance against ALL entities
            for i in range(len(h_ids)):
                h = h_ids[i].item()
                r = r_ids[i].item()
                true_t = t_ids[i].item()
                
                hr = hr_emb[i] # (embed_dim)
                
                # Compute L1 distance: norm(hr - all_entities, p=1)
                diff = hr.unsqueeze(0) - all_h_entity
                scores = -torch.norm(diff, p=1, dim=1) # (num_entities,)
                
                # Filtered Setting: mask out other valid true tails
                valid_tails = all_true_triplets.get((h, r), set())
                filter_mask = list(valid_tails - {true_t})
                if filter_mask:
                    scores[filter_mask] = -1e9
                    
                # Calculate rank
                true_score = scores[true_t]
                # Rank is the number of entities with a strictly higher score + 1
                rank = (scores > true_score).sum().item() + 1
                ranks.append(rank)
                
    # 3. Compute Metrics
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    
    return mrr, hits1, hits3, hits10
