import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.data_engineering.dataset import SpatialMMSGDataset
from core.models.end_to_end import GeometryMMKGModel
from core.models.kg_reasoning import JointLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 1. Dataset
    print("Loading dataset (Sanity Check Mode: taking only 16 triplets)...")
    dataset = SpatialMMSGDataset(r'd:\University\Thesis\Code\data\Spatial-MMSG', split='test')
    
    # Limit to 16 triplets for a very quick sanity check
    dataset.triplets = dataset.triplets[:16]
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Model
    print("Initializing GeometryMMKGModel...")
    num_relations = len(dataset.relation2id)
    model = GeometryMMKGModel(num_relations=num_relations, embed_dim=768).to(device)
    
    # 3. Loss & Optimizer
    criterion = JointLoss(margin=5.0, lambda_align=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Starting training loop...")
    model.train()
    for epoch in range(2):
        total_loss = 0
        total_link_loss = 0
        total_align_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            h_data = batch['h_data']
            t_data = batch['t_data']
            neg_t_data = batch['neg_t_data']
            r_id = batch['r_id'].to(device)
            
            # --- Head Forward ---
            h_text_emb, h_rgb_emb, h_depth_emb = model.forward_encoders(
                h_data['input_ids'].to(device), h_data['attention_mask'].to(device),
                h_data['rgb'].to(device), h_data['depth'].to(device)
            )
            h_entity = model.forward_fusion(h_text_emb, h_rgb_emb, h_depth_emb)
            
            # --- Tail Forward ---
            t_text_emb, t_rgb_emb, t_depth_emb = model.forward_encoders(
                t_data['input_ids'].to(device), t_data['attention_mask'].to(device),
                t_data['rgb'].to(device), t_data['depth'].to(device)
            )
            t_entity = model.forward_fusion(t_text_emb, t_rgb_emb, t_depth_emb)
            
            # --- Negative Tail Forward ---
            neg_t_text, neg_t_rgb, neg_t_depth = model.forward_encoders(
                neg_t_data['input_ids'].to(device), neg_t_data['attention_mask'].to(device),
                neg_t_data['rgb'].to(device), neg_t_data['depth'].to(device)
            )
            neg_t_entity = model.forward_fusion(neg_t_text, neg_t_rgb, neg_t_depth)
            
            # --- Scoring ---
            pos_score = model.scorer(h_entity, model.relation_embeddings(r_id), t_entity)
            neg_score = model.scorer(h_entity, model.relation_embeddings(r_id), neg_t_entity)
            
            # --- Loss ---
            # Compute align loss for the head entity to force modalities together
            loss, l_link, l_align = criterion(pos_score, neg_score, h_text_emb, h_rgb_emb, h_depth_emb)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_link_loss += l_link.item()
            total_align_loss += l_align.item()
            
        print(f"Epoch {epoch+1} | Total Loss: {total_loss/len(dataloader):.4f} "
              f"| Link Loss: {total_link_loss/len(dataloader):.4f} "
              f"| Align Loss: {total_align_loss/len(dataloader):.4f}")
        
    print("\nSanity check completed successfully! Model can forward, backward, and update weights without errors.")

if __name__ == '__main__':
    main()
