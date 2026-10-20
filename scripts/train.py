import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.data_engineering.fast_dataset import FastMMSGDataset, get_all_true_triplets
from core.models.end_to_end import GeometryMMKGModel
from core.models.kg_reasoning import JointLoss
from core.models.evaluation import evaluate_link_prediction
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    DATA_DIR = r'd:\University\Thesis\Code\data\Spatial-MMSG'
    
    # 1. Load Pre-Extracted Features
    print("Loading pre-extracted features into RAM...")
    all_text = torch.load(os.path.join(DATA_DIR, 'ent_text_emb.pt'), map_location='cpu', weights_only=False)
    all_rgb = torch.load(os.path.join(DATA_DIR, 'ent_rgb_emb.pt'), map_location='cpu', weights_only=False)
    all_depth = torch.load(os.path.join(DATA_DIR, 'ent_depth_emb.pt'), map_location='cpu', weights_only=False)
    
    # 2. Dataset
    print("Initializing Fast Datasets...")
    train_dataset = FastMMSGDataset(DATA_DIR, split='train')
    test_dataset = FastMMSGDataset(DATA_DIR, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    print("Building Filtered Triplet Dictionary for Evaluation...")
    all_true_triplets = get_all_true_triplets(DATA_DIR)
    
    # 3. Model
    print("Initializing GeometryMMKGModel...")
    num_relations = len(train_dataset.relation2id)
    model = GeometryMMKGModel(num_relations=num_relations, embed_dim=768).to(device)
    
    # 4. Loss & Optimizer
    criterion = JointLoss(margin=5.0, lambda_align=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    EPOCHS = 50
    print(f"Starting fast training loop for {EPOCHS} epochs...")
    
    best_h10 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_link_loss = 0
        total_align_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            h_ids = batch['h_id']
            r_ids = batch['r_id'].to(device)
            t_ids = batch['t_id']
            neg_t_ids = batch['neg_t_id']
            
            # Fetch embeddings directly from RAM (Bypassing ResNet/RoBERTa completely!)
            h_t, h_r, h_d = all_text[h_ids].to(device), all_rgb[h_ids].to(device), all_depth[h_ids].to(device)
            t_t, t_r, t_d = all_text[t_ids].to(device), all_rgb[t_ids].to(device), all_depth[t_ids].to(device)
            
            # Geometry-Guided Fusion (Fast forward)
            h_entity = model.forward_fusion(h_t, h_r, h_d)
            t_entity = model.forward_fusion(t_t, t_r, t_d)
            
            # In-Batch Hard Negative Mining using cdist
            r_emb = model.relation_embeddings(r_ids)
            hr_entity = h_entity + r_emb
            
            # dist[i, j] = || (h[i] + r[i]) - t[j] ||_1
            dist = torch.cdist(hr_entity, t_entity, p=1)
            scores = -dist
            
            # Positive scores are on the diagonal (i == j)
            pos_score = scores.diag()
            
            # Mask out the diagonal to find the hardest negative (max score) for each h
            mask = torch.eye(scores.size(0), device=device).bool()
            scores.masked_fill_(mask, -1e9)
            neg_score = scores.max(dim=1)[0]
            
            # Loss Calculation
            loss, l_link, l_align = criterion(pos_score, neg_score, h_t, h_r, h_d)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_link_loss += l_link.item()
            total_align_loss += l_align.item()
            
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"--- Evaluating at Epoch {epoch+1} ---")
            mrr, h1, h3, h10 = evaluate_link_prediction(
                model, all_text, all_rgb, all_depth, test_loader, all_true_triplets, device
            )
            print(f"Metrics -> MRR: {mrr:.4f} | Hits@1: {h1:.4f} | Hits@3: {h3:.4f} | Hits@10: {h10:.4f}")
            
            if h10 > best_h10:
                best_h10 = h10
                print(f"New best model! Saving to best_model.pth")
                torch.save(model.state_dict(), 'best_model.pth')
                
    print("Training complete! Saving final_model.pth")
    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == '__main__':
    main()
