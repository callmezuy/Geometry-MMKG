import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import RobertaTokenizer
import torchvision.transforms as T
from tqdm import tqdm

from core.features.encoders import TextEncoder, RGBEncoder, DepthEncoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class EntityFeatureDataset(Dataset):
    def __init__(self, data_dir, max_len=16):
        self.data_dir = data_dir
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_len = max_len
        
        # Load entity2id (we want to process strictly in order 0 to N-1)
        self.id2entity = {}
        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.num_entities = int(next(f).strip())
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    e_name, e_id = parts[0], int(parts[1])
                    self.id2entity[e_id] = e_name
                    
        self.entity2text = {}
        with open(os.path.join(data_dir, 'entity2text.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.entity2text[parts[0]] = parts[1]
                    
        self.rgb_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        # We process precisely by entity id `idx`
        eid = self.id2entity.get(idx, 'unknown')
        
        # Text
        text = self.entity2text.get(eid, 'unknown entity')
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        
        # RGB
        rgb_path = os.path.join(self.data_dir, 'images_rgb', f'{eid}.jpg')
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
        except:
            rgb_img = Image.new('RGB', (224, 224))
        rgb_tensor = self.rgb_transform(rgb_img)
        
        # Depth
        depth_path = os.path.join(self.data_dir, 'images_depth', f'{eid}.png')
        try:
            depth_img = Image.open(depth_path).convert('L')
        except:
            depth_img = Image.new('L', (224, 224))
        depth_tensor = self.depth_transform(depth_img)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'rgb': rgb_tensor,
            'depth': depth_tensor
        }

def extract_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(CODE_DIR, 'data', 'Spatial-VRD')
    
    # Init encoders
    print("Initializing Encoders...")
    text_enc = TextEncoder(out_dim=768).to(device)
    rgb_enc = RGBEncoder(out_dim=768, pretrained=True).to(device)
    depth_enc = DepthEncoder(out_dim=768).to(device)
    
    text_enc.eval()
    rgb_enc.eval()
    depth_enc.eval()
    
    dataset = EntityFeatureDataset(DATA_DIR)
    # Important: shuffle=False to ensure embeddings align with entity IDs
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0) 
    
    all_text = []
    all_rgb = []
    all_depth = []
    
    print(f"Starting extraction for {len(dataset)} entities...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            
            t_emb = text_enc(input_ids, attention_mask).cpu()
            r_emb = rgb_enc(rgb).cpu()
            d_emb = depth_enc(depth).cpu()
            
            all_text.append(t_emb)
            all_rgb.append(r_emb)
            all_depth.append(d_emb)
            
    # Concatenate all batches
    final_text = torch.cat(all_text, dim=0)
    final_rgb = torch.cat(all_rgb, dim=0)
    final_depth = torch.cat(all_depth, dim=0)
    
    print(f"Shapes: Text {final_text.shape}, RGB {final_rgb.shape}, Depth {final_depth.shape}")
    
    print("Saving tensors...")
    torch.save(final_text, os.path.join(DATA_DIR, 'ent_text_emb.pt'))
    torch.save(final_rgb, os.path.join(DATA_DIR, 'ent_rgb_emb.pt'))
    torch.save(final_depth, os.path.join(DATA_DIR, 'ent_depth_emb.pt'))
    print("Done! Features saved.")

if __name__ == '__main__':
    extract_all()
