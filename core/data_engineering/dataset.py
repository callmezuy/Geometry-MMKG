import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import RobertaTokenizer
import torchvision.transforms as T

class SpatialMMSGDataset(Dataset):
    def __init__(self, data_dir, split='train', max_len=16):
        self.data_dir = data_dir
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_len = max_len
        
        # Load entity texts
        self.entity2text = {}
        with open(os.path.join(data_dir, 'entity2text.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.entity2text[parts[0]] = parts[1]
                    
        # Load relation ids
        self.relation2id = {}
        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            next(f) # skip count
            for line in f:
                parts = line.strip().split('\t')
                self.relation2id[parts[0]] = int(parts[1])
                
        # Load triplets
        self.triplets = []
        with open(os.path.join(data_dir, f'{split}.txt'), 'r', encoding='utf-8') as f:
            next(f) # skip count
            for line in f:
                s, p, o = line.strip().split('\t')
                self.triplets.append((s, p, o)) # subject, predicate, object
                
        # Image Transforms
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
        return len(self.triplets)

    def load_entity_data(self, eid):
        # 1. Text
        text = self.entity2text.get(eid, 'unknown entity')
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        
        # 2. RGB
        rgb_path = os.path.join(self.data_dir, 'images_rgb', f'{eid}.jpg')
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
        except:
            rgb_img = Image.new('RGB', (224, 224))
        rgb_tensor = self.rgb_transform(rgb_img)
        
        # 3. Depth
        depth_path = os.path.join(self.data_dir, 'images_depth', f'{eid}.png')
        try:
            depth_img = Image.open(depth_path).convert('L') # Treat as grayscale for simplicity in sanity check
        except:
            depth_img = Image.new('L', (224, 224))
        depth_tensor = self.depth_transform(depth_img)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'rgb': rgb_tensor,
            'depth': depth_tensor
        }

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        
        h_data = self.load_entity_data(h)
        t_data = self.load_entity_data(t)
        
        # Generate random negative tail for contrastive margin loss
        neg_t = random.choice(self.triplets)[2]
        neg_t_data = self.load_entity_data(neg_t)
        
        return {
            'h_data': h_data,
            't_data': t_data,
            'neg_t_data': neg_t_data,
            'r_id': torch.tensor(self.relation2id[r], dtype=torch.long)
        }
