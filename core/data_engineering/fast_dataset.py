import os
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict

class FastMMSGDataset(Dataset):
    """
    Fast Dataset class that returns entity indices instead of heavy image tensors.
    The models will use these indices to lookup pre-extracted feature tensors.
    """
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        
        # Load relations
        self.relation2id = {}
        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                self.relation2id[parts[0]] = int(parts[1])
                
        # Load entities
        self.entity2id = {}
        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.num_entities = int(next(f).strip())
            for line in f:
                parts = line.strip().split('\t')
                self.entity2id[parts[0]] = int(parts[1])
                
        # Load triplets for this split
        self.triplets = []
        with open(os.path.join(data_dir, f'{split}.txt'), 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    s, p, o = parts
                    self.triplets.append((self.entity2id[s], self.relation2id[p], self.entity2id[o]))
                
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        
        # Random negative tail for training
        neg_t = random.randint(0, self.num_entities - 1)
        
        return {
            'h_id': torch.tensor(h, dtype=torch.long),
            'r_id': torch.tensor(r, dtype=torch.long),
            't_id': torch.tensor(t, dtype=torch.long),
            'neg_t_id': torch.tensor(neg_t, dtype=torch.long)
        }

def get_all_true_triplets(data_dir):
    """
    Builds a dictionary of all true triplets across train and test.
    Returns: dict mapping (h, r) -> set(t)
    Required for Filtered Evaluation.
    """
    all_true = defaultdict(set)
    entity2id = {}
    with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            entity2id[parts[0]] = int(parts[1])
            
    relation2id = {}
    with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            relation2id[parts[0]] = int(parts[1])

    for split in ['train.txt', 'valid.txt', 'test.txt']:
        path = os.path.join(data_dir, split)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        s, p, o = parts
                        h = entity2id[s]
                        r = relation2id[p]
                        t = entity2id[o]
                        all_true[(h, r)].add(t)
    return all_true
