import os
import re

DATA_DIR = r'd:\University\Thesis\Code\data\Spatial-MMSG'

def get_base_relation(rel):
    rel = rel.lower().strip()
    
    # 1. Spelling corrections & Exact Matches
    if rel in ['behing', 'ehind', 'behindt', 'behind']: return 'behind'
    if rel in ['aboe', 'abovw', 'over', 'above']: return 'above'
    if rel in ['undernearth', 'benesth', 'below', 'beneath', 'under']: return 'under'
    if 'front of' in rel or 'infront' in rel or 'in fron' in rel: return 'in front of'
    
    if 'near' in rel or 'next to' in rel or 'beside' in rel or 'adjacent' in rel or 'close to' in rel or 'nex to' in rel or 'next tp' in rel:
        return 'near'
        
    if 'wear' in rel or 'has on' in rel: return 'wearing'
    if 'hold' in rel or 'carry' in rel or 'grasp' in rel or 'grab' in rel: return 'holding'
    if 'look' in rel or 'watch' in rel or 'see' in rel: return 'looking at'
    if 'ride' in rel or 'driv' in rel: return 'riding'
    if 'play' in rel: return 'playing'
    if 'eat' in rel or 'feed' in rel or 'bite' in rel or 'graze' in rel: return 'eating'
    if 'walk' in rel or 'run' in rel or 'move' in rel or 'go' in rel: return 'moving'
    
    # 2. Spatial On/In (catch-all for "standing on", "sitting on", etc)
    if 'on' in rel.split() or 'upon' in rel or 'top of' in rel or 'stand' in rel or 'sit' in rel or 'lie' in rel or 'lay' in rel or 'rest' in rel or 'park' in rel:
        return 'on'
        
    if 'in' in rel.split() or 'inside' in rel: return 'in'
    if 'has' == rel or 'have' == rel: return 'has'
    
    return rel

def process_file(filename, new_rel2id):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_triplets = []
    for line in lines[1:]: # skip count header
        parts = line.strip().split('\t')
        if len(parts) == 3:
            s, o, p = parts
            base_p = get_base_relation(p)
            if base_p not in new_rel2id:
                new_rel2id[base_p] = len(new_rel2id)
            new_triplets.append((s, o, base_p))
            
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{len(new_triplets)}\n")
        for s, o, p in new_triplets:
            f.write(f"{s}\t{o}\t{p}\n")

def main():
    print("Cleaning MMSG Aliases...")
    
    # Process Train and Test triplets
    new_rel2id = {}
    process_file('train.txt', new_rel2id)
    process_file('test.txt', new_rel2id)
    
    # Write new relation2id.txt
    with open(os.path.join(DATA_DIR, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(new_rel2id)}\n")
        for r, idx in new_rel2id.items():
            f.write(f"{r}\t{idx}\n")
            
    print(f"Reduced relations down to {len(new_rel2id)} clean base relations!")
    print("Done! You must re-run 'python train.py' to evaluate the cleaned dataset.")

if __name__ == '__main__':
    main()
