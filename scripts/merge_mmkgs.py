import os
import json
import shutil
import urllib.parse
from tqdm import tqdm
from PIL import Image

DATA_DIR = r'd:\University\Thesis\Code\data'
BENCHMARK_DIR = r'd:\University\Thesis\Code\baselines\SNAG\SNAG_MKGC\benchmarks'
OUT_DIR = os.path.join(DATA_DIR, 'Merged-MMKG')
OUT_IMG_DIR = os.path.join(OUT_DIR, 'images')

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# 1. Load DB15K Image Index
db15k_img_index = {}
with open(os.path.join(DATA_DIR, 'raw_downloads', 'mmkb', 'DB15K', 'DB15K_ImageIndex.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            db15k_img_index[parts[0].strip('<>').strip()] = parts[1].strip()

# 2. Load Wikidata to Wikipedia Dictionary
WIKI_DICT_PATH = os.path.join(DATA_DIR, 'MKG-W', 'wikidata_to_wiki_dict.json')
wikidata_dict = {}
if os.path.exists(WIKI_DICT_PATH):
    with open(WIKI_DICT_PATH, 'r', encoding='utf-8') as f:
        wikidata_dict = json.load(f)

# Helper function to get image path for an entity
def get_image_path(dataset, entity):
    if dataset == 'DB15K':
        uri = entity.strip('<>').strip()
        img_id = db15k_img_index.get(uri)
        if not img_id: return None
        path = os.path.join(DATA_DIR, 'DB15K', 'images', f'{img_id}.jpg')
    elif dataset == 'MKG-W':
        qid = entity.split('/')[-1].replace('>', '')
        path = os.path.join(DATA_DIR, 'MKG-W', 'images', f'{qid}.jpg')
    elif dataset == 'MKG-Y':
        entity_clean = entity.replace('<', '').replace('>', '').replace('/', '_').replace(':', '_')
        path = os.path.join(DATA_DIR, 'MKG-Y', 'images', f'{entity_clean}.jpg')
    else:
        return None
        
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        # STRICT VALIDATION: Try to open the image with PIL to catch corrupt/DOS files
        try:
            with Image.open(path) as img:
                img.verify() # Quickly verifies if the file is a valid image without decoding
            return path
        except Exception:
            return None
    return None

# NEW: Semantic Namespace Alignment Logic
def get_unified_id(dataset, entity_str):
    if dataset == 'DB15K':
        # <http://dbpedia.org/resource/Elon_Musk> -> Elon_Musk
        uri = entity_str.strip('<>').split('/')[-1]
        return urllib.parse.unquote(uri)
    elif dataset == 'MKG-Y':
        # Gilles_Mbang_Ondo -> Gilles_Mbang_Ondo
        clean = entity_str.strip('<>').replace('Yago_', '')
        return urllib.parse.unquote(clean)
    elif dataset == 'MKG-W':
        # Q317521 -> Elon_Musk
        qid = entity_str.split('/')[-1].replace('>', '')
        return wikidata_dict.get(qid, qid) # Fallback to Q-node if not found
    return entity_str

def load_triplets(filepath):
    triplets = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                triplets.append((parts[0], parts[1], parts[2]))
    return triplets

# Global mappings
global_entity2id = {}     # unified_name -> new_id
global_relation2id = {}   # original_relation -> new_id
entity_image_copied = set() # Track which Unified IDs already have an image copied

merged_train = []
merged_valid = []
merged_test = []

# Dataset-specific Valid Entities
# ds_id -> (unified_id, new_global_id)
valid_entities_map = {'DB15K': {}, 'MKG-W': {}, 'MKG-Y': {}}

total_copied = 0

for dataset in ['DB15K', 'MKG-W', 'MKG-Y']:
    print(f"\nProcessing {dataset}...")
    ds_dir = os.path.join(BENCHMARK_DIR, dataset)
    
    # Load dataset mappings
    ds_ent2id = {}
    with open(os.path.join(ds_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                ds_ent2id[parts[-1]] = " ".join(parts[:-1]) # id -> str
                
    ds_rel2id = {}
    with open(os.path.join(ds_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                ds_rel2id[parts[-1]] = " ".join(parts[:-1]) # id -> str

    # Filter entities that have images and apply Semantic Alignment
    valid_count = 0
    for ent_id, ent_str in tqdm(ds_ent2id.items(), desc=f"Aligning {dataset}"):
        img_path = get_image_path(dataset, ent_str)
        if img_path:
            # ALIGNMENT: Get Universal Wikipedia Title
            unified_id = get_unified_id(dataset, ent_str)
            
            # Map unified ID to Global Index
            if unified_id not in global_entity2id:
                gid = len(global_entity2id)
                global_entity2id[unified_id] = gid
            else:
                gid = global_entity2id[unified_id]
                
            # Copy image ONLY ONCE per Unified Entity
            if unified_id not in entity_image_copied:
                shutil.copy2(img_path, os.path.join(OUT_IMG_DIR, f'{gid}.jpg'))
                entity_image_copied.add(unified_id)
                total_copied += 1
                
            valid_entities_map[dataset][ent_id] = gid
            valid_count += 1

    print(f"Valid & Aligned entities in {dataset}: {valid_count} / {len(ds_ent2id)}")

    # Process triplets
    for split, out_list in [('train2id.txt', merged_train), ('valid2id.txt', merged_valid), ('test2id.txt', merged_test)]:
        triplets = load_triplets(os.path.join(ds_dir, split))
        kept = 0
        for h, t, r in triplets:
            if h in valid_entities_map[dataset] and t in valid_entities_map[dataset]:
                r_str = f"{dataset}::{ds_rel2id[r]}"
                
                if r_str not in global_relation2id:
                    global_relation2id[r_str] = len(global_relation2id)
                    
                g_h = valid_entities_map[dataset][h]
                g_t = valid_entities_map[dataset][t]
                g_r = global_relation2id[r_str]
                
                out_list.append((g_h, g_t, g_r))
                kept += 1
        print(f"Kept {kept}/{len(triplets)} triplets for {dataset} {split}")

print("\n--- Final Statistics ---")
print(f"Total Unique Merged Entities (Deduplicated!): {len(global_entity2id)}")
print(f"Total Merged Relations: {len(global_relation2id)}")
print(f"Total Unique Images Copied: {total_copied}")
print(f"Total Train Triplets: {len(merged_train)}")
print(f"Total Valid Triplets: {len(merged_valid)}")
print(f"Total Test Triplets: {len(merged_test)}")

# Write outputs
def write_triplets(filename, triplets):
    # Deduplicate Triplets across all datasets
    unique_triplets = list(set(triplets))
    print(f"Writing {filename}: {len(unique_triplets)} unique triplets (removed {len(triplets) - len(unique_triplets)} duplicates)")
    with open(os.path.join(OUT_DIR, filename), 'w', encoding='utf-8') as f:
        f.write(f"{len(unique_triplets)}\n")
        for h, t, r in unique_triplets:
            f.write(f"{h}\t{t}\t{r}\n")

write_triplets('train2id.txt', merged_train)
write_triplets('valid2id.txt', merged_valid)
write_triplets('test2id.txt', merged_test)

with open(os.path.join(OUT_DIR, 'entity2id.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{len(global_entity2id)}\n")
    for ent, gid in global_entity2id.items():
        f.write(f"{ent}\t{gid}\n")
        
with open(os.path.join(OUT_DIR, 'relation2id.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{len(global_relation2id)}\n")
    for rel, gid in global_relation2id.items():
        f.write(f"{rel}\t{gid}\n")
        
print("\nFederated Merge with Semantic Namespace Alignment Complete!")
