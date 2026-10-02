import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from zipdepth.model.architecture import create_model

# Configs
IMAGE_DIR = 'data/DB15K/images'
OUTPUT_FILE = 'data/DB15K/DB15K-depth-visual.pth'
CHECKPOINT_PATH = 'ZipDepth/checkpoints/zipdepth_base.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Preprocessing used by ZipDepth (standard ImageNet)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    print(f"Loading model on {DEVICE}...")
    model = create_model('zipdepth_base', pretrained=False)
    
    # Load weights safely
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    sd = checkpoint.get('model_state_dict', checkpoint)
    # Handle prefixes if any
    clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Register Hook for Stage 4
    features = {}
    def hook_fn(module, input, output):
        features['stage4'] = output
    
    model.encoder.stage4.register_forward_hook(hook_fn)
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(image_files)} images for extraction.")
    
    depth_embeddings = {}
    
    with torch.no_grad():
        for img_name in tqdm(image_files, desc="Extracting Depth Features"):
            entity_id = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # Forward pass
                _ = model(tensor)
                
                # Extract features from stage 4
                feat = features['stage4'] # Shape: [1, C, H, W]
                
                # Global Average Pooling -> [1, C]
                depth_emb = feat.mean(dim=[2, 3])
                
                # L2 Normalize
                depth_emb = F.normalize(depth_emb, p=2, dim=-1)
                
                # Convert to numpy/cpu and store
                depth_embeddings[entity_id] = depth_emb.squeeze(0).cpu().numpy()
                
            except Exception as e:
                print(f"Failed to process {img_name}: {e}")
    
    # Save the embeddings dictionary
    torch.save(depth_embeddings, OUTPUT_FILE)
    print(f"Extracted features saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
