import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from zipdepth.model.architecture import create_model

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_DIR = os.path.dirname(CODE_DIR)

# Configs
DATASET_DIR = os.path.join(THESIS_DIR, 'Dataset', 'VRD', 'vrd')
IMAGE_DIRS = ['sg_train_images', 'sg_test_images']
CHECKPOINT_PATH = os.path.join(CODE_DIR, 'ZipDepth', 'checkpoints', 'zipdepth_base.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ZipDepth transform
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    print(f"Loading model on {DEVICE}...")
    model = create_model('zipdepth_base', pretrained=False)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    sd = checkpoint.get('model_state_dict', checkpoint)
    clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    
    for folder in IMAGE_DIRS:
        in_dir = os.path.join(DATASET_DIR, folder)
        # sg_train_images -> sg_train_depths
        out_folder = folder.replace('images', 'depths')
        out_dir = os.path.join(DATASET_DIR, out_folder)
        os.makedirs(out_dir, exist_ok=True)
        
        if not os.path.exists(in_dir):
            print(f"Warning: {in_dir} not found. Skipping.")
            continue
            
        image_files = [f for f in os.listdir(in_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Found {len(image_files)} images in {folder}.")
        
        # skip if already processed
        to_process = []
        for f in image_files:
            out_name = os.path.splitext(f)[0] + '.png'
            if not os.path.exists(os.path.join(out_dir, out_name)):
                to_process.append(f)
                
        print(f"Processing {len(to_process)} missing depth maps...")
        
        with torch.no_grad():
            for img_name in tqdm(to_process, desc=f"Extracting {folder}"):
                img_path = os.path.join(in_dir, img_name)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = img.size
                    
                    tensor = transform(img).unsqueeze(0).to(DEVICE)
                    
                    # Forward pass
                    out = model(tensor) # Shape: [1, 1, 384, 384]
                    
                    depth = out.squeeze().cpu().numpy()
                    
                    # Normalize to 0-65535 uint16
                    d_min, d_max = depth.min(), depth.max()
                    if d_max > d_min:
                        depth_norm = (depth - d_min) / (d_max - d_min)
                    else:
                        depth_norm = np.zeros_like(depth)
                    
                    depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)
                    
                    # Resize back to original image size
                    depth_img = Image.fromarray(depth_uint16)
                    depth_img = depth_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
                    
                    out_name = os.path.splitext(img_name)[0] + '.png'
                    depth_img.save(os.path.join(out_dir, out_name))
                    
                except Exception as e:
                    print(f"Failed to process {img_name}: {e}")

if __name__ == '__main__':
    main()
