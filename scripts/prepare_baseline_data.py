import os
import torch

def main():
    CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(CODE_DIR, 'data', 'Spatial-VRD')
    
    rgb_path = os.path.join(DATA_DIR, 'ent_rgb_emb.pt')
    depth_path = os.path.join(DATA_DIR, 'ent_depth_emb.pt')
    
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print("Feature extraction not finished yet! Missing .pt files.")
        return
        
    print("Loading extracted RGB and Depth features...")
    rgb_emb = torch.load(rgb_path, weights_only=False)
    depth_emb = torch.load(depth_path, weights_only=False)
    
    print(f"RGB Shape: {rgb_emb.shape}, Depth Shape: {depth_emb.shape}")
    
    # 1. Mean Strategy (Maintains 768 dim, perfect plug-and-play without modifying baseline code)
    print("Generating Mean-blended features for extended SOTA comparison...")
    mean_emb = (rgb_emb + depth_emb) / 2.0
    torch.save(mean_emb, os.path.join(DATA_DIR, 'ent_rgb_depth_mean_emb.pt'))
    
    # 2. Concat Strategy (Doubles dimension to 1536, might require changing linear layer dim in baselines)
    print("Generating Concat-blended features for extended SOTA comparison...")
    concat_emb = torch.cat([rgb_emb, depth_emb], dim=-1)
    torch.save(concat_emb, os.path.join(DATA_DIR, 'ent_rgb_depth_concat_emb.pt'))
    
    print("\nSuccessfully created combined baseline features!")
    print(f"Mean features saved: ent_rgb_depth_mean_emb.pt (Shape: {mean_emb.shape})")
    print(f"Concat features saved: ent_rgb_depth_concat_emb.pt (Shape: {concat_emb.shape})")
    print("Ready to trick SNAG and NativE into learning Depth!")

if __name__ == '__main__':
    main()
