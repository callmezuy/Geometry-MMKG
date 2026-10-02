# Geometry-Guided Multi-Modal Knowledge Graph (MMKG)

This repository contains the implementation for the thesis on **Geometry-Guided Modality Fusion for Multi-Modal Knowledge Graphs (MMKGs)**. 

Unlike traditional MMKG models (like SNAG, NativE, MKGformer) that only leverage Text and 2D Visual (RGB) modalities, this project introduces the **Spatial/Depth modality** to enhance entity representations. By capturing spatial geometry, the model disambiguates visually similar entities (e.g., distinguishing a 2D painting of a car from a 3D physical car).

## Core Contributions
1. **Spatial-MMSG Dataset**: A novel dataset constructed from Visual Genome & VRD, providing millions of triplets where each entity strictly possesses Text, RGB, and Depth representations.
2. **AlexNet-BN Depth Encoder**: A custom CNN initialized from scratch to strictly learn spatial features (bypassing ImageNet texture bias).
3. **Geometry-Guided Fusion**: A gated cross-attention mechanism that dynamically balances semantic visual features and spatial structural features before aligning them with text.

## Project Structure

```
├── core/
│   ├── data_engineering/
│   │   ├── dataset.py            # Standard dataset (loads raw images/text)
│   │   └── fast_dataset.py       # Fast dataset (loads pre-extracted .pt features)
│   ├── features/
│   │   └── encoders.py           # Text (RoBERTa), RGB (ResNet50), Depth (AlexNet-BN)
│   ├── fusion/
│   │   └── geometry_guided_fusion.py # Gating & Cross-Attention module
│   └── models/
│       ├── end_to_end.py         # Main overarching GeometryMMKGModel
│       ├── evaluation.py         # Standard KG metrics (MRR, Hits@1, Hits@10)
│       └── kg_reasoning.py       # Graph Convolution, TransE Scorer, Joint Loss
├── scripts/
│   ├── download_db15k_images.py  # Image downloader for standard datasets
│   ├── construct_spatial_mmsg.py # Constructs the Spatial-MMSG dataset
│   ├── extract_features.py       # Pre-extracts features for fast training
│   └── prepare_baseline_data.py  # Blends RGB/Depth for SOTA ablation studies
├── baselines/                    # Contains SOTA model codes (SNAG, NativE, etc.)
├── train.py                      # Main training pipeline
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage Pipeline

**1. Data Construction**
```bash
python scripts/construct_spatial_mmsg.py
```
**2. Feature Pre-Extraction (For Kaggle/Fast Training)**
```bash
python scripts/extract_features.py
```
**3. Prepare Baseline Features (Ablation)**
```bash
python scripts/prepare_baseline_data.py
```
**4. Train Model**
```bash
python train.py
```

## Baselines & Ablation Studies
This repository is configured to benchmark against standard MMKG models. For baseline reproduction, we generate `ent_rgb_depth_mean_emb.pt` and `ent_rgb_depth_concat_emb.pt` to feed into unmodified SOTA codebases (SNAG, NativE), proving that our **Geometry-Guided Fusion** outperforms naive addition of the depth modality.
