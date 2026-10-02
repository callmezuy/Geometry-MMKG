#!/bin/bash
echo "Setting up environment for MMKG Depth Modality Integration (Phase 1)"

# 1. Install dependencies
echo "Installing dependencies..."
pip install -r SNAG/requirement.txt
pip install gdown

# 2. Create data directories
echo "Creating data directories..."
mkdir -p data/DB15K
mkdir -p data/MKG-W
mkdir -p data/MKG-Y

# 3. Note about dataset
echo "Make sure data.zip is downloaded and extracted to the data/ directory."
echo "Environment setup complete!"
