# GeometryMMKGModel (V1) - Merged-MMKG Results

**Dataset:** Merged-MMKG (15,458 entities)
**Model:** GeometryMMKGModel V1 (Standalone)
**Architecture Highlights:**
- **Fusion Module:** GeometryGuidedFusion (Cross-Attention between RGB and Depth, guided by Spatial Map)
- **Graph Aggregation:** GCN (Graph Convolutional Network)
- **Scoring Function:** TransE ($h + r \approx t$, $L1$ distance)
- **Loss Function:** JointLoss (MarginRankingLoss with In-Batch Negative Sampling + Contrastive Alignment)
**Training:** 100 Epochs, Batch Size 1024, Random Negative Sampling.

## Best Evaluation Metrics (Epoch 75)

| Metric | Score |
| :--- | :--- |
| **MRR** | **0.1777 (17.8%)** |
| **Hits@1** | 0.0923 (9.2%) |
| **Hits@3** | 0.2154 (21.5%) |
| **Hits@10** | **0.3301 (33.0%)** |


