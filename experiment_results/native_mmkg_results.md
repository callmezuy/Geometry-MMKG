# NativE Baseline - Merged-MMKG Results (1000 Epochs)

**Dataset:** Merged-MMKG (15,458 entities)
**Model:** NativE (AdvRelRotatE + WGAN-GP)

### Bảng Tổng Hợp Kết Quả (Filtered Setting)

| Metric | CONTROL (RGB + Text) | ABLATION 1 (Mean RGB+Depth) | ABLATION 2 (3-Modal RGB+Depth+Text) |
| :--- | :--- | :--- | :--- |
| **MRR** | **0.384820** | 0.381977 (⬇ Tụt) | 0.382994 (⬇ Tụt) |
| **MR** | **913.6368** | 936.2797 (⬇ Tệ hơn)* | 926.9304 (⬇ Tệ hơn)* |
| **Hits@1** | **0.303388** | 0.300232 (⬇ Tụt) | 0.301063 (⬇ Tụt) |
| **Hits@3** | **0.429508** | 0.427018 (⬇ Tụt) | 0.427350 (⬇ Tụt) |
| **Hits@10** | 0.532962 | 0.530306 (⬇ Tụt) | **0.533128** (⬆ Nhỉnh hơn 0.0001) |

*(Ghi chú: Riêng chỉ số **MR** (Mean Rank) thì **càng thấp càng tốt**, các chỉ số còn lại **càng cao càng tốt**).*

