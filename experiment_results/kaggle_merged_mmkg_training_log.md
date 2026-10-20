# GeometryMMKGModel - Merged-MMKG Training Log (Kaggle Offline)

**Dataset:** Merged-MMKG (15,458 entities, OpenKE format)
**Model:** GeometryMMKGModel (TransE Scorer, JointLoss)
**Hardware:** Kaggle GPU (Offline Mode)

## Training Logs

```text
Using device: cuda
Loading pre-extracted features into RAM...
Initializing MMKG Datasets...
Initializing GeometryMMKGModel...
Epoch 01/100 | Loss: 26.7362
Epoch 02/100 | Loss: 16.2772
Epoch 03/100 | Loss: 13.1516
Epoch 04/100 | Loss: 11.5092
Epoch 05/100 | Loss: 10.4026
--- Evaluating at Epoch 5 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.09it/s]
Metrics -> MRR: 0.0477 | Hits@1: 0.0108 | Hits@3: 0.0470 | Hits@10: 0.1423
Saved new best model!
Epoch 06/100 | Loss: 9.6251
Epoch 07/100 | Loss: 9.0386
Epoch 08/100 | Loss: 8.6265
Epoch 09/100 | Loss: 8.2630
Epoch 10/100 | Loss: 7.9955
--- Evaluating at Epoch 10 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.37it/s]
Metrics -> MRR: 0.0690 | Hits@1: 0.0153 | Hits@3: 0.0875 | Hits@10: 0.1817
Saved new best model!
Epoch 11/100 | Loss: 7.7393
Epoch 12/100 | Loss: 7.5335
Epoch 13/100 | Loss: 7.3549
Epoch 14/100 | Loss: 7.1917
Epoch 15/100 | Loss: 7.0625
--- Evaluating at Epoch 15 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.33it/s]
Metrics -> MRR: 0.0871 | Hits@1: 0.0204 | Hits@3: 0.1113 | Hits@10: 0.2260
Saved new best model!
Epoch 16/100 | Loss: 6.9365
Epoch 17/100 | Loss: 6.8205
Epoch 18/100 | Loss: 6.7242
Epoch 19/100 | Loss: 6.6371
Epoch 20/100 | Loss: 6.5539
--- Evaluating at Epoch 20 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.34it/s]
Metrics -> MRR: 0.1049 | Hits@1: 0.0259 | Hits@3: 0.1378 | Hits@10: 0.2629
Saved new best model!
Epoch 21/100 | Loss: 6.4776
Epoch 22/100 | Loss: 6.4112
Epoch 23/100 | Loss: 6.3329
Epoch 24/100 | Loss: 6.2705
Epoch 25/100 | Loss: 6.2156
--- Evaluating at Epoch 25 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.29it/s]
Metrics -> MRR: 0.1123 | Hits@1: 0.0289 | Hits@3: 0.1433 | Hits@10: 0.2818
Saved new best model!
Epoch 26/100 | Loss: 6.1534
Epoch 27/100 | Loss: 6.1043
Epoch 28/100 | Loss: 6.0544
Epoch 29/100 | Loss: 6.0116
Epoch 30/100 | Loss: 5.9610
--- Evaluating at Epoch 30 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.34it/s]
Metrics -> MRR: 0.1339 | Hits@1: 0.0462 | Hits@3: 0.1782 | Hits@10: 0.2948
Saved new best model!
Epoch 31/100 | Loss: 5.9222
Epoch 32/100 | Loss: 5.8827
Epoch 33/100 | Loss: 5.8455
Epoch 34/100 | Loss: 5.8239
Epoch 35/100 | Loss: 5.7836
--- Evaluating at Epoch 35 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.44it/s]
Metrics -> MRR: 0.1351 | Hits@1: 0.0447 | Hits@3: 0.1803 | Hits@10: 0.3045
Saved new best model!
Epoch 36/100 | Loss: 5.7516
Epoch 37/100 | Loss: 5.7222
Epoch 38/100 | Loss: 5.6947
Epoch 39/100 | Loss: 5.6697
Epoch 40/100 | Loss: 5.6472
--- Evaluating at Epoch 40 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.54it/s]
Metrics -> MRR: 0.1253 | Hits@1: 0.0365 | Hits@3: 0.1589 | Hits@10: 0.3044
Epoch 41/100 | Loss: 5.6147
Epoch 42/100 | Loss: 5.5893
Epoch 43/100 | Loss: 5.5716
Epoch 44/100 | Loss: 5.5536
Epoch 45/100 | Loss: 5.5229
--- Evaluating at Epoch 45 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.53it/s]
Metrics -> MRR: 0.1449 | Hits@1: 0.0551 | Hits@3: 0.1870 | Hits@10: 0.3089
Saved new best model!
Epoch 46/100 | Loss: 5.4996
Epoch 47/100 | Loss: 5.4828
Epoch 48/100 | Loss: 5.4614
Epoch 49/100 | Loss: 5.4479
Epoch 50/100 | Loss: 5.4338
--- Evaluating at Epoch 50 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.55it/s]
Metrics -> MRR: 0.1589 | Hits@1: 0.0707 | Hits@3: 0.2021 | Hits@10: 0.3160
Saved new best model!
Epoch 51/100 | Loss: 5.4204
Epoch 52/100 | Loss: 5.4022
Epoch 53/100 | Loss: 5.3876
Epoch 54/100 | Loss: 5.3702
Epoch 55/100 | Loss: 5.3564
--- Evaluating at Epoch 55 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.61it/s]
Metrics -> MRR: 0.1513 | Hits@1: 0.0603 | Hits@3: 0.1950 | Hits@10: 0.3178
Epoch 56/100 | Loss: 5.3428
Epoch 57/100 | Loss: 5.3302
Epoch 58/100 | Loss: 5.3140
Epoch 59/100 | Loss: 5.2979
Epoch 60/100 | Loss: 5.2899
--- Evaluating at Epoch 60 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.52it/s]
Metrics -> MRR: 0.1620 | Hits@1: 0.0729 | Hits@3: 0.2072 | Hits@10: 0.3190
Saved new best model!
Epoch 61/100 | Loss: 5.2815
Epoch 62/100 | Loss: 5.2626
Epoch 63/100 | Loss: 5.2544
Epoch 64/100 | Loss: 5.2421
Epoch 65/100 | Loss: 5.2366
--- Evaluating at Epoch 65 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.51it/s]
Metrics -> MRR: 0.1554 | Hits@1: 0.0616 | Hits@3: 0.2004 | Hits@10: 0.3271
Epoch 66/100 | Loss: 5.2172
Epoch 67/100 | Loss: 5.2168
Epoch 68/100 | Loss: 5.1986
Epoch 69/100 | Loss: 5.1868
Epoch 70/100 | Loss: 5.1782
--- Evaluating at Epoch 70 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.55it/s]
Metrics -> MRR: 0.1580 | Hits@1: 0.0668 | Hits@3: 0.2029 | Hits@10: 0.3288
Epoch 71/100 | Loss: 5.1696
Epoch 72/100 | Loss: 5.1604
Epoch 73/100 | Loss: 5.1477
Epoch 74/100 | Loss: 5.1407
Epoch 75/100 | Loss: 5.1304
--- Evaluating at Epoch 75 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.56it/s]
Metrics -> MRR: 0.1777 | Hits@1: 0.0923 | Hits@3: 0.2154 | Hits@10: 0.3301
Saved new best model!
Epoch 76/100 | Loss: 5.1246
Epoch 77/100 | Loss: 5.1207
Epoch 78/100 | Loss: 5.1076
Epoch 79/100 | Loss: 5.1016
Epoch 80/100 | Loss: 5.0943
--- Evaluating at Epoch 80 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.58it/s]
Metrics -> MRR: 0.1616 | Hits@1: 0.0689 | Hits@3: 0.2052 | Hits@10: 0.3295
Epoch 81/100 | Loss: 5.0866
Epoch 82/100 | Loss: 5.0724
Epoch 83/100 | Loss: 5.0694
Epoch 84/100 | Loss: 5.0586
Epoch 85/100 | Loss: 5.0547
--- Evaluating at Epoch 85 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.56it/s]
Metrics -> MRR: 0.1747 | Hits@1: 0.0847 | Hits@3: 0.2185 | Hits@10: 0.3323
Epoch 86/100 | Loss: 5.0571
Epoch 87/100 | Loss: 5.0442
Epoch 88/100 | Loss: 5.0336
Epoch 89/100 | Loss: 5.0280
Epoch 90/100 | Loss: 5.0220
--- Evaluating at Epoch 90 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.57it/s]
Metrics -> MRR: 0.1759 | Hits@1: 0.0885 | Hits@3: 0.2162 | Hits@10: 0.3343
Epoch 91/100 | Loss: 5.0139
Epoch 92/100 | Loss: 5.0073
Epoch 93/100 | Loss: 5.0040
Epoch 94/100 | Loss: 4.9921
Epoch 95/100 | Loss: 4.9877
--- Evaluating at Epoch 95 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.51it/s]
Metrics -> MRR: 0.1714 | Hits@1: 0.0789 | Hits@3: 0.2159 | Hits@10: 0.3398
Epoch 96/100 | Loss: 4.9842
Epoch 97/100 | Loss: 4.9822
Epoch 98/100 | Loss: 4.9726
Epoch 99/100 | Loss: 4.9642
Epoch 100/100 | Loss: 4.9652
--- Evaluating at Epoch 100 ---
Pre-computing fused representations for all 93k entities...
Evaluating: 100%|█████████████████████████████████| 6/6 [00:00<00:00,  8.56it/s]
Metrics -> MRR: 0.1734 | Hits@1: 0.0819 | Hits@3: 0.2157 | Hits@10: 0.3354
```
