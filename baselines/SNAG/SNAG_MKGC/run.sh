#  GPU | DATA | num_proj | use_intermediate | joint_way | noise ratio (0~1) | mask ratio (0~1) | noise-level (Epoch/Step) | num_hidden_layers | num_attention_heads | Exp ID

# # DATA=DB15K / MKG-W / MKG-Y
# num_proj: 1 / 2
# use_intermediate: 0 / 1
# joint_way: "Mformer_hd_mean" / "Mformer_hd_graph" / "Mformer_weight" / "atten_weight" / "learnable_weight"
# noise ratio: 0 ~ 1
# mask ratio: 0 ~ 1
# noise-level: epoch / step

bash scripts/run_base.sh 0 DB15K 2 0 Mformer_hd_graph 0.2 0.7 epoch 1 2 K001
bash scripts/run_base.sh 0 MKG-Y 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 Y001
bash scripts/run_base.sh 0 MKG-W 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 W001
