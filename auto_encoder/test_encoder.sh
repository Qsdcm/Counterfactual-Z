cd /data/data54/wanghaobo/Counterfactual-Z/auto-encoder
python visualize_results.py \
  --ckpt_path ae_results/checkpoints/best_ae.pth \
  --data_root /data/data54/wanghaobo/data/Third_full/ \
  --domain_mode both \
  --sequence_type all \
  --num_samples 5 \
  --gpu_id 4