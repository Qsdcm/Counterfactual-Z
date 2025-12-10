cd /data/data54/wanghaobo/Counterfactual-Z/Second_encoder/
python test.py \
  --data_root /data/data54/wanghaobo/data/Third_full/ \
  --ckpt_path paired_results/checkpoints/best_paired_model.pth \
  --out_dir paired_results \
  --sequence_type all \
  --gpu_id 0