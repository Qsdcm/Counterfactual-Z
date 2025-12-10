cd /data/data54/wanghaobo/Counterfactual-Z/Second_encoder/
python train.py \
  --data_root /data/data54/wanghaobo/data/Third_full/ \
  --out_dir paired_results \
  --sequence_type all \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 1e-4 \
  --num_workers 8 \
  --gpu_id 0