#全部数据进行训练
cd /data/data54/wanghaobo/Counterfactual-Z/auto-encoder
python train_encoder.py \
  --data_root /data/data54/wanghaobo/data/Third_full/ \
  --domain_mode both \
  --sequence_type all \
  --hidden_dims 16 32 64 16 \
  --batch_size 32 \
  --num_workers 16 \
  --num_epochs 300 \
  --lr 0.001 \
  --gpu_id 4 \
  --preload

#示例：只用 高场 + FSE + SE 两个序列
# python auto_encoder.py \
#   --data_root /data/data54/wanghaobo/data/Third_full/ \
#   --domain_mode high \
#   --seq_list fse se \
#   --hidden_dims 16 32 64 16 \
#   --batch_size 4 \
#   --num_epochs 50 \
#   --gpu_id 1

# python train_encoder.py \
#   --data_root /data/data54/wanghaobo/data/Third_full/ \
#   --domain_mode both \   # low / high / both
#   --sequence_type all \  # fse / se / irfse / all
#   --hidden_dims 16 32 64 16 \
#   --batch_size 16 \
#   --num_epochs 100 \
#   --gpu_id 4

