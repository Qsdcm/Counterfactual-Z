#!/bin/bash
# Train the classifier on GPU 3
export CUDA_VISIBLE_DEVICES=3

# Ensure we are in the correct directory
cd "$(dirname "$0")"

/home/wanghaobo/.conda/envs/pt1.10/bin/python train.py \
  --data_root /data/data54/wanghaobo/data/Third_full/ \
  --ae_checkpoint /data/data54/wanghaobo/Counterfactual-Z/auto_encoder/ae_results/checkpoints/best_ae.pth \
  --out_dir results \
  --batch_size 32 \
  --num_epochs 50 \
  --lr 1e-4 \
  --gpu_id 0
