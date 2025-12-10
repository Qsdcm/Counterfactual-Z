#!/bin/bash
# Run inference (counterfactual generation) on GPU 3
export CUDA_VISIBLE_DEVICES=3

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# Check if classifier checkpoint exists
if [ ! -f "results/best_classifier.pth" ]; then
    echo "Error: Classifier checkpoint not found at results/best_classifier.pth"
    echo "Please run train.sh first."
    exit 1
fi

/home/wanghaobo/.conda/envs/pt1.10/bin/python infer.py \
  --data_root /data/data54/wanghaobo/data/Third_full/ \
  --ae_checkpoint /data/data54/wanghaobo/Counterfactual-Z/auto_encoder/ae_results/checkpoints/best_ae.pth \
  --classifier_checkpoint results/best_classifier.pth \
  --out_dir results/inference \
  --target_domain high \
  --num_samples 10 \
  --gpu_id 0
