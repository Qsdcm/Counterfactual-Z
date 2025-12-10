#!/bin/bash
# set -e 表示如果任何一行命令报错（返回非0状态码），整个脚本立刻停止，不往下执行
set -e 

export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="/data/data54/wanghaobo/Counterfactual-Z/Second_encoder"
DATA_ROOT="/data/data54/wanghaobo/data/Third_full/"
PYTHON_EXEC="/home/wanghaobo/.conda/envs/pt1.10/bin/python"

cd $PROJECT_DIR

echo "========== Start Training =========="
$PYTHON_EXEC train.py \
  --data_root $DATA_ROOT \
  --out_dir paired_results \
  --sequence_type all \
  --batch_size 16 \
  --num_epochs 100 \
  --gpu_id 0

# 只有上面那句 train.py 成功跑完没报错，才会执行下面这句
echo "========== Training Finished. Start Testing =========="

$PYTHON_EXEC test.py \
  --data_root $DATA_ROOT \
  --ckpt_path paired_results/checkpoints/best_paired_model.pth \
  --out_dir paired_results \
  --sequence_type all \
  --gpu_id 0