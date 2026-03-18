#!/bin/bash
# V9: Everything combined - VLA best practices + debug findings
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

GPU=0

# V9 Exp 1: HRM + 1000 bins (VLA-0 resolution) + Gaussian soft + masked augmentation
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v6.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --H_layers 3 --L_layers 3 \
    --num_bins 1000 --loss_type gaussian_soft --soft_sigma 5.0 \
    --soft_decode_temp 0.3 \
    --use_regression_head --regression_weight 0.5 \
    --obs_noise_std 3.0 --act_noise_std 1.0 \
    --lr 1e-4 --batch_size 256 --action_horizon 8 \
    --epochs 2000 --warmup_epochs 30 \
    --eval_interval 50 --eval_episodes 22 \
    --wandb_entity junha \
    --exp_name v9_hrm_bins1000_gauss_s5 \
    --gpu $GPU > logs/v9_hrm_1000.log 2>&1 &

# V9 Exp 2: HRM + 512 bins + quantile normalization (pre-clip at 1-99 pctile)
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v6.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --H_layers 3 --L_layers 3 \
    --num_bins 512 --loss_type gaussian_soft --soft_sigma 4.0 \
    --soft_decode_temp 0.3 \
    --use_regression_head --regression_weight 0.3 \
    --obs_noise_std 3.0 --act_noise_std 1.0 \
    --lr 1e-4 --batch_size 256 --action_horizon 8 \
    --epochs 2000 --warmup_epochs 30 \
    --eval_interval 50 --eval_episodes 22 \
    --wandb_entity junha \
    --exp_name v9_hrm_bins512_gauss_s4 \
    --gpu $GPU > logs/v9_hrm_512.log 2>&1 &

echo "2 V9 experiments launched on GPU $GPU"
