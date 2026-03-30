#!/bin/bash
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

GPU=0
COMMON="--epochs 3000 --eval_interval 50 --eval_episodes 50 --wandb_entity junha --gpu $GPU"

# Exp 1: HRM h=256, bins=256, geo features (proven config)
CUDA_VISIBLE_DEVICES=$GPU nohup python train_final.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --H_layers 3 --L_layers 3 \
    --num_bins 256 --soft_sigma 3.0 --soft_decode_temp 0.3 \
    --action_horizon 8 --obs_noise_std 3.0 --act_noise_std 1.0 \
    --exp_name final_hrm_h256 \
    $COMMON > logs/final_hrm_h256.log 2>&1 &

# Exp 2: HRM h=384, bins=512 (larger model)
CUDA_VISIBLE_DEVICES=$GPU nohup python train_final.py \
    --model_type hrm --hidden_size 384 --num_heads 12 \
    --H_cycles 5 --L_cycles 6 --H_layers 3 --L_layers 3 \
    --num_bins 512 --soft_sigma 4.0 --soft_decode_temp 0.3 \
    --action_horizon 8 --obs_noise_std 3.0 --act_noise_std 1.0 \
    --exp_name final_hrm_h384 \
    $COMMON > logs/final_hrm_h384.log 2>&1 &

# Exp 3: TRM h=256 (simpler architecture comparison)
CUDA_VISIBLE_DEVICES=$GPU nohup python train_final.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --L_layers 3 \
    --num_bins 256 --soft_sigma 3.0 --soft_decode_temp 0.3 \
    --action_horizon 8 --obs_noise_std 3.0 --act_noise_std 1.0 \
    --exp_name final_trm_h256 \
    $COMMON > logs/final_trm_h256.log 2>&1 &

echo "3 final experiments launched on GPU $GPU"
