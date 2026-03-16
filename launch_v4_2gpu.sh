#!/bin/bash
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

ENTITY="junha"
ZARR="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
COMMON="--obs_horizon 2 --pred_horizon 16 --action_horizon 8 --batch_size 256 --epochs 1000 --warmup_epochs 20 --eval_interval 25 --eval_episodes 50 --obs_noise_std 2.0 --wandb_entity $ENTITY --zarr_path $ZARR"

# Exp 1: Gaussian soft labels (sigma=2) + soft decode - TRM - GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_gauss_s2_bins256 \
    $COMMON --gpu 0 > logs/v4_trm_gauss.log 2>&1 &

# Exp 2: Gaussian soft labels + HRM - GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python train_v4.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --H_layers 2 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_hrm_gauss_s2_bins256 \
    $COMMON --gpu 0 > logs/v4_hrm_gauss.log 2>&1 &

echo "2 V4 experiments launched (GPUs 0, 1 only)"
