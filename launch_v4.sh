#!/bin/bash
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

ENTITY="junha"
ZARR="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
COMMON="--obs_horizon 2 --pred_horizon 16 --action_horizon 8 --batch_size 256 --epochs 1000 --warmup_epochs 20 --eval_interval 25 --eval_episodes 50 --obs_noise_std 2.0 --wandb_entity $ENTITY --zarr_path $ZARR"

# ============================================================
# Exp 1: Gaussian soft labels (sigma=2) + soft decode - TRM - GPU 0
# Key idea: neighboring bins get credit, soft decode averages bin centers
# ============================================================
CUDA_VISIBLE_DEVICES=0 nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_gauss_s2_bins256 \
    $COMMON --gpu 0 > logs/v4_trm_gauss.log 2>&1 &

# ============================================================
# Exp 2: Gaussian soft labels + HRM - GPU 1
# ============================================================
CUDA_VISIBLE_DEVICES=1 nohup python train_v4.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --H_layers 2 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_hrm_gauss_s2_bins256 \
    $COMMON --gpu 0 > logs/v4_hrm_gauss.log 2>&1 &

# ============================================================
# Exp 3: Gaussian soft + more bins (512) - TRM - GPU 2
# Finer resolution with soft decode should work well
# ============================================================
CUDA_VISIBLE_DEVICES=2 nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 512 --loss_type gaussian_soft --soft_sigma 3.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_gauss_s3_bins512 \
    $COMMON --gpu 0 > logs/v4_trm_gauss512.log 2>&1 &

# ============================================================
# Exp 4: Delta tokenization + Gaussian soft - TRM - GPU 6
# Tokenize action changes, not absolute values
# ============================================================
CUDA_VISIBLE_DEVICES=6 nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode delta \
    --exp_name v4_trm_gauss_delta_bins256 \
    $COMMON --gpu 0 > logs/v4_trm_delta.log 2>&1 &

# ============================================================
# Exp 5: Focal loss + fine bins (1024) - TRM - GPU 7
# Focus on hard examples, very fine resolution
# ============================================================
CUDA_VISIBLE_DEVICES=7 nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 1024 --loss_type focal \
    --soft_decode_temp 0.3 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_focal_bins1024 \
    $COMMON --gpu 0 > logs/v4_trm_focal1024.log 2>&1 &

echo "All 5 V4 experiments launched"
