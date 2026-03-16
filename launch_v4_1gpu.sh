#!/bin/bash
# V4 experiments - ALL on GPU 0, running in parallel (small models fit together)
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

ENTITY="junha"
ZARR="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
GPU=0
COMMON="--obs_horizon 2 --pred_horizon 16 --action_horizon 8 --batch_size 256 --epochs 1000 --warmup_epochs 20 --eval_interval 25 --eval_episodes 50 --obs_noise_std 2.0 --wandb_entity $ENTITY --zarr_path $ZARR --gpu $GPU"

# All experiments share GPU 0 (~1.5GB each, A100 has 40GB)

# Exp 1: Gaussian soft labels + TRM
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_gauss_s2_bins256 \
    $COMMON > logs/v4_trm_gauss.log 2>&1 &

# Exp 2: Gaussian soft labels + HRM
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v4.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --H_layers 2 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_hrm_gauss_s2_bins256 \
    $COMMON > logs/v4_hrm_gauss.log 2>&1 &

# Exp 3: Focal loss + 512 bins + TRM
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 512 --loss_type focal \
    --soft_decode_temp 0.3 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_focal_bins512 \
    $COMMON > logs/v4_trm_focal.log 2>&1 &

# Exp 4: Gaussian soft + delta tokenization + TRM
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 2.0 \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode delta \
    --exp_name v4_trm_gauss_delta \
    $COMMON > logs/v4_trm_delta.log 2>&1 &

# Exp 5: Distance-weighted CE + TRM
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v4.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --loss_type distance_weighted \
    --soft_decode_temp 0.5 --lr 1e-4 \
    --action_mode absolute \
    --exp_name v4_trm_distweight_bins256 \
    $COMMON > logs/v4_trm_distw.log 2>&1 &

echo "5 V4 experiments launched on GPU $GPU"
