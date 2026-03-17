#!/bin/bash
# V6: All-out experiments to beat diffusion policy
# Run 3 experiments in parallel on GPU 0 (each ~2-4GB)
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

GPU=0
COMMON="--wandb_entity junha --gpu $GPU --epochs 2000 --eval_interval 50 --eval_episodes 22 --temporal_ensemble"

# Exp 1: HRM h=256, Gaussian soft σ=3, dual loss (cls+reg), deep recursion
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v6.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --H_layers 3 --L_layers 3 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 3.0 \
    --soft_decode_temp 0.5 \
    --use_regression_head --regression_weight 0.5 \
    --obs_noise_std 3.0 --act_noise_std 1.0 \
    --lr 1e-4 --batch_size 256 --action_horizon 8 \
    --exp_name v6_hrm_deep_dual \
    $COMMON > logs/v6_hrm_deep.log 2>&1 &

# Exp 2: HRM h=384, Gaussian soft, grad all steps with checkpointing
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v6.py \
    --model_type hrm --hidden_size 384 --num_heads 12 \
    --H_cycles 3 --L_cycles 4 --H_layers 2 --L_layers 2 \
    --num_bins 256 --loss_type gaussian_soft --soft_sigma 3.0 \
    --soft_decode_temp 0.5 \
    --use_regression_head --regression_weight 0.3 \
    --grad_all_steps --use_checkpoint \
    --obs_noise_std 3.0 --act_noise_std 1.0 \
    --lr 5e-5 --batch_size 128 --action_horizon 8 \
    --exp_name v6_hrm_h384_gradall \
    $COMMON > logs/v6_hrm_gradall.log 2>&1 &

# Exp 3: TRM h=256, Focal + regression, shorter action horizon
CUDA_VISIBLE_DEVICES=$GPU nohup python train_v6.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --L_layers 3 \
    --num_bins 512 --loss_type focal \
    --soft_decode_temp 0.3 \
    --use_regression_head --regression_weight 0.5 \
    --obs_noise_std 3.0 --act_noise_std 1.0 \
    --lr 1e-4 --batch_size 256 --action_horizon 4 \
    --exp_name v6_trm_focal512_ah4 \
    $COMMON > logs/v6_trm_focal.log 2>&1 &

echo "3 V6 experiments launched on GPU $GPU"
