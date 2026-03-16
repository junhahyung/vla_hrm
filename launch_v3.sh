#!/bin/bash
set -e
cd /home/junhahyung/vla_hrm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla_hrm
mkdir -p logs checkpoints

ENTITY="junha"
ZARR="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"

# V3 TRM regression - GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python train_v3.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --output_mode regression \
    --obs_horizon 2 --pred_horizon 16 --action_horizon 8 \
    --epochs 1000 --batch_size 256 --lr 1e-4 \
    --obs_noise_std 2.0 --warmup_epochs 20 \
    --eval_interval 25 --eval_episodes 50 \
    --wandb_entity $ENTITY --zarr_path $ZARR \
    --exp_name v3_trm_reg_h256_ah8 \
    --gpu 0 > logs/v3_trm_reg.log 2>&1 &

# V3 HRM regression - GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python train_v3.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --H_layers 2 --L_layers 2 \
    --output_mode regression \
    --obs_horizon 2 --pred_horizon 16 --action_horizon 8 \
    --epochs 1000 --batch_size 256 --lr 1e-4 \
    --obs_noise_std 2.0 --warmup_epochs 20 \
    --eval_interval 25 --eval_episodes 50 \
    --wandb_entity $ENTITY --zarr_path $ZARR \
    --exp_name v3_hrm_reg_h256_ah8 \
    --gpu 0 > logs/v3_hrm_reg.log 2>&1 &

# V3 TRM deep recursion, very reactive (ah=2) - GPU 2
CUDA_VISIBLE_DEVICES=2 nohup python train_v3.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 5 --L_cycles 6 --L_layers 3 \
    --output_mode regression \
    --obs_horizon 2 --pred_horizon 16 --action_horizon 2 \
    --epochs 1000 --batch_size 256 --lr 1e-4 \
    --obs_noise_std 3.0 --warmup_epochs 20 \
    --eval_interval 25 --eval_episodes 50 \
    --wandb_entity $ENTITY --zarr_path $ZARR \
    --exp_name v3_trm_reg_deep_ah2 \
    --gpu 0 > logs/v3_trm_reactive.log 2>&1 &

# V3 TRM large model - GPU 7
CUDA_VISIBLE_DEVICES=7 nohup python train_v3.py \
    --model_type trm --hidden_size 512 --num_heads 16 \
    --H_cycles 3 --L_cycles 4 --L_layers 3 \
    --output_mode regression \
    --obs_horizon 2 --pred_horizon 16 --action_horizon 4 \
    --epochs 1000 --batch_size 256 --lr 5e-5 \
    --obs_noise_std 2.0 --warmup_epochs 20 \
    --eval_interval 25 --eval_episodes 50 \
    --wandb_entity $ENTITY --zarr_path $ZARR \
    --exp_name v3_trm_reg_h512_ah4 \
    --gpu 0 > logs/v3_trm_large.log 2>&1 &

echo "All 4 V3 experiments launched"
