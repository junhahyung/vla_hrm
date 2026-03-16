#!/bin/bash
# Hyperparameter sweep for PushT TRM/HRM experiments
# Run on remote server with: bash run_sweep.sh

set -e

export CUDA_VISIBLE_DEVICES=0
ZARR_PATH="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"

# ============================================================
# Phase 1: Baseline experiments (TRM vs HRM vs GPT)
# ============================================================

echo "=== Phase 1: Baselines ==="

# TRM baseline
python train.py \
    --model_type trm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --L_layers 2 \
    --num_bins 256 --tokenizer_type uniform \
    --pos_encoding_type dim_aware \
    --epochs 500 --batch_size 256 --lr 1e-4 \
    --exp_name "baseline_trm_h256" \
    --zarr_path $ZARR_PATH --gpu 0

# HRM baseline
python train.py \
    --model_type hrm --hidden_size 256 --num_heads 8 \
    --H_cycles 3 --L_cycles 4 --H_layers 2 --L_layers 2 \
    --num_bins 256 --tokenizer_type uniform \
    --pos_encoding_type dim_aware \
    --epochs 500 --batch_size 256 --lr 1e-4 \
    --exp_name "baseline_hrm_h256" \
    --zarr_path $ZARR_PATH --gpu 0

# GPT baseline
python train.py \
    --model_type gpt --hidden_size 256 --num_heads 8 \
    --num_layers 4 \
    --num_bins 256 --tokenizer_type uniform \
    --pos_encoding_type dim_aware \
    --epochs 500 --batch_size 256 --lr 1e-4 \
    --exp_name "baseline_gpt_h256" \
    --zarr_path $ZARR_PATH --gpu 0

# ============================================================
# Phase 2: Tokenizer experiments
# ============================================================

echo "=== Phase 2: Tokenizer ==="

for BINS in 128 256 512 1024; do
    for TOK in uniform mulaw; do
        python train.py \
            --model_type trm --hidden_size 256 --num_heads 8 \
            --H_cycles 3 --L_cycles 4 --L_layers 2 \
            --num_bins $BINS --tokenizer_type $TOK \
            --pos_encoding_type dim_aware \
            --epochs 300 --batch_size 256 --lr 1e-4 \
            --exp_name "tok_trm_${TOK}_bins${BINS}" \
            --zarr_path $ZARR_PATH --gpu 0
    done
done

# ============================================================
# Phase 3: Positional encoding experiments
# ============================================================

echo "=== Phase 3: Positional Encoding ==="

for POS in dim_aware rope learned none; do
    python train.py \
        --model_type trm --hidden_size 256 --num_heads 8 \
        --H_cycles 3 --L_cycles 4 --L_layers 2 \
        --num_bins 256 --tokenizer_type uniform \
        --pos_encoding_type $POS \
        --epochs 300 --batch_size 256 --lr 1e-4 \
        --exp_name "pos_trm_${POS}" \
        --zarr_path $ZARR_PATH --gpu 0
done

# ============================================================
# Phase 4: Model size experiments
# ============================================================

echo "=== Phase 4: Model Size ==="

for HIDDEN in 128 256 384 512; do
    HEADS=$((HIDDEN / 32))
    python train.py \
        --model_type trm --hidden_size $HIDDEN --num_heads $HEADS \
        --H_cycles 3 --L_cycles 4 --L_layers 2 \
        --num_bins 256 --tokenizer_type uniform \
        --pos_encoding_type dim_aware \
        --epochs 300 --batch_size 256 --lr 1e-4 \
        --exp_name "size_trm_h${HIDDEN}" \
        --zarr_path $ZARR_PATH --gpu 0
done

# ============================================================
# Phase 5: Recursion depth experiments
# ============================================================

echo "=== Phase 5: Recursion Depth ==="

for HC in 1 2 3 5 8; do
    for LC in 2 4 6 8; do
        python train.py \
            --model_type trm --hidden_size 256 --num_heads 8 \
            --H_cycles $HC --L_cycles $LC --L_layers 2 \
            --num_bins 256 --tokenizer_type uniform \
            --pos_encoding_type dim_aware \
            --epochs 300 --batch_size 256 --lr 1e-4 \
            --exp_name "depth_trm_H${HC}_L${LC}" \
            --zarr_path $ZARR_PATH --gpu 0
    done
done

echo "=== All experiments complete ==="
