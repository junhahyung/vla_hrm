#!/bin/bash
# Launch all experiments for paper reproduction on GPU 2
# Runs sequentially to use 1 GPU

set -e
source /home/junhahyung/anaconda3/etc/profile.d/conda.sh
conda activate /home/nas5/junhahyung/conda_envs/vla_hrm

export CUDA_VISIBLE_DEVICES=2
export SDL_VIDEODRIVER=dummy
export TMPDIR=/home/nas5/junhahyung/tmp

cd /home/nas5/junhahyung/vla_hrm

ZARR="/home/nas5/junhahyung/vla_hrm/data/pusht/pusht_cchi_v7_replay.zarr"
GPU=0  # GPU 0 within CUDA_VISIBLE_DEVICES=2

echo "============================================"
echo "Paper Experiments: Much Ado About Noising"
echo "All on PushT with keypoint obs (20D)"
echo "GPU: 2, Started: $(date)"
echo "============================================"

# ---- Table 1: Main Comparison ----
# All methods with same hidden=256, n_layers=4 (except HRM/TRM which use 3 layers x cycles)

echo ""
echo "=== Experiment 1/6: Regression (RCP) ==="
python run_all_experiments.py --method regression --hidden 256 --n_layers 4 \
    --epochs 500 --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix main 2>&1 | tee logs/regression_main.log

echo ""
echo "=== Experiment 2/6: Flow Matching ==="
python run_all_experiments.py --method flow --hidden 256 --n_layers 4 \
    --epochs 500 --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix main 2>&1 | tee logs/flow_main.log

echo ""
echo "=== Experiment 3/6: MIP (2-step) ==="
python run_all_experiments.py --method mip --hidden 256 --n_layers 4 \
    --epochs 500 --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix main 2>&1 | tee logs/mip_main.log

echo ""
echo "=== Experiment 4/6: Diffusion (100 steps) ==="
python run_all_experiments.py --method diffusion --hidden 256 --n_layers 4 \
    --epochs 500 --gpu $GPU --zarr_path $ZARR --seed 42 --num_denoise_steps 100 --exp_suffix main 2>&1 | tee logs/diffusion_main.log

echo ""
echo "=== Experiment 5/6: HRM (ours) ==="
python run_all_experiments.py --method hrm --hidden 256 --n_layers 3 \
    --H_cycles 5 --L_cycles 6 --epochs 500 --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix main 2>&1 | tee logs/hrm_main.log

echo ""
echo "=== Experiment 6/6: TRM (shared weights) ==="
python run_all_experiments.py --method trm --hidden 256 --n_layers 3 \
    --H_cycles 5 --L_cycles 6 --epochs 500 --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix main 2>&1 | tee logs/trm_main.log

echo ""
echo "============================================"
echo "Main comparison table complete: $(date)"
echo "============================================"

# ---- Table 2: Ablation - H/L Cycles ----
echo ""
echo "=== Ablation: HRM Cycle Counts ==="

for H in 1 2 3 5 8; do
    for L in 1 2 4 6; do
        echo "--- HRM H=${H} L=${L} ---"
        python run_all_experiments.py --method hrm --hidden 256 --n_layers 3 \
            --H_cycles $H --L_cycles $L --epochs 300 --eval_interval 50 \
            --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix "ablation_cycles" 2>&1 | tee logs/hrm_H${H}_L${L}.log
    done
done

echo ""
echo "============================================"
echo "Cycle ablation complete: $(date)"
echo "============================================"

# ---- Table 3: Scaling - Hidden Size ----
echo ""
echo "=== Ablation: Hidden Size Scaling ==="

for H_SIZE in 64 128 256 384; do
    echo "--- Regression h=${H_SIZE} ---"
    python run_all_experiments.py --method regression --hidden $H_SIZE --n_layers 4 \
        --epochs 300 --eval_interval 50 --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix "scale" 2>&1 | tee logs/regression_h${H_SIZE}.log

    echo "--- HRM h=${H_SIZE} ---"
    python run_all_experiments.py --method hrm --hidden $H_SIZE --n_layers 3 \
        --H_cycles 5 --L_cycles 6 --epochs 300 --eval_interval 50 \
        --gpu $GPU --zarr_path $ZARR --seed 42 --exp_suffix "scale" 2>&1 | tee logs/hrm_h${H_SIZE}.log
done

echo ""
echo "============================================"
echo "Scaling ablation complete: $(date)"
echo "============================================"

# ---- Table 4: Noise Ablation ----
echo ""
echo "=== Ablation: Observation Noise ==="

for NOISE in 0.0 0.5 1.0 2.0 3.0 5.0; do
    echo "--- HRM noise_std=${NOISE} ---"
    python run_all_experiments.py --method hrm --hidden 256 --n_layers 3 \
        --H_cycles 5 --L_cycles 6 --noise_std $NOISE \
        --epochs 300 --eval_interval 50 --gpu $GPU --zarr_path $ZARR --seed 42 \
        --exp_suffix "noise${NOISE}" 2>&1 | tee logs/hrm_noise${NOISE}.log
done

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "============================================"
