# PushT with Hierarchical Recursive Models (HRM/TRM)

## Overview

This project applies **recursive reasoning models** — **TRM** (Tiny Recursive Models) and **HRM** (Hierarchical Reasoning Models) — to the **PushT** robotic manipulation task from Diffusion Policy.

These models, originally designed for discrete puzzles (Sudoku, Mazes, ARC-AGI), use iterative refinement via recursive weight-shared computation. We adapt them for continuous robotic control through:
1. Continuous observation encoding (MLP encoder)
2. Regression-based action prediction
3. Learned action query tokens for parallel action decoding

## Task: PushT

- **Observation**: 5D state — (agent_x, agent_y, block_x, block_y, block_angle)
- **Action**: 2D — (target_x, target_y) position commands
- **Metric**: Coverage score (intersection area / goal area)
- **Diffusion Policy baseline**: ~0.75 mean score (state-based)

## Architecture Evolution

### V1: Fully Discrete (Tokenized Obs + Actions)
- Both observations and actions tokenized into discrete bins
- Cross-entropy loss on action tokens
- Result: 100% train accuracy but poor env performance (~0.06 mean score)
- **Problem**: Memorizes training sequences, doesn't generalize

### V2: Hybrid (Continuous Obs + Discrete Actions)
- MLP observation encoder + tokenized actions
- Better generalization (~0.15 mean score)

### V3: Fully Continuous (Current Best)
- MLP observation encoder + regression action output
- MSE loss on normalized continuous actions
- **Best result: 0.34 max score, 0.26 mean score (HRM)**

## Model Architectures

### TRM (Tiny Recursive Model)
- Single shared reasoning module (L-level) for both low and high-level computation
- Recursive: H_cycles outer × L_cycles inner iterations
- Only backprops through final cycle (memory efficient)
- ~1.5M parameters (h=256)

### HRM (Hierarchical Reasoning Model) — Best
- **Separate** H-level (slow planning) and L-level (fast computation) modules
- H-level learns abstract trajectory planning
- L-level handles precise action computation
- ~2.8M parameters (h=256)

## Results

### V3 Regression Models (Current Best)
| Model | Params | Epochs | Mean Score | Max Score | Notes |
|-------|--------|--------|------------|-----------|-------|
| **TRM reg h=256** | **1.5M** | **375** | **0.326** | **0.420** | **Best overall** |
| HRM reg h=256 | 2.8M | 550 | 0.302 | 0.393 | Strong |
| TRM reg h=512 | 7.5M | 125 | 0.284 | 0.355 | Still training |
| HRM deep h=384 (H5L6) | ~6M | 75 | 0.208 | 0.280 | Still improving |
| TRM deep (H5L6) ah=2 | 2.2M | 50 | 0.091 | 0.143 | Deep recursion slow |

### V1 Tokenized Models (Baseline)
| Model | Mean Score | Max Score | Notes |
|-------|------------|-----------|-------|
| TRM tokenized | 0.061 | 0.067 | 100% train acc, poor generalization |
| HRM tokenized | 0.045 | 0.059 | Same issue |

### Key Findings
1. **HRM > TRM**: Separate H/L modules significantly outperform weight-shared TRM
2. **Regression > Classification**: Continuous outputs avoid quantization errors
3. **Continuous obs encoding critical**: Tokenizing observations destroys information
4. **Observation noise augmentation helps**: Prevents overfitting to exact training states
5. **Action horizon matters**: Shorter execution horizons (4-8) with re-planning helps

## Setup

```bash
conda create -n vla_hrm python=3.10 -y
conda activate vla_hrm
pip install -r requirements.txt

# Download PushT data
cd diffusion_policy && mkdir -p data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -O data/pusht.zip
cd data && unzip pusht.zip && cd ../..

# Train V3 HRM (best)
python train_v3.py --model_type hrm --hidden_size 256 --output_mode regression \
    --obs_noise_std 2.0 --wandb_entity junha

# Train V3 TRM
python train_v3.py --model_type trm --hidden_size 256 --output_mode regression \
    --obs_noise_std 2.0 --wandb_entity junha
```

## Wandb

Training tracked at: [wandb.ai/junha/pusht-hrm](https://wandb.ai/junha/pusht-hrm)

## References

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) — Chi et al., RSS 2023
- [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) — Samsung SAIL Montreal
- [HRM](https://github.com/sapientinc/HRM) — Sapient Inc
