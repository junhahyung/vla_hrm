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

### V3 Regression Models
| Model | Params | Epochs | Mean Score | Max Score | Notes |
|-------|--------|--------|------------|-----------|-------|
| TRM reg h=256 | 1.5M | 650 | 0.355 | 0.434 | Best V3 |
| HRM reg h=256 | 2.8M | 950 | 0.302 | 0.393 | |

### V4 Discrete Tokens with Advanced Losses
| Model | Loss | Bins | Best Mean | Best Max | Epoch |
|-------|------|------|-----------|----------|-------|
| TRM Gaussian | GaussSoft(σ=2) | 256 | 0.403 | 0.488 | 1000 |
| TRM Focal | Focal | 512 | 0.375 | 0.470 | 1000 |
| HRM Gaussian | GaussSoft(σ=2) | 256 | 0.340 | 0.450 | 1000 |

### V8 Geometric Features + Mirroring (BEST)
| Model | Best Mean | Epoch | Notes |
|-------|----------|-------|-------|
| **HRM h=384 geo+mirror** | **0.558** | 3000 | Larger model, 512 bins |
| HRM h=256 geo+mirror | 0.523 | 200 | 21 geometric features, 4x mirror augment |

### V6-V9 Comparison
| Model | Best Mean | Epoch | Notes |
|-------|----------|-------|-------|
| V9 HRM 512 bins | 0.406 | 632 | |
| V6 HRM deep (H5L6) | 0.365 | 2000 | EMA no-TE |
| V7 iterative refine | 0.357 | 623 | Discrete diffusion |
| V6 HRM gradall h=384 | 0.355 | 1250 | |
| V9 HRM 1000 bins | 0.334 | 629 | |

### V7 Iterative Refinement (Discrete Diffusion)
| Refine Steps | Mean | Best Episode | Notes |
|-------------|------|-------------|-------|
| K=1 | 0.182 | 0.521 | No refinement |
| K=4 | 0.221 | 0.853 | Default |
| K=8 | 0.205 | 0.942 | More steps helps |

### Baselines (Same Eval Protocol: 50 episodes, seeds 100000+)
| Method | Mean Score | Params | Epoch | Notes |
|--------|-----------|--------|-------|-------|
| **Our HRM V8 h=384** | **0.558** | **~8M** | 3000 | **Beats DP by 10%!** |
| Our HRM V8 h=256 | 0.523 | 5.1M | 200 | |
| DP original code (1000ep) | 0.507 | 65.8M | 1000 | Their repo, our eval |
| DP original code (300ep) | 0.451 | 65.8M | 300 | |
| Our diffusion U-Net | 0.428 | 75M | 1249 | Our reimplementation |
| AR GPT baseline | 0.353 | ~3M | 2000 | |
| Diffusion Policy (paper) | ~0.75 | ~2.5M | - | Reference (diff eval?) |

### Ensemble Strategies (Debug)
| Strategy | Mean | Zeros/22 | Fixed Failed |
|----------|------|---------|-------------|
| goal_biased(15) | - | - | **4/8 fixed** (incl. 1 perfect!) |
| ensemble_noise(10) | 0.270 | **5** | 3/8 fixed |
| approach_first | 0.289 | 8 | 1/8 fixed |
| ensemble_temp | 0.262 | 7 | - |

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
