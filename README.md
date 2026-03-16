# PushT with Hierarchical Recursive Models (HRM/TRM)

## Overview

This project applies **discrete recursive reasoning models** — specifically **TRM** (Tiny Recursive Models) and **HRM** (Hierarchical Reasoning Models) — to the **PushT** robotic manipulation task from the Diffusion Policy benchmark.

The key innovation is adapting models originally designed for discrete puzzle-solving (Sudoku, Mazes, ARC) to continuous robotic control through careful tokenization of continuous action/observation spaces.

## Task: PushT

PushT is a planar pushing task where an agent must push a T-shaped block to a target pose.

- **Observation**: 5D state — (agent_x, agent_y, block_x, block_y, block_angle)
- **Action**: 2D — (target_x, target_y) position commands
- **Metric**: Coverage score (intersection area / goal area), threshold at 95%
- **Diffusion Policy baseline**: ~0.75 mean score (state-based)

## Models

### TRM (Tiny Recursive Model)
- Single shared reasoning module (L-level) used for both low-level and high-level computation
- Recursive refinement: H_cycles outer loops × L_cycles inner loops
- Only backpropagates through the final cycle (memory efficient)
- ~1.4M parameters (h=256)

### HRM (Hierarchical Reasoning Model)
- **Separate** H-level (slow planning) and L-level (fast computation) modules
- H-level: abstract trajectory planning
- L-level: precise action computation
- ~2.5M parameters (h=256)

### GPT Baseline
- Standard causal transformer for autoregressive action prediction

## Key Design Decisions

### Tokenization
Continuous values are discretized into bins:
- **Uniform binning**: Divide value range into N equal bins
- **Mu-law encoding**: Logarithmic compression for better resolution near center

### Positional Encoding
Custom **dimension-aware positional encoding** that encodes:
1. Time step position (which timestep)
2. Dimension identity (which obs/action dimension)
3. Token type (observation vs action)

### Sequence Layout
Flat sequence: `[obs_0_x, obs_0_y, ..., obs_0_angle, obs_1_x, ..., act_0_x, act_0_y, ..., act_15_x, act_15_y]`
- Observation horizon: 2 steps × 5 dims = 10 tokens
- Action prediction horizon: 16 steps × 2 dims = 32 tokens
- Total: 42 tokens

## Experiments

### Phase 1: Baselines
| Model | Params | Val Loss | Val Acc | Mean Score | Status |
|-------|--------|----------|---------|------------|--------|
| TRM (h=256) | 1.4M | - | - | - | Training |
| HRM (h=256) | 2.5M | - | - | - | Training |
| GPT (h=256) | - | - | - | - | Training |

### Phase 2: Tokenizer Ablation
| Bins | Type | Model | Score | Notes |
|------|------|-------|-------|-------|
| 128 | uniform | TRM | - | Pending |
| 256 | uniform | TRM | - | Baseline |
| 512 | uniform | TRM | - | Pending |
| 1024 | uniform | TRM | - | Pending |
| 256 | mulaw | TRM | - | Pending |

### Phase 3: Positional Encoding
| Type | Score | Notes |
|------|-------|-------|
| dim_aware | - | Baseline (time + dim + type embeddings) |
| rope | - | Rotary positional encoding |
| learned | - | Standard learned positional encoding |
| none | - | No positional encoding |

### Phase 4: Model Size
| Hidden | Params | Score | Notes |
|--------|--------|-------|-------|
| 128 | - | - | Small |
| 256 | 1.4M | - | Baseline |
| 384 | - | - | Medium |
| 512 | - | - | Large |

### Phase 5: Recursion Depth
| H_cycles | L_cycles | Score | Notes |
|----------|----------|-------|-------|
| 1 | 2 | - | Minimal recursion |
| 3 | 4 | - | Baseline |
| 5 | 6 | - | Deep recursion |
| 8 | 8 | - | Very deep |

## Setup

```bash
# Create environment
conda create -n vla_hrm python=3.10 -y
conda activate vla_hrm
pip install -r requirements.txt

# Download PushT data
cd diffusion_policy && mkdir -p data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -O data/pusht.zip
cd data && unzip pusht.zip

# Train TRM
python train.py --model_type trm --hidden_size 256 --num_bins 256

# Train HRM
python train.py --model_type hrm --hidden_size 256 --num_bins 256
```

## References

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Chi et al.
- [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) - Samsung SAIL Montreal
- [HRM](https://github.com/sapientinc/HRM) - Sapient Inc

## Wandb

Training tracked at: [wandb project](https://wandb.ai/ejshin0310-korea-advanced-institute-of-science-and-techn/pusht-hrm)
