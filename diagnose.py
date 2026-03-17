"""
Diagnose which initial configurations the model fails on and why.
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import json
import zarr

from pusht_hrm.model_v6 import build_model_v6
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv


def analyze_episodes(model, dataset_info, device, num_episodes=22, seed_start=10000):
    """Run evaluation and log detailed per-episode info."""
    env = PushTEnv()
    results = []

    for ep in range(num_episodes):
        seed = seed_start + ep
        env.seed(seed)
        obs = env.reset()
        initial_obs = obs.copy()

        obs_history = [obs]
        episode_scores = []
        episode_actions = []
        done = False
        step_count = 0

        while not done and step_count < 200:
            while len(obs_history) < dataset_info["obs_horizon"]:
                obs_history.insert(0, obs_history[0])

            recent_obs = np.array(obs_history[-dataset_info["obs_horizon"]:])
            obs_norm = (recent_obs - dataset_info["obs_mean"]) / dataset_info["obs_std"]
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                out = model.forward(obs_t)
                logits = out["cls_logits"]
                # Soft decode
                probs = torch.softmax(logits / 0.5, dim=-1)
                bins = torch.arange(logits.shape[-1], dtype=torch.float32, device=device)
                pred = (probs * bins).sum(dim=-1)

            pred_np = pred.cpu().numpy()[0]
            pred_reshaped = pred_np.reshape(dataset_info["pred_horizon"], 2)
            actions = np.zeros_like(pred_reshaped)
            for d in range(2):
                tok = dataset_info["tokenizers"][d]
                actions[:, d] = tok.val_min + (pred_reshaped[:, d] + 0.5) * tok.bin_width

            for t in range(min(dataset_info["action_horizon"], len(actions))):
                action = np.clip(actions[t], 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(float(score))
                episode_actions.append(action.tolist())
                step_count += 1
                if done:
                    break

        max_score = max(episode_scores) if episode_scores else 0

        results.append({
            "seed": seed,
            "initial_obs": initial_obs.tolist(),
            "initial_agent_pos": initial_obs[:2].tolist(),
            "initial_block_pos": initial_obs[2:4].tolist(),
            "initial_block_angle": float(initial_obs[4]),
            "max_score": float(max_score),
            "final_score": float(episode_scores[-1]) if episode_scores else 0,
            "num_steps": step_count,
            "first_5_actions": episode_actions[:5] if episode_actions else [],
            "score_at_50": float(episode_scores[49]) if len(episode_scores) > 49 else 0,
            "score_at_100": float(episode_scores[99]) if len(episode_scores) > 99 else 0,
        })

    return results


def analyze_training_data(zarr_path):
    """Analyze training data distribution."""
    root = zarr.open(zarr_path, "r")
    state = root["data/state"][:]
    action = root["data/action"][:]
    episode_ends = root["meta/episode_ends"][:]

    # Initial states of each episode
    starts = np.concatenate([[0], episode_ends[:-1]])
    initial_states = state[starts]

    return {
        "num_episodes": len(episode_ends),
        "initial_agent_x": {"mean": float(initial_states[:, 0].mean()), "std": float(initial_states[:, 0].std()),
                           "min": float(initial_states[:, 0].min()), "max": float(initial_states[:, 0].max())},
        "initial_agent_y": {"mean": float(initial_states[:, 1].mean()), "std": float(initial_states[:, 1].std()),
                           "min": float(initial_states[:, 1].min()), "max": float(initial_states[:, 1].max())},
        "initial_block_x": {"mean": float(initial_states[:, 2].mean()), "std": float(initial_states[:, 2].std()),
                           "min": float(initial_states[:, 2].min()), "max": float(initial_states[:, 2].max())},
        "initial_block_y": {"mean": float(initial_states[:, 3].mean()), "std": float(initial_states[:, 3].std()),
                           "min": float(initial_states[:, 3].min()), "max": float(initial_states[:, 3].max())},
        "initial_block_angle": {"mean": float(initial_states[:, 4].mean()), "std": float(initial_states[:, 4].std()),
                               "min": float(initial_states[:, 4].min()), "max": float(initial_states[:, 4].max())},
        "all_state_range": {
            "min": state.min(axis=0).tolist(),
            "max": state.max(axis=0).tolist(),
            "mean": state.mean(axis=0).tolist(),
            "std": state.std(axis=0).tolist(),
        },
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--zarr_path", type=str,
                   default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load data stats
    root = zarr.open(args.zarr_path, "r")
    state = root["data/state"][:]
    action = root["data/action"][:]
    obs_mean = state.mean(axis=0)
    obs_std = state.std(axis=0) + 1e-6
    act_min = action.min(axis=0)
    act_max = action.max(axis=0)
    act_ranges = [(act_min[i], act_max[i]) for i in range(2)]
    act_tokenizer = PerDimTokenizer(256, act_ranges, "uniform")

    dataset_info = {
        "obs_horizon": 2, "pred_horizon": 16, "action_horizon": 8,
        "obs_mean": obs_mean, "obs_std": obs_std,
        "tokenizers": [act_tokenizer.tokenizers[i] for i in range(2)],
    }

    # Try to load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Build model (try to infer config)
    config = {
        "model_type": "hrm", "vocab_size": 256, "hidden_size": 256,
        "num_heads": 8, "expansion": 2.0, "H_layers": 3, "L_layers": 3,
        "H_cycles": 5, "L_cycles": 6, "obs_horizon": 2, "pred_horizon": 16,
        "obs_dim": 5, "act_dim": 2, "act_seq_len": 32,
        "loss_type": "gaussian_soft", "soft_sigma": 3.0, "soft_decode_temp": 0.5,
        "forward_dtype": "bfloat16",
    }
    model = build_model_v6(config).to(device)
    model.load_state_dict(state_dict, strict=False)

    # Analyze training data
    print("=== Training Data Analysis ===")
    train_stats = analyze_training_data(args.zarr_path)
    print(json.dumps(train_stats, indent=2))

    # Analyze per-episode performance
    print("\n=== Per-Episode Analysis ===")
    results = analyze_episodes(model, dataset_info, device)

    # Sort by score
    results.sort(key=lambda x: x["max_score"])

    print("\n--- WORST episodes (score = 0) ---")
    for r in results:
        if r["max_score"] < 0.05:
            print(f"  seed={r['seed']}: agent=({r['initial_agent_pos'][0]:.0f},{r['initial_agent_pos'][1]:.0f}), "
                  f"block=({r['initial_block_pos'][0]:.0f},{r['initial_block_pos'][1]:.0f}), "
                  f"angle={r['initial_block_angle']:.2f}, score={r['max_score']:.3f}")

    print("\n--- BEST episodes ---")
    for r in results[-5:]:
        print(f"  seed={r['seed']}: agent=({r['initial_agent_pos'][0]:.0f},{r['initial_agent_pos'][1]:.0f}), "
              f"block=({r['initial_block_pos'][0]:.0f},{r['initial_block_pos'][1]:.0f}), "
              f"angle={r['initial_block_angle']:.2f}, score={r['max_score']:.3f}")

    # Statistical comparison
    good = [r for r in results if r["max_score"] > 0.3]
    bad = [r for r in results if r["max_score"] < 0.05]

    print(f"\n--- Statistics ---")
    print(f"Good episodes (>0.3): {len(good)}/{len(results)}")
    print(f"Bad episodes (<0.05): {len(bad)}/{len(results)}")

    if good and bad:
        good_agents = np.array([r["initial_agent_pos"] for r in good])
        bad_agents = np.array([r["initial_agent_pos"] for r in bad])
        good_blocks = np.array([r["initial_block_pos"] for r in good])
        bad_blocks = np.array([r["initial_block_pos"] for r in bad])

        print(f"\nGood agent pos: mean=({good_agents[:,0].mean():.0f},{good_agents[:,1].mean():.0f})")
        print(f"Bad agent pos:  mean=({bad_agents[:,0].mean():.0f},{bad_agents[:,1].mean():.0f})")
        print(f"Good block pos: mean=({good_blocks[:,0].mean():.0f},{good_blocks[:,1].mean():.0f})")
        print(f"Bad block pos:  mean=({bad_blocks[:,0].mean():.0f},{bad_blocks[:,1].mean():.0f})")

        good_angles = np.array([r["initial_block_angle"] for r in good])
        bad_angles = np.array([r["initial_block_angle"] for r in bad])
        print(f"Good block angle: mean={good_angles.mean():.2f}, std={good_angles.std():.2f}")
        print(f"Bad block angle:  mean={bad_angles.mean():.2f}, std={bad_angles.std():.2f}")

    # Save full results
    with open("diagnosis_results.json", "w") as f:
        json.dump({"train_stats": train_stats, "episodes": results}, f, indent=2)
    print("\nFull results saved to diagnosis_results.json")
