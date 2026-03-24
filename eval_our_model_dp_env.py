"""
Evaluate our HRM model using the EXACT DP evaluation environment.
Uses PushTKeypointsEnv with legacy=True, same seeds, same max_steps.
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_policy'))
import numpy as np
import torch
import json

# Our model
from pusht_hrm.model_v6 import build_model_v6
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.losses import soft_decode
from pusht_hrm.obs_features import compute_geometric_features
import zarr

# DP's environment (the exact one they use for eval)
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


def load_our_model(checkpoint_path, device, hidden_size=256, num_bins=256, use_geo=True):
    """Load our V8 HRM model."""
    zarr_path = "/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    root = zarr.open(zarr_path, "r")
    state = root["data/state"][:]
    action = root["data/action"][:]

    obs_mean = state.mean(axis=0)
    obs_std = state.std(axis=0) + 1e-6
    act_min = action.min(axis=0)
    act_max = action.max(axis=0)
    act_ranges = [(act_min[i], act_max[i]) for i in range(2)]
    act_tokenizer = PerDimTokenizer(num_bins, act_ranges, "uniform")

    obs_dim = 5 + 21 if use_geo else 5  # 5 raw + 21 geometric

    config = {
        "model_type": "hrm", "vocab_size": num_bins,
        "hidden_size": hidden_size, "num_heads": hidden_size // 32,
        "expansion": 2.0, "H_layers": 3, "L_layers": 3,
        "H_cycles": 5, "L_cycles": 6, "obs_horizon": 2, "pred_horizon": 16,
        "obs_dim": obs_dim, "act_dim": 2, "act_seq_len": 32,
        "loss_type": "gaussian_soft", "soft_sigma": 3.0 if num_bins == 256 else 4.0,
        "soft_decode_temp": 0.3,
        "use_regression_head": True, "regression_weight": 0.5,
        "forward_dtype": "bfloat16",
    }

    model = build_model_v6(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    return model, {
        "obs_mean": obs_mean, "obs_std": obs_std,
        "act_tokenizer": act_tokenizer, "act_min": act_min, "act_max": act_max,
        "obs_horizon": 2, "pred_horizon": 16, "num_bins": num_bins,
        "use_geo": use_geo,
    }


def get_state_from_keypoint_env(env):
    """Extract 5D state from the keypoint environment."""
    agent_pos = np.array(env.agent.position)
    block_pos = np.array(env.block.position)
    block_angle = env.block.angle % (2 * np.pi)
    return np.array([agent_pos[0], agent_pos[1], block_pos[0], block_pos[1], block_angle])


def build_obs(state_5d, data):
    """Build observation with optional geometric features."""
    if data["use_geo"]:
        geo = compute_geometric_features(state_5d)
        return np.concatenate([state_5d, geo])
    return state_5d


def evaluate(model, data, device, num_episodes=50, seed_start=100000,
             action_horizon=16, temperature=0.3):
    """Evaluate using DP's exact PushTKeypointsEnv with legacy=True."""
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    max_scores = []

    for ep in range(num_episodes):
        seed = seed_start + ep
        env = PushTKeypointsEnv(legacy=True, agent_keypoints=False, **kp_kwargs)
        env.seed(seed)
        kp_obs = env.reset()  # keypoint obs (40D)

        # Get 5D state for our model
        state = get_state_from_keypoint_env(env)
        obs_history = [state]
        episode_rewards = []
        done = False
        step_count = 0

        while not done and step_count < 300:
            while len(obs_history) < data["obs_horizon"]:
                obs_history.insert(0, obs_history[0])

            recent = np.array(obs_history[-data["obs_horizon"]:])

            # Build features and normalize
            obs_features = []
            for t in range(data["obs_horizon"]):
                obs_features.append(build_obs(recent[t], data))
            obs_features = np.array(obs_features)

            # Normalize
            if data["use_geo"]:
                # Normalize raw 5D part and geo part separately
                obs_norm = obs_features.copy()
                obs_norm[:, :5] = (obs_features[:, :5] - data["obs_mean"]) / data["obs_std"]
                # Geo features are already in reasonable range, just normalize
                if obs_norm.shape[0] > 0 and obs_norm.shape[1] > 5:
                    geo_mean = obs_norm[:, 5:].mean()
                    geo_std = obs_norm[:, 5:].std() + 1e-6
                    obs_norm[:, 5:] = (obs_norm[:, 5:] - geo_mean) / geo_std
            else:
                obs_norm = (obs_features - data["obs_mean"]) / data["obs_std"]

            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                out = model.forward(obs_t)
                logits = out["cls_logits"]
                soft_bins = soft_decode(logits, data["num_bins"], temperature)

            pred_np = soft_bins.cpu().numpy()[0].reshape(data["pred_horizon"], 2)
            actions = np.zeros_like(pred_np)
            for d in range(2):
                tok = data["act_tokenizer"].tokenizers[d]
                actions[:, d] = tok.val_min + (pred_np[:, d] + 0.5) * tok.bin_width

            for t in range(min(action_horizon, len(actions))):
                action = np.clip(actions[t], 0, 512)
                kp_obs, reward, done, info = env.step(action)
                state = get_state_from_keypoint_env(env)
                obs_history.append(state)
                episode_rewards.append(reward)
                step_count += 1
                if done:
                    break

        max_reward = max(episode_rewards) if episode_rewards else 0
        max_scores.append(max_reward)
        if ep < 5 or max_reward > 0.5:
            print(f"  ep {ep} (seed {seed}): {max_reward:.3f}")

    return {
        "mean_score": float(np.mean(max_scores)),
        "std_score": float(np.std(max_scores)),
        "min_score": float(np.min(max_scores)),
        "max_score": float(np.max(max_scores)),
        "num_episodes": num_episodes,
        "zeros": int(sum(1 for s in max_scores if s < 0.05)),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--no_geo", action="store_true")
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model, data = load_our_model(args.checkpoint, device, args.hidden_size,
                                  args.num_bins, not args.no_geo)
    print(f"Model loaded. use_geo={not args.no_geo}, bins={args.num_bins}")

    results = evaluate(model, data, device, args.num_episodes,
                       action_horizon=args.action_horizon, temperature=args.temperature)

    print(f"\n=== Results (DP exact env) ===")
    print(f"Mean: {results['mean_score']:.4f}")
    print(f"Range: {results['min_score']:.3f} - {results['max_score']:.3f}")
    print(f"Zeros: {results['zeros']}/{args.num_episodes}")

    with open("our_model_dp_env_results.json", "w") as f:
        json.dump(results, f, indent=2)
