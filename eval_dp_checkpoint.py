"""
Evaluate a trained Diffusion Policy checkpoint using our working PushT env.
Bypasses the broken gym vectorized env.
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_policy'))

import numpy as np
import torch
import json
from omegaconf import OmegaConf
import dill

# Use DP's own modules
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Use our working env
from pusht_hrm.pusht_env import PushTEnv


def load_dp_checkpoint(ckpt_path, device):
    """Load a DP checkpoint and return the policy."""
    payload = torch.load(ckpt_path, map_location=device, pickle_module=dill, weights_only=False)
    cfg = payload['cfg']

    # Print config for debugging
    print("=== DP Config ===")
    print(f"horizon: {cfg.horizon}")
    print(f"n_obs_steps: {cfg.n_obs_steps}")
    print(f"n_action_steps: {cfg.n_action_steps}")
    print(f"obs_dim: {cfg.obs_dim}")
    print(f"action_dim: {cfg.action_dim}")
    print(f"obs_as_global_cond: {cfg.obs_as_global_cond}")
    print(f"pred_action_steps_only: {cfg.pred_action_steps_only}")
    print(f"num_inference_steps: {cfg.policy.num_inference_steps}")

    # Reconstruct model
    import hydra
    policy = hydra.utils.instantiate(cfg.policy)

    # Load state dict (DP stores EMA weights)
    state_dict = payload['state_dicts']['model']
    ema_state_dict = payload['state_dicts'].get('ema_model', None)

    if ema_state_dict is not None:
        print("Using EMA weights")
        policy.load_state_dict(ema_state_dict)
    else:
        policy.load_state_dict(state_dict)

    # Load normalizer
    normalizer_state = payload['state_dicts']['model']
    # The normalizer is part of the policy

    policy.to(device)
    policy.eval()

    return policy, cfg


def evaluate_dp_policy(policy, cfg, device, num_episodes=50, seed_start=100000):
    """
    Evaluate DP policy using our PushT env.

    Key: DP uses keypoint observations (20D), but we generate them from state.
    The PushTKeypointsEnv generates 9 keypoints from the T-block shape.
    For state-based lowdim, the obs is actually keypoints + agent pos.
    """
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    max_steps = 300

    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

    max_scores = []

    for ep in range(num_episodes):
        seed = seed_start + ep

        # Create env (not vectorized - avoids gym issues)
        env = PushTKeypointsEnv(legacy=True, agent_keypoints=False, **kp_kwargs)
        env.seed(seed)
        obs = env.reset()

        # obs is (obs_dim * 2,) = keypoints + visibility mask
        Do = obs.shape[0] // 2

        obs_history = []
        episode_rewards = []
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # Build observation buffer
            obs_data = obs[:Do]  # just keypoints, no mask
            obs_history.append(obs_data)

            # Pad history if needed
            while len(obs_history) < n_obs_steps:
                obs_history.insert(0, obs_history[0])

            # Take last n_obs_steps
            obs_seq = np.array(obs_history[-n_obs_steps:])  # (n_obs_steps, Do)

            # Convert to tensor and normalize through policy
            obs_dict = {
                'obs': torch.tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
            }

            # Get action from policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            actions = action_dict['action'][0].cpu().numpy()  # (n_action_steps, action_dim)

            # Execute actions
            for t in range(min(n_action_steps, len(actions))):
                action = actions[t]
                obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                step_count += 1
                if done:
                    break

        max_reward = max(episode_rewards) if episode_rewards else 0
        max_scores.append(max_reward)

        if ep < 5 or max_reward > 0.5:
            print(f"  Episode {ep} (seed {seed}): max_reward={max_reward:.3f}, steps={step_count}")

    results = {
        "mean_score": float(np.mean(max_scores)),
        "std_score": float(np.std(max_scores)),
        "min_score": float(np.min(max_scores)),
        "max_score": float(np.max(max_scores)),
        "num_episodes": num_episodes,
        "all_scores": [float(s) for s in max_scores],
    }
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--seed_start", type=int, default=100000)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, cfg = load_dp_checkpoint(args.checkpoint, device)

    print(f"\nEvaluating {args.num_episodes} episodes (seeds {args.seed_start}-{args.seed_start + args.num_episodes - 1})...")
    results = evaluate_dp_policy(policy, cfg, device,
                                  num_episodes=args.num_episodes,
                                  seed_start=args.seed_start)

    print(f"\n=== Results ===")
    print(f"Mean score: {results['mean_score']:.4f}")
    print(f"Std: {results['std_score']:.4f}")
    print(f"Range: {results['min_score']:.3f} - {results['max_score']:.3f}")

    # Count zeros and high scores
    zeros = sum(1 for s in results['all_scores'] if s < 0.05)
    high = sum(1 for s in results['all_scores'] if s > 0.5)
    print(f"Zero episodes (<0.05): {zeros}/{args.num_episodes}")
    print(f"High episodes (>0.5): {high}/{args.num_episodes}")

    with open("dp_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to dp_eval_results.json")


if __name__ == "__main__":
    main()
