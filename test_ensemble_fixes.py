"""
Test various strategies to fix stuck episodes:
1. Random action perturbation when stuck
2. Ensemble of multiple forward passes with temperature
3. Fallback to random actions for initial approach
4. Different initial action seeds for V7 iterative refinement
5. Longer obs horizon
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import zarr

from pusht_hrm.model_v6 import build_model_v6
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv
from pusht_hrm.losses import soft_decode


def load_best_model(device):
    zarr_path = "/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    root = zarr.open(zarr_path, "r")
    state, action = root["data/state"][:], root["data/action"][:]
    obs_mean, obs_std = state.mean(axis=0), state.std(axis=0) + 1e-6
    act_min, act_max = action.min(axis=0), action.max(axis=0)
    act_tokenizer = PerDimTokenizer(256, [(act_min[i], act_max[i]) for i in range(2)], "uniform")

    config = {
        "model_type": "hrm", "vocab_size": 256, "hidden_size": 256,
        "num_heads": 8, "expansion": 2.0, "H_layers": 3, "L_layers": 3,
        "H_cycles": 5, "L_cycles": 6, "obs_horizon": 2, "pred_horizon": 16,
        "obs_dim": 5, "act_dim": 2, "act_seq_len": 32,
        "loss_type": "gaussian_soft", "soft_sigma": 3.0, "soft_decode_temp": 0.5,
        "use_regression_head": True, "regression_weight": 0.5,
        "forward_dtype": "bfloat16",
    }
    model = build_model_v6(config).to(device)
    ckpt = torch.load("checkpoints/v6_hrm_deep_dual/best_eval.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    return model, {"obs_mean": obs_mean, "obs_std": obs_std, "act_tokenizer": act_tokenizer,
                    "obs_horizon": 2, "pred_horizon": 16, "action_horizon": 16}


def decode_logits_to_actions(logits, data, temp=0.3):
    soft_bins = soft_decode(logits, 256, temp).detach().cpu().numpy()[0].reshape(data["pred_horizon"], 2)
    actions = np.zeros_like(soft_bins)
    for d in range(2):
        tok = data["act_tokenizer"].tokenizers[d]
        actions[:, d] = tok.val_min + (soft_bins[:, d] + 0.5) * tok.bin_width
    return actions


def get_obs_tensor(obs_history, data, device):
    while len(obs_history) < data["obs_horizon"]:
        obs_history.insert(0, obs_history[0])
    recent = np.array(obs_history[-data["obs_horizon"]:])
    obs_norm = (recent - data["obs_mean"]) / data["obs_std"]
    return torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)


def run_episode_strategy(model, data, device, seed, strategy="default", **kwargs):
    env = PushTEnv()
    env.seed(seed)
    obs = env.reset()
    obs_history = [obs]
    scores = []
    done = False
    step_count = 0
    ah = kwargs.get("action_horizon", data["action_horizon"])

    while not done and step_count < 200:
        obs_t = get_obs_tensor(obs_history, data, device)

        with torch.no_grad():
            out = model.forward(obs_t)
            logits = out["cls_logits"]

        if strategy == "default":
            actions = decode_logits_to_actions(logits, data, temp=kwargs.get("temp", 0.3))

        elif strategy == "ensemble_temp":
            # Average predictions from multiple temperatures
            temps = [0.1, 0.3, 0.5, 1.0]
            all_actions = [decode_logits_to_actions(logits, data, t) for t in temps]
            actions = np.mean(all_actions, axis=0)

        elif strategy == "ensemble_noise":
            # Multiple forward passes with input noise, average results
            all_actions = []
            for _ in range(kwargs.get("n_ensemble", 5)):
                noisy_obs = obs_t + torch.randn_like(obs_t) * 0.1
                out_n = model.forward(noisy_obs)
                all_actions.append(decode_logits_to_actions(out_n["cls_logits"], data))
            all_actions.append(decode_logits_to_actions(logits, data))  # original too
            actions = np.mean(all_actions, axis=0)

        elif strategy == "perturb_when_stuck":
            # If score hasn't changed, add random perturbation
            actions = decode_logits_to_actions(logits, data)
            if len(scores) > 20 and max(scores[-20:]) < 0.01:
                noise = np.random.randn(*actions.shape) * 30
                actions = actions + noise

        elif strategy == "approach_first":
            # For first 50 steps: move toward block, then use model
            actions = decode_logits_to_actions(logits, data)
            if step_count < 50:
                curr_obs = obs_history[-1]
                agent_pos = curr_obs[:2]
                block_pos = curr_obs[2:4]
                # Move toward block
                direction = block_pos - agent_pos
                dist = np.linalg.norm(direction)
                if dist > 30:  # Only override if far from block
                    target = agent_pos + direction * 0.8
                    actions[0] = target  # Override first action in chunk

        elif strategy == "goal_biased":
            # Bias actions toward the goal position
            actions = decode_logits_to_actions(logits, data)
            goal = np.array([256.0, 256.0])
            curr_obs = obs_history[-1]
            block_pos = curr_obs[2:4]
            block_to_goal = goal - block_pos
            # Add small bias toward pushing block to goal
            bias_strength = kwargs.get("bias", 10.0)
            push_dir = block_to_goal / (np.linalg.norm(block_to_goal) + 1e-6)
            actions += push_dir * bias_strength

        else:
            actions = decode_logits_to_actions(logits, data)

        for t in range(min(ah, len(actions))):
            action = np.clip(actions[t], 0, 512)
            obs, score, done, info = env.step(action)
            obs_history.append(obs)
            scores.append(float(score))
            step_count += 1
            if done:
                break

    return max(scores) if scores else 0


def main():
    device = torch.device("cuda:0")
    model, data = load_best_model(device)

    seeds = list(range(10000, 10022))

    strategies = [
        ("default", {}),
        ("ensemble_temp", {}),
        ("ensemble_noise", {"n_ensemble": 5}),
        ("ensemble_noise", {"n_ensemble": 10}),
        ("perturb_when_stuck", {}),
        ("approach_first", {}),
        ("goal_biased", {"bias": 5.0}),
        ("goal_biased", {"bias": 10.0}),
        ("goal_biased", {"bias": 20.0}),
        ("default", {"temp": 0.1}),
        ("default", {"temp": 0.3}),
        ("default", {"temp": 0.5}),
        ("default", {"temp": 1.0}),
        ("default", {"action_horizon": 4}),
        ("default", {"action_horizon": 8}),
        ("default", {"action_horizon": 16}),
    ]

    print(f"{'Strategy':<35} {'Mean':>6} {'Zeros':>6} {'High':>6} {'Max':>6}")
    print("-" * 65)

    for strat_name, kwargs in strategies:
        label = f"{strat_name}"
        if kwargs:
            label += f"({','.join(f'{k}={v}' for k,v in kwargs.items())})"

        scores = []
        for seed in seeds:
            s = run_episode_strategy(model, data, device, seed, strat_name, **kwargs)
            scores.append(s)

        mean = np.mean(scores)
        zeros = sum(1 for s in scores if s < 0.05)
        high = sum(1 for s in scores if s > 0.5)
        mx = max(scores)
        print(f"{label:<35} {mean:>6.3f} {zeros:>6d} {high:>6d} {mx:>6.3f}")

    # Also test: for failed seeds, what if we retry with random initial actions?
    print("\n\n=== FAILED SEED ANALYSIS ===")
    failed_seeds = []
    for seed in seeds:
        s = run_episode_strategy(model, data, device, seed, "default")
        if s < 0.05:
            failed_seeds.append(seed)

    print(f"Failed seeds: {failed_seeds}")
    print(f"\nRetrying with different strategies on failed seeds only:")

    for strat, kw in [("ensemble_noise", {"n_ensemble": 10}),
                       ("approach_first", {}),
                       ("goal_biased", {"bias": 15.0}),
                       ("perturb_when_stuck", {})]:
        fixed = 0
        for seed in failed_seeds:
            s = run_episode_strategy(model, data, device, seed, strat, **kw)
            if s > 0.05:
                fixed += 1
                print(f"  {strat}: seed {seed} FIXED! score={s:.3f}")
        print(f"  {strat}: fixed {fixed}/{len(failed_seeds)} failed episodes")


if __name__ == "__main__":
    main()
