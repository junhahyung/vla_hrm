"""
Deep debugging: why do some episodes fail and some succeed?

Tests:
1. Determinism: do failed episodes ALWAYS fail?
2. Temperature sweep: does adding randomness help?
3. Trajectory analysis: where does the model go wrong?
4. Initial prediction quality: are first actions reasonable?
5. Training data coverage: are failed configs OOD?
6. Per-episode breakdown with detailed logging
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import json
import zarr

from pusht_hrm.model_v6 import build_model_v6, EMA
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv
from pusht_hrm.losses import soft_decode


def load_model_and_data(checkpoint_path, device, model_cfg=None):
    zarr_path = "/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    root = zarr.open(zarr_path, "r")
    state = root["data/state"][:]
    action = root["data/action"][:]
    episode_ends = root["meta/episode_ends"][:]

    obs_mean = state.mean(axis=0)
    obs_std = state.std(axis=0) + 1e-6
    act_min = action.min(axis=0)
    act_max = action.max(axis=0)
    act_ranges = [(act_min[i], act_max[i]) for i in range(2)]
    act_tokenizer = PerDimTokenizer(256, act_ranges, "uniform")

    if model_cfg is None:
        model_cfg = {
            "model_type": "hrm", "vocab_size": 256, "hidden_size": 256,
            "num_heads": 8, "expansion": 2.0, "H_layers": 3, "L_layers": 3,
            "H_cycles": 5, "L_cycles": 6, "obs_horizon": 2, "pred_horizon": 16,
            "obs_dim": 5, "act_dim": 2, "act_seq_len": 32,
            "loss_type": "gaussian_soft", "soft_sigma": 3.0, "soft_decode_temp": 0.5,
            "use_regression_head": True, "regression_weight": 0.5,
            "forward_dtype": "bfloat16",
        }

    model = build_model_v6(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, {
        "obs_mean": obs_mean, "obs_std": obs_std,
        "act_tokenizer": act_tokenizer, "act_min": act_min, "act_max": act_max,
        "state": state, "action": action, "episode_ends": episode_ends,
        "obs_horizon": 2, "pred_horizon": 16, "action_horizon": 8,
    }


def predict_actions(model, obs_norm, data, device, temperature=0.5):
    """Get predicted actions from model."""
    obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model.forward(obs_t)
        logits = out["cls_logits"]
        soft_bins = soft_decode(logits, 256, temperature)
    pred_np = soft_bins.cpu().numpy()[0].reshape(data["pred_horizon"], 2)
    actions = np.zeros_like(pred_np)
    for d in range(2):
        tok = data["act_tokenizer"].tokenizers[d]
        actions[:, d] = tok.val_min + (pred_np[:, d] + 0.5) * tok.bin_width
    return actions


def run_episode(model, data, device, seed, temperature=0.5, action_horizon=8):
    """Run single episode with detailed logging."""
    env = PushTEnv()
    env.seed(seed)
    obs = env.reset()
    initial_obs = obs.copy()
    obs_history = [obs]

    trajectory = []
    done = False
    step_count = 0
    replan_count = 0

    while not done and step_count < 200:
        while len(obs_history) < data["obs_horizon"]:
            obs_history.insert(0, obs_history[0])
        recent_obs = np.array(obs_history[-data["obs_horizon"]:])
        obs_norm = (recent_obs - data["obs_mean"]) / data["obs_std"]

        actions = predict_actions(model, obs_norm, data, device, temperature)

        replan_count += 1
        for t in range(min(action_horizon, len(actions))):
            action = np.clip(actions[t], 0, 512)
            obs, score, done, info = env.step(action)
            obs_history.append(obs)
            trajectory.append({
                "step": step_count,
                "obs": obs.tolist(),
                "action": action.tolist(),
                "score": float(score),
                "replan": replan_count,
                "action_idx_in_chunk": t,
            })
            step_count += 1
            if done:
                break

    scores = [t["score"] for t in trajectory]
    return {
        "seed": seed,
        "initial_obs": initial_obs.tolist(),
        "max_score": float(max(scores)) if scores else 0,
        "final_score": float(scores[-1]) if scores else 0,
        "num_steps": step_count,
        "num_replans": replan_count,
        "trajectory": trajectory,
        "scores_at_milestones": {
            "step_25": float(scores[24]) if len(scores) > 24 else 0,
            "step_50": float(scores[49]) if len(scores) > 49 else 0,
            "step_100": float(scores[99]) if len(scores) > 99 else 0,
            "step_150": float(scores[149]) if len(scores) > 149 else 0,
        },
    }


def test_determinism(model, data, device, seeds=[10000, 10001, 10002], n_runs=5):
    """Test: do the same seeds always produce the same result?"""
    print("\n" + "="*60)
    print("TEST 1: DETERMINISM")
    print("="*60)
    for seed in seeds:
        scores = []
        for run in range(n_runs):
            result = run_episode(model, data, device, seed)
            scores.append(result["max_score"])
        print(f"  Seed {seed}: scores = {[f'{s:.3f}' for s in scores]}, "
              f"std = {np.std(scores):.4f} {'(DETERMINISTIC)' if np.std(scores) < 0.001 else '(STOCHASTIC!)'}")


def test_temperature(model, data, device, seeds=range(10000, 10022)):
    """Test: does temperature/randomness help?"""
    print("\n" + "="*60)
    print("TEST 2: TEMPERATURE SWEEP")
    print("="*60)
    temps = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    for temp in temps:
        max_scores = []
        for seed in seeds:
            result = run_episode(model, data, device, seed, temperature=temp)
            max_scores.append(result["max_score"])
        mean = np.mean(max_scores)
        zeros = sum(1 for s in max_scores if s < 0.05)
        high = sum(1 for s in max_scores if s > 0.5)
        print(f"  temp={temp:.1f}: mean={mean:.3f}, zeros={zeros}/22, high(>0.5)={high}/22, "
              f"min={min(max_scores):.3f}, max={max(max_scores):.3f}")


def test_action_horizon(model, data, device, seeds=range(10000, 10022)):
    """Test: does shorter action horizon (more replanning) help?"""
    print("\n" + "="*60)
    print("TEST 3: ACTION HORIZON SWEEP")
    print("="*60)
    for ah in [1, 2, 4, 8, 16]:
        max_scores = []
        for seed in seeds:
            result = run_episode(model, data, device, seed, action_horizon=ah)
            max_scores.append(result["max_score"])
        mean = np.mean(max_scores)
        zeros = sum(1 for s in max_scores if s < 0.05)
        print(f"  action_horizon={ah:2d}: mean={mean:.3f}, zeros={zeros}/22, "
              f"min={min(max_scores):.3f}, max={max(max_scores):.3f}")


def analyze_failures(model, data, device):
    """Deep analysis of which episodes fail and why."""
    print("\n" + "="*60)
    print("TEST 4: FAILURE ANALYSIS")
    print("="*60)

    results = []
    for seed in range(10000, 10022):
        r = run_episode(model, data, device, seed)
        results.append(r)

    results.sort(key=lambda x: x["max_score"])

    # Separate good and bad
    bad = [r for r in results if r["max_score"] < 0.05]
    medium = [r for r in results if 0.05 <= r["max_score"] <= 0.5]
    good = [r for r in results if r["max_score"] > 0.5]

    print(f"\nBad (<0.05): {len(bad)}, Medium (0.05-0.5): {len(medium)}, Good (>0.5): {len(good)}")

    # Analyze initial states
    goal = np.array([256.0, 256.0, np.pi/4])  # goal pose

    print("\n--- Bad episodes ---")
    for r in bad:
        obs = np.array(r["initial_obs"])
        agent = obs[:2]
        block = obs[2:4]
        angle = obs[4]
        dist_agent_block = np.linalg.norm(agent - block)
        dist_block_goal = np.linalg.norm(block - goal[:2])
        angle_diff = abs(angle - goal[2]) % (2*np.pi)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)

        # First action
        traj = r["trajectory"]
        first_action = np.array(traj[0]["action"]) if traj else [0,0]
        action_dir = first_action - agent
        block_dir = block - agent
        # Dot product: is the agent moving toward the block?
        if np.linalg.norm(action_dir) > 0 and np.linalg.norm(block_dir) > 0:
            cos_sim = np.dot(action_dir, block_dir) / (np.linalg.norm(action_dir) * np.linalg.norm(block_dir))
        else:
            cos_sim = 0

        print(f"  seed={r['seed']}: agent=({agent[0]:.0f},{agent[1]:.0f}), "
              f"block=({block[0]:.0f},{block[1]:.0f}), angle={angle:.2f}")
        print(f"    dist_a2b={dist_agent_block:.0f}, dist_b2goal={dist_block_goal:.0f}, "
              f"angle_diff={np.degrees(angle_diff):.0f}°")
        print(f"    first_action=({first_action[0]:.0f},{first_action[1]:.0f}), "
              f"moving_toward_block={cos_sim:.2f}")

        # Score progression
        milestones = r["scores_at_milestones"]
        print(f"    score@25={milestones['step_25']:.3f}, @50={milestones['step_50']:.3f}, "
              f"@100={milestones['step_100']:.3f}, @150={milestones['step_150']:.3f}")

    print("\n--- Good episodes ---")
    for r in good[:5]:
        obs = np.array(r["initial_obs"])
        agent = obs[:2]
        block = obs[2:4]
        angle = obs[4]
        dist_agent_block = np.linalg.norm(agent - block)
        dist_block_goal = np.linalg.norm(block - goal[:2])

        traj = r["trajectory"]
        first_action = np.array(traj[0]["action"]) if traj else [0,0]
        block_dir = block - agent
        action_dir = first_action - agent
        if np.linalg.norm(action_dir) > 0 and np.linalg.norm(block_dir) > 0:
            cos_sim = np.dot(action_dir, block_dir) / (np.linalg.norm(action_dir) * np.linalg.norm(block_dir))
        else:
            cos_sim = 0

        print(f"  seed={r['seed']}: agent=({agent[0]:.0f},{agent[1]:.0f}), "
              f"block=({block[0]:.0f},{block[1]:.0f}), angle={angle:.2f}, "
              f"score={r['max_score']:.3f}")
        print(f"    dist_a2b={dist_agent_block:.0f}, dist_b2goal={dist_block_goal:.0f}, "
              f"moving_toward_block={cos_sim:.2f}")

    return results


def check_training_coverage(data):
    """Check if eval initial states are covered by training data."""
    print("\n" + "="*60)
    print("TEST 5: TRAINING DATA COVERAGE")
    print("="*60)

    state = data["state"]
    episode_ends = data["episode_ends"]
    starts = np.concatenate([[0], episode_ends[:-1]])
    train_initial = state[starts]

    print(f"Training episodes: {len(starts)}")
    print(f"Training initial states range:")
    for i, name in enumerate(["agent_x", "agent_y", "block_x", "block_y", "angle"]):
        print(f"  {name}: [{train_initial[:, i].min():.0f}, {train_initial[:, i].max():.0f}], "
              f"mean={train_initial[:, i].mean():.0f}, std={train_initial[:, i].std():.0f}")

    # Check eval initial states
    env = PushTEnv()
    print(f"\nEval initial states:")
    for seed in range(10000, 10022):
        env.seed(seed)
        obs = env.reset()
        # Check if this is within training range
        in_range = all(
            train_initial[:, i].min() - 50 <= obs[i] <= train_initial[:, i].max() + 50
            for i in range(5)
        )
        if not in_range:
            print(f"  seed {seed} OUT OF TRAINING RANGE: {obs}")


def analyze_action_patterns(results):
    """Look at action patterns: are failed episodes getting stuck?"""
    print("\n" + "="*60)
    print("TEST 6: ACTION PATTERN ANALYSIS")
    print("="*60)

    for r in results:
        if r["max_score"] < 0.05 and r["trajectory"]:
            traj = r["trajectory"]
            actions = np.array([t["action"] for t in traj[:50]])
            if len(actions) > 1:
                # Check action variance (is the agent stuck?)
                action_var = actions.var(axis=0)
                # Check if agent position changes
                positions = np.array([t["obs"][:2] for t in traj[:50]])
                pos_range = positions.max(axis=0) - positions.min(axis=0)

                print(f"  seed={r['seed']}: action_var=({action_var[0]:.0f},{action_var[1]:.0f}), "
                      f"pos_range=({pos_range[0]:.0f},{pos_range[1]:.0f})")

                # Is the agent going to a single point repeatedly?
                if action_var[0] < 100 and action_var[1] < 100:
                    mean_action = actions.mean(axis=0)
                    print(f"    STUCK: agent keeps going to ({mean_action[0]:.0f},{mean_action[1]:.0f})")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    checkpoint = "checkpoints/v6_hrm_deep_dual/best_eval.pt"

    print("Loading model...")
    model, data = load_model_and_data(checkpoint, device)
    print("Model loaded.\n")

    # Run all tests
    test_determinism(model, data, device)
    test_temperature(model, data, device)
    test_action_horizon(model, data, device)
    results = analyze_failures(model, data, device)
    check_training_coverage(data)
    analyze_action_patterns(results)

    # Save results
    save_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "trajectory"}
        r_copy["trajectory_len"] = len(r["trajectory"])
        save_results.append(r_copy)
    with open("debug_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    print("\n\nDone! Results saved to debug_results.json")
