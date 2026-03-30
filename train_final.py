"""
FINAL: Train HRM on PushT using DP's EXACT environment for evaluation.
This is the fair comparison against Diffusion Policy (0.871 mean).
"""
import argparse, os, sys, math, json, numpy as np, torch
os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_policy'))
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import wandb
from tqdm import tqdm
import zarr

from pusht_hrm.model_v6 import build_model_v6, EMA
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.losses import GaussianSoftLabelLoss, soft_decode
from pusht_hrm.obs_features import compute_geometric_features, GEOMETRIC_FEATURE_DIM

# DP's exact environment
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


# ============== Dataset ==============
class PushTDatasetFinal(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, num_bins=256,
                 obs_noise_std=0.0, act_noise_std=0.0, use_geo=True, augment_mirror=True):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_bins = num_bins
        self.obs_noise_std = obs_noise_std
        self.act_noise_std = act_noise_std
        self.use_geo = use_geo
        self.augment_mirror = augment_mirror
        self.obs_dim = 5 + (GEOMETRIC_FEATURE_DIM if use_geo else 0)
        self.act_dim = 2
        self.act_seq_len = pred_horizon * self.act_dim

        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])

        self.indices = []
        for ep in range(len(episode_ends)):
            s, e = starts[ep], episode_ends[ep]
            for t in range(e - s - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(s + t)
        self.indices = np.array(self.indices)
        self.base_len = len(self.indices)
        self.num_augments = 4 if augment_mirror else 1

        # Compute normalization stats with geo features
        sample = self._build_features(self.state[:200])
        self.obs_mean = sample.mean(axis=0)
        self.obs_std = sample.std(axis=0) + 1e-6

        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6
        self.act_tokenizer = PerDimTokenizer(num_bins,
            [(self.act_min[i], self.act_max[i]) for i in range(2)], "uniform")

        print(f"Final dataset: {self.base_len}x{self.num_augments}={len(self)} samples, obs_dim={self.obs_dim}")

    def _build_features(self, raw_obs):
        if self.use_geo:
            geo = compute_geometric_features(raw_obs)
            return np.concatenate([raw_obs, geo], axis=-1)
        return raw_obs

    def _mirror(self, obs_raw, actions, mode):
        center = 256.0
        obs, acts = obs_raw.copy(), actions.copy()
        if mode in (1, 3):
            obs[:, 0] = 2*center - obs[:, 0]
            obs[:, 2] = 2*center - obs[:, 2]
            acts[:, 0] = 2*center - acts[:, 0]
        if mode in (2, 3):
            obs[:, 1] = 2*center - obs[:, 1]
            obs[:, 3] = 2*center - obs[:, 3]
            acts[:, 1] = 2*center - acts[:, 1]
        if mode in (1, 2, 3):
            if mode == 1: obs[:, 4] = np.pi - obs[:, 4]
            elif mode == 2: obs[:, 4] = -obs[:, 4]
            else: obs[:, 4] = np.pi + obs[:, 4]
        return obs, acts

    def normalize_obs(self, obs_feat):
        return (obs_feat - self.obs_mean) / self.obs_std

    def __len__(self):
        return self.base_len * self.num_augments

    def __getitem__(self, idx):
        aug = idx // self.base_len
        t = self.indices[idx % self.base_len]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()

        if aug > 0:
            obs, actions = self._mirror(obs, actions, aug)
        if self.obs_noise_std > 0:
            obs[:, :4] += np.random.randn(self.obs_horizon, 4).astype(np.float32) * self.obs_noise_std
        if self.act_noise_std > 0:
            actions += np.random.randn(*actions.shape).astype(np.float32) * self.act_noise_std
            actions = np.clip(actions, 0, 512)

        obs_feat = self._build_features(obs)
        obs_norm = self.normalize_obs(obs_feat)
        act_tokens = self.act_tokenizer.encode(np.clip(actions, self.act_min, self.act_max)).flatten()
        act_normalized = ((np.clip(actions, self.act_min, self.act_max) - self.act_min) / self.act_range).flatten()

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "act_normalized": torch.tensor(act_normalized, dtype=torch.float32),
        }


# ============== Evaluation using DP's exact env ==============
def evaluate_dp_env(model, dataset, device, num_episodes=50, seed_start=100000,
                    action_horizon=8, temperature=0.3):
    """Evaluate using DP's PushTKeypointsEnv with legacy=True. Matches DP paper protocol."""
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    max_scores = []

    for ep in range(num_episodes):
        seed = seed_start + ep
        env = PushTKeypointsEnv(legacy=True, agent_keypoints=False, **kp_kwargs)
        env.seed(seed)
        env.reset()

        state = np.array([env.agent.position[0], env.agent.position[1],
                         env.block.position[0], env.block.position[1], env.block.angle])
        obs_history = [state]
        rewards = []
        done, step_count = False, 0

        while not done and step_count < 300:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])

            recent = np.array(obs_history[-dataset.obs_horizon:])
            obs_feat = dataset._build_features(recent)
            obs_norm = dataset.normalize_obs(obs_feat)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                out = model.forward(obs_t)
                logits = out["cls_logits"]
                soft_bins = soft_decode(logits, dataset.num_bins, temperature)

            pred = soft_bins.cpu().numpy()[0].reshape(dataset.pred_horizon, 2)
            actions = np.zeros_like(pred)
            for d in range(2):
                tok = dataset.act_tokenizer.tokenizers[d]
                actions[:, d] = tok.val_min + (pred[:, d] + 0.5) * tok.bin_width

            for t in range(min(action_horizon, len(actions))):
                action = np.clip(actions[t], 0, 512)
                _, reward, done, _ = env.step(action)
                state = np.array([env.agent.position[0], env.agent.position[1],
                                 env.block.position[0], env.block.position[1], env.block.angle])
                obs_history.append(state)
                rewards.append(reward)
                step_count += 1
                if done:
                    break

        max_scores.append(max(rewards) if rewards else 0)

    return {
        "mean_score": float(np.mean(max_scores)),
        "std_score": float(np.std(max_scores)),
        "min_score": float(np.min(max_scores)),
        "max_score": float(np.max(max_scores)),
        "zeros": sum(1 for s in max_scores if s < 0.05),
    }


# ============== Training ==============
def get_lr(step, warmup, total, max_lr, min_lr=1e-6):
    if step < warmup: return max_lr * step / warmup
    r = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--model_type", default="hrm")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--H_layers", type=int, default=3)
    p.add_argument("--L_layers", type=int, default=3)
    p.add_argument("--H_cycles", type=int, default=5)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--soft_sigma", type=float, default=3.0)
    p.add_argument("--soft_decode_temp", type=float, default=0.3)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--obs_noise_std", type=float, default=3.0)
    p.add_argument("--act_noise_std", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--wandb_entity", default="junha")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device(f"cuda:{args.gpu}")

    dataset = PushTDatasetFinal(args.zarr_path, num_bins=args.num_bins,
                                 obs_noise_std=args.obs_noise_std, act_noise_std=args.act_noise_std)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(len(indices) * 0.9)
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    config = {
        "model_type": args.model_type, "vocab_size": args.num_bins,
        "hidden_size": args.hidden_size, "num_heads": args.num_heads,
        "expansion": 2.0, "H_layers": args.H_layers, "L_layers": args.L_layers,
        "H_cycles": args.H_cycles, "L_cycles": args.L_cycles,
        "obs_horizon": 2, "pred_horizon": 16, "obs_dim": dataset.obs_dim, "act_dim": 2,
        "act_seq_len": dataset.act_seq_len,
        "loss_type": "gaussian_soft", "soft_sigma": args.soft_sigma,
        "soft_decode_temp": args.soft_decode_temp,
        "use_regression_head": True, "regression_weight": 0.5,
        "forward_dtype": "bfloat16",
    }
    model = build_model_v6(config).to(device)
    ema = EMA(model, decay=0.999)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}, Params: {num_params:,}")

    if args.exp_name is None:
        args.exp_name = f"final_{args.model_type}_h{args.hidden_size}_bins{args.num_bins}"
    wandb.init(project="pusht-hrm", entity=args.wandb_entity, name=args.exp_name, config=vars(args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 30 * len(train_loader)
    save_dir = os.path.join("checkpoints", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_eval = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            global_step += 1
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups: pg["lr"] = lr

            obs = batch["obs"].to(device)
            act_tokens = batch["act_tokens"].to(device)
            act_norm = batch["act_normalized"].to(device)
            output = model(obs, act_tokens=act_tokens, act_normalized=act_norm)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch["obs"].to(device), act_tokens=batch["act_tokens"].to(device),
                             act_normalized=batch["act_normalized"].to(device))
                val_loss += output["loss"].item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} val={avg_val:.4f}")
        log = {"epoch": epoch+1, "train/loss": avg_loss, "val/loss": avg_val, "train/lr": lr}

        # Evaluate on DP's exact environment
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print(f"Evaluating on DP env ({args.eval_episodes} episodes)...")
            for name, use_ema in [("model", False), ("ema", True)]:
                if use_ema: ema.apply_shadow(model)
                r = evaluate_dp_env(model, dataset, device, args.eval_episodes,
                                    action_horizon=args.action_horizon, temperature=args.soft_decode_temp)
                if use_ema: ema.restore(model)
                print(f"  [{name}] mean={r['mean_score']:.4f} (range: {r['min_score']:.3f}-{r['max_score']:.3f}, zeros={r['zeros']})")
                log[f"eval/{name}_mean"] = r["mean_score"]
                log[f"eval/{name}_max"] = r["max_score"]
                if r["mean_score"] > best_eval:
                    best_eval = r["mean_score"]
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                    log["eval/best"] = best_eval

        wandb.log(log, step=global_step)

    print(f"\nDone. Best DP-env score: {best_eval:.4f} (DP baseline: 0.871)")
    wandb.finish()


if __name__ == "__main__":
    main()
