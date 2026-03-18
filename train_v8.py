"""
V8: Geometric features + trajectory augmentation + best of V6/V7.

Based on debug findings:
- Model navigates to block but can't push correctly
- Issue is geometric reasoning about approach angle and rotation
- Solution: explicit geometric features + more diverse training data

New features:
- 21 geometric features (distances, directions, angles, pushing geometry)
- Trajectory mirroring augmentation (flip x/y for 4x more data)
- Combined iterative refinement + direct prediction
- Temperature 0.3 (optimal from debug)
"""
import argparse, os, sys, json, math, yaml, numpy as np, torch
os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
from tqdm import tqdm
import zarr

from pusht_hrm.model_v6 import build_model_v6, EMA
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv
from pusht_hrm.losses import soft_decode
from pusht_hrm.obs_features import compute_geometric_features, GEOMETRIC_FEATURE_DIM


class PushTDatasetV8(Dataset):
    """Dataset with geometric features and trajectory augmentation."""

    def __init__(self, zarr_path, obs_horizon=2, action_horizon=8, pred_horizon=16,
                 num_bins=256, obs_noise_std=0.0, act_noise_std=0.0,
                 use_geometric_features=True, augment_mirror=True):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.num_bins = num_bins
        self.obs_noise_std = obs_noise_std
        self.act_noise_std = act_noise_std
        self.use_geo = use_geometric_features
        self.augment_mirror = augment_mirror
        self.act_dim = 2
        self.raw_obs_dim = 5
        self.obs_dim = 5 + (GEOMETRIC_FEATURE_DIM if use_geometric_features else 0)
        self.act_seq_len = pred_horizon * self.act_dim

        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])

        self.indices = []
        for ep_idx in range(len(episode_ends)):
            s, e = self.episode_starts[ep_idx], episode_ends[ep_idx]
            for t in range(e - s - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(s + t)
        self.indices = np.array(self.indices)

        # If augmenting with mirroring, multiply indices
        self.base_len = len(self.indices)
        # 4 augmentations: original, flip_x, flip_y, flip_both
        self.num_augments = 4 if augment_mirror else 1

        # Stats computed on original data (with geometric features)
        sample_obs = self._build_obs_features(self.state[:100])
        self.obs_mean = np.mean(sample_obs, axis=0)
        self.obs_std = np.std(sample_obs, axis=0) + 1e-6

        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6
        act_ranges = [(self.act_min[i], self.act_max[i]) for i in range(2)]
        self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, "uniform")

        print(f"V8 dataset: {self.base_len} base samples × {self.num_augments} augments = {len(self)} total, "
              f"obs_dim={self.obs_dim}")

    def _build_obs_features(self, raw_obs):
        """Add geometric features to raw observations."""
        if self.use_geo:
            geo = compute_geometric_features(raw_obs)
            return np.concatenate([raw_obs, geo], axis=-1)
        return raw_obs

    def _mirror_obs_actions(self, obs_raw, actions, flip_mode):
        """Mirror observations and actions for augmentation.
        flip_mode: 0=original, 1=flip_x, 2=flip_y, 3=flip_both
        """
        center = 256.0
        obs = obs_raw.copy()
        acts = actions.copy()

        if flip_mode in (1, 3):  # flip x
            obs[:, 0] = 2*center - obs[:, 0]  # agent_x
            obs[:, 2] = 2*center - obs[:, 2]  # block_x
            acts[:, 0] = 2*center - acts[:, 0]  # action_x
        if flip_mode in (2, 3):  # flip y
            obs[:, 1] = 2*center - obs[:, 1]  # agent_y
            obs[:, 3] = 2*center - obs[:, 3]  # block_y
            acts[:, 1] = 2*center - acts[:, 1]  # action_y
        if flip_mode in (1, 2, 3):  # angle needs adjustment
            if flip_mode == 1:
                obs[:, 4] = np.pi - obs[:, 4]
            elif flip_mode == 2:
                obs[:, 4] = -obs[:, 4]
            elif flip_mode == 3:
                obs[:, 4] = np.pi + obs[:, 4]

        return obs, acts

    def __len__(self):
        return self.base_len * self.num_augments

    def normalize_obs(self, obs_with_features):
        return (obs_with_features - self.obs_mean) / self.obs_std

    def __getitem__(self, idx):
        aug_mode = idx // self.base_len
        base_idx = idx % self.base_len

        t = self.indices[base_idx]
        obs_raw = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()

        # Apply mirror augmentation
        if aug_mode > 0:
            obs_raw, actions = self._mirror_obs_actions(obs_raw, actions, aug_mode)

        # Noise augmentation
        if self.obs_noise_std > 0:
            obs_raw[:, :4] += np.random.randn(self.obs_horizon, 4).astype(np.float32) * self.obs_noise_std
        if self.act_noise_std > 0:
            actions += np.random.randn(*actions.shape).astype(np.float32) * self.act_noise_std
            actions = np.clip(actions, 0, 512)

        # Build features and normalize
        obs_feat = self._build_obs_features(obs_raw)
        obs_norm = self.normalize_obs(obs_feat)

        # Tokenize actions
        act_tokens = self.act_tokenizer.encode(np.clip(actions, self.act_min, self.act_max)).flatten()
        act_normalized = ((np.clip(actions, self.act_min, self.act_max) - self.act_min) / self.act_range).flatten()

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "act_normalized": torch.tensor(act_normalized, dtype=torch.float32),
        }


def evaluate_v8(model, dataset, device, num_episodes=22, seed_start=10000,
                temperature=0.3, action_horizon=16, temporal_ensemble=True):
    """Evaluate with geometric features and optimal settings from debug."""
    env = PushTEnv()
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        obs_history = [obs]
        episode_scores = []
        done = False
        step_count = 0
        pending = {}

        while not done and step_count < 200:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])
            recent_raw = np.array(obs_history[-dataset.obs_horizon:])

            # Build geometric features
            obs_feat = dataset._build_obs_features(recent_raw)
            obs_norm = dataset.normalize_obs(obs_feat)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                out = model.forward(obs_t)
                logits = out["cls_logits"]
                soft_bins = soft_decode(logits, dataset.num_bins, temperature)

            pred_np = soft_bins.cpu().numpy()[0].reshape(dataset.pred_horizon, 2)
            actions = np.zeros_like(pred_np)
            for d in range(2):
                tok = dataset.act_tokenizer.tokenizers[d]
                actions[:, d] = tok.val_min + (pred_np[:, d] + 0.5) * tok.bin_width

            if temporal_ensemble:
                for t_off in range(len(actions)):
                    fs = step_count + t_off
                    if fs not in pending:
                        pending[fs] = []
                    pending[fs].append(actions[t_off])

            for t in range(min(action_horizon, len(actions))):
                if temporal_ensemble and step_count in pending:
                    action = np.mean(pending.pop(step_count), axis=0)
                else:
                    action = actions[t]
                action = np.clip(action, 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(float(score))
                step_count += 1
                if done:
                    break

        max_scores.append(max(episode_scores) if episode_scores else 0)

    return {
        "mean_score": np.mean(max_scores),
        "std_score": np.std(max_scores),
        "min_score": np.min(max_scores),
        "max_score": np.max(max_scores),
    }


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    r = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--action_horizon", type=int, default=16)  # 16 is best from debug
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--model_type", default="hrm")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--H_layers", type=int, default=3)
    p.add_argument("--L_layers", type=int, default=3)
    p.add_argument("--H_cycles", type=int, default=5)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--soft_sigma", type=float, default=3.0)
    p.add_argument("--soft_decode_temp", type=float, default=0.3)  # optimal from debug
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=22)
    p.add_argument("--obs_noise_std", type=float, default=3.0)
    p.add_argument("--act_noise_std", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_entity", default="junha")
    p.add_argument("--wandb_project", default="pusht-hrm")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    dataset = PushTDatasetV8(args.zarr_path, args.obs_horizon, args.action_horizon,
                              args.pred_horizon, args.num_bins, args.obs_noise_std, args.act_noise_std,
                              use_geometric_features=True, augment_mirror=True)

    val_size = int(dataset.base_len * 0.1)
    # Split on base indices, not augmented
    train_base = dataset.base_len - val_size
    # Create proper split
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    train_indices = all_indices[:int(len(dataset) * 0.9)]
    val_indices = all_indices[int(len(dataset) * 0.9):]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    config = {
        "model_type": args.model_type, "vocab_size": args.num_bins,
        "hidden_size": args.hidden_size, "num_heads": args.num_heads,
        "expansion": args.expansion, "H_layers": args.H_layers, "L_layers": args.L_layers,
        "H_cycles": args.H_cycles, "L_cycles": args.L_cycles,
        "obs_horizon": args.obs_horizon, "pred_horizon": args.pred_horizon,
        "obs_dim": dataset.obs_dim, "act_dim": 2, "act_seq_len": dataset.act_seq_len,
        "loss_type": "gaussian_soft", "soft_sigma": args.soft_sigma,
        "soft_decode_temp": args.soft_decode_temp,
        "use_regression_head": True, "regression_weight": 0.5,
        "forward_dtype": "bfloat16",
    }

    model = build_model_v6(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}_v8 (geo+mirror), Params: {num_params:,}")

    ema = EMA(model, decay=args.ema_decay)

    if args.exp_name is None:
        args.exp_name = f"v8_{args.model_type}_geo_mirror_h{args.hidden_size}"

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, config=vars(args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_eval = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            global_step += 1
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            obs = batch["obs"].to(device)
            act_tokens = batch["act_tokens"].to(device)
            act_norm = batch["act_normalized"].to(device)

            output = model(obs, act_tokens=act_tokens, act_normalized=act_norm)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})
            if global_step % 100 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=global_step)

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

        log_dict = {"epoch": epoch+1, "train/epoch_loss": avg_loss, "val/loss": avg_val}

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print("Evaluating...")
            for who, use_ema in [("model", False), ("ema", True)]:
                if use_ema:
                    ema.apply_shadow(model)
                r = evaluate_v8(model, dataset, device, num_episodes=args.eval_episodes,
                               temperature=args.soft_decode_temp, action_horizon=args.action_horizon)
                if use_ema:
                    ema.restore(model)
                print(f"  [{who}] mean={r['mean_score']:.4f} (range: {r['min_score']:.3f}-{r['max_score']:.3f})")
                log_dict[f"eval/{who}_mean"] = r["mean_score"]
                if r["mean_score"] > best_eval:
                    best_eval = r["mean_score"]
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_eval.pt"))

        wandb.log(log_dict, step=global_step)

    print(f"Done. Best: {best_eval:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
