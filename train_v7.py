"""
V7: Iterative Action Refinement training.
Train: corrupt action tokens at random noise levels, predict clean tokens.
Eval: start from random tokens, iteratively refine over K steps.
"""
import argparse, os, sys, json, math, yaml, numpy as np, torch
os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
from tqdm import tqdm
import zarr

from pusht_hrm.model_v7 import build_model_v7, EMA
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv
from pusht_hrm.losses import soft_decode


class PushTDatasetV7(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, action_horizon=8, pred_horizon=16,
                 num_bins=256, obs_noise_std=0.0, act_noise_std=0.0):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.num_bins = num_bins
        self.obs_noise_std = obs_noise_std
        self.act_noise_std = act_noise_std
        self.obs_dim, self.act_dim = 5, 2
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

        self.obs_mean = self.state.mean(axis=0)
        self.obs_std = self.state.std(axis=0) + 1e-6
        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6
        act_ranges = [(self.act_min[i], self.act_max[i]) for i in range(2)]
        self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, "uniform")
        print(f"V7 dataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()
        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std
        if self.act_noise_std > 0:
            actions = actions + np.random.randn(*actions.shape).astype(np.float32) * self.act_noise_std
            actions = np.clip(actions, self.act_min - 5, self.act_max + 5)

        obs_norm = self.normalize_obs(obs)
        act_tokens = self.act_tokenizer.encode(actions).flatten()
        act_normalized = ((actions - self.act_min) / self.act_range).flatten()

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "act_normalized": torch.tensor(act_normalized, dtype=torch.float32),
        }


def evaluate_v7(model, dataset, device, num_episodes=22, seed_start=10000,
                num_refine_steps=4, temperature=0.5, action_horizon=8,
                temporal_ensemble=True):
    env = PushTEnv()
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        obs_history = [obs]
        episode_scores = []
        done = False
        step_count = 0
        pending_actions = {}

        while not done and step_count < 200:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])

            recent_obs = np.array(obs_history[-dataset.obs_horizon:])
            obs_norm = dataset.normalize_obs(recent_obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            pred = model.predict_actions(obs_t, num_steps=num_refine_steps,
                                         temperature=temperature)
            pred_np = pred.cpu().numpy()[0]
            pred_reshaped = pred_np.reshape(dataset.pred_horizon, 2)
            actions = np.zeros_like(pred_reshaped)
            for d in range(2):
                tok = dataset.act_tokenizer.tokenizers[d]
                actions[:, d] = tok.val_min + (pred_reshaped[:, d] + 0.5) * tok.bin_width

            if temporal_ensemble:
                for t_off in range(len(actions)):
                    fs = step_count + t_off
                    if fs not in pending_actions:
                        pending_actions[fs] = []
                    pending_actions[fs].append(actions[t_off])

            for t in range(min(action_horizon, len(actions))):
                if temporal_ensemble and step_count in pending_actions:
                    action = np.mean(pending_actions.pop(step_count), axis=0)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--model_type", default="hrm")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--H_layers", type=int, default=2)
    p.add_argument("--L_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--L_cycles", type=int, default=4)
    p.add_argument("--num_refine_steps", type=int, default=4)
    p.add_argument("--soft_sigma", type=float, default=3.0)
    p.add_argument("--soft_decode_temp", type=float, default=0.5)
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
    return args


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    r = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Device: {device}")

    dataset = PushTDatasetV7(args.zarr_path, args.obs_horizon, args.action_horizon,
                              args.pred_horizon, args.num_bins, args.obs_noise_std, args.act_noise_std)

    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(args.seed))
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
        "obs_dim": 5, "act_dim": 2, "num_refine_steps": args.num_refine_steps,
        "soft_sigma": args.soft_sigma, "soft_decode_temp": args.soft_decode_temp,
        "forward_dtype": "bfloat16",
    }

    model = build_model_v7(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}_v7 (iterative refine), Params: {num_params:,}")

    ema = EMA(model, decay=args.ema_decay)

    if args.exp_name is None:
        args.exp_name = f"v7_{args.model_type}_h{args.hidden_size}_refine{args.num_refine_steps}"

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

            output = model(obs, clean_act_tokens=act_tokens, act_normalized=act_norm)
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

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                act_tokens = batch["act_tokens"].to(device)
                act_norm = batch["act_normalized"].to(device)
                output = model(obs, clean_act_tokens=act_tokens, act_normalized=act_norm)
                val_loss += output["loss"].item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} val={avg_val:.4f}")
        log_dict = {"epoch": epoch + 1, "train/epoch_loss": avg_loss, "val/loss": avg_val}

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print(f"Evaluating (refine_steps={args.num_refine_steps})...")

            # Eval with different refine steps
            for n_steps in [1, args.num_refine_steps, args.num_refine_steps * 2]:
                results = evaluate_v7(model, dataset, device,
                                      num_episodes=args.eval_episodes,
                                      num_refine_steps=n_steps,
                                      temperature=args.soft_decode_temp,
                                      action_horizon=args.action_horizon)
                print(f"  [model K={n_steps}] mean={results['mean_score']:.4f} (range: {results['min_score']:.3f}-{results['max_score']:.3f})")
                log_dict[f"eval/model_K{n_steps}_mean"] = results["mean_score"]

            # EMA eval
            ema.apply_shadow(model)
            results_ema = evaluate_v7(model, dataset, device,
                                      num_episodes=args.eval_episodes,
                                      num_refine_steps=args.num_refine_steps,
                                      temperature=args.soft_decode_temp,
                                      action_horizon=args.action_horizon)
            ema.restore(model)
            print(f"  [EMA K={args.num_refine_steps}] mean={results_ema['mean_score']:.4f}")
            log_dict["eval/ema_mean"] = results_ema["mean_score"]

            best_this = max(results["mean_score"], results_ema["mean_score"])
            if best_this > best_eval:
                best_eval = best_this
                torch.save(model.state_dict(), os.path.join(save_dir, "best_eval.pt"))
                log_dict["eval/best"] = best_eval

        wandb.log(log_dict, step=global_step)

    print(f"Done. Best eval: {best_eval:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
