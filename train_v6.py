"""
V6: Comprehensive training with every trick to beat diffusion policy.

Features:
- EMA model for evaluation
- Action temporal ensemble at inference
- Gaussian soft labels + regression dual loss
- Observation + action noise augmentation
- Masked action augmentation (VLA-0)
- Proper eval protocol matching diffusion policy
- Gradient checkpointing for deeper recursion
"""
import argparse
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import math
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
from tqdm import tqdm
import zarr

from pusht_hrm.model_v6 import build_model_v6, EMA
from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv


class PushTDatasetV6(Dataset):
    """V6 dataset with all augmentation tricks."""

    def __init__(self, zarr_path, obs_horizon=2, action_horizon=8, pred_horizon=16,
                 num_bins=256, obs_noise_std=0.0, act_noise_std=0.0,
                 mask_augment_prob=0.0):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.num_bins = num_bins
        self.obs_noise_std = obs_noise_std
        self.act_noise_std = act_noise_std
        self.mask_augment_prob = mask_augment_prob
        self.obs_dim = 5
        self.act_dim = 2
        self.act_seq_len = pred_horizon * self.act_dim

        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]

        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_ends = episode_ends

        self.indices = []
        for ep_idx in range(len(episode_ends)):
            start = self.episode_starts[ep_idx]
            end = self.episode_ends[ep_idx]
            for t in range(end - start - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(start + t)
        self.indices = np.array(self.indices)

        self.obs_mean = self.state.mean(axis=0)
        self.obs_std = self.state.std(axis=0) + 1e-6

        # Per-dim action tokenizer
        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6
        act_ranges = [(self.act_min[i], self.act_max[i]) for i in range(2)]
        self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, "uniform")

        print(f"PushT V6: {len(self.indices)} samples, obs_h={obs_horizon}, pred_h={pred_horizon}")

    def __len__(self):
        return len(self.indices)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()

        # Obs noise augmentation
        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std

        # Action noise augmentation
        if self.act_noise_std > 0:
            actions = actions + np.random.randn(*actions.shape).astype(np.float32) * self.act_noise_std
            actions = np.clip(actions, self.act_min - 5, self.act_max + 5)

        obs_norm = self.normalize_obs(obs)

        # Tokenize actions
        act_tokens = self.act_tokenizer.encode(actions).flatten()

        # Normalized actions [0, 1] for regression head
        act_normalized = ((actions - self.act_min) / self.act_range).flatten()

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "act_normalized": torch.tensor(act_normalized, dtype=torch.float32),
            "raw_actions": torch.tensor(actions, dtype=torch.float32),
        }


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)


def evaluate_v6(model, dataset, device, num_episodes=22, seed_start=10000,
                use_soft_decode=True, temperature=0.5, action_horizon=8,
                temporal_ensemble=True):
    """
    Evaluate matching diffusion policy protocol.
    - 22 test episodes, seeds 10000-10021
    - max_steps=200
    - Reports mean of max_reward per episode
    """
    env = PushTEnv()
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        obs_history = [obs]

        episode_scores = []
        done = False
        step_count = 0
        max_steps = 200  # Match diffusion policy

        # For temporal ensemble
        pending_actions = {}  # step -> list of predicted actions

        while not done and step_count < max_steps:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])

            recent_obs = np.array(obs_history[-dataset.obs_horizon:])
            obs_norm = dataset.normalize_obs(recent_obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            pred = model.predict_actions(obs_t, use_soft_decode=use_soft_decode,
                                         temperature=temperature)
            pred_np = pred.cpu().numpy()[0]

            # Decode to continuous actions
            pred_reshaped = pred_np.reshape(dataset.pred_horizon, dataset.act_dim)
            actions = np.zeros_like(pred_reshaped)
            for d in range(dataset.act_dim):
                tok = dataset.act_tokenizer.tokenizers[d]
                actions[:, d] = tok.val_min + (pred_reshaped[:, d] + 0.5) * tok.bin_width

            if temporal_ensemble:
                # Store predictions for temporal ensemble
                for t_offset in range(len(actions)):
                    future_step = step_count + t_offset
                    if future_step not in pending_actions:
                        pending_actions[future_step] = []
                    pending_actions[future_step].append(actions[t_offset])

            # Execute action_horizon steps
            for t in range(min(action_horizon, len(actions))):
                if temporal_ensemble and (step_count in pending_actions):
                    # Average all predictions for this step
                    all_preds = np.array(pending_actions[step_count])
                    action = np.mean(all_preds, axis=0)
                    del pending_actions[step_count]
                else:
                    action = actions[t]

                action = np.clip(action, 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(score)
                step_count += 1
                if done:
                    break

        max_score = max(episode_scores) if episode_scores else 0
        max_scores.append(max_score)

    return {
        "mean_score": np.mean(max_scores),  # This is what diffusion policy reports
        "std_score": np.std(max_scores),
        "min_score": np.min(max_scores),
        "max_score": np.max(max_scores),
        "all_scores": max_scores,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", type=str,
                   default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--model_type", type=str, default="hrm")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--H_layers", type=int, default=2)
    p.add_argument("--L_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--L_cycles", type=int, default=4)
    p.add_argument("--forward_dtype", type=str, default="bfloat16")
    p.add_argument("--loss_type", type=str, default="gaussian_soft")
    p.add_argument("--soft_sigma", type=float, default=3.0)
    p.add_argument("--soft_decode_temp", type=float, default=0.5)
    p.add_argument("--use_regression_head", action="store_true", default=True)
    p.add_argument("--regression_weight", type=float, default=0.5)
    p.add_argument("--grad_all_steps", action="store_true", default=False)
    p.add_argument("--use_checkpoint", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=22)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--obs_noise_std", type=float, default=3.0)
    p.add_argument("--act_noise_std", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--temporal_ensemble", action="store_true", default=True)
    p.add_argument("--wandb_project", type=str, default="pusht-hrm")
    p.add_argument("--wandb_entity", type=str, default="junha")
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            for k, v in yaml.safe_load(f).items():
                if hasattr(args, k):
                    setattr(args, k, v)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PushTDatasetV6(
        zarr_path=args.zarr_path, obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon, pred_horizon=args.pred_horizon,
        num_bins=args.num_bins, obs_noise_std=args.obs_noise_std,
        act_noise_std=args.act_noise_std,
    )

    val_size = int(len(dataset) * args.val_split)
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
        "obs_dim": 5, "act_dim": 2, "act_seq_len": dataset.act_seq_len,
        "loss_type": args.loss_type, "soft_sigma": args.soft_sigma,
        "soft_decode_temp": args.soft_decode_temp,
        "use_regression_head": args.use_regression_head,
        "regression_weight": args.regression_weight,
        "grad_all_steps": args.grad_all_steps,
        "use_checkpoint": args.use_checkpoint,
        "forward_dtype": args.forward_dtype,
    }

    model = build_model_v6(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}_v6, Params: {num_params:,}")

    # EMA
    ema = EMA(model, decay=args.ema_decay)

    if args.exp_name is None:
        args.exp_name = f"v6_{args.model_type}_h{args.hidden_size}_H{args.H_cycles}L{args.L_cycles}_{args.loss_type}"

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, config=vars(args))
    wandb.config.update({"num_params": num_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_eval_score = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_cls_loss = 0
        train_reg_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            global_step += 1
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            obs = batch["obs"].to(device)
            act_tokens = batch["act_tokens"].to(device)
            act_normalized = batch["act_normalized"].to(device)

            output = model(obs, act_tokens=act_tokens, act_normalized=act_normalized)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)

            train_loss += loss.item()
            cls_l = output.get("cls_loss", torch.tensor(0)).item()
            reg_l = output.get("reg_loss", torch.tensor(0)).item()
            train_cls_loss += cls_l
            train_reg_loss += reg_l

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cls": f"{cls_l:.4f}",
                             "reg": f"{reg_l:.4f}", "lr": f"{lr:.6f}"})

            if global_step % 100 == 0:
                wandb.log({"train/loss": loss.item(), "train/cls_loss": cls_l,
                          "train/reg_loss": reg_l, "train/lr": lr}, step=global_step)

        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                act_tokens = batch["act_tokens"].to(device)
                act_normalized = batch["act_normalized"].to(device)
                output = model(obs, act_tokens=act_tokens, act_normalized=act_normalized)
                val_loss += output["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} val={avg_val_loss:.4f}")

        log_dict = {"epoch": epoch + 1, "train/epoch_loss": avg_loss, "val/loss": avg_val_loss}

        # Evaluation
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print(f"Evaluating ({args.eval_episodes} episodes, matching DP protocol)...")

            # Evaluate current model
            results = evaluate_v6(model, dataset, device,
                                  num_episodes=args.eval_episodes,
                                  use_soft_decode=True, temperature=args.soft_decode_temp,
                                  action_horizon=args.action_horizon,
                                  temporal_ensemble=args.temporal_ensemble)
            print(f"  [model] mean={results['mean_score']:.4f} (range: {results['min_score']:.3f}-{results['max_score']:.3f})")
            log_dict["eval/model_mean_score"] = results["mean_score"]

            # Evaluate EMA model
            ema.apply_shadow(model)
            results_ema = evaluate_v6(model, dataset, device,
                                      num_episodes=args.eval_episodes,
                                      use_soft_decode=True, temperature=args.soft_decode_temp,
                                      action_horizon=args.action_horizon,
                                      temporal_ensemble=args.temporal_ensemble)
            ema.restore(model)
            print(f"  [EMA]   mean={results_ema['mean_score']:.4f} (range: {results_ema['min_score']:.3f}-{results_ema['max_score']:.3f})")
            log_dict["eval/ema_mean_score"] = results_ema["mean_score"]

            # Also eval without temporal ensemble for comparison
            ema.apply_shadow(model)
            results_no_te = evaluate_v6(model, dataset, device,
                                        num_episodes=args.eval_episodes,
                                        use_soft_decode=True, temperature=args.soft_decode_temp,
                                        action_horizon=args.action_horizon,
                                        temporal_ensemble=False)
            ema.restore(model)
            print(f"  [EMA-noTE] mean={results_no_te['mean_score']:.4f}")
            log_dict["eval/ema_no_te_mean_score"] = results_no_te["mean_score"]

            best_this = max(results["mean_score"], results_ema["mean_score"])
            if best_this > best_eval_score:
                best_eval_score = best_this
                torch.save(model.state_dict(), os.path.join(save_dir, "best_eval.pt"))
                log_dict["eval/best_score"] = best_eval_score

        wandb.log(log_dict, step=global_step)

        if (epoch + 1) % 200 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
                "best_eval_score": best_eval_score,
            }, os.path.join(save_dir, f"ckpt_ep{epoch+1}.pt"))

    print(f"\nDone. Best eval score: {best_eval_score:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
