"""
V3 training: regression + optional classification with recursive reasoning.
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
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm

from pusht_hrm.dataset_v3 import PushTDatasetV3
from pusht_hrm.model_v3 import build_model_v3
from pusht_hrm.pusht_env import PushTEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", type=str,
                   default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--tokenizer_type", type=str, default="uniform")
    p.add_argument("--mu", type=float, default=255.0)
    p.add_argument("--model_type", type=str, default="trm")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--H_layers", type=int, default=2)
    p.add_argument("--L_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--L_cycles", type=int, default=4)
    p.add_argument("--forward_dtype", type=str, default="bfloat16")
    p.add_argument("--output_mode", type=str, default="regression",
                   choices=["regression", "classification", "both"])
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--obs_noise_std", type=float, default=0.0)
    p.add_argument("--wandb_project", type=str, default="pusht-hrm")
    p.add_argument("--wandb_entity", type=str, default=None)
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


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)


def evaluate_v3(model, dataset, device, num_episodes=50, seed_start=0):
    """Evaluate V3 model (regression output) in PushT env."""
    env = PushTEnv()
    scores, max_scores = [], []

    for ep in range(num_episodes):
        env.seed(seed_start + ep * 1000)
        obs = env.reset()
        obs_history = [obs]
        episode_scores = []
        done = False
        step_count = 0

        while not done and step_count < 300:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])

            recent_obs = np.array(obs_history[-dataset.obs_horizon:])
            obs_norm = dataset.normalize_obs(recent_obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            pred = model.predict_actions(obs_t)  # (1, T_a, act_dim) or (1, T_a, act_dim) tokens

            if pred.dtype == torch.float32 or pred.dtype == torch.float16 or pred.dtype == torch.bfloat16:
                # Regression output: denormalize
                actions = dataset.denormalize_actions(pred[0].cpu()).numpy()
            else:
                # Classification output: decode tokens
                tokens = pred[0].cpu().numpy()
                actions = dataset.act_tokenizer.decode(tokens)

            for t in range(min(dataset.action_horizon, len(actions))):
                action = np.clip(actions[t], 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(score)
                step_count += 1
                if done:
                    break

        max_score = max(episode_scores) if episode_scores else 0
        scores.append(episode_scores[-1] if episode_scores else 0)
        max_scores.append(max_score)

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "mean_max_score": np.mean(max_scores),
        "std_max_score": np.std(max_scores),
        "num_episodes": num_episodes,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PushTDatasetV3(
        zarr_path=args.zarr_path, obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon, pred_horizon=args.pred_horizon,
        num_bins=args.num_bins, tokenizer_type=args.tokenizer_type,
        mu=args.mu, obs_noise_std=args.obs_noise_std,
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
        "obs_dim": 5, "act_dim": 2, "output_mode": args.output_mode,
        "forward_dtype": args.forward_dtype,
    }

    model = build_model_v3(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}_v3 ({args.output_mode}), Params: {num_params:,}")

    if args.exp_name is None:
        args.exp_name = f"v3_{args.model_type}_{args.output_mode}_h{args.hidden_size}_H{args.H_cycles}L{args.L_cycles}"

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, config=vars(args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_eval_score = 0.0
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
            actions = batch["actions"].to(device)
            act_tokens = batch["act_tokens"].to(device)

            output = model(obs, target_actions=actions, act_tokens=act_tokens,
                          label_smoothing=args.label_smoothing)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

            if global_step % 100 == 0:
                log = {"train/loss": loss.item(), "train/lr": lr}
                if "reg_loss" in output:
                    log["train/reg_loss"] = output["reg_loss"].item()
                if "cls_loss" in output:
                    log["train/cls_loss"] = output["cls_loss"].item()
                wandb.log(log, step=global_step)

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                actions = batch["actions"].to(device)
                act_tokens = batch["act_tokens"].to(device)
                output = model(obs, target_actions=actions, act_tokens=act_tokens)
                val_loss += output["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        log_dict = {"epoch": epoch + 1, "train/epoch_loss": avg_train_loss, "val/loss": avg_val_loss}

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_val.pt"))

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print(f"Evaluating ({args.eval_episodes} episodes)...")
            eval_results = evaluate_v3(model, dataset, device,
                                       num_episodes=args.eval_episodes, seed_start=100000)
            print(f"  mean_score={eval_results['mean_score']:.4f} max_score={eval_results['mean_max_score']:.4f}")

            log_dict.update({
                "eval/mean_score": eval_results["mean_score"],
                "eval/mean_max_score": eval_results["mean_max_score"],
            })
            if eval_results["mean_max_score"] > best_eval_score:
                best_eval_score = eval_results["mean_max_score"]
                torch.save(model.state_dict(), os.path.join(save_dir, "best_eval.pt"))

        wandb.log(log_dict, step=global_step)

    print(f"Done. Best eval: {best_eval_score:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
