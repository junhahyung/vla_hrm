"""
V2 training: continuous obs encoder + discrete action decoder.
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

from pusht_hrm.dataset_v2 import PushTDatasetV2
from pusht_hrm.model_v2 import build_model_v2
from pusht_hrm.pusht_env import evaluate_policy_v2


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--zarr_path", type=str,
                   default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--action_horizon", type=int, default=8)

    # Tokenizer
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--tokenizer_type", type=str, default="uniform")
    p.add_argument("--mu", type=float, default=255.0)

    # Model
    p.add_argument("--model_type", type=str, default="trm")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--H_layers", type=int, default=2)
    p.add_argument("--L_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--L_cycles", type=int, default=4)
    p.add_argument("--forward_dtype", type=str, default="bfloat16")

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--obs_noise_std", type=float, default=0.0)

    # Logging
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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = PushTDatasetV2(
        zarr_path=args.zarr_path,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        pred_horizon=args.pred_horizon,
        num_bins=args.num_bins,
        tokenizer_type=args.tokenizer_type,
        mu=args.mu,
        obs_noise_std=args.obs_noise_std,
    )

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model_config = {
        "model_type": args.model_type,
        "vocab_size": args.num_bins,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "expansion": args.expansion,
        "H_layers": args.H_layers,
        "L_layers": args.L_layers,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "obs_dim": 5,
        "act_dim": 2,
        "forward_dtype": args.forward_dtype,
    }

    model = build_model_v2(model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}_v2, Parameters: {num_params:,}")

    if args.exp_name is None:
        args.exp_name = f"v2_{args.model_type}_h{args.hidden_size}_bins{args.num_bins}_H{args.H_cycles}L{args.L_cycles}"

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, config=vars(args))
    wandb.config.update({"num_params": num_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_val_loss = float("inf")
    best_eval_score = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            global_step += 1
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            obs = batch["obs"].to(device)
            act_tokens = batch["act_tokens"].to(device)

            output = model(obs, act_tokens, label_smoothing=args.label_smoothing)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                preds = output["logits"].argmax(dim=-1)
                correct = (preds == act_tokens).sum().item()
                train_correct += correct
                train_total += act_tokens.numel()

            train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.6f}",
                "acc": f"{train_correct/max(train_total,1)*100:.1f}%"
            })

            if global_step % 100 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/accuracy": train_correct / max(train_total, 1),
                }, step=global_step)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                act_tokens = batch["act_tokens"].to(device)
                output = model(obs, act_tokens)
                val_loss += output["loss"].item()
                preds = output["logits"].argmax(dim=-1)
                val_correct += (preds == act_tokens).sum().item()
                val_total += act_tokens.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"train_acc={train_acc*100:.1f}% val_acc={val_acc*100:.1f}%")

        log_dict = {
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "train/epoch_accuracy": train_acc,
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_val.pt"))

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print(f"Evaluating in PushT environment ({args.eval_episodes} episodes)...")
            eval_results = evaluate_policy_v2(
                model, dataset, device,
                num_episodes=args.eval_episodes,
                seed_start=100000,
            )
            print(f"  mean_score={eval_results['mean_score']:.4f} "
                  f"mean_max_score={eval_results['mean_max_score']:.4f}")

            log_dict.update({
                "eval/mean_score": eval_results["mean_score"],
                "eval/std_score": eval_results["std_score"],
                "eval/mean_max_score": eval_results["mean_max_score"],
            })

            if eval_results["mean_max_score"] > best_eval_score:
                best_eval_score = eval_results["mean_max_score"]
                torch.save(model.state_dict(), os.path.join(save_dir, "best_eval.pt"))
                log_dict["eval/best_max_score"] = best_eval_score

        wandb.log(log_dict, step=global_step)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_eval_score": best_eval_score,
        }, os.path.join(save_dir, "latest.pt"))

    print(f"\nDone. Best val loss: {best_val_loss:.4f}, Best eval score: {best_eval_score:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
