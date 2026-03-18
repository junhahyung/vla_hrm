"""
Run actual diffusion policy on PushT for fair comparison.
Uses a simple DDPM implementation (no need for the full repo).
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
from tqdm import tqdm
import zarr
import math

from pusht_hrm.pusht_env import PushTEnv


# ============== Simple DDPM for action prediction ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalMLP(nn.Module):
    """Simple MLP that takes obs + noisy actions + timestep → predicted noise."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, num_layers=6):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = [nn.Linear(action_dim + hidden_dim * 2, hidden_dim), nn.Mish()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Mish()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_action, obs, timestep):
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_proj(obs)
        x = torch.cat([noisy_action, obs_emb, t_emb], dim=-1)
        return self.net(x)


class DDPM:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register = {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
            'sqrt_recip_alphas': torch.sqrt(1 / alphas),
            'posterior_variance': betas * (1 - torch.cat([torch.tensor([0.0]), alphas_cumprod[:-1]])) / (1 - alphas_cumprod),
        }

    def to(self, device):
        for k, v in self.register.items():
            self.register[k] = v.to(device)
        return self

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.register['sqrt_alphas_cumprod'][t][:, None]
        sqrt_omac = self.register['sqrt_one_minus_alphas_cumprod'][t][:, None]
        return sqrt_ac * x_start + sqrt_omac * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, obs, t):
        pred_noise = model(x, obs, t.float())
        alpha = self.register['alphas'][t][:, None]
        alpha_cumprod = self.register['alphas_cumprod'][t][:, None]
        beta = self.register['betas'][t][:, None]
        sqrt_recip = self.register['sqrt_recip_alphas'][t][:, None]
        sqrt_omac = self.register['sqrt_one_minus_alphas_cumprod'][t][:, None]

        mean = sqrt_recip * (x - beta / sqrt_omac * pred_noise)

        if t[0] > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.register['posterior_variance'][t])[:, None]
            return mean + sigma * noise
        return mean


class PushTDiffusionDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, obs_noise_std=3.0):
        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_noise_std = obs_noise_std

        self.indices = []
        for ep_idx in range(len(episode_ends)):
            s, e = self.episode_starts[ep_idx], episode_ends[ep_idx]
            for t in range(e - s - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(s + t)
        self.indices = np.array(self.indices)

        self.obs_mean = self.state.mean(axis=0)
        self.obs_std = self.state.std(axis=0) + 1e-6
        self.act_mean = self.action.mean(axis=0)
        self.act_std = self.action.std(axis=0) + 1e-6
        print(f"Diffusion dataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()
        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std
        obs_norm = ((obs - self.obs_mean) / self.obs_std).flatten()
        act_norm = (actions - self.act_mean) / self.act_std
        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "actions": torch.tensor(act_norm.flatten(), dtype=torch.float32),
        }


def evaluate_diffusion(model, ddpm, dataset, device, num_episodes=22, seed_start=10000,
                        num_denoise_steps=100, action_horizon=8):
    env = PushTEnv()
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        obs_history = [obs]
        episode_scores = []
        done = False
        step_count = 0

        while not done and step_count < 200:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])
            recent = np.array(obs_history[-dataset.obs_horizon:])
            obs_norm = ((recent - dataset.obs_mean) / dataset.obs_std).flatten()
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            # Denoise from random noise
            model.eval()
            x = torch.randn(1, dataset.pred_horizon * 2, device=device)
            for i in reversed(range(num_denoise_steps)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                x = ddpm.p_sample(model, x, obs_t, t)

            actions = x[0].cpu().numpy().reshape(dataset.pred_horizon, 2)
            actions = actions * dataset.act_std + dataset.act_mean

            for t in range(min(action_horizon, len(actions))):
                action = np.clip(actions[t], 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(float(score))
                step_count += 1
                if done:
                    break

        max_scores.append(max(episode_scores) if episode_scores else 0)

    return {"mean_score": np.mean(max_scores), "std_score": np.std(max_scores),
            "min_score": np.min(max_scores), "max_score": np.max(max_scores)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_denoise_steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--wandb_entity", default="junha")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = PushTDiffusionDataset(args.zarr_path)
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    obs_dim = dataset.obs_horizon * 5
    action_dim = dataset.pred_horizon * 2

    model = ConditionalMLP(obs_dim, action_dim, args.hidden_size, args.num_layers).to(device)
    ddpm = DDPM(args.num_denoise_steps).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Diffusion MLP: {num_params:,} params")

    wandb.init(project="pusht-hrm", entity=args.wandb_entity, name=f"diffusion_mlp_h{args.hidden_size}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_eval = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs = batch["obs"].to(device)
            actions = batch["actions"].to(device)

            t = torch.randint(0, args.num_denoise_steps, (obs.shape[0],), device=device)
            noisy_actions, noise = ddpm.q_sample(actions, t)
            pred_noise = model(noisy_actions, obs, t.float())
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train/loss": avg_loss})

        if (epoch + 1) % args.eval_interval == 0:
            results = evaluate_diffusion(model, ddpm, dataset, device,
                                          action_horizon=args.action_horizon)
            print(f"  mean={results['mean_score']:.4f} (range: {results['min_score']:.3f}-{results['max_score']:.3f})")
            wandb.log({"eval/mean_score": results["mean_score"]})
            if results["mean_score"] > best_eval:
                best_eval = results["mean_score"]
                torch.save(model.state_dict(), "checkpoints/diffusion_best.pt")

    print(f"Done. Best: {best_eval:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
