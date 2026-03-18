"""
Proper diffusion policy baseline matching the original paper.

Key differences from our simple MLP version:
1. 1D U-Net (Conv1D) - not MLP
2. obs_as_global_cond with FiLM modulation
3. Cosine noise schedule (squaredcos_cap_v2)
4. Per-dimension linear normalization
5. EMA model
6. 100 diffusion steps
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
import copy

from pusht_hrm.pusht_env import PushTEnv


# ============== Cosine Beta Schedule ==============
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ============== 1D U-Net Components ==============
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


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        # Ensure n_groups divides out_channels
        while out_channels % n_groups != 0 and n_groups > 1:
            n_groups = n_groups // 2
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
        ])
        # FiLM: scale + bias
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, 2*C, 1)
        scale, bias = embed.chunk(2, dim=1)
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=[256,512,1024], kernel_size=5):
        super().__init__()
        self.input_dim = input_dim
        dsed = 256  # diffusion step embed dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4), nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        # Project input to start_dim to avoid GroupNorm issues
        start_dim = down_dims[0]
        self.input_proj = nn.Conv1d(input_dim, start_dim, 1)
        self.output_proj = nn.Conv1d(start_dim, input_dim, 1)

        all_dims = [start_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Down
        self.down_modules = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size),
                nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1),  # downsample
            ]))

        # Mid
        mid_dim = all_dims[-1]
        self.mid = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size),
        ])

        # Up
        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out):
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size),
                nn.ConvTranspose1d(dim_out, dim_out, 4, stride=2, padding=1),  # upsample
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(all_dims[1], all_dims[1], kernel_size),
            nn.Conv1d(all_dims[1], start_dim, 1),
        )

    def forward(self, x, timestep, global_cond):
        """x: (B, T, D) -> (B, T, D), timestep: (B,), global_cond: (B, cond_dim)"""
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.input_proj(x)  # (B, start_dim, T)

        t_emb = self.diffusion_step_encoder(timestep.float())
        cond = torch.cat([t_emb, global_cond], dim=-1)

        # Down
        h = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            h.append(x)
            x = downsample(x)

        # Mid
        for mid_block in self.mid:
            x = mid_block(x, cond)

        # Up
        for (res1, res2, upsample), skip in zip(self.up_modules, reversed(h)):
            x = upsample(x)
            # Handle size mismatch
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        x = self.final_conv(x)
        x = self.output_proj(x)
        return x.permute(0, 2, 1)  # (B, T, D)


# ============== EMA ==============
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


# ============== DDPM ==============
class DDPMScheduler:
    def __init__(self, num_steps=100, schedule="cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            betas = torch.linspace(1e-4, 0.02, num_steps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)

        self.num_steps = num_steps
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / alphas)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    def to(self, device):
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'sqrt_alphas_cumprod',
                      'sqrt_one_minus_alphas_cumprod', 'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, x, noise, t):
        s_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s_omac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return s_ac * x + s_omac * noise

    @torch.no_grad()
    def step(self, model_output, t, sample):
        alpha = self.alphas[t].view(-1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip = self.sqrt_recip_alphas[t].view(-1, 1, 1)

        mean = sqrt_recip * (sample - beta / sqrt_omac * model_output)

        if t[0] > 0:
            noise = torch.randn_like(sample)
            sigma = torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1)
            return mean + sigma * noise
        return mean


# ============== Dataset ==============
class PushTDiffDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, obs_noise_std=3.0):
        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_noise_std = obs_noise_std

        self.indices = []
        for ep in range(len(episode_ends)):
            s, e = starts[ep], episode_ends[ep]
            for t in range(e - s - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(s + t)
        self.indices = np.array(self.indices)

        # Per-dimension normalization (like LinearNormalizer)
        self.obs_min = self.state.min(axis=0)
        self.obs_max = self.state.max(axis=0)
        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        print(f"Diffusion dataset: {len(self.indices)} samples")

    def normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 1

    def normalize_act(self, act):
        return 2 * (act - self.act_min) / (self.act_max - self.act_min + 1e-6) - 1

    def denormalize_act(self, act):
        return (act + 1) / 2 * (self.act_max - self.act_min + 1e-6) + self.act_min

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()
        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std
        return {
            "obs": torch.tensor(self.normalize_obs(obs), dtype=torch.float32),
            "actions": torch.tensor(self.normalize_act(actions), dtype=torch.float32),
        }


def evaluate(model, scheduler, dataset, device, num_episodes=22, seed_start=10000,
             n_action_steps=8, num_denoise=100):
    env = PushTEnv()
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        obs_history = [obs]
        scores = []
        done, step_count = False, 0

        while not done and step_count < 200:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])
            recent = np.array(obs_history[-dataset.obs_horizon:])
            obs_norm = dataset.normalize_obs(recent)
            # Global cond: flatten obs
            global_cond = torch.tensor(obs_norm.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

            # Denoise
            model.eval()
            noisy = torch.randn(1, dataset.pred_horizon, 2, device=device)

            for i in reversed(range(num_denoise)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                pred_noise = model(noisy, t, global_cond)
                noisy = scheduler.step(pred_noise, t, noisy)

            actions_norm = noisy[0].cpu().numpy()
            actions = dataset.denormalize_act(actions_norm)

            for t in range(min(n_action_steps, len(actions))):
                action = np.clip(actions[t], 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                scores.append(float(score))
                step_count += 1
                if done:
                    break

        max_scores.append(max(scores) if scores else 0)

    return {"mean_score": np.mean(max_scores), "std_score": np.std(max_scores),
            "min_score": np.min(max_scores), "max_score": np.max(max_scores)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--num_denoise", type=int, default=100)
    p.add_argument("--wandb_entity", default="junha")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = PushTDiffDataset(args.zarr_path)
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    # Proper U-Net (matching DP paper)
    obs_dim = dataset.obs_horizon * 5
    model = ConditionalUnet1D(input_dim=2, global_cond_dim=obs_dim,
                               down_dims=[256, 512, 1024], kernel_size=5).to(device)
    scheduler = DDPMScheduler(args.num_denoise, schedule="cosine").to(device)
    ema = EMAModel(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Proper Diffusion U-Net: {num_params:,} params")

    wandb.init(project="pusht-hrm", entity=args.wandb_entity,
               name=f"diffusion_unet1d_proper", config=vars(args))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best_eval = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs = batch["obs"].to(device)  # (B, obs_h, 5)
            actions = batch["actions"].to(device)  # (B, T, 2)
            global_cond = obs.reshape(obs.shape[0], -1)  # flatten

            noise = torch.randn_like(actions)
            t = torch.randint(0, args.num_denoise, (actions.shape[0],), device=device)
            noisy = scheduler.add_noise(actions, noise, t)

            pred_noise = model(noisy, t, global_cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}")
        wandb.log({"epoch": epoch+1, "train/loss": avg_loss})

        if (epoch + 1) % args.eval_interval == 0:
            # Eval with EMA
            ema.apply(model)
            results = evaluate(model, scheduler, dataset, device,
                               n_action_steps=args.n_action_steps,
                               num_denoise=args.num_denoise)
            ema.restore(model)
            print(f"  [EMA] mean={results['mean_score']:.4f} (range: {results['min_score']:.3f}-{results['max_score']:.3f})")
            wandb.log({"eval/mean_score": results["mean_score"]})
            if results["mean_score"] > best_eval:
                best_eval = results["mean_score"]
                torch.save(model.state_dict(), "checkpoints/diffusion_proper_best.pt")

    print(f"Done. Best: {best_eval:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
