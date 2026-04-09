"""
Reproduce "Much Ado About Noising" experiments using HRM architecture.

Methods implemented:
1. Regression (RCP) - simple L2 regression baseline
2. Flow Matching - flow-based generative control policy
3. MIP - Minimal Iterative Policy (2-step from the paper)
4. Diffusion Policy (DDPM) - noise-prediction diffusion
5. HRM (ours) - Hierarchical Reasoning Model with recursive cycles
6. TRM (ours) - shared-weight variant

All methods use:
- Same 20D keypoint observations (matching DP exactly)
- Same pred_horizon=16, obs_horizon=2, action_horizon=8
- Same dataset, normalization, and evaluation protocol
"""
import argparse, os, sys, math, json, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import wandb
from tqdm import tqdm
import zarr

os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_policy'))
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


# ============== Shared Building Blocks ==============

def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d, exp=2.0):
        super().__init__()
        inter = round(exp * d * 2 / 3)
        inter = ((inter + 255) // 256) * 256
        self.w = nn.Linear(d, inter * 2, bias=False)
        self.down = nn.Linear(inter, d, bias=False)
    def forward(self, x):
        g, u = self.w(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)

class Attn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nh, self.hd)
        q, k, v = [qkv[:,:,i].transpose(1,2) for i in range(3)]
        o = F.scaled_dot_product_attention(q, k, v).transpose(1,2).reshape(B, L, -1)
        return self.out(o)

class Block(nn.Module):
    def __init__(self, d, nh, exp=2.0):
        super().__init__()
        self.attn = Attn(d, nh)
        self.mlp = SwiGLU(d, exp)
    def forward(self, x):
        x = rms_norm(x + self.attn(x))
        return rms_norm(x + self.mlp(x))

class RecursiveModule(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    def forward(self, h, inject):
        h = h + inject
        for b in self.blocks:
            h = b(h)
        return h


# ============== Shared Encoder ==============

class ObsEncoder(nn.Module):
    """Shared obs encoder for all methods."""
    def __init__(self, obs_dim=20, d=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d),
        )
    def forward(self, obs):
        return self.enc(obs.float())


# ============== Method 1: Regression (RCP) ==============

class RegressionPolicy(nn.Module):
    """Simple regression baseline (L2 loss). The RCP from the paper."""
    def __init__(self, obs_dim=20, act_dim=2, d=256, nh=8, n_layers=4,
                 obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_len = obs_horizon
        self.act_len = pred_horizon
        self.total_len = self.obs_len + self.act_len

        self.obs_enc = ObsEncoder(obs_dim, d)
        self.act_q = nn.Parameter(torch.randn(pred_horizon, d) * 0.02)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)

        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))

    def forward(self, obs, target_actions=None):
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_enc(obs)
        act_emb = self.act_q.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos = self.pos_emb(torch.arange(self.total_len, device=device))
        typ = self.type_emb(torch.cat([
            torch.zeros(self.obs_len, dtype=torch.long, device=device),
            torch.ones(self.act_len, dtype=torch.long, device=device),
        ]))
        seq = seq + pos + typ
        for b in self.blocks:
            seq = b(seq)
        pred = self.head(seq[:, self.obs_len:].float())
        loss = F.mse_loss(pred, target_actions) if target_actions is not None else None
        return pred, loss


# ============== Method 2: Flow Matching ==============

class FlowPolicy(nn.Module):
    """Flow matching policy. Predicts velocity field v_t for interpolant I_t = t*a + (1-t)*z."""
    def __init__(self, obs_dim=20, act_dim=2, d=256, nh=8, n_layers=4,
                 obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.obs_len = obs_horizon
        self.act_len = pred_horizon
        self.total_len = self.obs_len + self.act_len

        self.obs_enc = ObsEncoder(obs_dim, d)
        self.act_proj = nn.Linear(act_dim, d)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d),
        )

        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))

    def forward_vel(self, obs, noisy_actions, t):
        """Predict velocity field given obs, noisy actions I_t, and time t."""
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_enc(obs)
        act_emb = self.act_proj(noisy_actions)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos = self.pos_emb(torch.arange(self.total_len, device=device))
        typ = self.type_emb(torch.cat([
            torch.zeros(self.obs_len, dtype=torch.long, device=device),
            torch.ones(self.act_len, dtype=torch.long, device=device),
        ]))
        t_emb = self.time_emb(t.view(-1, 1)).unsqueeze(1)
        seq = seq + pos + typ + t_emb
        for b in self.blocks:
            seq = b(seq)
        return self.head(seq[:, self.obs_len:].float())

    def forward(self, obs, target_actions=None):
        B, device = obs.shape[0], obs.device
        if target_actions is not None:
            # Training: sample t, create interpolant, predict velocity
            t = torch.rand(B, device=device)
            z = torch.randn_like(target_actions)
            I_t = t.view(-1,1,1) * target_actions + (1 - t.view(-1,1,1)) * z
            target_vel = target_actions - z  # dI_t/dt = a - z
            pred_vel = self.forward_vel(obs, I_t, t)
            loss = F.mse_loss(pred_vel, target_vel)
            return None, loss
        else:
            # Inference: Euler integration from noise
            x = torch.randn(B, self.pred_horizon, self.act_dim, device=device)
            n_steps = 10
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                v = self.forward_vel(obs, x, t)
                x = x + v * dt
            return x, None

    @torch.no_grad()
    def sample(self, obs, n_steps=10):
        """Inference with Euler integration."""
        B, device = obs.shape[0], obs.device
        x = torch.randn(B, self.pred_horizon, self.act_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward_vel(obs, x, t)
            x = x + v * dt
        return x


# ============== Method 3: MIP (Minimal Iterative Policy) ==============

class MIPPolicy(nn.Module):
    """
    MIP from the paper: two-step iterative computation.
    Step 1: a_0 = pi(obs, zeros, t=0)  (predict from nothing)
    Step 2: a = pi(obs, t*a_0 + (1-t)*z, t*)  (refine with stochastic interpolant)
    t* = 0.9 fixed.
    """
    def __init__(self, obs_dim=20, act_dim=2, d=256, nh=8, n_layers=4,
                 obs_horizon=2, pred_horizon=16, t_star=0.9):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.t_star = t_star
        self.obs_len = obs_horizon
        self.act_len = pred_horizon
        self.total_len = self.obs_len + self.act_len

        self.obs_enc = ObsEncoder(obs_dim, d)
        self.act_proj = nn.Linear(act_dim, d)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)
        self.time_emb = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d),
        )

        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        # MIP predicts actions directly (not velocity)
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))

    def forward_pred(self, obs, input_actions, t):
        """Predict actions given obs, input (noisy/zero) actions, and time t."""
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_enc(obs)
        act_emb = self.act_proj(input_actions)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos = self.pos_emb(torch.arange(self.total_len, device=device))
        typ = self.type_emb(torch.cat([
            torch.zeros(self.obs_len, dtype=torch.long, device=device),
            torch.ones(self.act_len, dtype=torch.long, device=device),
        ]))
        t_emb = self.time_emb(t.view(-1, 1)).unsqueeze(1)
        seq = seq + pos + typ + t_emb
        for b in self.blocks:
            seq = b(seq)
        return self.head(seq[:, self.obs_len:].float())

    def forward(self, obs, target_actions=None):
        B, device = obs.shape[0], obs.device
        if target_actions is not None:
            # Training: supervise both steps
            # Step 1: from zeros at t=0
            zeros = torch.zeros(B, self.pred_horizon, self.act_dim, device=device)
            t0 = torch.zeros(B, device=device)
            pred_0 = self.forward_pred(obs, zeros, t0)
            loss_0 = F.mse_loss(pred_0, target_actions)

            # Step 2: from interpolant at t*
            z = torch.randn_like(target_actions)
            t_star = torch.full((B,), self.t_star, device=device)
            I_t = self.t_star * target_actions + (1 - self.t_star) * z
            pred_1 = self.forward_pred(obs, I_t, t_star)
            loss_1 = F.mse_loss(pred_1, target_actions)

            return pred_0, loss_0 + loss_1
        else:
            # Inference: two-step
            zeros = torch.zeros(B, self.pred_horizon, self.act_dim, device=device)
            t0 = torch.zeros(B, device=device)
            a0 = self.forward_pred(obs, zeros, t0)
            # Deterministic second step (z=0 at inference)
            t_star = torch.full((B,), self.t_star, device=device)
            I_t = self.t_star * a0  # z=0 at inference
            pred = self.forward_pred(obs, I_t, t_star)
            return pred, None


# ============== Method 4: Diffusion (DDPM) ==============

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

class DiffusionPolicy(nn.Module):
    """Transformer-based diffusion policy (noise prediction) for fair comparison."""
    def __init__(self, obs_dim=20, act_dim=2, d=256, nh=8, n_layers=4,
                 obs_horizon=2, pred_horizon=16, num_steps=100):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.num_steps = num_steps
        self.obs_len = obs_horizon
        self.act_len = pred_horizon
        self.total_len = self.obs_len + self.act_len

        self.obs_enc = ObsEncoder(obs_dim, d)
        self.act_proj = nn.Linear(act_dim, d)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)

        self.time_enc = nn.Sequential(
            SinusoidalPosEmb(d),
            nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d),
        )

        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))

        # Cosine schedule
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        ac = torch.cos(((x / num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)

        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1 / alphas))
        acp = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.register_buffer('posterior_var', betas * (1 - acp) / (1 - alphas_cumprod))

    def forward_noise(self, obs, noisy_actions, timestep):
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_enc(obs)
        act_emb = self.act_proj(noisy_actions)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos = self.pos_emb(torch.arange(self.total_len, device=device))
        typ = self.type_emb(torch.cat([
            torch.zeros(self.obs_len, dtype=torch.long, device=device),
            torch.ones(self.act_len, dtype=torch.long, device=device),
        ]))
        t_emb = self.time_enc(timestep.float()).unsqueeze(1)
        seq = seq + pos + typ + t_emb
        for b in self.blocks:
            seq = b(seq)
        return self.head(seq[:, self.obs_len:].float())

    def add_noise(self, x, noise, t):
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return s1 * x + s2 * noise

    def forward(self, obs, target_actions=None):
        B, device = obs.shape[0], obs.device
        if target_actions is not None:
            noise = torch.randn_like(target_actions)
            t = torch.randint(0, self.num_steps, (B,), device=device)
            noisy = self.add_noise(target_actions, noise, t)
            pred_noise = self.forward_noise(obs, noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            return None, loss
        else:
            return self.sample(obs), None

    @torch.no_grad()
    def sample(self, obs, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        B, device = obs.shape[0], obs.device
        x = torch.randn(B, self.pred_horizon, self.act_dim, device=device)
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_noise = self.forward_noise(obs, x, t)
            alpha = (1 - self.betas[t]).view(-1, 1, 1)
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_recip = self.sqrt_recip_alphas[t].view(-1, 1, 1)
            mean = sqrt_recip * (x - self.betas[t].view(-1,1,1) / sqrt_omac * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_var[t]).view(-1, 1, 1)
                x = mean + sigma * noise
            else:
                x = mean
        return x


# ============== Method 5: HRM (ours) ==============

class HRM(nn.Module):
    """Hierarchical Reasoning Model with separate H/L modules."""
    def __init__(self, obs_dim=20, act_dim=2, d=256, nh=8, n_layers=3,
                 H_cycles=5, L_cycles=6, obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.obs_len = obs_horizon
        self.act_len = pred_horizon
        self.total_len = self.obs_len + self.act_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.obs_enc = ObsEncoder(obs_dim, d)
        self.act_q = nn.Parameter(torch.randn(pred_horizon, d) * 0.02)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)

        self.H = RecursiveModule([Block(d, nh) for _ in range(n_layers)])
        self.L = RecursiveModule([Block(d, nh) for _ in range(n_layers)])

        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))
        self.z_H = nn.Parameter(torch.randn(d) * 0.02)
        self.z_L = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, obs, target_actions=None):
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_enc(obs)
        act_emb = self.act_q.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos = self.pos_emb(torch.arange(self.total_len, device=device))
        typ = self.type_emb(torch.cat([
            torch.zeros(self.obs_len, dtype=torch.long, device=device),
            torch.ones(self.act_len, dtype=torch.long, device=device),
        ]))
        seq = seq + pos + typ
        z_H = self.z_H.view(1,1,-1).expand(B, self.total_len, -1).clone()
        z_L = self.z_L.view(1,1,-1).expand(B, self.total_len, -1).clone()

        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    if not (h == self.H_cycles-1 and l == self.L_cycles-1):
                        z_L = self.L(z_L, z_H + seq)
                if not (h == self.H_cycles-1):
                    z_H = self.H(z_H, z_L)
        z_L = self.L(z_L, z_H + seq)
        z_H = self.H(z_H, z_L)

        pred = self.head(z_H[:, self.obs_len:].float())
        loss = F.mse_loss(pred, target_actions) if target_actions is not None else None
        return pred, loss


# ============== Method 6: TRM (shared weights) ==============

class TRM(nn.Module):
    """Tiny Recursive Model with shared H/L module (single reasoning module)."""
    def __init__(self, obs_dim=20, act_dim=2, d=256, nh=8, n_layers=3,
                 H_cycles=5, L_cycles=6, obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.obs_len = obs_horizon
        self.act_len = pred_horizon
        self.total_len = self.obs_len + self.act_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.obs_enc = ObsEncoder(obs_dim, d)
        self.act_q = nn.Parameter(torch.randn(pred_horizon, d) * 0.02)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)

        # Single shared module for both H and L
        self.shared = RecursiveModule([Block(d, nh) for _ in range(n_layers)])

        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))
        self.z_H = nn.Parameter(torch.randn(d) * 0.02)
        self.z_L = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, obs, target_actions=None):
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_enc(obs)
        act_emb = self.act_q.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos = self.pos_emb(torch.arange(self.total_len, device=device))
        typ = self.type_emb(torch.cat([
            torch.zeros(self.obs_len, dtype=torch.long, device=device),
            torch.ones(self.act_len, dtype=torch.long, device=device),
        ]))
        seq = seq + pos + typ
        z_H = self.z_H.view(1,1,-1).expand(B, self.total_len, -1).clone()
        z_L = self.z_L.view(1,1,-1).expand(B, self.total_len, -1).clone()

        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    if not (h == self.H_cycles-1 and l == self.L_cycles-1):
                        z_L = self.shared(z_L, z_H + seq)
                if not (h == self.H_cycles-1):
                    z_H = self.shared(z_H, z_L)
        z_L = self.shared(z_L, z_H + seq)
        z_H = self.shared(z_H, z_L)

        pred = self.head(z_H[:, self.obs_len:].float())
        loss = F.mse_loss(pred, target_actions) if target_actions is not None else None
        return pred, loss


# ============== Dataset ==============

class PushTKeypointDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, noise_std=0.0):
        root = zarr.open(zarr_path, "r")
        state = root["data/state"][:]
        keypoints = root["data/keypoint"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])
        self.obs = np.concatenate([keypoints.reshape(-1, 18), state[:, :2]], axis=1)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.noise_std = noise_std

        self.indices = []
        for ep in range(len(episode_ends)):
            s, e = starts[ep], episode_ends[ep]
            for t in range(e - s - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(s + t)
        self.indices = np.array(self.indices)

        self.obs_min = self.obs.min(axis=0)
        self.obs_max = self.obs.max(axis=0)
        self.obs_range = self.obs_max - self.obs_min + 1e-6
        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6
        print(f"Dataset: {len(self.indices)} samples, obs={self.obs.shape[1]}D")

    def normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / self.obs_range - 1

    def normalize_act(self, act):
        return 2 * (act - self.act_min) / self.act_range - 1

    def denormalize_act(self, act):
        return (act + 1) / 2 * self.act_range + self.act_min

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.obs[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()
        if self.noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.noise_std
        return {
            "obs": torch.tensor(self.normalize_obs(obs), dtype=torch.float32),
            "actions": torch.tensor(self.normalize_act(actions), dtype=torch.float32),
        }


# ============== Evaluation ==============

def evaluate(model, dataset, device, method_name, num_episodes=50, seed_start=100000,
             action_horizon=8, num_denoise_steps=100):
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    max_scores = []

    for ep in range(num_episodes):
        env = PushTKeypointsEnv(legacy=True, agent_keypoints=False, **kp_kwargs)
        env.seed(seed_start + ep)
        raw_obs = env.reset()
        Do = raw_obs.shape[0] // 2
        obs_history = [raw_obs[:Do]]
        rewards = []
        done, steps = False, 0

        while not done and steps < 300:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])
            recent = np.array(obs_history[-dataset.obs_horizon:])
            obs_norm = dataset.normalize_obs(recent)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                if method_name == 'diffusion':
                    actions_norm = model.sample(obs_t, num_denoise_steps)
                    actions_norm = actions_norm[0].cpu().numpy()
                elif method_name == 'flow':
                    actions_norm = model.sample(obs_t, n_steps=10)
                    actions_norm = actions_norm[0].cpu().numpy()
                else:
                    pred, _ = model(obs_t)
                    actions_norm = pred[0].cpu().numpy()

            actions = dataset.denormalize_act(actions_norm)
            for t in range(min(action_horizon, len(actions))):
                raw_obs, reward, done, _ = env.step(np.clip(actions[t], 0, 512))
                obs_history.append(raw_obs[:Do])
                rewards.append(reward)
                steps += 1
                if done:
                    break

        max_scores.append(max(rewards) if rewards else 0)

    mean_s = np.mean(max_scores)
    return {"mean": mean_s, "std": np.std(max_scores), "max": np.max(max_scores),
            "min": np.min(max_scores), "zeros": sum(1 for s in max_scores if s < 0.05),
            "scores": max_scores}


# ============== Training ==============

def get_lr(step, warmup, total, max_lr):
    if step < warmup:
        return max_lr * step / warmup
    r = (step - warmup) / (total - warmup)
    return 1e-6 + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - 1e-6)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    ds = PushTKeypointDataset(args.zarr_path, noise_std=args.noise_std)
    idx = list(range(len(ds)))
    np.random.shuffle(idx)
    split = int(len(idx) * 0.9)
    train_dl = DataLoader(Subset(ds, idx[:split]), batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    nh = max(args.hidden // 32, 4)

    if args.method == 'regression':
        model = RegressionPolicy(d=args.hidden, nh=nh, n_layers=args.n_layers).to(device)
    elif args.method == 'flow':
        model = FlowPolicy(d=args.hidden, nh=nh, n_layers=args.n_layers).to(device)
    elif args.method == 'mip':
        model = MIPPolicy(d=args.hidden, nh=nh, n_layers=args.n_layers).to(device)
    elif args.method == 'diffusion':
        model = DiffusionPolicy(d=args.hidden, nh=nh, n_layers=args.n_layers,
                                num_steps=args.num_denoise_steps).to(device)
    elif args.method == 'hrm':
        model = HRM(d=args.hidden, nh=nh, n_layers=args.n_layers,
                    H_cycles=args.H_cycles, L_cycles=args.L_cycles).to(device)
    elif args.method == 'trm':
        model = TRM(d=args.hidden, nh=nh, n_layers=args.n_layers,
                    H_cycles=args.H_cycles, L_cycles=args.L_cycles).to(device)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Method: {args.method} | Params: {nparams:,} | Hidden: {args.hidden}")
    print(f"{'='*60}\n")

    exp_name = f"{args.method}_h{args.hidden}_seed{args.seed}"
    if args.method in ('hrm', 'trm'):
        exp_name += f"_H{args.H_cycles}L{args.L_cycles}"
    if args.exp_suffix:
        exp_name += f"_{args.exp_suffix}"

    wandb.init(project="pusht-paper-reproduce",
               name=exp_name, config=vars(args))
    wandb.log({"model/params": nparams})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.95, 0.999))
    total_steps = args.epochs * len(train_dl)
    warmup = 10 * len(train_dl)

    save_dir = f"checkpoints/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    best = 0.0
    step = 0

    for epoch in range(args.epochs):
        model.train()
        tloss = 0
        for batch in tqdm(train_dl, desc=f"[{args.method}] Ep {epoch+1}/{args.epochs}", leave=False):
            step += 1
            lr = get_lr(step, warmup, total_steps, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            obs = batch["obs"].to(device)
            acts = batch["actions"].to(device)
            _, loss = model(obs, acts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()

        avg = tloss / len(train_dl)
        wandb.log({"train/loss": avg, "train/lr": lr, "epoch": epoch+1}, step=step)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            r = evaluate(model, ds, device, args.method,
                        num_episodes=args.eval_episodes,
                        action_horizon=args.action_horizon,
                        num_denoise_steps=args.num_denoise_steps)
            print(f"  [{args.method}] Ep {epoch+1}: loss={avg:.6f} | mean={r['mean']:.4f} max={r['max']:.3f} zeros={r['zeros']}/{args.eval_episodes}")
            wandb.log({"eval/mean": r["mean"], "eval/max": r["max"],
                       "eval/std": r["std"], "eval/zeros": r["zeros"]}, step=step)
            if r["mean"] > best:
                best = r["mean"]
                torch.save(model.state_dict(), f"{save_dir}/best.pt")
                wandb.log({"eval/best_mean": best}, step=step)

    # Final eval with more episodes
    print(f"\nFinal evaluation (100 episodes)...")
    r = evaluate(model, ds, device, args.method, num_episodes=100,
                action_horizon=args.action_horizon,
                num_denoise_steps=args.num_denoise_steps)
    print(f"  [{args.method}] FINAL: mean={r['mean']:.4f} std={r['std']:.4f} max={r['max']:.3f}")
    wandb.log({"eval_final/mean": r["mean"], "eval_final/std": r["std"],
               "eval_final/max": r["max"]}, step=step)

    # Save final
    torch.save(model.state_dict(), f"{save_dir}/final.pt")
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump({"method": args.method, "params": nparams, "best_mean": best,
                   "final_mean": r["mean"], "final_std": r["std"],
                   "scores": r["scores"], "config": vars(args)}, f, indent=2)

    print(f"\nDone. Best: {best:.4f}")
    wandb.finish()
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=['regression', 'flow', 'mip', 'diffusion', 'hrm', 'trm'])
    p.add_argument("--zarr_path", default="/home/nas5/junhahyung/vla_hrm/data/pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--H_cycles", type=int, default=5)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise_std", type=float, default=2.0)
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--num_denoise_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--exp_suffix", default="")
    args = p.parse_args()
    train(args)
