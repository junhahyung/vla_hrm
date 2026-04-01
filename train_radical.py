"""
RADICAL: Use keypoint observations (matching DP exactly) + regression output.
This should close the observation gap with DP.

Key changes:
- Use 20D keypoint obs from zarr (9 keypoints × 2D + agent pos 2D) = same as DP
- Regression output (continuous actions, not discrete tokens)
- LinearNormalizer matching DP's normalization
- Evaluate on DP's exact PushTKeypointsEnv
"""
import argparse, os, sys, math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ['SDL_VIDEODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_policy'))
from torch.utils.data import DataLoader, Dataset, Subset
import wandb
from tqdm import tqdm
import zarr

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


# ============== Simple but effective HRM for regression ==============
def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d, exp=2.0):
        super().__init__()
        inter = round(exp * d * 2 / 3)
        inter = ((inter + 255) // 256) * 256  # align
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


class HRM_Radical(nn.Module):
    """HRM with keypoint obs + regression output."""
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

        # Obs encoder (handles 20D keypoint obs per timestep)
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d),
        )
        # Action queries
        self.act_q = nn.Parameter(torch.randn(pred_horizon, d) * 0.02)
        self.pos_emb = nn.Embedding(self.total_len, d)
        self.type_emb = nn.Embedding(2, d)

        # H and L modules (separate for HRM)
        self.H = RecursiveModule([Block(d, nh) for _ in range(n_layers)])
        self.L = RecursiveModule([Block(d, nh) for _ in range(n_layers)])

        # Regression head → continuous actions
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, act_dim))

        self.z_H = nn.Parameter(torch.randn(d) * 0.02)
        self.z_L = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, obs, target_actions=None):
        """obs: (B, obs_h, obs_dim), target_actions: (B, pred_h, act_dim) normalized."""
        B, device = obs.shape[0], obs.device

        obs_emb = self.obs_enc(obs.float())
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

        # Recursive reasoning
        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    if not (h == self.H_cycles-1 and l == self.L_cycles-1):
                        z_L = self.L(z_L, z_H + seq)
                if not (h == self.H_cycles-1):
                    z_H = self.H(z_H, z_L)
        # Last step with grad
        z_L = self.L(z_L, z_H + seq)
        z_H = self.H(z_H, z_L)

        act_feat = z_H[:, self.obs_len:].float()
        pred = self.head(act_feat.float())  # (B, pred_h, act_dim)

        loss = None
        if target_actions is not None:
            loss = F.mse_loss(pred, target_actions)
        return pred, loss


# ============== Dataset with keypoint obs ==============
class PushTKeypointDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, noise_std=0.0):
        root = zarr.open(zarr_path, "r")
        state = root["data/state"][:]  # (N, 5) for agent pos
        keypoints = root["data/keypoint"][:]  # (N, 9, 2)
        self.action = root["data/action"][:]  # (N, 2)
        episode_ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])

        # Build 20D obs: 9 keypoints (18D) + agent pos (2D) = matching DP exactly
        self.obs = np.concatenate([keypoints.reshape(-1, 18), state[:, :2]], axis=1)  # (N, 20)

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

        # Normalization (DP uses per-dim linear normalization)
        self.obs_min = self.obs.min(axis=0)
        self.obs_max = self.obs.max(axis=0)
        self.obs_range = self.obs_max - self.obs_min + 1e-6
        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6

        print(f"Keypoint dataset: {len(self.indices)} samples, obs={self.obs.shape[1]}D")

    def normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / self.obs_range - 1  # [-1, 1]

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


# ============== Eval on DP's exact env ==============
def evaluate(model, dataset, device, num_episodes=50, seed_start=100000, action_horizon=8):
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    max_scores = []

    for ep in range(num_episodes):
        env = PushTKeypointsEnv(legacy=True, agent_keypoints=False, **kp_kwargs)
        env.seed(seed_start + ep)
        raw_obs = env.reset()
        Do = raw_obs.shape[0] // 2
        obs_history = [raw_obs[:Do]]  # 20D keypoints
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
                pred, _ = model(obs_t)
            actions = dataset.denormalize_act(pred[0].cpu().numpy())

            for t in range(min(action_horizon, len(actions))):
                raw_obs, reward, done, _ = env.step(np.clip(actions[t], 0, 512))
                obs_history.append(raw_obs[:Do])
                rewards.append(reward)
                steps += 1
                if done: break

        max_scores.append(max(rewards) if rewards else 0)

    mean = np.mean(max_scores)
    zeros = sum(1 for s in max_scores if s < 0.05)
    return {"mean": mean, "std": np.std(max_scores), "max": np.max(max_scores),
            "zeros": zeros, "scores": max_scores}


def get_lr(step, warmup, total, max_lr):
    if step < warmup: return max_lr * step / warmup
    r = (step - warmup) / (total - warmup)
    return 1e-6 + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", default="/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
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
    p.add_argument("--exp_name", default=None)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device(f"cuda:{args.gpu}")

    ds = PushTKeypointDataset(args.zarr_path, noise_std=args.noise_std)
    idx = list(range(len(ds))); np.random.shuffle(idx)
    split = int(len(idx) * 0.9)
    train_dl = DataLoader(Subset(ds, idx[:split]), batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    nh = args.hidden // 32
    model = HRM_Radical(obs_dim=20, d=args.hidden, nh=nh, n_layers=args.n_layers,
                        H_cycles=args.H_cycles, L_cycles=args.L_cycles).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HRM Radical: {nparams:,} params, obs=20D keypoints, output=regression")

    if args.exp_name is None:
        args.exp_name = f"radical_hrm_h{args.hidden}_H{args.H_cycles}L{args.L_cycles}"
    wandb.init(project="pusht-hrm", entity="junha", name=args.exp_name, config=vars(args))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.95, 0.999))
    total_steps = args.epochs * len(train_dl)
    warmup = 10 * len(train_dl)

    save_dir = f"checkpoints/{args.exp_name}"; os.makedirs(save_dir, exist_ok=True)
    best = 0.0; step = 0

    for epoch in range(args.epochs):
        model.train(); tloss = 0
        for batch in tqdm(train_dl, desc=f"Ep {epoch+1}/{args.epochs}", leave=False):
            step += 1
            lr = get_lr(step, warmup, total_steps, args.lr)
            for pg in opt.param_groups: pg["lr"] = lr
            obs = batch["obs"].to(device)
            acts = batch["actions"].to(device)
            _, loss = model(obs, acts)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tloss += loss.item()

        avg = tloss / len(train_dl)
        print(f"Epoch {epoch+1}: loss={avg:.6f}")
        wandb.log({"train/loss": avg, "train/lr": lr, "epoch": epoch+1}, step=step)

        if (epoch+1) % args.eval_interval == 0 or epoch == args.epochs-1:
            r = evaluate(model, ds, device, num_episodes=50, action_horizon=args.action_horizon)
            print(f"  EVAL: mean={r['mean']:.4f} max={r['max']:.3f} zeros={r['zeros']}/50")
            wandb.log({"eval/mean": r["mean"], "eval/max": r["max"], "eval/zeros": r["zeros"]}, step=step)
            if r["mean"] > best:
                best = r["mean"]
                torch.save(model.state_dict(), f"{save_dir}/best.pt")

    print(f"\nBest: {best:.4f} (DP: 0.871)")
    wandb.finish()


if __name__ == "__main__":
    main()
