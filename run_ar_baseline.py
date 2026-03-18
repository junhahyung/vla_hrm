"""
Simple Autoregressive discrete token baseline for comparison.
Standard GPT-style: predict action tokens one by one given obs.
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

from pusht_hrm.tokenizer import PerDimTokenizer
from pusht_hrm.pusht_env import PushTEnv
from pusht_hrm.losses import soft_decode


class CausalTransformer(nn.Module):
    """Standard causal GPT for autoregressive action prediction."""
    def __init__(self, obs_dim, num_bins, hidden_size=256, num_heads=8, num_layers=6,
                 obs_horizon=2, pred_horizon=16, act_dim=2):
        super().__init__()
        self.num_bins = num_bins
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.act_seq_len = pred_horizon * act_dim
        self.hidden_size = hidden_size

        # Obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Token embedding for actions
        self.token_embed = nn.Embedding(num_bins + 1, hidden_size)  # +1 for BOS
        self.bos_token = num_bins

        # Position embedding
        total_len = obs_horizon + self.act_seq_len
        self.pos_embed = nn.Embedding(total_len, hidden_size)
        self.type_embed = nn.Embedding(2, hidden_size)

        # Transformer
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4,
                                            dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.causal_mask = None

        # Output
        self.head = nn.Linear(hidden_size, num_bins)

    def _get_causal_mask(self, L, device):
        if self.causal_mask is None or self.causal_mask.shape[0] != L:
            self.causal_mask = torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)
        return self.causal_mask

    def forward(self, obs, act_tokens=None):
        """
        obs: (B, obs_horizon, obs_dim)
        act_tokens: (B, act_seq_len) target tokens
        """
        B, device = obs.shape[0], obs.device

        # Encode obs
        obs_emb = self.obs_encoder(obs.float())  # (B, obs_h, D)

        if act_tokens is not None:
            # Teacher forcing: shift right, prepend BOS
            bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)
            shifted = torch.cat([bos, act_tokens[:, :-1]], dim=1)  # (B, act_seq_len)
            act_emb = self.token_embed(shifted)  # (B, act_seq_len, D)
        else:
            bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)
            act_emb = self.token_embed(bos)

        seq = torch.cat([obs_emb, act_emb], dim=1)
        L = seq.shape[1]

        pos_ids = torch.arange(L, device=device)
        type_ids = torch.cat([torch.zeros(self.obs_horizon, dtype=torch.long, device=device),
                              torch.ones(L - self.obs_horizon, dtype=torch.long, device=device)])
        seq = seq + self.pos_embed(pos_ids) + self.type_embed(type_ids)

        mask = self._get_causal_mask(L, device)
        h = self.transformer(seq, mask=mask)

        # Only predict at action positions
        logits = self.head(h[:, self.obs_horizon:])  # (B, act_seq_len, num_bins)

        loss = None
        if act_tokens is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.num_bins), act_tokens.reshape(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict_actions(self, obs, temperature=0.3):
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_encoder(obs.float())

        # Autoregressive generation
        tokens = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)
        generated = []

        for i in range(self.act_seq_len):
            act_emb = self.token_embed(tokens)
            seq = torch.cat([obs_emb, act_emb], dim=1)
            L = seq.shape[1]
            pos_ids = torch.arange(L, device=device)
            type_ids = torch.cat([torch.zeros(self.obs_horizon, dtype=torch.long, device=device),
                                  torch.ones(L - self.obs_horizon, dtype=torch.long, device=device)])
            seq = seq + self.pos_embed(pos_ids)[:L] + self.type_embed(type_ids)
            mask = self._get_causal_mask(L, device)
            h = self.transformer(seq, mask=mask)
            next_logits = self.head(h[:, -1:])  # (B, 1, num_bins)

            # Soft decode for continuous output
            probs = F.softmax(next_logits / temperature, dim=-1)
            bins = torch.arange(self.num_bins, dtype=torch.float32, device=device)
            soft_idx = (probs * bins).sum(dim=-1)  # (B, 1)
            next_token = soft_idx.round().long().clamp(0, self.num_bins - 1)

            tokens = torch.cat([tokens, next_token], dim=1)
            generated.append(soft_idx.squeeze(-1))

        return torch.stack(generated, dim=1)  # (B, act_seq_len)


class ARDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16, num_bins=256, obs_noise_std=3.0):
        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]
        self.action = root["data/action"][:]
        episode_ends = root["meta/episode_ends"][:]
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_bins = num_bins
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
        act_min, act_max = self.action.min(axis=0), self.action.max(axis=0)
        self.act_tokenizer = PerDimTokenizer(num_bins, [(act_min[i], act_max[i]) for i in range(2)], "uniform")
        print(f"AR dataset: {len(self.indices)} samples")

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
        return {
            "obs": torch.tensor(self.normalize_obs(obs), dtype=torch.float32),
            "act_tokens": torch.tensor(self.act_tokenizer.encode(actions).flatten(), dtype=torch.long),
        }


def evaluate_ar(model, dataset, device, num_episodes=22, seed_start=10000, action_horizon=8):
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
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model.predict_actions(obs_t).cpu().numpy()[0].reshape(dataset.pred_horizon, 2)
            actions = np.zeros_like(pred)
            for d in range(2):
                tok = dataset.act_tokenizer.tokenizers[d]
                actions[:, d] = tok.val_min + (pred[:, d] + 0.5) * tok.bin_width
            for t in range(min(action_horizon, len(actions))):
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
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    np.random.seed(42)

    zarr_path = "/home/junhahyung/vla_hrm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    dataset = ARDataset(zarr_path)
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = CausalTransformer(obs_dim=5, num_bins=256, hidden_size=256, num_heads=8, num_layers=6).to(device)
    print(f"AR model: {sum(p.numel() for p in model.parameters()):,} params")

    wandb.init(project="pusht-hrm", entity="junha", name="ar_gpt_h256_l6")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    best_eval = 0.0
    for epoch in range(2000):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs = batch["obs"].to(device)
            act_tokens = batch["act_tokens"].to(device)
            out = model(obs, act_tokens)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train/loss": avg_loss})

        if (epoch + 1) % 50 == 0:
            r = evaluate_ar(model, dataset, device, action_horizon=16)
            print(f"  mean={r['mean_score']:.4f} (range: {r['min_score']:.3f}-{r['max_score']:.3f})")
            wandb.log({"eval/mean_score": r["mean_score"]})
            if r["mean_score"] > best_eval:
                best_eval = r["mean_score"]

    print(f"Done. Best: {best_eval:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
