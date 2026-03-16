"""
V4 dataset: continuous obs + discrete action tokens with advanced tokenization.

Supports:
- Absolute tokenization (standard)
- Delta tokenization (tokenize action changes, not absolute values)
- Residual tokenization (coarse bin + fine residual)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from pusht_hrm.tokenizer import PerDimTokenizer


class PushTDatasetV4(Dataset):
    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pred_horizon: int = 16,
        num_bins: int = 256,
        tokenizer_type: str = "uniform",
        mu: float = 255.0,
        obs_noise_std: float = 0.0,
        action_mode: str = "absolute",  # "absolute", "delta", "residual"
        num_coarse_bins: int = 32,  # for residual mode
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.obs_noise_std = obs_noise_std
        self.num_bins = num_bins
        self.action_mode = action_mode
        self.num_coarse_bins = num_coarse_bins
        self.obs_dim = 5
        self.act_dim = 2

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

        # Normalization stats
        self.obs_mean = self.state.mean(axis=0)
        self.obs_std = self.state.std(axis=0) + 1e-6

        # Action tokenization depends on mode
        if action_mode == "absolute":
            self.act_min = self.action.min(axis=0)
            self.act_max = self.action.max(axis=0)
            act_ranges = [(self.act_min[i], self.act_max[i]) for i in range(2)]
            self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, tokenizer_type, mu)
            self.act_seq_len = pred_horizon * self.act_dim
            self.vocab_size = num_bins

        elif action_mode == "delta":
            # Compute action deltas
            deltas = np.diff(self.action, axis=0)
            # Add zero delta for first step
            deltas = np.vstack([np.zeros((1, 2)), deltas])
            self.action_deltas = deltas
            self.delta_min = deltas.min(axis=0)
            self.delta_max = deltas.max(axis=0)
            # Expand range slightly for safety
            pad = (self.delta_max - self.delta_min) * 0.05
            delta_ranges = [(self.delta_min[i] - pad[i], self.delta_max[i] + pad[i]) for i in range(2)]
            self.delta_tokenizer = PerDimTokenizer(num_bins, delta_ranges, tokenizer_type, mu)
            # Also need absolute tokenizer for first action
            abs_ranges = [(self.action.min(axis=0)[i], self.action.max(axis=0)[i]) for i in range(2)]
            self.abs_tokenizer = PerDimTokenizer(num_bins, abs_ranges, tokenizer_type, mu)
            self.act_seq_len = pred_horizon * self.act_dim
            self.vocab_size = num_bins

        elif action_mode == "residual":
            # Coarse + fine tokenization
            self.act_min = self.action.min(axis=0)
            self.act_max = self.action.max(axis=0)
            act_ranges = [(self.act_min[i], self.act_max[i]) for i in range(2)]
            # Coarse tokenizer
            self.coarse_tokenizer = PerDimTokenizer(num_coarse_bins, act_ranges, "uniform")
            # Fine tokenizer within each coarse bin
            coarse_width = (self.act_max - self.act_min) / num_coarse_bins
            num_fine = num_bins // num_coarse_bins
            self.num_fine_bins = num_fine
            # Fine covers [-coarse_width/2, coarse_width/2]
            fine_ranges = [(-coarse_width[i]/2, coarse_width[i]/2) for i in range(2)]
            self.fine_tokenizer = PerDimTokenizer(num_fine, fine_ranges, "uniform")
            # Two tokens per action dim per timestep: coarse + fine
            self.act_seq_len = pred_horizon * self.act_dim * 2
            self.vocab_size = max(num_coarse_bins, num_fine)

        print(f"PushT V4: {len(self.indices)} samples, mode={action_mode}, "
              f"bins={num_bins}, seq_len={self.act_seq_len}, vocab={self.vocab_size}")

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

        obs_norm = self.normalize_obs(obs)

        if self.action_mode == "absolute":
            act_tokens = self.act_tokenizer.encode(actions).flatten()

        elif self.action_mode == "delta":
            # First action is absolute, rest are deltas
            deltas = np.diff(actions, axis=0)
            first_tokens = self.abs_tokenizer.encode(actions[:1]).flatten()
            delta_tokens = self.delta_tokenizer.encode(deltas).flatten()
            act_tokens = np.concatenate([first_tokens, delta_tokens])

        elif self.action_mode == "residual":
            coarse_tokens = self.coarse_tokenizer.encode(actions)  # (T, 2)
            # Compute residuals
            coarse_centers = self.coarse_tokenizer.decode(coarse_tokens)
            residuals = actions - coarse_centers
            fine_tokens = self.fine_tokenizer.encode(residuals)  # (T, 2)
            # Interleave: [coarse_d0, fine_d0, coarse_d1, fine_d1, ...]
            act_tokens = np.stack([coarse_tokens, fine_tokens], axis=-1).reshape(-1)

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "raw_actions": torch.tensor(actions, dtype=torch.float32),
        }

    def decode_actions(self, act_tokens):
        """Decode action tokens back to continuous actions."""
        if isinstance(act_tokens, torch.Tensor):
            act_tokens = act_tokens.cpu().numpy()

        if self.action_mode == "absolute":
            if act_tokens.ndim == 1:
                return self.act_tokenizer.decode(act_tokens.reshape(self.pred_horizon, self.act_dim))
            else:
                B = act_tokens.shape[0]
                return self.act_tokenizer.decode(act_tokens.reshape(B, self.pred_horizon, self.act_dim))

        elif self.action_mode == "delta":
            if act_tokens.ndim == 1:
                first = self.abs_tokenizer.decode(act_tokens[:2].reshape(1, 2))
                deltas = self.delta_tokenizer.decode(act_tokens[2:].reshape(self.pred_horizon - 1, 2))
                actions = np.cumsum(np.vstack([first, deltas]), axis=0)
                return actions
            else:
                B = act_tokens.shape[0]
                results = []
                for b in range(B):
                    results.append(self.decode_actions(act_tokens[b]))
                return np.stack(results)

        elif self.action_mode == "residual":
            if act_tokens.ndim == 1:
                tokens = act_tokens.reshape(self.pred_horizon, self.act_dim, 2)
                coarse_tokens = tokens[:, :, 0]
                fine_tokens = tokens[:, :, 1]
                coarse_centers = self.coarse_tokenizer.decode(coarse_tokens)
                fine_offsets = self.fine_tokenizer.decode(fine_tokens)
                return coarse_centers + fine_offsets
            else:
                B = act_tokens.shape[0]
                results = []
                for b in range(B):
                    results.append(self.decode_actions(act_tokens[b]))
                return np.stack(results)

    def soft_decode_actions(self, logits, temperature=1.0):
        """
        Soft decode: use weighted average of bin centers by softmax probabilities.
        logits: (B, act_seq_len, vocab_size)
        Returns: (B, pred_horizon, act_dim) continuous actions
        """
        from pusht_hrm.losses import soft_decode
        B = logits.shape[0]

        if self.action_mode == "absolute":
            # Get soft bin indices (continuous)
            soft_bins = soft_decode(logits, self.vocab_size, temperature)  # (B, act_seq_len)
            soft_bins = soft_bins.reshape(B, self.pred_horizon, self.act_dim)
            # Convert bin indices to continuous values
            actions = torch.zeros_like(soft_bins)
            for d in range(self.act_dim):
                tok = self.act_tokenizer.tokenizers[d]
                actions[:, :, d] = tok.val_min + (soft_bins[:, :, d] + 0.5) * tok.bin_width
            return actions.cpu().numpy()

        # For delta/residual, fall back to argmax then decode
        tokens = logits.argmax(dim=-1).cpu().numpy()
        if tokens.ndim == 1:
            return self.decode_actions(tokens)
        return np.stack([self.decode_actions(tokens[b]) for b in range(B)])
