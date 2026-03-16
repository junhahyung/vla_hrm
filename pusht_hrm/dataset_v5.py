"""
V5 dataset: Inspired by VLA-0's integer text tokenization.

Instead of bins, normalize actions to integers [0, max_int] and predict each digit.
This gives arbitrary resolution without changing vocabulary.

Also supports:
- Per-digit tokenization (each digit 0-9 is a token, vocab=10)
- Direct integer tokenization (vocab=max_int+1, like V1 but with proper normalization)
- Action chunking with temporal ensemble support
- Masked action augmentation (randomly mask some action values during training)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class PushTDatasetV5(Dataset):
    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pred_horizon: int = 16,
        action_int_max: int = 256,  # normalize actions to [0, action_int_max]
        obs_noise_std: float = 0.0,
        mask_augment_prob: float = 0.0,  # probability of masking each action token
        tokenize_mode: str = "direct",  # "direct" (vocab=max+1) or "digits" (vocab=10)
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.action_int_max = action_int_max
        self.obs_noise_std = obs_noise_std
        self.mask_augment_prob = mask_augment_prob
        self.tokenize_mode = tokenize_mode
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

        # Action normalization: per-dim to [0, action_int_max]
        self.act_min = self.action.min(axis=0)
        self.act_max = self.action.max(axis=0)
        self.act_range = self.act_max - self.act_min + 1e-6

        if tokenize_mode == "direct":
            self.vocab_size = action_int_max + 1
            self.act_seq_len = pred_horizon * self.act_dim
        elif tokenize_mode == "digits":
            # Each integer is represented as multiple digit tokens
            self.num_digits = len(str(action_int_max))
            self.vocab_size = 10  # digits 0-9
            self.act_seq_len = pred_horizon * self.act_dim * self.num_digits

        # Mask token (for augmentation)
        self.mask_token_id = self.vocab_size  # extra token for masking
        self.effective_vocab_size = self.vocab_size + 1  # +1 for mask

        print(f"PushT V5: {len(self.indices)} samples, mode={tokenize_mode}, "
              f"int_max={action_int_max}, vocab={self.effective_vocab_size}, "
              f"seq_len={self.act_seq_len}")

    def __len__(self):
        return len(self.indices)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def action_to_int(self, actions):
        """Convert continuous actions to integers [0, action_int_max]."""
        normalized = (actions - self.act_min) / self.act_range  # [0, 1]
        integers = np.round(normalized * self.action_int_max).astype(np.int64)
        return np.clip(integers, 0, self.action_int_max)

    def int_to_action(self, integers):
        """Convert integers back to continuous actions."""
        if isinstance(integers, torch.Tensor):
            integers = integers.float()
            normalized = integers / self.action_int_max
            act_min = torch.tensor(self.act_min, device=integers.device)
            act_range = torch.tensor(self.act_range, device=integers.device)
            return normalized * act_range + act_min
        normalized = integers.astype(np.float32) / self.action_int_max
        return normalized * self.act_range + self.act_min

    def int_to_digits(self, integers):
        """Convert integers to digit tokens. E.g., 42 -> [0, 4, 2] for 3 digits."""
        digits = []
        for _ in range(self.num_digits):
            digits.append(integers % 10)
            integers = integers // 10
        return np.stack(digits[::-1], axis=-1)  # (..., num_digits), MSB first

    def digits_to_int(self, digits):
        """Convert digit tokens back to integers."""
        result = np.zeros(digits.shape[:-1], dtype=np.int64)
        for i in range(self.num_digits):
            result = result * 10 + digits[..., i]
        return result

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()

        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std

        obs_norm = self.normalize_obs(obs)

        # Convert actions to integers
        act_ints = self.action_to_int(actions)  # (T, 2)

        if self.tokenize_mode == "direct":
            act_tokens = act_ints.flatten()  # (T * 2,)
        elif self.tokenize_mode == "digits":
            act_digits = self.int_to_digits(act_ints)  # (T, 2, num_digits)
            act_tokens = act_digits.reshape(-1)  # (T * 2 * num_digits,)

        # Masked action augmentation (VLA-0 inspired)
        mask = np.zeros_like(act_tokens, dtype=bool)
        if self.mask_augment_prob > 0:
            mask = np.random.rand(len(act_tokens)) < self.mask_augment_prob

        # Create input tokens (with potential masking) and targets
        input_tokens = act_tokens.copy()
        if mask.any():
            input_tokens[mask] = self.mask_token_id

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "input_act_tokens": torch.tensor(input_tokens, dtype=torch.long),
            "raw_actions": torch.tensor(actions, dtype=torch.float32),
        }

    def decode_actions(self, act_tokens):
        """Decode action tokens to continuous actions."""
        if isinstance(act_tokens, torch.Tensor):
            act_tokens = act_tokens.cpu().numpy()

        if self.tokenize_mode == "direct":
            if act_tokens.ndim == 1:
                ints = act_tokens.reshape(self.pred_horizon, self.act_dim)
            else:
                ints = act_tokens.reshape(-1, self.pred_horizon, self.act_dim)
            return self.int_to_action(ints)

        elif self.tokenize_mode == "digits":
            if act_tokens.ndim == 1:
                digits = act_tokens.reshape(self.pred_horizon, self.act_dim, self.num_digits)
                ints = self.digits_to_int(digits)
            else:
                B = act_tokens.shape[0]
                digits = act_tokens.reshape(B, self.pred_horizon, self.act_dim, self.num_digits)
                ints = self.digits_to_int(digits)
            return self.int_to_action(ints)

    def soft_decode_actions(self, logits, temperature=1.0):
        """Soft decode using weighted average."""
        import torch.nn.functional as F
        probs = F.softmax(logits / temperature, dim=-1)
        bins = torch.arange(logits.shape[-1], dtype=torch.float32, device=logits.device)
        soft_indices = (probs * bins).sum(dim=-1)  # (B, seq_len)

        if self.tokenize_mode == "direct":
            B = soft_indices.shape[0]
            soft_ints = soft_indices.reshape(B, self.pred_horizon, self.act_dim)
            return self.int_to_action(soft_ints)
        else:
            # For digits, round and decode
            return self.decode_actions(soft_indices.round().long())
