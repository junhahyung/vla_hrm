"""
PushT dataset loader that tokenizes continuous state/action into discrete tokens.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from pusht_hrm.tokenizer import PerDimTokenizer


class PushTTokenDataset(Dataset):
    """
    PushT dataset that returns tokenized observation and action sequences.

    Observation: T_o steps of 5D state (agent_x, agent_y, block_x, block_y, block_angle)
    Action: T_a steps of 2D actions (target_x, target_y)

    Sequence layout (flattened):
        [obs_0_dim0, obs_0_dim1, ..., obs_0_dim4, obs_1_dim0, ..., act_0_dim0, act_0_dim1, ..., act_{T_a-1}_dim1]
    """

    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        pred_horizon: int = 16,
        num_bins: int = 256,
        tokenizer_type: str = "uniform",
        mu: float = 255.0,
        normalize_obs: bool = True,
        obs_noise_std: float = 0.0,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.obs_noise_std = obs_noise_std

        # Load data
        root = zarr.open(zarr_path, "r")
        self.state = root["data/state"][:]  # (N, 5)
        self.action = root["data/action"][:]  # (N, 2)
        episode_ends = root["meta/episode_ends"][:]

        # Compute episode starts
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_ends = episode_ends

        # Compute valid indices
        self.indices = []
        for ep_idx in range(len(episode_ends)):
            start = self.episode_starts[ep_idx]
            end = self.episode_ends[ep_idx]
            ep_len = end - start
            # Need at least obs_horizon + pred_horizon steps
            for t in range(ep_len - pred_horizon + 1):
                # Ensure we have enough past observations
                if t >= obs_horizon - 1:
                    self.indices.append(start + t)
        self.indices = np.array(self.indices)

        # Compute normalization stats
        if normalize_obs:
            self.state_min = self.state.min(axis=0)
            self.state_max = self.state.max(axis=0)
            self.action_min = self.action.min(axis=0)
            self.action_max = self.action.max(axis=0)
        else:
            self.state_min = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.state_max = np.array([512.0, 512.0, 512.0, 512.0, 2 * np.pi])
            self.action_min = np.array([0.0, 0.0])
            self.action_max = np.array([512.0, 512.0])

        # Create per-dimension tokenizers
        obs_ranges = [(self.state_min[i], self.state_max[i]) for i in range(5)]
        act_ranges = [(self.action_min[i], self.action_max[i]) for i in range(2)]

        self.obs_tokenizer = PerDimTokenizer(num_bins, obs_ranges, tokenizer_type, mu)
        self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, tokenizer_type, mu)

        self.num_bins = num_bins
        self.obs_dim = 5
        self.act_dim = 2

        # Sequence layout
        self.obs_seq_len = obs_horizon * self.obs_dim
        self.act_seq_len = pred_horizon * self.act_dim
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        print(f"PushT dataset: {len(self.indices)} samples, "
              f"obs_horizon={obs_horizon}, pred_horizon={pred_horizon}, "
              f"seq_len={self.total_seq_len}, vocab={num_bins}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # Get observations (past obs_horizon steps including current)
        obs_start = t - self.obs_horizon + 1
        obs = self.state[obs_start:t + 1].copy()  # (obs_horizon, 5)

        # Get actions (future pred_horizon steps)
        actions = self.action[t:t + self.pred_horizon]  # (pred_horizon, 2)

        # Add observation noise for augmentation
        if self.obs_noise_std > 0:
            noise = np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std
            obs = obs + noise

        # Tokenize
        obs_tokens = self.obs_tokenizer.encode(obs).flatten()  # (obs_horizon * 5,)
        act_tokens = self.act_tokenizer.encode(actions).flatten()  # (pred_horizon * 2,)

        # Full input sequence: obs_tokens + MASK for actions
        # For training: input has obs tokens, target has action tokens
        input_tokens = np.concatenate([obs_tokens, act_tokens])  # for teacher forcing
        target_tokens = np.full(self.total_seq_len, -100, dtype=np.int64)  # -100 = ignore
        target_tokens[self.obs_seq_len:] = act_tokens  # only predict actions

        return {
            "input_tokens": torch.tensor(input_tokens, dtype=torch.long),
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "raw_obs": torch.tensor(obs, dtype=torch.float32),
            "raw_actions": torch.tensor(actions, dtype=torch.float32),
        }

    def decode_actions(self, act_tokens: np.ndarray) -> np.ndarray:
        """Decode action tokens back to continuous values.
        act_tokens: (pred_horizon * 2,) or (batch, pred_horizon * 2)
        """
        if act_tokens.ndim == 1:
            act_tokens = act_tokens.reshape(self.pred_horizon, self.act_dim)
            return self.act_tokenizer.decode(act_tokens)
        else:
            batch_size = act_tokens.shape[0]
            act_tokens = act_tokens.reshape(batch_size, self.pred_horizon, self.act_dim)
            return self.act_tokenizer.decode(act_tokens)

    def decode_actions_torch(self, act_tokens: torch.Tensor) -> torch.Tensor:
        """Decode action tokens back to continuous values (torch version)."""
        if act_tokens.ndim == 1:
            act_tokens = act_tokens.reshape(self.pred_horizon, self.act_dim)
            return self.act_tokenizer.decode_torch(act_tokens)
        else:
            batch_size = act_tokens.shape[0]
            act_tokens = act_tokens.reshape(batch_size, self.pred_horizon, self.act_dim)
            return self.act_tokenizer.decode_torch(act_tokens)
