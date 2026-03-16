"""
V2 dataset: Returns continuous observations + tokenized actions.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from pusht_hrm.tokenizer import PerDimTokenizer


class PushTDatasetV2(Dataset):
    """
    PushT dataset with continuous observations and discrete action tokens.
    Observations are normalized to [-1, 1].
    Actions are discretized into bins.
    """

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

        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_ends = episode_ends

        # Valid indices
        self.indices = []
        for ep_idx in range(len(episode_ends)):
            start = self.episode_starts[ep_idx]
            end = self.episode_ends[ep_idx]
            ep_len = end - start
            for t in range(ep_len - pred_horizon + 1):
                if t >= obs_horizon - 1:
                    self.indices.append(start + t)
        self.indices = np.array(self.indices)

        # Normalization stats (fit on training data)
        self.obs_mean = self.state.mean(axis=0)
        self.obs_std = self.state.std(axis=0) + 1e-6

        # Action tokenization
        self.action_min = self.action.min(axis=0)
        self.action_max = self.action.max(axis=0)
        act_ranges = [(self.action_min[i], self.action_max[i]) for i in range(2)]
        self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, tokenizer_type, mu)

        self.num_bins = num_bins
        self.obs_dim = 5
        self.act_dim = 2
        self.act_seq_len = pred_horizon * self.act_dim

        print(f"PushT V2 dataset: {len(self.indices)} samples, "
              f"obs_horizon={obs_horizon}, pred_horizon={pred_horizon}, "
              f"act_seq_len={self.act_seq_len}, vocab={num_bins}")
        print(f"  obs_mean={self.obs_mean}, obs_std={self.obs_std}")

    def __len__(self):
        return len(self.indices)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def __getitem__(self, idx):
        t = self.indices[idx]

        # Observations
        obs_start = t - self.obs_horizon + 1
        obs = self.state[obs_start:t + 1].copy()  # (obs_horizon, 5)

        # Actions
        actions = self.action[t:t + self.pred_horizon]  # (pred_horizon, 2)

        # Augmentation: add noise to obs
        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std

        # Normalize observations
        obs_norm = self.normalize_obs(obs)

        # Tokenize actions
        act_tokens = self.act_tokenizer.encode(actions).flatten()  # (pred_horizon * 2,)

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "raw_obs": torch.tensor(obs, dtype=torch.float32),
            "raw_actions": torch.tensor(actions, dtype=torch.float32),
        }

    def decode_actions(self, act_tokens):
        """act_tokens: (act_seq_len,) or (B, act_seq_len) -> continuous actions"""
        if isinstance(act_tokens, torch.Tensor):
            act_tokens = act_tokens.cpu().numpy()
        if act_tokens.ndim == 1:
            return self.act_tokenizer.decode(
                act_tokens.reshape(self.pred_horizon, self.act_dim)
            )
        else:
            B = act_tokens.shape[0]
            return self.act_tokenizer.decode(
                act_tokens.reshape(B, self.pred_horizon, self.act_dim)
            )
