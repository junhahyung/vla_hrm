"""
V3 dataset: continuous obs + continuous actions (normalized) + optional discrete tokens.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from pusht_hrm.tokenizer import PerDimTokenizer


class PushTDatasetV3(Dataset):
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
        self.num_bins = num_bins

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

        # Normalization
        self.obs_mean = self.state.mean(axis=0)
        self.obs_std = self.state.std(axis=0) + 1e-6
        self.act_mean = self.action.mean(axis=0)
        self.act_std = self.action.std(axis=0) + 1e-6

        # Action tokenizer
        act_ranges = [(self.action.min(axis=0)[i], self.action.max(axis=0)[i]) for i in range(2)]
        self.act_tokenizer = PerDimTokenizer(num_bins, act_ranges, tokenizer_type, mu)

        self.obs_dim = 5
        self.act_dim = 2

        print(f"PushT V3: {len(self.indices)} samples, obs_h={obs_horizon}, pred_h={pred_horizon}, bins={num_bins}")

    def __len__(self):
        return len(self.indices)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def normalize_actions(self, actions):
        return (actions - self.act_mean) / self.act_std

    def denormalize_actions(self, actions):
        if isinstance(actions, torch.Tensor):
            return actions * torch.tensor(self.act_std, device=actions.device) + torch.tensor(self.act_mean, device=actions.device)
        return actions * self.act_std + self.act_mean

    def __getitem__(self, idx):
        t = self.indices[idx]
        obs = self.state[t - self.obs_horizon + 1:t + 1].copy()
        actions = self.action[t:t + self.pred_horizon].copy()

        if self.obs_noise_std > 0:
            obs += np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std

        obs_norm = self.normalize_obs(obs)
        act_norm = self.normalize_actions(actions)
        act_tokens = self.act_tokenizer.encode(actions).flatten()

        return {
            "obs": torch.tensor(obs_norm, dtype=torch.float32),
            "actions": torch.tensor(act_norm, dtype=torch.float32),
            "act_tokens": torch.tensor(act_tokens, dtype=torch.long),
            "raw_actions": torch.tensor(actions, dtype=torch.float32),
        }
