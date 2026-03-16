"""
Tokenization strategies for converting continuous PushT actions/observations to discrete tokens.
"""
import numpy as np
import torch
import torch.nn as nn


class UniformTokenizer:
    """Uniform binning tokenizer for continuous values."""

    def __init__(self, num_bins: int, val_min: float, val_max: float):
        self.num_bins = num_bins
        self.val_min = val_min
        self.val_max = val_max
        self.bin_width = (val_max - val_min) / num_bins

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Convert continuous values to bin indices."""
        clipped = np.clip(values, self.val_min, self.val_max - 1e-6)
        bins = ((clipped - self.val_min) / self.bin_width).astype(np.int64)
        return np.clip(bins, 0, self.num_bins - 1)

    def decode(self, bins: np.ndarray) -> np.ndarray:
        """Convert bin indices back to continuous values (bin centers)."""
        return self.val_min + (bins.astype(np.float32) + 0.5) * self.bin_width

    def encode_torch(self, values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, self.val_min, self.val_max - 1e-6)
        bins = ((clipped - self.val_min) / self.bin_width).long()
        return torch.clamp(bins, 0, self.num_bins - 1)

    def decode_torch(self, bins: torch.Tensor) -> torch.Tensor:
        return self.val_min + (bins.float() + 0.5) * self.bin_width


class MuLawTokenizer:
    """Mu-law companding tokenizer - better resolution near center of range."""

    def __init__(self, num_bins: int, val_min: float, val_max: float, mu: float = 255.0):
        self.num_bins = num_bins
        self.val_min = val_min
        self.val_max = val_max
        self.mu = mu
        self.center = (val_max + val_min) / 2.0
        self.half_range = (val_max - val_min) / 2.0

    def _mu_law_encode(self, x: np.ndarray) -> np.ndarray:
        """Apply mu-law companding."""
        x_norm = (x - self.center) / self.half_range  # [-1, 1]
        x_norm = np.clip(x_norm, -1.0, 1.0)
        compressed = np.sign(x_norm) * np.log1p(self.mu * np.abs(x_norm)) / np.log1p(self.mu)
        return compressed  # [-1, 1]

    def _mu_law_decode(self, y: np.ndarray) -> np.ndarray:
        """Inverse mu-law."""
        expanded = np.sign(y) * (np.power(1.0 + self.mu, np.abs(y)) - 1.0) / self.mu
        return expanded * self.half_range + self.center

    def encode(self, values: np.ndarray) -> np.ndarray:
        compressed = self._mu_law_encode(values)
        bins = ((compressed + 1.0) / 2.0 * self.num_bins).astype(np.int64)
        return np.clip(bins, 0, self.num_bins - 1)

    def decode(self, bins: np.ndarray) -> np.ndarray:
        compressed = (bins.astype(np.float32) + 0.5) / self.num_bins * 2.0 - 1.0
        return self._mu_law_decode(compressed)

    def encode_torch(self, values: torch.Tensor) -> torch.Tensor:
        x_norm = (values - self.center) / self.half_range
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        compressed = torch.sign(x_norm) * torch.log1p(self.mu * torch.abs(x_norm)) / np.log1p(self.mu)
        bins = ((compressed + 1.0) / 2.0 * self.num_bins).long()
        return torch.clamp(bins, 0, self.num_bins - 1)

    def decode_torch(self, bins: torch.Tensor) -> torch.Tensor:
        compressed = (bins.float() + 0.5) / self.num_bins * 2.0 - 1.0
        expanded = torch.sign(compressed) * (torch.pow(1.0 + self.mu, torch.abs(compressed)) - 1.0) / self.mu
        return expanded * self.half_range + self.center


class PerDimTokenizer:
    """Per-dimension tokenizer with separate bins for each dimension."""

    def __init__(self, num_bins: int, dim_ranges: list, tokenizer_type: str = "uniform", mu: float = 255.0):
        """
        Args:
            num_bins: number of bins per dimension
            dim_ranges: list of (min, max) tuples for each dimension
            tokenizer_type: "uniform" or "mulaw"
        """
        self.num_bins = num_bins
        self.num_dims = len(dim_ranges)
        self.vocab_size = num_bins  # shared vocab across dims

        if tokenizer_type == "uniform":
            self.tokenizers = [
                UniformTokenizer(num_bins, r[0], r[1]) for r in dim_ranges
            ]
        elif tokenizer_type == "mulaw":
            self.tokenizers = [
                MuLawTokenizer(num_bins, r[0], r[1], mu=mu) for r in dim_ranges
            ]
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    def encode(self, values: np.ndarray) -> np.ndarray:
        """values: (..., num_dims) -> (..., num_dims) int64"""
        result = np.zeros_like(values, dtype=np.int64)
        for d in range(self.num_dims):
            result[..., d] = self.tokenizers[d].encode(values[..., d])
        return result

    def decode(self, bins: np.ndarray) -> np.ndarray:
        """bins: (..., num_dims) int64 -> (..., num_dims) float32"""
        result = np.zeros_like(bins, dtype=np.float32)
        for d in range(self.num_dims):
            result[..., d] = self.tokenizers[d].decode(bins[..., d])
        return result

    def encode_torch(self, values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values, dtype=torch.long)
        for d in range(self.num_dims):
            result[..., d] = self.tokenizers[d].encode_torch(values[..., d])
        return result

    def decode_torch(self, bins: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(bins.shape, dtype=torch.float32, device=bins.device)
        for d in range(self.num_dims):
            result[..., d] = self.tokenizers[d].decode_torch(bins[..., d])
        return result
