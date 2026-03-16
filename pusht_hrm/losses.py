"""
Custom loss functions for discrete token prediction in continuous control.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianSoftLabelLoss(nn.Module):
    """
    Instead of hard one-hot targets, use a Gaussian kernel centered on the true bin.
    Neighboring bins get partial credit proportional to their distance.
    This teaches the model that bin 127 and bin 128 are almost the same.

    target_distribution[i] = exp(-(i - true_bin)^2 / (2 * sigma^2))
    Then normalize to sum to 1 and compute KL divergence.
    """
    def __init__(self, num_bins, sigma=1.0, loss_type="kl"):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.loss_type = loss_type  # "kl" or "ce" (cross-entropy with soft targets)

        # Precompute Gaussian kernel weights for efficiency
        # For each target bin, compute the soft label distribution
        bins = torch.arange(num_bins).float()
        # Create kernel: (num_bins, num_bins) where [t, i] = weight for bin i when target is t
        diff = bins.unsqueeze(0) - bins.unsqueeze(1)  # (num_bins, num_bins)
        kernel = torch.exp(-diff.pow(2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum(dim=1, keepdim=True)  # normalize each row
        self.register_buffer("kernel", kernel)

    def forward(self, logits, targets, ignore_index=-100):
        """
        logits: (B*L, num_bins) raw logits
        targets: (B*L,) integer bin indices
        """
        mask = targets != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        logits = logits[mask]
        targets = targets[mask]

        # Get soft target distributions
        soft_targets = self.kernel[targets]  # (N, num_bins)

        if self.loss_type == "kl":
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")
        else:
            # Cross-entropy with soft targets
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(soft_targets * log_probs).sum(dim=-1).mean()

        return loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression: predict cumulative probabilities P(Y > k) for k=0,...,K-1.
    This preserves the ordering of bins - predicting bin 128 when truth is 127
    is much better than predicting bin 0.

    The model outputs K-1 logits, and the loss is binary cross-entropy on
    cumulative indicators: 1[target > k] for each k.
    """
    def __init__(self, num_bins):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, logits, targets, ignore_index=-100):
        """
        logits: (B*L, num_bins-1) cumulative logits
        targets: (B*L,) integer bin indices
        """
        mask = targets != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        logits = logits[mask]
        targets = targets[mask]

        # Create ordinal targets: 1[target > k] for k=0,...,K-2
        k = torch.arange(self.num_bins - 1, device=targets.device).unsqueeze(0)
        ordinal_targets = (targets.unsqueeze(1) > k).float()  # (N, K-1)

        loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets)
        return loss

    @staticmethod
    def decode(logits):
        """Convert ordinal logits to bin predictions."""
        probs = torch.sigmoid(logits)  # P(Y > k)
        # Expected bin = sum of P(Y > k) for k=0,...,K-2
        # This gives a continuous estimate
        return probs.sum(dim=-1)


class FocalLoss(nn.Module):
    """
    Focal loss: down-weight easy examples, focus on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, num_bins=256):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights
        self.num_bins = num_bins

    def forward(self, logits, targets, ignore_index=-100):
        mask = targets != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        logits = logits[mask]
        targets = targets[mask]

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        loss = (focal_weight * ce_loss).mean()
        return loss


class DistanceWeightedCELoss(nn.Module):
    """
    Cross-entropy weighted by distance: penalize predictions proportional
    to their distance from the true bin.

    Loss = CE(logits, target) * (1 + lambda * |predicted_bin - target_bin| / num_bins)

    This makes predicting a far-away bin much worse than predicting a neighbor.
    """
    def __init__(self, num_bins, distance_weight=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.distance_weight = distance_weight

    def forward(self, logits, targets, ignore_index=-100):
        mask = targets != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        logits = logits[mask]
        targets = targets[mask]

        # Standard CE loss per sample
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Distance penalty based on predicted bin
        pred_bins = logits.argmax(dim=-1)
        distance = (pred_bins - targets).abs().float() / self.num_bins
        weight = 1.0 + self.distance_weight * distance

        return (weight * ce_loss).mean()


class GaussianMixtureLoss(nn.Module):
    """
    Predict parameters of a Gaussian mixture model over bin space.
    Each component has: weight (pi), mean (mu), std (sigma).
    This allows multi-modal predictions naturally.
    """
    def __init__(self, num_bins, num_components=5):
        super().__init__()
        self.num_bins = num_bins
        self.num_components = num_components

    def forward(self, params, targets, ignore_index=-100):
        """
        params: (B*L, num_components * 3) - pi, mu, log_sigma for each component
        targets: (B*L,) integer bin indices (converted to continuous)
        """
        mask = targets != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=params.device)

        params = params[mask]
        targets = targets[mask].float() / self.num_bins  # normalize to [0, 1]

        K = self.num_components
        pi_logits = params[:, :K]
        mu = torch.sigmoid(params[:, K:2*K])  # [0, 1]
        log_sigma = params[:, 2*K:3*K]
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=1.0)

        # Log probabilities under each component
        log_pi = F.log_softmax(pi_logits, dim=-1)
        targets_expanded = targets.unsqueeze(1)  # (N, 1)

        log_probs = log_pi - 0.5 * math.log(2 * math.pi) - log_sigma - \
                    0.5 * ((targets_expanded - mu) / sigma).pow(2)

        # Log-sum-exp for mixture
        loss = -torch.logsumexp(log_probs, dim=-1).mean()
        return loss

    @staticmethod
    def decode(params, num_bins, num_components):
        """Decode GMM params to continuous predictions."""
        K = num_components
        pi = F.softmax(params[:, :K], dim=-1)
        mu = torch.sigmoid(params[:, K:2*K])
        # Use weighted mean of components
        pred = (pi * mu).sum(dim=-1)
        return pred * num_bins


def soft_decode(logits, num_bins, temperature=1.0):
    """
    Soft decoding: weighted average of bin centers by softmax probabilities.
    Much better than argmax for continuous control.

    Args:
        logits: (B, L, num_bins)
        num_bins: vocabulary size
        temperature: softmax temperature (lower = more peaked)
    Returns:
        continuous values: (B, L) in [0, num_bins-1] range
    """
    probs = F.softmax(logits / temperature, dim=-1)
    bin_centers = torch.arange(num_bins, dtype=torch.float32, device=logits.device)
    return (probs * bin_centers).sum(dim=-1)
