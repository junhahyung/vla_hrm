"""
V4: Discrete tokens done right.

Continuous obs encoder + discrete action tokens with:
- Gaussian soft label loss
- Soft decoding (weighted average instead of argmax)
- Ordinal regression option
- Focal loss option
- Support for delta/residual tokenization
- Hierarchical coarse-to-fine via recursive reasoning
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pusht_hrm.losses import (
    GaussianSoftLabelLoss, OrdinalRegressionLoss, FocalLoss,
    DistanceWeightedCELoss, soft_decode,
)


# ============== Building blocks ==============

def rms_norm(x, eps=1e-5):
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


def _find_multiple(a, b):
    return (-(a // -b)) * b


class CastedLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.trunc_normal_(self.weight, std=1.0 / (in_f ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2)
        self.down_proj = CastedLinear(inter, hidden_size)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    return ((q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2)),
            (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2)))


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size)
        self.o_proj = CastedLinear(hidden_size, hidden_size)

    def forward(self, h, cos_sin=None):
        B, L, _ = h.shape
        qkv = self.qkv_proj(h).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, self.hidden_size))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, expansion, causal=False, eps=1e-5):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, causal=causal)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.eps = eps

    def forward(self, h, cos_sin=None):
        h = rms_norm(h + self.attn(h, cos_sin=cos_sin), self.eps)
        h = rms_norm(h + self.mlp(h), self.eps)
        return h


class ReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, h, injection, cos_sin=None):
        h = h + injection
        for layer in self.layers:
            h = layer(h, cos_sin=cos_sin)
        return h


class ObsEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_size, num_layers=3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.SiLU()])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs.float())


# ============== V4 Model ==============

class PushTRecursiveV4(nn.Module):
    """
    V4: Discrete token model with advanced losses and soft decoding.

    Supports both TRM (shared L) and HRM (separate H/L) modes.
    """

    def __init__(self, config):
        super().__init__()
        self.model_type = config.get("model_type", "trm")
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.expansion = config["expansion"]
        self.H_layers = config.get("H_layers", 2)
        self.L_layers = config["L_layers"]
        self.H_cycles = config["H_cycles"]
        self.L_cycles = config["L_cycles"]
        self.obs_horizon = config["obs_horizon"]
        self.pred_horizon = config["pred_horizon"]
        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]
        self.num_bins = config["vocab_size"]
        self.act_seq_len = config.get("act_seq_len", self.pred_horizon * self.act_dim)
        self.loss_type = config.get("loss_type", "gaussian_soft")
        self.soft_sigma = config.get("soft_sigma", 2.0)
        self.soft_decode_temp = config.get("soft_decode_temp", 1.0)
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        self.obs_seq_len = self.obs_horizon
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        # Obs encoder
        self.obs_encoder = ObsEncoder(self.obs_dim, self.hidden_size, num_layers=3)

        # Action queries
        self.act_queries = nn.Parameter(torch.randn(self.act_seq_len, self.hidden_size) * 0.02)

        # Embeddings
        self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
        self.type_embed = nn.Embedding(2, self.hidden_size)

        # Output head
        if self.loss_type == "ordinal":
            self.act_head = nn.Linear(self.hidden_size, self.num_bins - 1)
        else:
            self.act_head = nn.Linear(self.hidden_size, self.num_bins)

        # Reasoning modules
        L_blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                    for _ in range(self.L_layers)]
        self.L_level = ReasoningModule(L_blocks)

        if self.model_type == "hrm":
            H_blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                        for _ in range(self.H_layers)]
            self.H_level = ReasoningModule(H_blocks)

        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // self.num_heads, self.total_seq_len
        )

        self.z_H_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

        # Loss function
        if self.loss_type == "gaussian_soft":
            self.loss_fn = GaussianSoftLabelLoss(self.num_bins, sigma=self.soft_sigma)
        elif self.loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=2.0, num_bins=self.num_bins)
        elif self.loss_type == "distance_weighted":
            self.loss_fn = DistanceWeightedCELoss(self.num_bins, distance_weight=2.0)
        elif self.loss_type == "ordinal":
            self.loss_fn = OrdinalRegressionLoss(self.num_bins)
        elif self.loss_type == "ce":
            self.loss_fn = None  # plain cross-entropy
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _build_input(self, obs):
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_encoder(obs).to(self.forward_dtype)
        act_emb = self.act_queries.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([obs_emb, act_emb], dim=1)
        pos_ids = torch.arange(self.total_seq_len, device=device)
        type_ids = torch.cat([
            torch.zeros(self.obs_seq_len, dtype=torch.long, device=device),
            torch.ones(self.act_seq_len, dtype=torch.long, device=device),
        ])
        seq = seq + self.pos_embed(pos_ids).to(self.forward_dtype) + self.type_embed(type_ids).to(self.forward_dtype)
        return seq

    def _recursive_forward(self, input_emb, B):
        cos_sin = self.rotary_emb()

        z_H = self.z_H_init.view(1, 1, -1).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()
        z_L = self.z_L_init.view(1, 1, -1).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()

        if self.model_type == "trm":
            # TRM: shared L for both levels
            with torch.no_grad():
                for h in range(self.H_cycles - 1):
                    for l in range(self.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                    z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

            for l in range(self.L_cycles):
                z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
            z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        elif self.model_type == "hrm":
            # HRM: separate H and L modules
            with torch.no_grad():
                for h in range(self.H_cycles):
                    for l in range(self.L_cycles):
                        if not (h == self.H_cycles - 1 and l == self.L_cycles - 1):
                            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                    if not (h == self.H_cycles - 1):
                        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
            z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        return z_H

    def forward(self, obs, act_tokens=None, label_smoothing=0.0):
        """
        obs: (B, T_o, obs_dim)
        act_tokens: (B, act_seq_len) discrete targets
        """
        B = obs.shape[0]
        input_emb = self._build_input(obs)
        z_H = self._recursive_forward(input_emb, B)

        act_features = z_H[:, self.obs_seq_len:].float()
        logits = self.act_head(act_features)  # (B, act_seq_len, num_bins or num_bins-1)

        loss = None
        if act_tokens is not None:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = act_tokens.reshape(-1)

            if self.loss_fn is not None:
                loss = self.loss_fn(flat_logits, flat_targets)
            else:
                loss = F.cross_entropy(flat_logits, flat_targets,
                                      label_smoothing=label_smoothing)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict_actions(self, obs, use_soft_decode=True, temperature=None):
        """
        Predict actions from observations.

        If use_soft_decode=True, returns continuous bin indices (float).
        Otherwise returns argmax bin indices (int).
        """
        out = self.forward(obs)
        logits = out["logits"]

        if self.loss_type == "ordinal":
            # Ordinal: decode via cumulative probabilities
            continuous = OrdinalRegressionLoss.decode(logits)
            return continuous  # (B, act_seq_len)

        if use_soft_decode:
            temp = temperature or self.soft_decode_temp
            return soft_decode(logits, self.num_bins, temp)  # (B, act_seq_len) float
        else:
            return logits.argmax(dim=-1)  # (B, act_seq_len) int


def build_model_v4(config):
    return PushTRecursiveV4(config)
