"""
V6: Kitchen-sink model with every trick to beat diffusion policy.

Improvements over V4:
1. EMA (Exponential Moving Average) model for inference
2. Gradient through ALL recursion steps (not just last) via gradient checkpointing
3. Action temporal ensemble at inference (blend overlapping predictions)
4. Deeper obs encoder with residual connections
5. Configurable: soft labels, focal loss, ordinal, etc.
6. Support for varying obs_horizon (2-8)
7. Noise augmentation on both obs AND actions during training
8. Masked action augmentation (VLA-0 inspired)
9. Multi-resolution prediction heads (coarse + fine)
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pusht_hrm.losses import GaussianSoftLabelLoss, FocalLoss, DistanceWeightedCELoss, soft_decode


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
    def __init__(self, hidden_size, num_heads, expansion, eps=1e-5):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)
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


class DeepObsEncoder(nn.Module):
    """Deeper obs encoder with residual connections."""
    def __init__(self, obs_dim, hidden_size, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size),
            ))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, obs):
        h = self.input_proj(obs.float())
        for layer in self.layers:
            h = h + layer(h)  # residual
        return self.norm(h)


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class PushTRecursiveV6(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.get("model_type", "hrm")
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
        self.soft_sigma = config.get("soft_sigma", 3.0)
        self.soft_decode_temp = config.get("soft_decode_temp", 0.5)
        self.grad_all_steps = config.get("grad_all_steps", False)
        self.use_checkpoint = config.get("use_checkpoint", False)
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        # Also output regression head for combined loss
        self.use_regression_head = config.get("use_regression_head", True)
        self.regression_weight = config.get("regression_weight", 0.5)

        self.obs_seq_len = self.obs_horizon
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        # Deep obs encoder
        self.obs_encoder = DeepObsEncoder(self.obs_dim, self.hidden_size, num_layers=4)

        # Action queries
        self.act_queries = nn.Parameter(torch.randn(self.act_seq_len, self.hidden_size) * 0.02)

        # Embeddings
        self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
        self.type_embed = nn.Embedding(2, self.hidden_size)

        # Classification head
        self.cls_head = nn.Linear(self.hidden_size, self.num_bins)

        # Regression head (predicts continuous normalized actions)
        if self.use_regression_head:
            self.reg_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 1),  # 1 value per token position
            )

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
            self.cls_loss_fn = GaussianSoftLabelLoss(self.num_bins, sigma=self.soft_sigma)
        elif self.loss_type == "focal":
            self.cls_loss_fn = FocalLoss(gamma=2.0, num_bins=self.num_bins)
        elif self.loss_type == "distance_weighted":
            self.cls_loss_fn = DistanceWeightedCELoss(self.num_bins, distance_weight=2.0)
        else:
            self.cls_loss_fn = None

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

    def _one_L_step(self, z_L, z_H, input_emb, cos_sin):
        return self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)

    def _one_H_step(self, z_H, z_L, cos_sin):
        if self.model_type == "hrm":
            return self.H_level(z_H, z_L, cos_sin=cos_sin)
        else:
            return self.L_level(z_H, z_L, cos_sin=cos_sin)

    def _recursive_forward(self, input_emb, B):
        cos_sin = self.rotary_emb()
        z_H = self.z_H_init.view(1, 1, -1).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()
        z_L = self.z_L_init.view(1, 1, -1).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()

        if self.grad_all_steps:
            # Gradient through ALL recursion steps (expensive but potentially better)
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    if self.use_checkpoint:
                        z_L = checkpoint(self._one_L_step, z_L, z_H, input_emb, cos_sin, use_reentrant=False)
                    else:
                        z_L = self._one_L_step(z_L, z_H, input_emb, cos_sin)
                if self.use_checkpoint:
                    z_H = checkpoint(self._one_H_step, z_H, z_L, cos_sin, use_reentrant=False)
                else:
                    z_H = self._one_H_step(z_H, z_L, cos_sin)
        else:
            # Standard: no grad for all but last cycle
            with torch.no_grad():
                for h in range(self.H_cycles - 1):
                    for l in range(self.L_cycles):
                        z_L = self._one_L_step(z_L, z_H, input_emb, cos_sin)
                    z_H = self._one_H_step(z_H, z_L, cos_sin)

                # Last H cycle: no grad for all but last L step
                for l in range(self.L_cycles - 1):
                    z_L = self._one_L_step(z_L, z_H, input_emb, cos_sin)

            # Final steps with grad
            z_L = self._one_L_step(z_L, z_H, input_emb, cos_sin)
            z_H = self._one_H_step(z_H, z_L, cos_sin)

        return z_H

    def forward(self, obs, act_tokens=None, act_normalized=None, label_smoothing=0.0):
        """
        obs: (B, T_o, obs_dim)
        act_tokens: (B, act_seq_len) discrete targets
        act_normalized: (B, act_seq_len) normalized continuous targets [0, 1]
        """
        B = obs.shape[0]
        input_emb = self._build_input(obs)
        z_H = self._recursive_forward(input_emb, B)

        act_features = z_H[:, self.obs_seq_len:].float()

        # Classification logits
        cls_logits = self.cls_head(act_features)  # (B, act_seq_len, num_bins)

        outputs = {"cls_logits": cls_logits}
        loss = torch.tensor(0.0, device=obs.device)

        # Classification loss
        if act_tokens is not None:
            flat_logits = cls_logits.reshape(-1, self.num_bins)
            flat_targets = act_tokens.reshape(-1)
            if self.cls_loss_fn is not None:
                cls_loss = self.cls_loss_fn(flat_logits, flat_targets)
            else:
                cls_loss = F.cross_entropy(flat_logits, flat_targets, label_smoothing=label_smoothing)
            loss = loss + cls_loss
            outputs["cls_loss"] = cls_loss

        # Regression loss
        if self.use_regression_head and act_normalized is not None:
            reg_pred = self.reg_head(act_features).squeeze(-1)  # (B, act_seq_len)
            reg_loss = F.mse_loss(reg_pred, act_normalized)
            loss = loss + self.regression_weight * reg_loss
            outputs["reg_loss"] = reg_loss
            outputs["reg_pred"] = reg_pred

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def predict_actions(self, obs, use_soft_decode=True, temperature=None):
        out = self.forward(obs)
        logits = out["cls_logits"]

        if use_soft_decode:
            temp = temperature or self.soft_decode_temp
            return soft_decode(logits, self.num_bins, temp)
        else:
            return logits.argmax(dim=-1)


def build_model_v6(config):
    return PushTRecursiveV6(config)
