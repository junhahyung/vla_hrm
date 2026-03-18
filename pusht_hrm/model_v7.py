"""
V7: Iterative Action Refinement — making recursive models work like diffusion.

KEY INSIGHT: Diffusion policy works by iteratively denoising random actions.
TRM/HRM's recursive reasoning IS iterative refinement, but we weren't
exploiting it at the token level. V7 changes this:

Training:
  1. Add noise to ground-truth action tokens (like diffusion forward process)
  2. Model sees obs + noisy actions, predicts clean actions
  3. Train at multiple noise levels (like diffusion timesteps)

Inference:
  1. Start with random/uniform action tokens
  2. Feed obs + noisy actions to model → get refined actions
  3. Feed obs + refined actions back → get even better actions
  4. Repeat K times (like diffusion reverse process)

This bridges the gap between discrete token models and diffusion.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pusht_hrm.losses import GaussianSoftLabelLoss, soft_decode


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
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size)
        self.o_proj = CastedLinear(hidden_size, hidden_size)

    def forward(self, h, cos_sin=None):
        B, L, _ = h.shape
        qkv = self.qkv_proj(h).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        v = v.to(q.dtype)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
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
    def __init__(self, obs_dim, hidden_size, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.SiLU(),
                         nn.Linear(hidden_size * 2, hidden_size))
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, obs):
        h = self.input_proj(obs.float())
        for layer in self.layers:
            h = h + layer(h)
        return self.norm(h)


class PushTIterativeRefineV7(nn.Module):
    """
    V7: Iterative Action Refinement model.

    The model takes obs + (possibly noisy) action embeddings and predicts
    clean action tokens. At inference, predictions are fed back iteratively.
    """

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
        self.act_seq_len = self.pred_horizon * self.act_dim
        self.soft_sigma = config.get("soft_sigma", 3.0)
        self.soft_decode_temp = config.get("soft_decode_temp", 0.5)
        self.num_refine_steps = config.get("num_refine_steps", 4)
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        # Noise levels for training (like diffusion timesteps)
        self.noise_levels = config.get("noise_levels", [0.0, 0.1, 0.3, 0.5, 0.8, 1.0])

        self.obs_seq_len = self.obs_horizon
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        # Obs encoder
        self.obs_encoder = DeepObsEncoder(self.obs_dim, self.hidden_size)

        # Action token embedding (for conditioning on noisy/predicted actions)
        self.act_embed = nn.Embedding(self.num_bins + 1, self.hidden_size)  # +1 for mask/noise token
        self.noise_token_id = self.num_bins

        # Noise level embedding (like diffusion timestep embedding)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Positional + type embeddings
        self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
        self.type_embed = nn.Embedding(2, self.hidden_size)

        # Output head
        self.cls_head = nn.Linear(self.hidden_size, self.num_bins)

        # Also regression head
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
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

        # Loss
        self.cls_loss_fn = GaussianSoftLabelLoss(self.num_bins, sigma=self.soft_sigma)

    def _build_input(self, obs, act_tokens, noise_level):
        """
        Build input with obs embeddings + action token embeddings + noise embedding.

        act_tokens: (B, act_seq_len) - can be ground truth, noisy, or predicted
        noise_level: (B,) or scalar - noise level [0, 1]
        """
        B, device = obs.shape[0], obs.device
        obs_emb = self.obs_encoder(obs).to(self.forward_dtype)  # (B, obs_h, D)
        act_emb = self.act_embed(act_tokens).to(self.forward_dtype)  # (B, act_seq_len, D)

        # Add noise level embedding to action tokens
        if isinstance(noise_level, (int, float)):
            noise_level = torch.full((B,), noise_level, device=device)
        noise_emb = self.noise_embed(noise_level.unsqueeze(-1).float()).to(self.forward_dtype)  # (B, D)
        act_emb = act_emb + noise_emb.unsqueeze(1)  # broadcast to all action positions

        seq = torch.cat([obs_emb, act_emb], dim=1)  # (B, total_len, D)

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

        if self.model_type == "hrm":
            with torch.no_grad():
                for h in range(self.H_cycles):
                    for l in range(self.L_cycles):
                        if not (h == self.H_cycles - 1 and l == self.L_cycles - 1):
                            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                    if not (h == self.H_cycles - 1):
                        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
            z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
        else:
            with torch.no_grad():
                for h in range(self.H_cycles - 1):
                    for l in range(self.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                    z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)
            for l in range(self.L_cycles):
                z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
            z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        return z_H

    def _add_noise_to_tokens(self, clean_tokens, noise_level):
        """
        Add noise to clean action tokens (like diffusion forward process).
        noise_level: 0 = clean, 1 = fully random.

        Strategy: with probability noise_level, replace each token with a random token.
        """
        B, L = clean_tokens.shape
        device = clean_tokens.device
        mask = torch.rand(B, L, device=device) < noise_level.unsqueeze(1)
        random_tokens = torch.randint(0, self.num_bins, (B, L), device=device)
        noisy_tokens = torch.where(mask, random_tokens, clean_tokens)
        return noisy_tokens

    def forward(self, obs, clean_act_tokens=None, act_normalized=None):
        """
        Training forward: randomly corrupt action tokens and predict clean ones.
        Samples a random noise level for each batch element.
        """
        B = obs.shape[0]
        device = obs.device

        # Sample random noise level per batch element
        noise_idx = torch.randint(0, len(self.noise_levels), (B,), device=device)
        noise_level = torch.tensor(self.noise_levels, device=device)[noise_idx]

        # Create noisy action tokens
        if clean_act_tokens is not None:
            noisy_tokens = self._add_noise_to_tokens(clean_act_tokens, noise_level)
        else:
            # Full noise (inference mode)
            noisy_tokens = torch.randint(0, self.num_bins, (B, self.act_seq_len), device=device)
            noise_level = torch.ones(B, device=device)

        # Forward
        input_emb = self._build_input(obs, noisy_tokens, noise_level)
        z_H = self._recursive_forward(input_emb, B)

        act_features = z_H[:, self.obs_seq_len:].float()
        cls_logits = self.cls_head(act_features)
        reg_pred = self.reg_head(act_features).squeeze(-1)

        outputs = {"cls_logits": cls_logits, "reg_pred": reg_pred}

        loss = torch.tensor(0.0, device=device)
        if clean_act_tokens is not None:
            cls_loss = self.cls_loss_fn(
                cls_logits.reshape(-1, self.num_bins),
                clean_act_tokens.reshape(-1),
            )
            loss = loss + cls_loss
            outputs["cls_loss"] = cls_loss

        if act_normalized is not None:
            reg_loss = F.mse_loss(reg_pred, act_normalized)
            loss = loss + 0.5 * reg_loss
            outputs["reg_loss"] = reg_loss

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def predict_actions(self, obs, num_steps=None, temperature=None):
        """
        Iterative action refinement at inference.
        Start from random tokens, iteratively refine.
        """
        B = obs.shape[0]
        device = obs.device
        num_steps = num_steps or self.num_refine_steps
        temp = temperature or self.soft_decode_temp

        # Start from random tokens
        act_tokens = torch.randint(0, self.num_bins, (B, self.act_seq_len), device=device)

        # Iteratively refine
        for step in range(num_steps):
            # Decreasing noise level
            noise_level = 1.0 - (step + 1) / num_steps  # 1→0
            noise_t = torch.full((B,), noise_level, device=device)

            input_emb = self._build_input(obs, act_tokens, noise_t)
            z_H = self._recursive_forward(input_emb, B)
            act_features = z_H[:, self.obs_seq_len:].float()
            logits = self.cls_head(act_features)

            # Update tokens via soft decode then discretize
            soft_bins = soft_decode(logits, self.num_bins, temp)
            act_tokens = soft_bins.round().long().clamp(0, self.num_bins - 1)

        # Final soft decode for continuous output
        input_emb = self._build_input(obs, act_tokens, torch.zeros(B, device=device))
        z_H = self._recursive_forward(input_emb, B)
        act_features = z_H[:, self.obs_seq_len:].float()
        logits = self.cls_head(act_features)

        return soft_decode(logits, self.num_bins, temp)


class EMA:
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


def build_model_v7(config):
    return PushTIterativeRefineV7(config)
