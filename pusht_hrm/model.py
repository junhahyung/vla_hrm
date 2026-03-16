"""
TRM and HRM models adapted for PushT continuous action prediction.

Key adaptations from original puzzle models:
- Remove puzzle embeddings (not needed for PushT)
- Add dimension-aware positional encoding
- Support continuous output via discrete token prediction
- Simplified ACT (no halting needed for fixed-length action prediction)
"""
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ============== Building blocks ==============

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def _find_multiple(a, b):
    return (-(a // -b)) * b


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.weight, std=1.0 / (in_features ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2)
        self.down_proj = CastedLinear(inter, hidden_size)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
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
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size)
        self.o_proj = CastedLinear(hidden_size, hidden_size)

    def forward(self, hidden_states, cos_sin=None):
        B, L, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)  # B, H, L, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, expansion, causal=False, rms_eps=1e-5):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, causal=causal)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.rms_eps = rms_eps

    def forward(self, hidden_states, cos_sin=None):
        # Post-norm (like original TRM/HRM)
        hidden_states = rms_norm(
            hidden_states + self.attn(hidden_states, cos_sin=cos_sin),
            self.rms_eps
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            self.rms_eps
        )
        return hidden_states


class MLPBlock(nn.Module):
    """MLP variant for the L-level (as in TRM's mlp_t option)."""
    def __init__(self, seq_len, hidden_size, expansion, rms_eps=1e-5):
        super().__init__()
        self.mlp_t = SwiGLU(seq_len, expansion)  # operates on transposed seq dim
        self.mlp = SwiGLU(hidden_size, expansion)
        self.rms_eps = rms_eps

    def forward(self, hidden_states, cos_sin=None):
        # Transpose mixing (like TRM's MLP-T)
        h = hidden_states.transpose(1, 2)
        h = rms_norm(h + self.mlp_t(h), self.rms_eps)
        hidden_states = h.transpose(1, 2)
        # Channel mixing
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), self.rms_eps)
        return hidden_states


class ReasoningModule(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states, input_injection, cos_sin=None):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states


# ============== Dimension-aware positional encoding ==============

class DimPositionalEncoding(nn.Module):
    """
    Positional encoding that encodes both:
    1. Time step position (which timestep in the sequence)
    2. Dimension identity (which dimension of obs/action)

    This helps the model understand the structure of the flattened sequence.
    """

    def __init__(self, hidden_size, obs_horizon, pred_horizon, obs_dim, act_dim):
        super().__init__()
        total_len = obs_horizon * obs_dim + pred_horizon * act_dim

        # Time position embedding
        max_time = max(obs_horizon, pred_horizon)
        self.time_embed = nn.Embedding(max_time, hidden_size)

        # Dimension identity embedding
        max_dim = max(obs_dim, act_dim)
        self.dim_embed = nn.Embedding(max_dim, hidden_size)

        # Token type embedding (obs vs action)
        self.type_embed = nn.Embedding(2, hidden_size)

        # Precompute position info
        time_ids = []
        dim_ids = []
        type_ids = []
        for t in range(obs_horizon):
            for d in range(obs_dim):
                time_ids.append(t)
                dim_ids.append(d)
                type_ids.append(0)  # observation
        for t in range(pred_horizon):
            for d in range(act_dim):
                time_ids.append(t)
                dim_ids.append(d)
                type_ids.append(1)  # action

        self.register_buffer("time_ids", torch.tensor(time_ids, dtype=torch.long))
        self.register_buffer("dim_ids", torch.tensor(dim_ids, dtype=torch.long))
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long))

    def forward(self) -> torch.Tensor:
        return self.time_embed(self.time_ids) + self.dim_embed(self.dim_ids) + self.type_embed(self.type_ids)


# ============== Main Models ==============

class PushTTRM(nn.Module):
    """
    Tiny Recursive Model adapted for PushT.

    Architecture: Single reasoning module (L-level) that recursively refines
    both the hidden state and predictions.

    Input: tokenized observations (given) + masked action positions
    Output: predicted action tokens
    """

    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.expansion = config["expansion"]
        self.L_layers = config["L_layers"]
        self.H_cycles = config["H_cycles"]
        self.L_cycles = config["L_cycles"]
        self.obs_horizon = config["obs_horizon"]
        self.pred_horizon = config["pred_horizon"]
        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]
        self.pos_encoding_type = config.get("pos_encoding_type", "dim_aware")
        self.use_mlp_t = config.get("use_mlp_t", False)
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        self.obs_seq_len = self.obs_horizon * self.obs_dim
        self.act_seq_len = self.pred_horizon * self.act_dim
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        # Token embedding
        embed_scale = math.sqrt(self.hidden_size)
        self.embed_scale = embed_scale
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        nn.init.trunc_normal_(self.embed_tokens.weight, std=1.0 / embed_scale)

        # Mask token for action positions during inference
        self.mask_token = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

        # Output head
        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size)

        # Positional encoding
        if self.pos_encoding_type == "dim_aware":
            self.pos_enc = DimPositionalEncoding(
                self.hidden_size, self.obs_horizon, self.pred_horizon,
                self.obs_dim, self.act_dim
            )
        elif self.pos_encoding_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // self.num_heads,
                max_position_embeddings=self.total_seq_len
            )
        elif self.pos_encoding_type == "learned":
            self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
            nn.init.trunc_normal_(self.pos_embed.weight, std=1.0 / embed_scale)

        # Reasoning layers (L-level)
        if self.use_mlp_t:
            blocks = [MLPBlock(self.total_seq_len, self.hidden_size, self.expansion)
                      for _ in range(self.L_layers)]
        else:
            blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                      for _ in range(self.L_layers)]
        self.L_level = ReasoningModule(nn.ModuleList(blocks))

        # Initial latent states
        self.z_H_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

    def _get_cos_sin(self):
        if hasattr(self, "rotary_emb"):
            return self.rotary_emb()
        return None

    def _get_input_embeddings(self, input_tokens: torch.Tensor, mask_actions: bool = False):
        """
        Embed input tokens. If mask_actions=True, replace action token embeddings
        with the learned mask token (for inference).
        """
        B = input_tokens.shape[0]
        embeddings = self.embed_tokens(input_tokens)  # (B, L, D)

        if mask_actions:
            mask = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, self.act_seq_len, -1)
            embeddings = torch.cat([embeddings[:, :self.obs_seq_len], mask], dim=1)

        # Add positional encoding
        if self.pos_encoding_type == "dim_aware":
            pos = self.pos_enc()
            embeddings = embeddings + pos.unsqueeze(0)
        elif self.pos_encoding_type == "learned":
            pos_ids = torch.arange(self.total_seq_len, device=input_tokens.device)
            embeddings = 0.707106781 * (embeddings + self.pos_embed(pos_ids).unsqueeze(0))

        return self.embed_scale * embeddings.to(self.forward_dtype)

    def forward(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor = None,
                mask_actions: bool = False, label_smoothing: float = 0.0):
        """
        Forward pass with recursive reasoning.

        Args:
            input_tokens: (B, total_seq_len) - tokenized obs + actions
            target_tokens: (B, total_seq_len) - targets with -100 for obs positions
            mask_actions: if True, mask action tokens (for inference)
            label_smoothing: label smoothing factor for cross-entropy
        """
        B = input_tokens.shape[0]
        cos_sin = self._get_cos_sin()

        input_emb = self._get_input_embeddings(input_tokens, mask_actions=mask_actions)

        # Initialize latent states
        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).clone()
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).clone()

        # Recursive reasoning: H_cycles of H-level, each with L_cycles of L-level
        # TRM uses shared L_level for both L and H updates
        with torch.no_grad():
            for h in range(self.H_cycles - 1):
                for l in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Last cycle with grad
        for l in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Predict action tokens
        logits = self.lm_head(z_H.float())  # (B, L, vocab_size)

        loss = None
        if target_tokens is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_tokens.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict_actions(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """
        Given observation tokens, predict action tokens.

        Args:
            obs_tokens: (B, obs_seq_len) tokenized observations
        Returns:
            act_tokens: (B, act_seq_len) predicted action tokens
        """
        B = obs_tokens.shape[0]
        device = obs_tokens.device

        # Create dummy action tokens (will be masked)
        dummy_acts = torch.zeros(B, self.act_seq_len, dtype=torch.long, device=device)
        input_tokens = torch.cat([obs_tokens, dummy_acts], dim=1)

        out = self.forward(input_tokens, mask_actions=True)
        logits = out["logits"][:, self.obs_seq_len:]  # (B, act_seq_len, vocab)
        act_tokens = logits.argmax(dim=-1)  # (B, act_seq_len)
        return act_tokens


class PushTHRM(nn.Module):
    """
    Hierarchical Reasoning Model adapted for PushT.

    Key difference from TRM: separate H-level and L-level modules
    (H for slow abstract planning, L for fast detailed computation).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.expansion = config["expansion"]
        self.H_layers = config["H_layers"]
        self.L_layers = config["L_layers"]
        self.H_cycles = config["H_cycles"]
        self.L_cycles = config["L_cycles"]
        self.obs_horizon = config["obs_horizon"]
        self.pred_horizon = config["pred_horizon"]
        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]
        self.pos_encoding_type = config.get("pos_encoding_type", "dim_aware")
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        self.obs_seq_len = self.obs_horizon * self.obs_dim
        self.act_seq_len = self.pred_horizon * self.act_dim
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        # Token embedding
        embed_scale = math.sqrt(self.hidden_size)
        self.embed_scale = embed_scale
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        nn.init.trunc_normal_(self.embed_tokens.weight, std=1.0 / embed_scale)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

        # Output head
        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size)

        # Positional encoding
        if self.pos_encoding_type == "dim_aware":
            self.pos_enc = DimPositionalEncoding(
                self.hidden_size, self.obs_horizon, self.pred_horizon,
                self.obs_dim, self.act_dim
            )
        elif self.pos_encoding_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // self.num_heads,
                max_position_embeddings=self.total_seq_len
            )
        elif self.pos_encoding_type == "learned":
            self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
            nn.init.trunc_normal_(self.pos_embed.weight, std=1.0 / embed_scale)

        # Separate H-level and L-level reasoning modules
        H_blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                    for _ in range(self.H_layers)]
        L_blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                    for _ in range(self.L_layers)]
        self.H_level = ReasoningModule(nn.ModuleList(H_blocks))
        self.L_level = ReasoningModule(nn.ModuleList(L_blocks))

        # Initial latent states
        self.z_H_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

    def _get_cos_sin(self):
        if hasattr(self, "rotary_emb"):
            return self.rotary_emb()
        return None

    def _get_input_embeddings(self, input_tokens, mask_actions=False):
        B = input_tokens.shape[0]
        embeddings = self.embed_tokens(input_tokens)

        if mask_actions:
            mask = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, self.act_seq_len, -1)
            embeddings = torch.cat([embeddings[:, :self.obs_seq_len], mask], dim=1)

        if self.pos_encoding_type == "dim_aware":
            pos = self.pos_enc()
            embeddings = embeddings + pos.unsqueeze(0)
        elif self.pos_encoding_type == "learned":
            pos_ids = torch.arange(self.total_seq_len, device=input_tokens.device)
            embeddings = 0.707106781 * (embeddings + self.pos_embed(pos_ids).unsqueeze(0))

        return self.embed_scale * embeddings.to(self.forward_dtype)

    def forward(self, input_tokens, target_tokens=None, mask_actions=False, label_smoothing=0.0):
        B = input_tokens.shape[0]
        cos_sin = self._get_cos_sin()

        input_emb = self._get_input_embeddings(input_tokens, mask_actions=mask_actions)

        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).clone()
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).clone()

        # Recursive reasoning with separate H and L modules
        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    if not (h == self.H_cycles - 1 and l == self.L_cycles - 1):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                if not (h == self.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # 1-step grad (last L + last H)
        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        logits = self.lm_head(z_H.float())

        loss = None
        if target_tokens is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_tokens.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict_actions(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dummy_acts = torch.zeros(B, self.act_seq_len, dtype=torch.long, device=device)
        input_tokens = torch.cat([obs_tokens, dummy_acts], dim=1)
        out = self.forward(input_tokens, mask_actions=True)
        logits = out["logits"][:, self.obs_seq_len:]
        return logits.argmax(dim=-1)


class PushTGPT(nn.Module):
    """
    Autoregressive GPT baseline for comparison.
    Predicts action tokens one by one given observation tokens.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.expansion = config["expansion"]
        self.num_layers = config.get("num_layers", 4)
        self.obs_horizon = config["obs_horizon"]
        self.pred_horizon = config["pred_horizon"]
        self.obs_dim = config["obs_dim"]
        self.act_dim = config["act_dim"]

        self.obs_seq_len = self.obs_horizon * self.obs_dim
        self.act_seq_len = self.pred_horizon * self.act_dim
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        embed_scale = math.sqrt(self.hidden_size)
        self.embed_scale = embed_scale
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        nn.init.trunc_normal_(self.embed_tokens.weight, std=1.0 / embed_scale)

        self.pos_enc = DimPositionalEncoding(
            self.hidden_size, self.obs_horizon, self.pred_horizon,
            self.obs_dim, self.act_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.expansion, causal=True)
            for _ in range(self.num_layers)
        ])

        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size)

    def forward(self, input_tokens, target_tokens=None, mask_actions=False, label_smoothing=0.0):
        embeddings = self.embed_scale * self.embed_tokens(input_tokens)
        pos = self.pos_enc()
        embeddings = embeddings + pos.unsqueeze(0)

        h = embeddings.to(torch.bfloat16)
        for block in self.blocks:
            h = block(h)

        logits = self.lm_head(h.float())

        loss = None
        if target_tokens is not None:
            # Shift for autoregressive: predict next token
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = target_tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict_actions(self, obs_tokens):
        B = obs_tokens.shape[0]
        device = obs_tokens.device

        # Start with obs tokens, generate action tokens autoregressively
        tokens = obs_tokens.clone()
        for _ in range(self.act_seq_len):
            out = self.forward(tokens)
            next_token = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens[:, self.obs_seq_len:]


def build_model(config: dict) -> nn.Module:
    """Factory function to build model from config."""
    model_type = config.get("model_type", "trm")
    if model_type == "trm":
        return PushTTRM(config)
    elif model_type == "hrm":
        return PushTHRM(config)
    elif model_type == "gpt":
        return PushTGPT(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
