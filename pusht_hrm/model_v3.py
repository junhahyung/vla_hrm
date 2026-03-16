"""
V3: Continuous obs + continuous action output (regression) with recursive reasoning.

Key changes from V2:
- Regression output (MSE loss) instead of discrete token prediction
- Support for both discrete and regression modes
- Temporal ensemble over recursion depths
- Action normalization to [-1, 1]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PushTTRM_V3(nn.Module):
    """
    TRM V3: Continuous obs + continuous action regression with recursive reasoning.

    Supports both:
    - Regression mode (MSE loss on normalized actions)
    - Classification mode (cross-entropy on discretized actions)
    - Mixed mode (both losses combined)
    """

    def __init__(self, config):
        super().__init__()
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
        self.output_mode = config.get("output_mode", "regression")  # "regression", "classification", "both"
        self.num_bins = config.get("vocab_size", 256)
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        self.obs_seq_len = self.obs_horizon
        self.act_seq_len = self.pred_horizon  # one query per timestep, predicts act_dim values
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        # Obs encoder
        self.obs_encoder = ObsEncoder(self.obs_dim, self.hidden_size, num_layers=3)

        # Action query tokens (one per timestep)
        self.act_queries = nn.Parameter(torch.randn(self.act_seq_len, self.hidden_size) * 0.02)

        # Positional + type embeddings
        self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
        self.type_embed = nn.Embedding(2, self.hidden_size)

        # Output heads
        if self.output_mode in ("regression", "both"):
            self.reg_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.act_dim),
            )
        if self.output_mode in ("classification", "both"):
            self.cls_head = CastedLinear(self.hidden_size, self.num_bins * self.act_dim, bias=True)

        # Reasoning module
        blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                  for _ in range(self.L_layers)]
        self.L_level = ReasoningModule(blocks)

        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // self.num_heads, self.total_seq_len
        )

        self.z_H_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

    def _build_input(self, obs):
        B = obs.shape[0]
        device = obs.device
        obs_emb = self.obs_encoder(obs).to(self.forward_dtype)  # (B, T_o, D)
        act_emb = self.act_queries.unsqueeze(0).expand(B, -1, -1)  # (B, T_a, D)
        seq = torch.cat([obs_emb, act_emb], dim=1)  # (B, total_len, D)

        pos_ids = torch.arange(self.total_seq_len, device=device)
        type_ids = torch.cat([
            torch.zeros(self.obs_seq_len, dtype=torch.long, device=device),
            torch.ones(self.act_seq_len, dtype=torch.long, device=device),
        ])
        seq = seq + self.pos_embed(pos_ids).to(self.forward_dtype) + self.type_embed(type_ids).to(self.forward_dtype)
        return seq

    def forward(self, obs, target_actions=None, act_tokens=None, label_smoothing=0.0):
        """
        obs: (B, T_o, obs_dim)
        target_actions: (B, T_a, act_dim) normalized continuous actions for regression
        act_tokens: (B, T_a * act_dim) discrete tokens for classification
        """
        B = obs.shape[0]
        cos_sin = self.rotary_emb()
        input_emb = self._build_input(obs)

        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()

        # Recursive reasoning
        with torch.no_grad():
            for h in range(self.H_cycles - 1):
                for l in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        for l in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Extract action features
        act_features = z_H[:, self.obs_seq_len:].float()  # (B, T_a, D)

        outputs = {}
        loss = torch.tensor(0.0, device=obs.device)

        # Regression output
        if self.output_mode in ("regression", "both"):
            pred_actions = self.reg_head(act_features)  # (B, T_a, act_dim)
            outputs["pred_actions"] = pred_actions
            if target_actions is not None:
                reg_loss = F.mse_loss(pred_actions, target_actions)
                loss = loss + reg_loss
                outputs["reg_loss"] = reg_loss

        # Classification output
        if self.output_mode in ("classification", "both"):
            cls_logits = self.cls_head(act_features)  # (B, T_a, num_bins * act_dim)
            cls_logits = cls_logits.view(B, self.act_seq_len, self.act_dim, self.num_bins)
            outputs["cls_logits"] = cls_logits
            if act_tokens is not None:
                act_tokens_reshaped = act_tokens.view(B, self.pred_horizon, self.act_dim)
                cls_loss = F.cross_entropy(
                    cls_logits.reshape(-1, self.num_bins),
                    act_tokens_reshaped.reshape(-1),
                    label_smoothing=label_smoothing,
                )
                loss = loss + cls_loss
                outputs["cls_loss"] = cls_loss

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def predict_actions(self, obs):
        """
        obs: (B, T_o, obs_dim) -> (B, T_a, act_dim) continuous actions
        """
        out = self.forward(obs)
        if "pred_actions" in out:
            return out["pred_actions"]
        elif "cls_logits" in out:
            # Decode from logits to bin centers
            return out["cls_logits"].argmax(dim=-1)  # (B, T_a, act_dim) token indices
        else:
            raise RuntimeError("No action output available")


class PushTHRM_V3(nn.Module):
    """HRM V3: Separate H/L modules with regression output."""

    def __init__(self, config):
        super().__init__()
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
        self.output_mode = config.get("output_mode", "regression")
        self.num_bins = config.get("vocab_size", 256)
        self.forward_dtype = getattr(torch, config.get("forward_dtype", "bfloat16"))

        self.obs_seq_len = self.obs_horizon
        self.act_seq_len = self.pred_horizon
        self.total_seq_len = self.obs_seq_len + self.act_seq_len

        self.obs_encoder = ObsEncoder(self.obs_dim, self.hidden_size, num_layers=3)
        self.act_queries = nn.Parameter(torch.randn(self.act_seq_len, self.hidden_size) * 0.02)
        self.pos_embed = nn.Embedding(self.total_seq_len, self.hidden_size)
        self.type_embed = nn.Embedding(2, self.hidden_size)

        if self.output_mode in ("regression", "both"):
            self.reg_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.act_dim),
            )
        if self.output_mode in ("classification", "both"):
            self.cls_head = CastedLinear(self.hidden_size, self.num_bins * self.act_dim, bias=True)

        H_blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                    for _ in range(self.H_layers)]
        L_blocks = [TransformerBlock(self.hidden_size, self.num_heads, self.expansion)
                    for _ in range(self.L_layers)]
        self.H_level = ReasoningModule(H_blocks)
        self.L_level = ReasoningModule(L_blocks)

        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // self.num_heads, self.total_seq_len
        )

        self.z_H_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

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

    def forward(self, obs, target_actions=None, act_tokens=None, label_smoothing=0.0):
        B = obs.shape[0]
        cos_sin = self.rotary_emb()
        input_emb = self._build_input(obs)

        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, self.total_seq_len, -1).to(self.forward_dtype).clone()

        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    if not (h == self.H_cycles - 1 and l == self.L_cycles - 1):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                if not (h == self.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        act_features = z_H[:, self.obs_seq_len:].float()

        outputs = {}
        loss = torch.tensor(0.0, device=obs.device)

        if self.output_mode in ("regression", "both"):
            pred_actions = self.reg_head(act_features)
            outputs["pred_actions"] = pred_actions
            if target_actions is not None:
                reg_loss = F.mse_loss(pred_actions, target_actions)
                loss = loss + reg_loss
                outputs["reg_loss"] = reg_loss

        if self.output_mode in ("classification", "both"):
            cls_logits = self.cls_head(act_features)
            cls_logits = cls_logits.view(B, self.act_seq_len, self.act_dim, self.num_bins)
            outputs["cls_logits"] = cls_logits
            if act_tokens is not None:
                act_tokens_reshaped = act_tokens.view(B, self.pred_horizon, self.act_dim)
                cls_loss = F.cross_entropy(
                    cls_logits.reshape(-1, self.num_bins),
                    act_tokens_reshaped.reshape(-1),
                    label_smoothing=label_smoothing,
                )
                loss = loss + cls_loss
                outputs["cls_loss"] = cls_loss

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def predict_actions(self, obs):
        out = self.forward(obs)
        if "pred_actions" in out:
            return out["pred_actions"]
        elif "cls_logits" in out:
            return out["cls_logits"].argmax(dim=-1)


def build_model_v3(config):
    model_type = config.get("model_type", "trm")
    if model_type == "trm":
        return PushTTRM_V3(config)
    elif model_type == "hrm":
        return PushTHRM_V3(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
