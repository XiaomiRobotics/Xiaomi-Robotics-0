# Copyright (C) 2026 Xiaomi Corporation.
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, LogisticNormal

from transformers import Qwen3VLTextConfig
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from mibot.models import MIMODEL
from mibot.models.VLM.qwen3vl import Qwen3VLForConditionalGeneration
from mibot.utils.device import resolve_attn_implementation
from mibot.utils.model_utils import auto_cast


# ============================================================
# Helper functions
# ============================================================


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: shift-scale transformation used in DiT.

    Returns ``x * (1 + scale) + shift``, following the DiT / AdaLN-Zero
    formulation that lets the network learn whether to skip or amplify
    each sub-layer.
    """
    return x * (1 + scale) + shift


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA).

    Args:
        hidden_states: ``(batch, num_kv_heads, seq_len, head_dim)``
        n_rep: Number of repetitions per KV head (``num_q_heads // num_kv_heads``).

    Returns:
        Tensor with shape ``(batch, num_q_heads, seq_len, head_dim)``.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return hidden_states.repeat_interleave(n_rep, dim=1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor of shape ``(B, H, S, D)``.
        k: Key tensor of shape ``(B, H, S, D)``.
        cos: Cosine component of the rotary embedding.
        sin: Sine component of the rotary embedding.
        position_ids: Unused, kept for API compatibility.
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin for broadcast.

    Returns:
        Tuple of rotated (query, key) tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================
# Projectors & Embedders
# ============================================================


class MLPProjector(nn.Module):
    """Multi-layer perceptron projector with optional GELU activation.

    Used to project between different dimensional spaces (e.g. state/action
    dimensions to DiT hidden size).

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        num_layers: Number of linear layers (intermediate layers use GELU).
        bias: Whether to include bias in linear layers.
    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, output_dim, bias=bias)]
        for _ in range(1, num_layers):
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(output_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP.

    Used for conditioning the DiT on the diffusion timestep *t*.

    Args:
        hidden_size: Output dimension of the MLP (matches DiT hidden size).
        frequency_embedding_size: Dimension of the sinusoidal frequency embedding.
        dtype: Data type for the frequency embedding computation.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Compute sinusoidal timestep embedding.

        Args:
            t: Timestep tensor of shape ``(B,)``.
            dim: Embedding dimension (should equal ``frequency_embedding_size``).
            max_period: Controls the frequency range of the embedding.

        Returns:
            Embedding tensor of shape ``(B, dim)``.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep *t* and return with a sequence dimension.

        Args:
            t: Timestep tensor of shape ``(B,)``.

        Returns:
            Embedding of shape ``(B, 1, hidden_size)``.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        # Add sequence dimension: (B, 1, D)
        return t_emb[:, None]


# ============================================================
# DiT components
# ============================================================


class DiTAttention(nn.Module):
    """Multi-head attention with GQA, QK-norm, and VLM KV-cache for DiT decoder.

    Cross-attends to the VLM's cached key-value pairs while applying
    QK-RMSNorm for training stability.

    Args:
        hidden_size: Total attention dimension (``num_heads * head_dim``).
        head_dim: Dimension per attention head.
        kv_heads: Number of KV heads (grouped-query attention).
        dropout: Attention dropout probability (only active during training).
    """

    def __init__(self, hidden_size: int = 768, head_dim: int = 64, kv_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(self.head_dim)
        self.k_norm = Qwen2RMSNorm(self.head_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, torch.Tensor],
        position_embeds: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: QKV projection → RoPE → cross-attend with cached KV → output.

        Args:
            hidden_state: Input of shape ``(B, S, D)``.
            past_key_values: Tuple of (cached_key, cached_value) from the VLM.
            position_embeds: (cos, sin) rotary embedding tensors.
            attn_mask: Boolean attention mask.

        Returns:
            Attended output of shape ``(B, S, D)``.
        """
        bsz, q_len, _ = hidden_state.size()

        qkv = self.qkv_proj(hidden_state)
        qkv = qkv.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary position embedding
        cos, sin = position_embeds
        if cos.ndim == 4:
            cos = cos[0]
            sin = sin[0]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Prepend cached KV from VLM
        k_cache, v_cache = past_key_values
        k_cache = repeat_kv(k_cache, self.kv_group)
        v_cache = repeat_kv(v_cache, self.kv_group)

        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DiTMLP(nn.Module):
    """SwiGLU MLP used in DiT decoder layers.

    Args:
        hidden_size: Input and output dimension.  Intermediate size is ``4 * hidden_size``.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: ``down(gelu(gate(x)) * up(x))``."""
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DecoderLayer(nn.Module):
    """DiT decoder layer with AdaLN modulation conditioned on diffusion timestep.

    Each layer produces 6 modulation parameters (shift/scale/gate for both
    the attention and FFN sub-layers) from the timestep embedding.

    Args:
        hidden_size: Model hidden dimension.
        head_dim: Dimension per attention head.
        kv_heads: Number of KV heads for GQA.
    """

    def __init__(self, hidden_size: int = 768, head_dim: int = 64, kv_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = DiTAttention(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads)
        self.mlp = DiTMLP(hidden_size=hidden_size)

        # LayerNorms: input → attn → middle, post → ffn → final
        self.input_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.middle_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.post_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.final_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)

        # AdaLN: produces 6 modulation parameters (shift/scale/gate for attn & ffn)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, torch.Tensor],
        position_embeds: Tuple[torch.Tensor, torch.Tensor],
        t_embeds: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN modulation.

        Args:
            hidden_states: ``(B, S, D)`` input.
            past_key_values: VLM KV-cache for joint self-attention.
            position_embeds: (cos, sin) rotary embedding tensors.
            t_embeds: Timestep modulation parameters ``(B, 6, D)``.
            attn_mask: Boolean attention mask.

        Returns:
            Modulated output of shape ``(B, S, D)``.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + t_embeds).chunk(
            6, dim=1
        )

        # Attention block with AdaLN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        hidden_states = self.attn(hidden_states, past_key_values, position_embeds, attn_mask=attn_mask)
        hidden_states = residual + gate_msa * hidden_states
        hidden_states = self.middle_layernorm(hidden_states)

        # FFN block with AdaLN
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class DiT(nn.Module):
    """Diffusion Transformer that cross-attends to VLM KV-cache with AdaLN timestep conditioning.

    The DiT layers align with the *tail* of the VLM's KV-cache so that
    deeper DiT layers attend to later VLM layers.

    Args:
        hidden_size: Model hidden dimension.
        layer_num: Number of decoder layers.
        head_dim: Dimension per attention head.
        kv_heads: Number of KV heads for GQA.
    """

    def __init__(self, hidden_size: int = 768, layer_num: int = 8, head_dim: int = 128, kv_heads: int = 2):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads) for _ in range(layer_num)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        position_embeds: Tuple[torch.Tensor, torch.Tensor],
        t_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all DiT layers.

        Args:
            hidden_states: ``(B, S, D)`` input.
            past_key_values: Per-layer VLM KV-cache.
            attn_mask: Boolean attention mask.
            position_embeds: (cos, sin) rotary embedding tensors.
            t_embeds: Timestep modulation parameters ``(B, 6, D)``.

        Returns:
            Output of shape ``(B, S, D)``.
        """
        # Align DiT layers with the tail of VLM KV-cache
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, past_key_values[start_idx + i], position_embeds, t_embeds, attn_mask=attn_mask
            )
        return hidden_states


# ============================================================
# XR0 Model
# ============================================================


@MIMODEL.register_module()
class XR0(nn.Module):
    """Vision-Language-Action model: VLM (Qwen3-VL) encodes vision+language, DiT decodes actions via rectified flow."""

    def __init__(
        self,
        state_shape: Tuple[int, int] = (1, 32),
        action_shape: Tuple[int, int] = (30, 32),
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        num_steps: int = 5,
        flow_sampling: str = "beta",
        local_window: int = 4,
        training_repeat: int = 4,
        enable_freq: bool = False,
        prefix_mask_prob: float = 0.5,
        async_train: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        # None picks flash_attention_2 on CUDA when flash_attn is installed, otherwise sdpa.
        self.attn_implementation = resolve_attn_implementation(requested=attn_implementation)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dit_num_layers = dit_num_layers
        self.dit_hidden_size = dit_hidden_size
        self.num_steps = num_steps
        self.local_window = local_window
        self.training_repeat = training_repeat
        self.freq_coefficient = 1.0 if enable_freq else 0.0
        self.prefix_mask_prob = prefix_mask_prob
        self.async_train = async_train

        # Rectified flow timestep sampling distributions
        self.flow_sampling = flow_sampling
        self.logistic_normal = LogisticNormal(0.0, 1.0)
        self.beta = Beta(1.5, 1.0)

        self._build_model()

    def _build_model(self) -> None:
        """Instantiate all sub-modules: VLM backbone, DiT head, projectors, and embeddings."""
        # VLM backbone
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", attn_implementation=self.attn_implementation, dtype=torch.bfloat16
        ).train()
        self.vlm.model.get_input_embeddings().requires_grad_(False)
        self.vlm.model.visual.gradient_checkpointing_enable()

        # DiT policy head
        self.dit = DiT(hidden_size=self.dit_hidden_size, kv_heads=8, layer_num=self.dit_num_layers)

        # State / action projectors
        self.state_projector = MLPProjector(
            input_dim=self.state_shape[-1], output_dim=self.dit_hidden_size, num_layers=2
        )
        self.action_projector = MLPProjector(
            input_dim=self.action_shape[-1], output_dim=self.dit_hidden_size, num_layers=2
        )
        self.action_output_layer = MLPProjector(
            input_dim=self.dit_hidden_size, output_dim=self.action_shape[-1], num_layers=2
        )

        # Timestep embedding for diffusion t
        self.t_embedder = TimestepEmbedder(self.dit_hidden_size)
        self.t_projector = MLPProjector(input_dim=self.dit_hidden_size, output_dim=6 * self.dit_hidden_size, bias=True)

        # RoPE for DiT (same config as VLM)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct"))

        # Sink token prepended to DiT input
        self.sink = nn.Embedding(1, self.dit_hidden_size)

        # P2_Local-style local causal mask for [sink + state + action].
        s_len = self.state_shape[-2] + 1
        a_len = self.action_shape[-2]
        mask_ss = torch.tril(torch.ones(s_len, s_len))
        mask_sa = torch.zeros(s_len, a_len)
        mask_as = torch.ones(a_len, s_len)
        mask_aa = torch.tril(torch.ones(a_len, a_len))
        mask_aa = mask_aa * torch.triu(torch.ones(a_len, a_len), diagonal=-self.local_window)

        top = torch.cat([mask_ss, mask_sa], dim=1)
        bottom = torch.cat([mask_as, mask_aa], dim=1)
        full_mask = torch.cat([top, bottom], dim=0)
        self.register_buffer("saved_causal_mask", full_mask.unsqueeze(0).int(), persistent=False)

        self.to(torch.bfloat16)

    # --------------------------------------------------------
    # Rectified flow methods
    # --------------------------------------------------------

    @torch.no_grad()
    def _sample_timestep(self, batch_size: int, dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> torch.Tensor:
        """Sample random timesteps for rectified flow training.

        The distribution is controlled by ``self.flow_sampling``:
        - ``"logit_normal"``: LogisticNormal(0, 1)
        - ``"beta"``: Beta(1.5, 1.0) rescaled to (0, 0.999)
        - otherwise: Uniform(0, 1)

        Args:
            batch_size: Number of timesteps to sample.
            dtype: Output tensor dtype.
            device: Output tensor device.

        Returns:
            Timestep tensor of shape ``(batch_size,)``.
        """
        if self.flow_sampling == "logit_normal":
            u = self.logistic_normal.sample((batch_size,))[:, 0].to(device)
        elif self.flow_sampling == "beta":
            u = self.beta.sample((batch_size,)).to(device)
            u = (1 - u) * 0.999
        else:
            u = torch.rand(size=(batch_size,), device=device)
        return u.to(dtype)

    @torch.no_grad()
    def _flow_interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between noise and data: ``z_t = (1-t)*x0 + t*x1``.

        Args:
            x0: Noise sample (source distribution).
            x1: Data sample (target distribution).
            t: Interpolation coefficient in [0, 1].

        Returns:
            Interpolated tensor with the same shape as x0/x1.
        """
        return (1 - t) * x0 + t * x1

    @torch.no_grad()
    def _flow_velocity_target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Velocity target for rectified flow: ``v = x1 - x0``.

        Args:
            x0: Noise sample.
            x1: Data sample.

        Returns:
            Velocity tensor of the same shape.
        """
        return x1 - x0

    @torch.no_grad()
    def _flow_generate(self, x0: torch.Tensor, dit_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Euler integration: generate action from noise over ``num_steps`` steps.

        Args:
            x0: Initial noise tensor of shape ``(B, action_len, action_dim)``.
            dit_kwargs: Keyword arguments forwarded to ``dit_forward``.

        Returns:
            Denoised action prediction.
        """
        dt = 1.0 / self.num_steps
        z = x0.clone()
        for step in range(self.num_steps):
            t = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * step / self.num_steps
            v = self.dit_forward(z, t, **dit_kwargs)
            z = z + v * dt
        return z

    # --------------------------------------------------------
    # DiT forward
    # --------------------------------------------------------

    def dit_forward(
        self,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        """Single forward pass of DiT.

        1. Embed timestep *t* → 6 AdaLN modulation parameters per layer.
        2. Project noisy action to hidden dim.
        3. Prepend [sink, state] tokens.
        4. Run DiT decoder layers (cross-attending to VLM KV-cache).
        5. Extract action tokens and project back to action dim.

        Args:
            noisy_action: Noisy action tensor ``(B, action_len, action_dim)``.
            t: Timestep ``(B, 1, 1)``.
            action_mask: Binary mask ``(B, action_len, action_dim)``.
            state_embed: Projected state ``(B, state_len, D)``.
            position_embeds: (cos, sin) rotary embeddings for DiT tokens.
            past_key_values: Per-layer VLM KV-cache.
            attn_mask: Boolean attention mask.

        Returns:
            Predicted velocity (training) or action (inference) of shape
            ``(B, action_len, action_dim)``.
        """
        # Timestep conditioning: embed t → 6 modulation parameters per layer
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        # Project noisy action to DiT hidden dim
        noisy_action = noisy_action * action_mask
        noisy_action = self.action_projector(noisy_action)

        # Concatenate: [sink, state, noisy_action]
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()

        # DiT forward with VLM KV-cache (joint self-attention)
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        # Extract action tokens and project back to action dim
        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        output = self.action_output_layer(hidden_states)
        if prefix_length > 0:
            output[:, :prefix_length] = 0.0

        return output

    def _normalize_prefix_length(self, prefix_length: Any, action_length: int) -> int:
        if isinstance(prefix_length, torch.Tensor):
            if prefix_length.numel() == 0:
                prefix_length = 0
            else:
                prefix_length = int(prefix_length.flatten()[0].item())
        elif prefix_length is None:
            prefix_length = 0
        else:
            prefix_length = int(prefix_length)

        return max(0, min(prefix_length, action_length))

    def _make_local_causal_mask(
        self,
        batch_size: int,
        state_length: int,
        action_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected_state_length = self.state_shape[-2]
        expected_action_length = self.action_shape[-2]
        if state_length == expected_state_length and action_length == expected_action_length:
            return self.saved_causal_mask.expand(batch_size, -1, -1)

        s_len = state_length + 1
        a_len = action_length
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        mask_aa = mask_aa * torch.triu(torch.ones(a_len, a_len, device=device), diagonal=-self.local_window)

        top = torch.cat([mask_ss, mask_sa], dim=1)
        bottom = torch.cat([mask_as, mask_aa], dim=1)
        return torch.cat([top, bottom], dim=0).unsqueeze(0).expand(batch_size, -1, -1)
    
    def _random_mask_prefix(self, causal_mask, prefix_length, state_length, keep_last_k=2):
        if prefix_length <= keep_last_k:
            return causal_mask

        action_start = 1 + state_length
        masked_prefix_end = action_start + prefix_length - keep_last_k
        suffix_start = action_start + prefix_length

        if suffix_start >= causal_mask.shape[-1]:
            return causal_mask

        causal_mask = causal_mask.clone()
        num_maskable = prefix_length - keep_last_k
        rand_mask = torch.rand(num_maskable, device=causal_mask.device) < self.prefix_mask_prob
        causal_mask[:, suffix_start:, action_start:masked_prefix_end] *= (~rand_mask).int()
        return causal_mask

    def _repeat_tensor(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if not self.training or self.training_repeat <= 1:
            return x
        return x.repeat_interleave(self.training_repeat, dim=dim)

    def _repeat_past_key_values(
        self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.training or self.training_repeat <= 1:
            return past_key_values
        return [(self._repeat_tensor(key), self._repeat_tensor(value)) for key, value in past_key_values]

    # --------------------------------------------------------
    # Forward / generate / loss
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run inference: generate action predictions from the given batch.

        Args:
            batch: Input dict containing VLM inputs, action, action_mask, and state.

        Returns:
            Predicted action tensor.
        """
        return self.forward(batch, return_loss=False)

    @auto_cast
    def forward(self, batch: Dict[str, Any], return_loss: bool = False) -> Dict[str, torch.Tensor]:
        """Full forward pass: VLM encoding → rectified flow training or inference.

        During training, samples a timestep, interpolates noise with the target
        action, and predicts the velocity.  During inference, runs Euler
        integration to denoise from pure noise.

        Args:
            batch: Input dict with VLM inputs, action, action_mask, state.
            return_loss: If True, return a loss dict; otherwise return predictions.

        Returns:
            - If ``return_loss``: ``{"loss": mse_loss}``
            - Otherwise: predicted action tensor.
        """
        prefix_length = batch.pop("prefix_length", 0)

        # VLM forward with KV-cache
        vlm_outputs = self.vlm(**batch, use_cache=True)

        # Extract action, action_mask, state from batch
        action, action_mask, state = self.get_action_input(batch)
        action_bs, action_length, _ = action.shape
        _, state_length, _ = state.shape
        q_len = action_length + state_length + 1  # +1 for sink token
        prefix_length = self._normalize_prefix_length(prefix_length, action_length)

        # KV-cache: DynamicCache → list of (key, value) tuples for DiT indexing
        past_key_values = list(vlm_outputs.past_key_values)

        if self.training:
            prefix_length = 0
            if self.async_train and random.random() < 0.5:
                prefix_length = random.randint(1, min(6, action_length))

        prefix = action[:, :prefix_length]

        # Position ids for DiT: continue from VLM's last position (MRoPE)
        position_ids = (
            torch.arange(0, q_len, device=action.device).view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        if action_length > prefix_length:
            position_ids[:, :, -(action_length - prefix_length) :] += 10

        # Attention mask: [VLM cache mask | local causal mask for DiT tokens]
        cache_mask = batch["attention_mask"][:, None, :].expand(-1, q_len, -1)
        causal_mask = self._make_local_causal_mask(action_bs, state_length, action_length, action.device)
        if self.training and prefix_length > 2:
            causal_mask = self._random_mask_prefix(causal_mask, prefix_length, state_length)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        # Project state to DiT hidden dim
        state_embed = self.state_projector(state)

        if self.training and self.training_repeat > 1:
            position_ids = self._repeat_tensor(position_ids, dim=1)
            action = self._repeat_tensor(action)
            prefix = self._repeat_tensor(prefix)
            action_mask = self._repeat_tensor(action_mask)
            state_embed = self._repeat_tensor(state_embed)
            attn_mask = self._repeat_tensor(attn_mask)
            past_key_values = self._repeat_past_key_values(past_key_values)

        position_embeds = self.rotary_emb(action, position_ids)

        # Rectified flow: training or inference
        noise = torch.randn_like(action)

        if self.training:
            # Sample timestep, interpolate, compute velocity target
            t = self._sample_timestep(action.shape[0], dtype=action.dtype, device=action.device)
            t = t.unsqueeze(dim=1).unsqueeze(dim=1)
            noisy_action = self._flow_interpolate(noise, action, t)
            target = self._flow_velocity_target(noise, action)
            pred = self.dit_forward(
                torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1),
                t,
                action_mask,
                state_embed,
                position_embeds,
                past_key_values,
                attn_mask,
                prefix_length=prefix_length,
            )[:, prefix_length:]
            target = target[:, prefix_length:]

            if prefix_length > 0:
                with torch.no_grad():
                    pred_prefix = self._flow_generate(
                        torch.cat([prefix, noise[:, prefix_length:]], dim=1),
                        dict(
                            action_mask=action_mask,
                            state_embed=state_embed,
                            position_embeds=position_embeds,
                            past_key_values=past_key_values,
                            attn_mask=attn_mask,
                            prefix_length=prefix_length,
                        ),
                    )
                weight = (pred_prefix[:, prefix_length:] - action[:, prefix_length:]).abs()
            else:
                weight = torch.ones_like(pred)

            action_mask = action_mask[:, prefix_length:]
        else:
            target = action
            dit_kwargs = dict(
                action_mask=action_mask,
                state_embed=state_embed,
                position_embeds=position_embeds,
                past_key_values=past_key_values,
                attn_mask=attn_mask,
                prefix_length=prefix_length,
            )
            pred = self._flow_generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), dit_kwargs)

        if return_loss:
            return self.compute_loss(pred, target, action_mask, weight if self.training else None)
        else:
            return pred

    def get_action_input(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract action, action_mask, and state from batch.

        Provides zero-filled defaults for inference when action/state are not
        present in the batch.

        Args:
            batch: Input dict that may contain "action", "action_mask", "state".

        Returns:
            Tuple of (action, action_mask, state) tensors.
        """
        device = batch["input_ids"].device
        if "action" in batch:
            action = batch.pop("action")
            action_mask = batch.pop("action_mask", None)
            if action_mask is None:
                action_mask = torch.ones_like(action, dtype=torch.int32)
            state = batch.pop("state")
        else:
            action = torch.zeros((1, *self.action_shape), device=device, dtype=torch.bfloat16)
            action_mask = torch.ones_like(action, dtype=torch.int32)
            state = torch.zeros((1, *self.state_shape), device=device, dtype=torch.bfloat16)

        return action, action_mask, state

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_mask: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """MSE + optional frequency-domain loss on action prediction, masked by ``action_mask``.

        Args:
            pred: Predicted velocity or action ``(B, L, D)``.
            target: Ground-truth target ``(B, L, D)``.
            action_mask: Binary mask ``(B, L, D)``.

        Returns:
            Loss dictionary containing the total loss and component terms.
        """
        pred = pred.float()
        target = target.float()
        action_mask = action_mask.bool()
        if weight is None:
            weight = torch.ones_like(pred)
        weight = weight.float()

        if not torch.any(action_mask):
            loss_mse = (pred.reshape(-1)[0] - target.reshape(-1)[0]) * 0.0
            loss_freq = loss_mse
            return {"loss": loss_mse, "loss_mse": loss_mse, "loss_freq": loss_freq}

        with torch.no_grad():
            masked_weight = weight[action_mask]
            if masked_weight.numel() > 0:
                weight = weight.clone()
                weight[action_mask] = weight[action_mask] / masked_weight.mean()
                weight = torch.clamp(weight, min=0.5, max=5.0)

        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[action_mask].mean()

        if self.freq_coefficient > 0.0:
            loss_freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
            weight_dct = weight.mean(dim=[1, 2])
            loss_freq = (loss_freq * weight_dct.unsqueeze(1).unsqueeze(2)).mean()

        else:
            loss_freq = loss_mse * 0.0

        loss = 0.5 * loss_mse + self.freq_coefficient * loss_freq

        return {"loss": loss, "loss_mse": loss_mse, "loss_freq": loss_freq}
