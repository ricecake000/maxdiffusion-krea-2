"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# JAX/Flax implementation of the Krea 2 (K2) single-stream MMDiT.
# Mirrors the diffusers reference implementation `Krea2Transformer2DModel`.

import math
from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ...common_types import BlockSizes
from ...configuration_utils import ConfigMixin, flax_register_to_config
from ...utils import BaseOutput
from ..modeling_flax_utils import FlaxModelMixin
from ..attention_flax import AttentionOp, apply_rope
from ..embeddings_flax import FluxPosEmbed


@flax.struct.dataclass
class Krea2Transformer2DModelOutput(BaseOutput):
  """
  Output of `Krea2Transformer2DModel`: the predicted flow-matching velocity for
  the packed image tokens, shape `(batch_size, image_seq_len, in_channels)`.
  """

  sample: jnp.ndarray


class Krea2RMSNorm(nn.Module):
  """RMSNorm with a zero-centered scale: the effective multiplier is `1 + weight`,
  matching the Krea 2 checkpoint format. Runs in float32 and casts back to the
  input dtype; the scale weight is kept in float32."""

  dim: int
  eps: float = 1e-5

  @nn.compact
  def __call__(self, x):
    weight = self.param("weight", nn.initializers.zeros, (self.dim,), jnp.float32)
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
    return (normed * (1.0 + weight)).astype(x.dtype)


def apply_explicit_lora(output, inputs, adapters=(), dtype=jnp.float32, precision=None):
  """Adds explicitly supplied LoRA updates to a linear projection.

  Each adapter is a ``(down_kernel, up_kernel, multiplier)`` tuple. Keeping
  these tensors as ordinary call arguments lets the staged transformer reuse a
  standalone block executable without relying on its shortened Flax module
  path to match a model-level LoRA interceptor.
  """
  for down_kernel, up_kernel, multiplier in adapters:
    lora_input = inputs.astype(dtype)
    down = jnp.matmul(lora_input, down_kernel.astype(dtype), precision=precision)
    update = jnp.matmul(down, up_kernel.astype(dtype), precision=precision)
    output = output + update * jnp.asarray(multiplier, dtype=dtype)
  return output


class Krea2SwiGLU(nn.Module):
  """SwiGLU feed-forward with separate gate/up/down projections (no bias)."""

  dim: int
  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    self.gate_proj = nn.Dense(
        self.hidden_dim,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.up_proj = nn.Dense(
        self.hidden_dim,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.down_proj = nn.Dense(
        self.dim,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, x, lora_params=None):
    lora_params = lora_params or {}
    gate = apply_explicit_lora(
        self.gate_proj(x), x, lora_params.get("gate_proj", ()), self.dtype, self.precision
    )
    up = apply_explicit_lora(self.up_proj(x), x, lora_params.get("up_proj", ()), self.dtype, self.precision)
    down_input = nn.silu(gate) * up
    return apply_explicit_lora(
        self.down_proj(down_input),
        down_input,
        lora_params.get("down_proj", ()),
        self.dtype,
        self.precision,
    )


class Krea2Attention(nn.Module):
  """Self-attention with grouped-query projections, per-head zero-centered q/k
  RMSNorm, optional rotary embeddings, and a sigmoid output gate applied to the
  attention output before the output projection."""

  dim: int
  num_heads: int
  num_kv_heads: int
  head_dim: int = 128
  eps: float = 1e-5
  use_rope: bool = True
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 512
  flash_block_sizes: Optional[BlockSizes] = None
  mesh: Optional[jax.sharding.Mesh] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  mask_padding_tokens: bool = True

  def setup(self):
    dense_kwargs = dict(
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.to_q = nn.Dense(
        self.num_heads * self.head_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        **dense_kwargs,
    )
    self.to_k = nn.Dense(
        self.num_kv_heads * self.head_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        **dense_kwargs,
    )
    self.to_v = nn.Dense(
        self.num_kv_heads * self.head_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        **dense_kwargs,
    )
    self.to_gate = nn.Dense(
        self.num_heads * self.head_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        **dense_kwargs,
    )
    self.to_out = nn.Dense(
        self.dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("heads", "embed")),
        **dense_kwargs,
    )
    self.norm_q = Krea2RMSNorm(self.head_dim, eps=self.eps)
    self.norm_k = Krea2RMSNorm(self.head_dim, eps=self.eps)

    if self.attention_kernel != "dot_product":
      self.attention_op = AttentionOp(
          mesh=self.mesh,
          attention_kernel=self.attention_kernel,
          scale=1.0 / math.sqrt(self.head_dim),
          heads=self.num_heads,
          dim_head=self.head_dim,
          flash_min_seq_length=self.flash_min_seq_length,
          flash_block_sizes=self.flash_block_sizes,
          mask_padding_tokens=self.mask_padding_tokens,
          dtype=self.dtype,
      )

  def _masked_dot_product_attention(self, query, key, value, attention_mask):
    # query/key/value: (B, H, L, D). Softmax in float32 for stability.
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(self.head_dim)
    if attention_mask is not None:
      # attention_mask: (B, L_kv) key-padding mask, True/1 = valid.
      bias = jnp.where(attention_mask[:, None, None, :].astype(jnp.bool_), 0.0, -1e9).astype(jnp.float32)
      scores = scores + bias
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, value.astype(jnp.float32))
    return out.astype(self.dtype)

  def __call__(self, hidden_states, attention_mask=None, image_rotary_emb=None, lora_params=None):
    batch_size, seq_len, _ = hidden_states.shape
    lora_params = lora_params or {}

    query = apply_explicit_lora(
        self.to_q(hidden_states), hidden_states, lora_params.get("to_q", ()), self.dtype, self.precision
    ).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    key = apply_explicit_lora(
        self.to_k(hidden_states), hidden_states, lora_params.get("to_k", ()), self.dtype, self.precision
    ).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    value = apply_explicit_lora(
        self.to_v(hidden_states), hidden_states, lora_params.get("to_v", ()), self.dtype, self.precision
    ).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    gate = apply_explicit_lora(
        self.to_gate(hidden_states), hidden_states, lora_params.get("to_gate", ()), self.dtype, self.precision
    )

    query = self.norm_q(query)
    key = self.norm_k(key)

    # (B, H, L, D)
    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))

    if self.use_rope and image_rotary_emb is not None:
      query, key = apply_rope(query, key, image_rotary_emb)

    if self.num_kv_heads != self.num_heads:
      repeats = self.num_heads // self.num_kv_heads
      key = jnp.repeat(key, repeats, axis=1)
      value = jnp.repeat(value, repeats, axis=1)

    if self.attention_kernel == "dot_product":
      attn_output = self._masked_dot_product_attention(query, key, value, attention_mask)
      attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
    else:
      # Flatten to (B, L, H*D) as expected by the shared attention op.
      q_flat = jnp.transpose(query, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
      k_flat = jnp.transpose(key, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
      v_flat = jnp.transpose(value, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
      mask = None
      if attention_mask is not None:
        mask = attention_mask.astype(jnp.int32)
      attn_output = self.attention_op.apply_attention(q_flat, k_flat, v_flat, attention_mask=mask)

    attn_output = attn_output * jax.nn.sigmoid(gate)
    return apply_explicit_lora(
        self.to_out(attn_output), attn_output, lora_params.get("to_out", ()), self.dtype, self.precision
    )


class Krea2TextFusionBlock(nn.Module):
  """Pre-norm transformer block (no rotary embeddings, no time modulation) used by
  the text fusion stage."""

  dim: int
  num_heads: int
  num_kv_heads: int
  intermediate_size: int
  eps: float = 1e-5
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    self.norm1 = Krea2RMSNorm(self.dim, eps=self.eps)
    self.norm2 = Krea2RMSNorm(self.dim, eps=self.eps)
    self.attn = Krea2Attention(
        dim=self.dim,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.dim // self.num_heads,
        eps=self.eps,
        use_rope=False,
        attention_kernel="dot_product",
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.ff = Krea2SwiGLU(
        dim=self.dim,
        hidden_dim=self.intermediate_size,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, hidden_states, attention_mask=None):
    hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
    hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
    return hidden_states


class Krea2TextFusion(nn.Module):
  """Fuses the stack of tapped text-encoder hidden states into a single sequence.

  Two `layerwise_blocks` attend across the `num_text_layers` axis independently
  for every token, a linear `projector` collapses that axis, and two
  `refiner_blocks` attend across the token sequence.
  """

  num_text_layers: int
  dim: int
  num_heads: int
  num_kv_heads: int
  intermediate_size: int
  num_layerwise_blocks: int = 2
  num_refiner_blocks: int = 2
  eps: float = 1e-5
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    block_kwargs = dict(
        dim=self.dim,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        intermediate_size=self.intermediate_size,
        eps=self.eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.layerwise_blocks = [Krea2TextFusionBlock(**block_kwargs) for _ in range(self.num_layerwise_blocks)]
    self.projector = nn.Dense(
        1,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.refiner_blocks = [Krea2TextFusionBlock(**block_kwargs) for _ in range(self.num_refiner_blocks)]

  def __call__(self, encoder_hidden_states, attention_mask=None):
    batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

    hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
    for block in self.layerwise_blocks:
      hidden_states = block(hidden_states)

    hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim)
    # Collapse the tapped-layer axis with a linear projector: (B, S, D, L) -> (B, S, D).
    hidden_states = jnp.transpose(hidden_states, (0, 1, 3, 2))
    hidden_states = self.projector(hidden_states)[..., 0]

    for block in self.refiner_blocks:
      hidden_states = block(hidden_states, attention_mask=attention_mask)

    return hidden_states


class Krea2TextProjection(nn.Module):
  """Projects the fused text features into the transformer width."""

  text_dim: int
  hidden_size: int
  eps: float = 1e-5
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    self.norm = Krea2RMSNorm(self.text_dim, eps=self.eps)
    self.linear_1 = nn.Dense(
        self.hidden_size,
        use_bias=True,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.linear_2 = nn.Dense(
        self.hidden_size,
        use_bias=True,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, hidden_states):
    hidden_states = self.linear_1(self.norm(hidden_states))
    return self.linear_2(jax.nn.gelu(hidden_states, approximate=True))


class Krea2TimestepEmbedding(nn.Module):
  """Sinusoidal flow-time embedding (cos-first, input scaled by 1000) followed by
  a two-layer MLP. Keeps the sequence dimension at size 1 so the per-block
  modulations broadcast over tokens."""

  embed_dim: int
  hidden_size: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    self.linear_1 = nn.Dense(
        self.hidden_size,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.linear_2 = nn.Dense(
        self.hidden_size,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, timestep):
    half = self.embed_dim // 2
    freqs = jnp.exp(-math.log(1e4) * jnp.arange(half, dtype=jnp.float32) / half)
    args = (timestep.astype(jnp.float32) * 1e3)[:, None, None] * freqs
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(self.dtype)
    return self.linear_2(jax.nn.gelu(self.linear_1(emb), approximate=True))


class Krea2TransformerBlock(nn.Module):
  """Single-stream MMDiT block: adaptive RMSNorm modulation driven by one shared
  timestep modulation vector plus a per-block learned table."""

  hidden_size: int
  intermediate_size: int
  num_heads: int
  num_kv_heads: int
  norm_eps: float = 1e-5
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 512
  flash_block_sizes: Optional[BlockSizes] = None
  mesh: Optional[jax.sharding.Mesh] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  mask_padding_tokens: bool = True

  def setup(self):
    self.scale_shift_table = self.param("scale_shift_table", nn.initializers.zeros, (6, self.hidden_size), jnp.float32)
    self.norm1 = Krea2RMSNorm(self.hidden_size, eps=self.norm_eps)
    self.norm2 = Krea2RMSNorm(self.hidden_size, eps=self.norm_eps)
    self.attn = Krea2Attention(
        dim=self.hidden_size,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.hidden_size // self.num_heads,
        eps=self.norm_eps,
        use_rope=True,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mask_padding_tokens=self.mask_padding_tokens,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.ff = Krea2SwiGLU(
        dim=self.hidden_size,
        hidden_dim=self.intermediate_size,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, hidden_states, temb_mod, image_rotary_emb=None, attention_mask=None, lora_params=None):
    # temb_mod: (B, 1, 6 * hidden_size), shared across all blocks; each block only
    # learns an additive table. Modulation arithmetic runs in float32.
    batch_size = hidden_states.shape[0]
    modulation = temb_mod.astype(jnp.float32).reshape(batch_size, 1, 6, self.hidden_size)
    modulation = modulation + self.scale_shift_table[None, None]
    prescale, preshift, pregate, postscale, postshift, postgate = [
        modulation[:, :, i, :].astype(hidden_states.dtype) for i in range(6)
    ]

    attn_input = (1.0 + prescale) * self.norm1(hidden_states) + preshift
    lora_params = lora_params or {}
    attn_output = self.attn(
        attn_input,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
        lora_params=lora_params.get("attn"),
    )
    hidden_states = hidden_states + pregate * attn_output

    ff_input = (1.0 + postscale) * self.norm2(hidden_states) + postshift
    ff_output = self.ff(ff_input, lora_params=lora_params.get("ff"))
    hidden_states = hidden_states + postgate * ff_output
    return hidden_states


class Krea2FinalLayer(nn.Module):
  """Final adaptive RMSNorm and output projection. The modulation uses `temb`
  (before the shared modulation projection) plus a learned (2, hidden) table."""

  hidden_size: int
  out_channels: int
  eps: float = 1e-5
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    self.scale_shift_table = self.param("scale_shift_table", nn.initializers.zeros, (2, self.hidden_size), jnp.float32)
    self.norm = Krea2RMSNorm(self.hidden_size, eps=self.eps)
    self.linear = nn.Dense(
        self.out_channels,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(self, hidden_states, temb):
    # temb: (B, 1, hidden). Broadcast against the (2, hidden) table -> (B, 2, hidden).
    modulation = temb.astype(jnp.float32) + self.scale_shift_table[None]
    scale = modulation[:, 0:1, :].astype(hidden_states.dtype)
    shift = modulation[:, 1:2, :].astype(hidden_states.dtype)
    hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
    return self.linear(hidden_states)


@flax_register_to_config
class Krea2Transformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
  """
  The Krea 2 single-stream MMDiT flow-matching backbone (JAX/Flax).

  Text conditioning enters as a stack of hidden states tapped from several layers
  of the Qwen3-VL text encoder. A small text-fusion transformer collapses the
  layer axis and refines the token sequence; the result is concatenated with the
  patchified image latents into a single `[text, image]` sequence.
  """

  in_channels: int = 64
  num_layers: int = 28
  attention_head_dim: int = 128
  num_attention_heads: int = 48
  num_key_value_heads: int = 12
  intermediate_size: int = 16384
  timestep_embed_dim: int = 256
  text_hidden_dim: int = 2560
  num_text_layers: int = 12
  text_num_attention_heads: int = 20
  text_num_key_value_heads: int = 20
  text_intermediate_size: int = 6912
  num_layerwise_text_blocks: int = 2
  num_refiner_text_blocks: int = 2
  axes_dims_rope: Tuple[int, ...] = (32, 48, 48)
  rope_theta: float = 1000.0
  norm_eps: float = 1e-5
  attention_kernel: str = "dot_product"
  flash_min_seq_length: int = 512
  flash_block_sizes: Optional[BlockSizes] = None
  mesh: Optional[jax.sharding.Mesh] = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  mask_padding_tokens: bool = True

  def setup(self):
    if sum(self.axes_dims_rope) != self.attention_head_dim:
      raise ValueError(
          f"sum(axes_dims_rope)={sum(self.axes_dims_rope)} must equal attention_head_dim={self.attention_head_dim}"
      )
    hidden_size = self.attention_head_dim * self.num_attention_heads

    self.img_in = nn.Dense(
        hidden_size,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.time_embed = Krea2TimestepEmbedding(
        embed_dim=self.timestep_embed_dim,
        hidden_size=hidden_size,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.time_mod_proj = nn.Dense(
        6 * hidden_size,
        use_bias=True,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.text_fusion = Krea2TextFusion(
        num_text_layers=self.num_text_layers,
        dim=self.text_hidden_dim,
        num_heads=self.text_num_attention_heads,
        num_kv_heads=self.text_num_key_value_heads,
        intermediate_size=self.text_intermediate_size,
        num_layerwise_blocks=self.num_layerwise_text_blocks,
        num_refiner_blocks=self.num_refiner_text_blocks,
        eps=self.norm_eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.txt_in = Krea2TextProjection(
        text_dim=self.text_hidden_dim,
        hidden_size=hidden_size,
        eps=self.norm_eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )
    self.pos_embed = FluxPosEmbed(theta=self.rope_theta, axes_dim=self.axes_dims_rope, return_tuple=True)

    self.blocks = [
        Krea2TransformerBlock(
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            norm_eps=self.norm_eps,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mask_padding_tokens=self.mask_padding_tokens,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
        )
        for _ in range(self.num_layers)
    ]

    self.final_layer = Krea2FinalLayer(
        hidden_size=hidden_size,
        out_channels=self.in_channels,
        eps=self.norm_eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def __call__(
      self,
      hidden_states,
      encoder_hidden_states,
      timestep,
      img_ids,
      txt_ids,
      encoder_attention_mask=None,
      return_dict: bool = True,
  ):
    """
    Args:
      hidden_states: `(batch, image_seq_len, in_channels)` packed noisy latents.
      encoder_hidden_states: `(batch, text_seq_len, num_text_layers, text_hidden_dim)`
        stack of tapped text-encoder hidden states.
      timestep: `(batch,)` flow-matching time in [0, 1] (1 = pure noise).
      img_ids: `(image_seq_len, 3)` or `(batch, image_seq_len, 3)` rotary coords `(0, h, w)`.
      txt_ids: `(text_seq_len, 3)` or `(batch, text_seq_len, 3)` all-zero rotary coords.
      encoder_attention_mask: optional `(batch, text_seq_len)` boolean mask, True = valid.
    """
    hidden_states, temb, temb_mod, concat_rotary_emb, attention_mask = self.prepare_inputs(
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        encoder_attention_mask,
    )
    text_seq_len = encoder_hidden_states.shape[1]

    for block in self.blocks:
      hidden_states = block(
          hidden_states,
          temb_mod=temb_mod,
          image_rotary_emb=concat_rotary_emb,
          attention_mask=attention_mask,
      )

    output = self.finalize_output(hidden_states, temb, text_seq_len)

    if not return_dict:
      return (output,)
    return Krea2Transformer2DModelOutput(sample=output)

  def prepare_inputs(
      self,
      hidden_states,
      encoder_hidden_states,
      timestep,
      img_ids,
      txt_ids,
      encoder_attention_mask=None,
  ):
    """Projects and combines text/image inputs before the repeated DiT blocks."""
    batch_size, image_seq_len, _ = hidden_states.shape

    temb = self.time_embed(timestep)
    temb_mod = self.time_mod_proj(jax.nn.gelu(temb, approximate=True))

    text_attention_mask = None
    attention_mask = None
    if encoder_attention_mask is not None:
      text_attention_mask = encoder_attention_mask
      image_mask = jnp.ones((batch_size, image_seq_len), dtype=encoder_attention_mask.dtype)
      attention_mask = jnp.concatenate([encoder_attention_mask, image_mask], axis=1)

    encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    hidden_states = self.img_in(hidden_states)
    hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
      img_ids = img_ids[0]
    text_rotary_emb = self.pos_embed(txt_ids)
    image_rotary_emb = self.pos_embed(img_ids)
    concat_rotary_emb = (
        jnp.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        jnp.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )
    return hidden_states, temb, temb_mod, concat_rotary_emb, attention_mask

  def finalize_output(self, hidden_states, temb, text_seq_len: int):
    """Removes text tokens and applies the final adaptive projection."""
    hidden_states = hidden_states[:, text_seq_len:]
    return self.final_layer(hidden_states, temb)
