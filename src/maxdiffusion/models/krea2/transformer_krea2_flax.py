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
# With attention 'flash_custom' and rope_layout 'rotate_half', Krea2Attention
# hands the raw q/k projections to the attention wrapper, which runs the q/k
# norm, RoPE, the q softmax scale and the padding in one Pallas pass per tensor
# (kernels/krea2_qk_prep.py; float32 math with a single final cast).

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
from ..attention_flax import AttentionOp, apply_rope, krea2_flash_custom_io_layout
from .transformer_quant import (
    KREA2_TRANSPOSED_KERNEL_TARGETS,
    Krea2QuantDense,
    normalize_quant_targets,
    quantize_activation,
)

ROPE_LAYOUTS = ("interleaved", "rotate_half")
# Bump when the traced attention glue changes; it is part of the AOT cache key
# (generate_krea2.attention_glue_aot_meta), so this invalidates cached executables.
# 1: flash_custom + rotate_half runs q/k norm, RoPE, the q scale and the padding
#    in one Pallas pass per tensor (kernels/krea2_qk_prep.py); W8A8 projections
#    share one materialized attention-input quantization (optimization barrier).
# 2: the hybrid kernel takes v as the flat unpadded (B, L, Hkv*D) projection and
#    writes its output as (B, L, Hq*D) (direct I/O layout): no v pad, no output copy;
#    the W8A8 to_v then rescales flat, and the to_q/to_k int8 kernels are stored transposed.
KREA2_ATTENTION_GLUE_REVISION = 2


def _validate_rope_layout(layout):
  if layout not in ROPE_LAYOUTS:
    raise ValueError(f"rope_layout must be one of {ROPE_LAYOUTS}, got {layout!r}.")


def krea2_rotary_tables(ids, axes_dim, theta, layout="interleaved"):
  """Rotary cos/sin tables for per-token position ids.

  Args:
    ids: `(L, n_axes)` position ids (cast to float32).
    axes_dim: per-axis rotary widths, summing to head_dim.
    theta: rotary base.
    layout: "interleaved" returns `(L, head_dim)` tables with every frequency
      repeated twice (pairs `(2i, 2i+1)` share an angle), numerically identical
      to `FluxPosEmbed(return_tuple=True)`. "rotate_half" returns `(L, head_dim // 2)`
      tables with one entry per frequency, for `apply_rope_rotate_half`.

  Returns:
    `(cos, sin)` float32 tables.
  """
  _validate_rope_layout(layout)
  pos = ids.astype(jnp.float32)
  cos_out = []
  sin_out = []
  for i, dim in enumerate(axes_dim):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.outer(pos[..., i], freqs)
    if layout == "interleaved":
      freqs = jnp.repeat(freqs, 2, axis=-1)
    cos_out.append(jnp.cos(freqs))
    sin_out.append(jnp.sin(freqs))
  return jnp.concatenate(cos_out, axis=-1), jnp.concatenate(sin_out, axis=-1)


def apply_rope_rotate_half(x, cos, sin):
  """Rotate-half RoPE on `x` of shape `(B, H, L, D)` with `(L, D/2)` cos/sin tables.

  `out = concat([x1*cos - x2*sin, x2*cos + x1*sin])` for `x1, x2 = x[..., :D/2], x[..., D/2:]`,
  computed in float32 and cast back to `x.dtype`. With q/k projection columns
  permuted by `permute_rope_weights_to_rotate_half`, this equals interleaved
  RoPE on the original layout (up to the same permutation).
  """
  half = x.shape[-1] // 2
  x_f32 = x.astype(jnp.float32)
  x1 = x_f32[..., :half]
  x2 = x_f32[..., half:]
  cos = cos.astype(jnp.float32)[None, None]
  sin = sin.astype(jnp.float32)[None, None]
  out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return out.astype(x.dtype)


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

  def setup(self):
    # setup (not compact): Krea2Attention hands `weight` to the fused q/k
    # post-processing kernel without calling this module.
    self.weight = self.param("weight", nn.initializers.zeros, (self.dim,), jnp.float32)

  def __call__(self, x):
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
    return (normed * (1.0 + self.weight)).astype(x.dtype)


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


def _projection(
    features, kernel_axes, quantized, dtype, weights_dtype, precision, unflatten=None, transposed_kernel=False
):
  """A bias-free block projection: `Krea2QuantDense` (W8A8) when `quantized`, else `nn.Dense`.

  `unflatten` is the caller's output layout and `transposed_kernel` the int8
  kernel's stored layout (see `Krea2QuantDense`); `nn.Dense` ignores both.
  """
  if quantized:
    return Krea2QuantDense(
        features,
        kernel_axes=kernel_axes,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        unflatten=unflatten,
        transposed_kernel=transposed_kernel,
    )
  return nn.Dense(
      features,
      use_bias=False,
      kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), kernel_axes),
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
  )


def _apply_projection(proj, x, quantized_x):
  """Calls a W8A8 projection with the shared `(x_q, x_scale)` pair, a float one with `x` only."""
  if isinstance(proj, Krea2QuantDense):
    return proj(x, quantized_x)
  return proj(x)


class Krea2SwiGLU(nn.Module):
  """SwiGLU feed-forward with separate gate/up/down projections (no bias).

  Projections named in `quant_targets` run as W8A8 (`Krea2QuantDense`);
  `gate_proj` and `up_proj` share one quantization of `x`.
  """

  dim: int
  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  quant_targets: Tuple[str, ...] = ()

  def setup(self):
    normalize_quant_targets(self.quant_targets)  # rejects unknown names
    proj_kwargs = dict(dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision)
    self.gate_proj = _projection(self.hidden_dim, ("embed", "mlp"), "gate_proj" in self.quant_targets, **proj_kwargs)
    self.up_proj = _projection(self.hidden_dim, ("embed", "mlp"), "up_proj" in self.quant_targets, **proj_kwargs)
    self.down_proj = _projection(self.dim, ("mlp", "embed"), "down_proj" in self.quant_targets, **proj_kwargs)

  def __call__(self, x, lora_params=None):
    lora_params = lora_params or {}
    quantized_x = None
    if "gate_proj" in self.quant_targets or "up_proj" in self.quant_targets:
      quantized_x = quantize_activation(x, self.dtype)
    gate = apply_explicit_lora(
        _apply_projection(self.gate_proj, x, quantized_x), x, lora_params.get("gate_proj", ()), self.dtype, self.precision
    )
    up = apply_explicit_lora(
        _apply_projection(self.up_proj, x, quantized_x), x, lora_params.get("up_proj", ()), self.dtype, self.precision
    )
    down_input = nn.silu(gate) * up
    quantized_down = quantize_activation(down_input, self.dtype) if "down_proj" in self.quant_targets else None
    return apply_explicit_lora(
        _apply_projection(self.down_proj, down_input, quantized_down),
        down_input,
        lora_params.get("down_proj", ()),
        self.dtype,
        self.precision,
    )


class Krea2Attention(nn.Module):
  """Self-attention with grouped-query projections, per-head zero-centered q/k
  RMSNorm, optional rotary embeddings, and a sigmoid output gate applied to the
  attention output before the output projection.

  Projections named in `quant_targets` run as W8A8 (`Krea2QuantDense`);
  `to_q/to_k/to_v/to_gate` share one quantization of `hidden_states`.
  """

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
  rope_layout: str = "interleaved"
  quant_targets: Tuple[str, ...] = ()

  def setup(self):
    _validate_rope_layout(self.rope_layout)
    normalize_quant_targets(self.quant_targets)  # rejects unknown names
    proj_kwargs = dict(dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision)
    q_features = self.num_heads * self.head_dim
    kv_features = self.num_kv_heads * self.head_dim
    # q/k/v are reshaped to heads right after the projection; W8A8 rescales in that layout.
    q_layout = (self.num_heads, self.head_dim)
    kv_layout = (self.num_kv_heads, self.head_dim)
    # Except v for a flash_custom kernel with the "direct" I/O layout (the hybrid kernel), which reads the
    # flat (B, L, Hkv*D) projection: a head layout rescale there makes XLA emit the to_v matmul head-major
    # (with an s8 weight relayout) and copy it back to flat. Layout only: the values are the same.
    v_layout = kv_layout
    if self.attention_kernel == "flash_custom":
      if krea2_flash_custom_io_layout(self.flash_block_sizes, self.mesh) == "direct":
        v_layout = None

    def head_projection(name, features, layout):
      # The int8 kernels of KREA2_TRANSPOSED_KERNEL_TARGETS are stored (features, in): see transformer_quant.
      return _projection(
          features,
          ("embed", "heads"),
          name in self.quant_targets,
          unflatten=layout,
          transposed_kernel=name in KREA2_TRANSPOSED_KERNEL_TARGETS,
          **proj_kwargs,
      )

    self.to_q = head_projection("to_q", q_features, q_layout)
    self.to_k = head_projection("to_k", kv_features, kv_layout)
    self.to_v = head_projection("to_v", kv_features, v_layout)
    self.to_gate = _projection(q_features, ("embed", "heads"), "to_gate" in self.quant_targets, **proj_kwargs)
    self.to_out = _projection(self.dim, ("heads", "embed"), "to_out" in self.quant_targets, **proj_kwargs)
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

  def _gated_output_projection(self, attn_output, gate, lora_params):
    """`to_out(attn_output * sigmoid(gate))` plus its explicit LoRA updates."""
    attn_output = attn_output * jax.nn.sigmoid(gate)
    quantized = quantize_activation(attn_output, self.dtype) if "to_out" in self.quant_targets else None
    return apply_explicit_lora(
        _apply_projection(self.to_out, attn_output, quantized),
        attn_output,
        lora_params.get("to_out", ()),
        self.dtype,
        self.precision,
    )

  def __call__(self, hidden_states, attention_mask=None, image_rotary_emb=None, lora_params=None):
    batch_size, seq_len, _ = hidden_states.shape
    lora_params = lora_params or {}

    quantized_hidden = None
    if any(name in self.quant_targets for name in ("to_q", "to_k", "to_v", "to_gate")):
      # The barrier makes the projections share one materialized (x_q, scale):
      # without it XLA fuses the quantization into a separate producer per
      # operand form (to_q/to_gate vs to_k/to_v), quantizing the input twice.
      quantized_hidden = jax.lax.optimization_barrier(quantize_activation(hidden_states, self.dtype))

    def project(name):
      out = _apply_projection(getattr(self, name), hidden_states, quantized_hidden)
      return apply_explicit_lora(out, hidden_states, lora_params.get(name, ()), self.dtype, self.precision)

    query = project("to_q").reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    key = project("to_k").reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    value = project("to_v").reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    gate = project("to_gate")

    mask = None
    if attention_mask is not None:
      mask = attention_mask.astype(jnp.int32)

    use_rope = self.use_rope and image_rotary_emb is not None
    use_custom_kernel = self.attention_kernel == "flash_custom"
    use_custom_path = use_custom_kernel and seq_len >= self.flash_min_seq_length
    if use_custom_path and use_rope and self.rope_layout == "rotate_half":
      # q/k go to the flash_custom wrapper un-normed: it runs norm_q/norm_k,
      # RoPE, the softmax scale (q) and the padding to its block sizes in one
      # Pallas pass per tensor (kernels/krea2_qk_prep.py), which reads the
      # projection once and writes the kernel operand once; done here in jnp,
      # XLA splits it into three HBM-bound fusions per tensor on TPU. The
      # (B, H, L, D) transposes are free: XLA writes the projections head-major.
      cos, sin = image_rotary_emb
      qk_prep = {
          "q_norm_weight": self.norm_q.weight,
          "k_norm_weight": self.norm_k.weight,
          "eps": self.norm_q.eps,
          "cos": cos,
          "sin": sin,
      }
      attn_output = self.attention_op.apply_attention(
          jnp.transpose(query, (0, 2, 1, 3)),
          jnp.transpose(key, (0, 2, 1, 3)),
          jnp.transpose(value, (0, 2, 1, 3)),
          attention_mask=mask,
          extra_context={"krea2_qk_prep": qk_prep},
      )
      return self._gated_output_projection(attn_output, gate, lora_params)

    query = self.norm_q(query)
    key = self.norm_k(key)

    # (B, H, L, D)
    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))

    if use_rope:
      if self.rope_layout == "rotate_half":
        cos, sin = image_rotary_emb
        query = apply_rope_rotate_half(query, cos, sin)
        key = apply_rope_rotate_half(key, cos, sin)
      else:
        query, key = apply_rope(query, key, image_rotary_emb)

    if use_custom_path:
      # The Krea 2 kernel takes 4-D (B, H, L, D) queries and un-expanded GQA
      # (B, H_kv, L, D) keys/values, unscaled; the mask must be a prefix
      # key-validity mask ([image | text] with tail-padded, compacted text).
      attn_output = self.attention_op.apply_attention(query, key, value, attention_mask=mask)
      return self._gated_output_projection(attn_output, gate, lora_params)

    if self.num_kv_heads != self.num_heads:
      repeats = self.num_heads // self.num_kv_heads
      key = jnp.repeat(key, repeats, axis=1)
      value = jnp.repeat(value, repeats, axis=1)

    # Below flash_min_seq_length flash_custom uses this module's masked
    # attention: the shared dispatcher's dot_product fallback ignores the mask.
    if self.attention_kernel == "dot_product" or use_custom_kernel:
      attn_output = self._masked_dot_product_attention(query, key, value, attention_mask)
      attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
    else:
      # Flatten to (B, L, H*D) as expected by the shared attention op.
      q_flat = jnp.transpose(query, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
      k_flat = jnp.transpose(key, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
      v_flat = jnp.transpose(value, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
      attn_output = self.attention_op.apply_attention(q_flat, k_flat, v_flat, attention_mask=mask)

    return self._gated_output_projection(attn_output, gate, lora_params)


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
  rope_layout: str = "interleaved"
  quant_targets: Tuple[str, ...] = ()

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
        rope_layout=self.rope_layout,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        quant_targets=self.quant_targets,
    )
    self.ff = Krea2SwiGLU(
        dim=self.hidden_size,
        hidden_dim=self.intermediate_size,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        quant_targets=self.quant_targets,
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
  layer axis and refines the token sequence; the result is appended to the
  patchified image latents into a single `[image | text]` sequence. Attention is
  permutation-equivariant given per-token rotary ids (text ids are all zero), so
  this order matches the reference `[text, image]` order exactly while keeping
  text padding at the tail of the key sequence (a prefix validity mask).

  `rope_layout="rotate_half"` expects q/k projection weights permuted by
  `util.permute_rope_weights_to_rotate_half`; outputs are then identical to the
  default interleaved layout with the original weights.

  `quant_targets` (see `transformer_quant.KREA2_QUANT_TARGETS`) selects the DiT
  block projections that run as int8 W8A8 matmuls; the params must then come
  from `transformer_quant.quantize_transformer_params`. Text fusion and the
  input/output projections always stay float.
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
  rope_layout: str = "interleaved"
  quant_targets: Tuple[str, ...] = ()

  def setup(self):
    _validate_rope_layout(self.rope_layout)
    # Raises on unknown names; order and duplicates are normalized away.
    quant_targets = normalize_quant_targets(self.quant_targets)
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
            rope_layout=self.rope_layout,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
            quant_targets=quant_targets,
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
    text_hidden = self.encode_text_context(encoder_hidden_states, encoder_attention_mask)
    output = self.forward_with_text_context(
        hidden_states, text_hidden, timestep, img_ids, txt_ids, encoder_attention_mask
    )

    if not return_dict:
      return (output,)
    return Krea2Transformer2DModelOutput(sample=output)

  def encode_text_context(self, encoder_hidden_states, encoder_attention_mask=None):
    """Prompt-only text path: text fusion + projection into the transformer width.

    Depends only on the prompt, so the pipeline runs it once per prompt rather
    than once per denoise step. Returns `(batch, text_seq_len, hidden_size)`.
    """
    text_hidden = self.text_fusion(encoder_hidden_states, attention_mask=encoder_attention_mask)
    return self.txt_in(text_hidden)

  def forward_with_text_context(
      self,
      hidden_states,
      text_hidden,
      timestep,
      img_ids,
      txt_ids,
      encoder_attention_mask=None,
  ):
    """One denoise-step forward from a precomputed `encode_text_context` output;
    returns the `(batch, image_seq_len, in_channels)` velocity."""
    image_seq_len = hidden_states.shape[1]
    hidden_states, temb, temb_mod, concat_rotary_emb, attention_mask = self.prepare_inputs(
        hidden_states,
        text_hidden,
        timestep,
        img_ids,
        txt_ids,
        encoder_attention_mask,
    )

    for block in self.blocks:
      hidden_states = block(
          hidden_states,
          temb_mod=temb_mod,
          image_rotary_emb=concat_rotary_emb,
          attention_mask=attention_mask,
      )

    return self.finalize_output(hidden_states, temb, image_seq_len)

  def prepare_inputs(
      self,
      hidden_states,
      text_hidden,
      timestep,
      img_ids,
      txt_ids,
      encoder_attention_mask=None,
  ):
    """Builds the `[image | text]` sequence, time modulation, rotary tables and
    key-validity mask before the repeated DiT blocks.

    `text_hidden` is the `encode_text_context` output. The mask is
    `[ones(image), encoder_attention_mask]`, a prefix mask whenever the text
    mask is (tail padding).
    """
    batch_size, image_seq_len, _ = hidden_states.shape

    temb = self.time_embed(timestep)
    temb_mod = self.time_mod_proj(jax.nn.gelu(temb, approximate=True))

    attention_mask = None
    if encoder_attention_mask is not None:
      image_mask = jnp.ones((batch_size, image_seq_len), dtype=encoder_attention_mask.dtype)
      attention_mask = jnp.concatenate([image_mask, encoder_attention_mask], axis=1)

    hidden_states = self.img_in(hidden_states)
    hidden_states = jnp.concatenate([hidden_states, text_hidden.astype(hidden_states.dtype)], axis=1)

    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
      img_ids = img_ids[0]
    ids = jnp.concatenate([img_ids, txt_ids], axis=0)
    concat_rotary_emb = krea2_rotary_tables(ids, self.axes_dims_rope, self.rope_theta, self.rope_layout)
    return hidden_states, temb, temb_mod, concat_rotary_emb, attention_mask

  def finalize_output(self, hidden_states, temb, image_seq_len: int):
    """Keeps the leading image tokens and applies the final adaptive projection."""
    hidden_states = hidden_states[:, :image_seq_len]
    return self.final_layer(hidden_states, temb)
