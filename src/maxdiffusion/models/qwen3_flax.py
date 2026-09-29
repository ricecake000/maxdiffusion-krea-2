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

import math
from typing import Any, List, Optional, Tuple
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import max_logging
from maxdiffusion.safetensors_utils import SafetensorsShardReader

# -----------------------------------------------------------------------------
# Qwen3 Configuration
# -----------------------------------------------------------------------------


class FlaxQwen3Config:

  def __init__(
      self,
      vocab_size: int = 151936,
      hidden_size: int = 2560,
      intermediate_size: int = 9728,
      num_hidden_layers: int = 36,
      num_attention_heads: int = 32,
      num_key_value_heads: int = 8,
      head_dim: int = 128,
      rms_norm_eps: float = 1e-6,
      rope_theta: float = 1000000.0,
      max_position_embeddings: int = 40960,
      dtype=jnp.float32,
  ):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.head_dim = head_dim
    self.rms_norm_eps = rms_norm_eps
    self.rope_theta = rope_theta
    self.max_position_embeddings = max_position_embeddings
    self.dtype = dtype


# -----------------------------------------------------------------------------
# Core Model Layers
# -----------------------------------------------------------------------------


class FlaxQwen3RMSNorm(nn.Module):
  dim: int
  eps: float = 1e-6
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    scale = self.param("weight", nn.initializers.ones, (self.dim,), self.dtype)
    normed = x_float * jax.lax.rsqrt(variance + self.eps)
    return (normed.astype(self.dtype)) * scale


class FlaxQwen3MLP(nn.Module):
  config: FlaxQwen3Config

  @nn.compact
  def __call__(self, x):
    gate_proj = nn.Dense(
        self.config.intermediate_size,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.config.dtype,
        name="gate_proj",
    )
    up_proj = nn.Dense(
        self.config.intermediate_size,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=self.config.dtype,
        name="up_proj",
    )
    down_proj = nn.Dense(
        self.config.hidden_size,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        dtype=self.config.dtype,
        name="down_proj",
    )

    return down_proj(jax.nn.silu(gate_proj(x)) * up_proj(x))


# -----------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# -----------------------------------------------------------------------------


def precompute_qwen3_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 1000000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Precomputes the cosine and sine tables for RoPE.
  Matches standard Llama/Qwen half-half rotation layout.
  """
  inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
  t = jnp.arange(max_seq_len, dtype=jnp.float32)
  freqs = jnp.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)

  # Concatenate [freqs, freqs] to match Hugging Face's rotate_half layout
  emb = jnp.concatenate([freqs, freqs], axis=-1)  # (max_seq_len, head_dim)

  cos = jnp.cos(emb)
  sin = jnp.sin(emb)
  return cos, sin


def apply_qwen3_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Applies RoPE to Q and K tensors.
  q shape: (batch, seq_len, num_heads, head_dim)
  k shape: (batch, seq_len, num_kv_heads, head_dim)
  cos, sin shape: (seq_len, head_dim) or (batch, seq_len, head_dim)
  """
  if cos.ndim == 3:
    # Per-batch positions: (batch, seq_len, head_dim) -> (batch, seq_len, 1, head_dim)
    cos = cos[:, :, jnp.newaxis, :]
    sin = sin[:, :, jnp.newaxis, :]
  else:
    # Reshape cos/sin to (1, seq_len, 1, head_dim) for broadcasting
    cos = cos[jnp.newaxis, :, jnp.newaxis, :]
    sin = sin[jnp.newaxis, :, jnp.newaxis, :]

  def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)

  q_rot = (q * cos) + (rotate_half(q) * sin)
  k_rot = (k * cos) + (rotate_half(k) * sin)
  return q_rot, k_rot


# -----------------------------------------------------------------------------
# Self Attention (Grouped Query Attention)
# -----------------------------------------------------------------------------


class FlaxQwen3Attention(nn.Module):
  config: FlaxQwen3Config

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
      cos_table: Optional[jnp.ndarray] = None,
      sin_table: Optional[jnp.ndarray] = None,
      position_ids: Optional[jnp.ndarray] = None,
  ):
    batch_size, seq_len, _ = x.shape

    # 1. Project Q, K, V
    # Output sizes: Q: 4096, K: 1024, V: 1024
    q_proj = nn.Dense(
        self.config.num_attention_heads * self.config.head_dim,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        dtype=self.config.dtype,
        name="q_proj",
    )
    k_proj = nn.Dense(
        self.config.num_key_value_heads * self.config.head_dim,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        dtype=self.config.dtype,
        name="k_proj",
    )
    v_proj = nn.Dense(
        self.config.num_key_value_heads * self.config.head_dim,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "heads")),
        dtype=self.config.dtype,
        name="v_proj",
    )
    o_proj = nn.Dense(
        self.config.hidden_size,
        use_bias=False,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("heads", "embed")),
        dtype=self.config.dtype,
        name="o_proj",
    )

    # QK-Norm Layers (Head-wise, sharing scale weights of size head_dim = 128)
    q_norm = FlaxQwen3RMSNorm(
        dim=self.config.head_dim,
        eps=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        name="q_norm",
    )
    k_norm = FlaxQwen3RMSNorm(
        dim=self.config.head_dim,
        eps=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        name="k_norm",
    )

    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    # 2. Reshape to heads first: (batch, seq_len, num_heads, head_dim)
    q = q.reshape((batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim))
    k = k.reshape((batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim))
    v = v.reshape((batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim))

    # Apply QK-Norm head-wise (normalizes over the last axis of size 128)
    q = q_norm(q)
    k = k_norm(k)

    # 3. Apply RoPE
    if cos_table is not None and sin_table is not None:
      if position_ids is not None:
        # Gather per-token positions: (batch, seq_len, head_dim). Used by Krea 2,
        # whose prompt template pads mid-sequence so positions count only valid tokens.
        cos = cos_table[position_ids]
        sin = sin_table[position_ids]
      else:
        # Extract cos/sin for the current sequence length
        cos = cos_table[:seq_len, :]
        sin = sin_table[:seq_len, :]
      q, k = apply_qwen3_rotary_pos_emb(q, k, cos, sin)

    # 4. Repeat KV heads to match Query heads (GQA)
    gqa_ratio = self.config.num_attention_heads // self.config.num_key_value_heads
    if gqa_ratio > 1:
      k = jnp.repeat(k, gqa_ratio, axis=-2)
      v = jnp.repeat(v, gqa_ratio, axis=-2)

    # 5. Transpose to (batch, num_heads, seq_len, head_dim) for attention
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # 6. Compute attention logits in float32
    q_f = q.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    v_f = v.astype(jnp.float32)

    scores = jnp.matmul(q_f, jnp.transpose(k_f, (0, 1, 3, 2))) / math.sqrt(self.config.head_dim)

    # 7. Apply causal attention mask
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    scores = jnp.where(causal_mask, scores, -1e4)

    # 8. Apply padding attention mask if provided
    if attention_mask is not None:
      p_mask = attention_mask[:, jnp.newaxis, jnp.newaxis, :].astype(jnp.bool_)
      scores = jnp.where(p_mask, scores, -1e4)

    # 9. Softmax & Weighted Sum in float32
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(probs, v_f).astype(self.config.dtype)

    # 10. Reshape back and project out: (batch, seq_len, hidden_size)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, -1))
    return o_proj(out)


# -----------------------------------------------------------------------------
# Decoder Block Layer
# -----------------------------------------------------------------------------


class FlaxQwen3DecoderLayer(nn.Module):
  config: FlaxQwen3Config

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
      cos_table: Optional[jnp.ndarray] = None,
      sin_table: Optional[jnp.ndarray] = None,
      position_ids: Optional[jnp.ndarray] = None,
  ):
    # input_layernorm
    input_layernorm = FlaxQwen3RMSNorm(
        dim=self.config.hidden_size,
        eps=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        name="input_layernorm",
    )
    # self_attn
    self_attn = FlaxQwen3Attention(
        config=self.config,
        name="self_attn",
    )
    # post_attention_layernorm
    post_attention_layernorm = FlaxQwen3RMSNorm(
        dim=self.config.hidden_size,
        eps=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        name="post_attention_layernorm",
    )
    # mlp
    mlp = FlaxQwen3MLP(
        config=self.config,
        name="mlp",
    )

    # Self-Attention block (with residual)
    attn_out = self_attn(
        input_layernorm(x),
        attention_mask=attention_mask,
        cos_table=cos_table,
        sin_table=sin_table,
        position_ids=position_ids,
    )
    x = x + attn_out

    # MLP block (with residual)
    mlp_out = mlp(post_attention_layernorm(x))
    x = x + mlp_out

    return x


# -----------------------------------------------------------------------------
# Full Transformer Model
# -----------------------------------------------------------------------------


class FlaxQwen3Model(nn.Module):
  config: FlaxQwen3Config

  @nn.compact
  def __call__(
      self,
      input_ids: Optional[jnp.ndarray] = None,
      attention_mask: Optional[jnp.ndarray] = None,
      position_ids: Optional[jnp.ndarray] = None,
      inputs_embeds: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    Runs the full Qwen3-4B model.

    Args:
        input_ids: (batch, seq_len) token ids. Exactly one of `input_ids` and
            `inputs_embeds` must be given.
        attention_mask: optional (batch, seq_len) padding mask.
        position_ids: optional (batch, seq_len) per-token rotary positions. When
            omitted, positions default to `arange(seq_len)`.
        inputs_embeds: optional (batch, seq_len, hidden_size) token embeddings,
            e.g. `embedding_table[input_ids]` gathered on the host. When given,
            the `embed_tokens` module is not created, so the parameter tree has
            no embedding table (the lookup is exact either way).
    Returns:
        last_hidden_state: Output of the final layer (batch, seq_len, 2560)
        all_hidden_states: List of activations from every layer, including token embeddings (length 37)
    """
    if (input_ids is None) == (inputs_embeds is None):
      raise ValueError("FlaxQwen3Model needs exactly one of input_ids and inputs_embeds.")

    # 1. Token Embeddings
    if inputs_embeds is not None:
      hidden_states = inputs_embeds.astype(self.config.dtype)
    else:
      embed_tokens = nn.Embed(
          num_embeddings=self.config.vocab_size,
          features=self.config.hidden_size,
          embedding_init=nn.with_logical_partitioning(
              nn.initializers.normal(stddev=self.config.hidden_size**-0.5), ("vocab", "embed")
          ),
          dtype=self.config.dtype,
          name="embed_tokens",
      )
      hidden_states = embed_tokens(input_ids)

    # Track all layer activations (including embedding layer)
    all_hidden_states = [hidden_states]

    # 2. Precompute RoPE cos/sin tables
    cos_table, sin_table = precompute_qwen3_freqs_cis(
        head_dim=self.config.head_dim,
        max_seq_len=self.config.max_position_embeddings,
        theta=self.config.rope_theta,
    )

    # 3. Stacked Decoder Layers
    for i in range(self.config.num_hidden_layers):
      layer = FlaxQwen3DecoderLayer(
          config=self.config,
          name=f"layers_{i}",
      )
      hidden_states = layer(
          hidden_states,
          attention_mask=attention_mask,
          cos_table=cos_table,
          sin_table=sin_table,
          position_ids=position_ids,
      )
      all_hidden_states.append(hidden_states)

    # 4. Final RMSNorm
    norm = FlaxQwen3RMSNorm(
        dim=self.config.hidden_size,
        eps=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        name="norm",
    )
    hidden_states = norm(hidden_states)

    # Keep all_hidden_states as the raw outputs of the layers, do not overwrite with final norm.

    return hidden_states, all_hidden_states


# -----------------------------------------------------------------------------
# NNX Transformer Model Implementations for Qwen3
# -----------------------------------------------------------------------------


class NNXFlaxQwen3RMSNorm(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, dim: int, eps: float = 1e-6, dtype: jnp.dtype = jnp.float32):
    self.eps = eps
    self.dtype = dtype
    self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    normed = x_float * jax.lax.rsqrt(variance + self.eps)
    return (normed.astype(self.dtype)) * self.weight[...]


class NNXFlaxQwen3MLP(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3Config):
    self.config = config
    self.gate_proj = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.intermediate_size,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=config.dtype,
        rngs=rngs,
    )
    self.up_proj = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.intermediate_size,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
        dtype=config.dtype,
        rngs=rngs,
    )
    self.down_proj = nnx.Linear(
        in_features=config.intermediate_size,
        out_features=config.hidden_size,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
        dtype=config.dtype,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class NNXFlaxQwen3Attention(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3Config):
    self.config = config
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.head_dim = config.head_dim

    self.q_proj = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.num_attention_heads * config.head_dim,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        dtype=config.dtype,
        rngs=rngs,
    )
    self.k_proj = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.num_key_value_heads * config.head_dim,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        dtype=config.dtype,
        rngs=rngs,
    )
    self.v_proj = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.num_key_value_heads * config.head_dim,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads")),
        dtype=config.dtype,
        rngs=rngs,
    )
    self.o_proj = nnx.Linear(
        in_features=config.num_attention_heads * config.head_dim,
        out_features=config.hidden_size,
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        dtype=config.dtype,
        rngs=rngs,
    )

    self.q_norm = NNXFlaxQwen3RMSNorm(rngs=rngs, dim=config.head_dim, eps=config.rms_norm_eps, dtype=config.dtype)
    self.k_norm = NNXFlaxQwen3RMSNorm(rngs=rngs, dim=config.head_dim, eps=config.rms_norm_eps, dtype=config.dtype)

  def __call__(
      self,
      x: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
      cos_table: Optional[jnp.ndarray] = None,
      sin_table: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    batch_size, seq_len, _ = x.shape

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)

    if cos_table is not None and sin_table is not None:
      cos_seq = cos_table[:seq_len, :]
      sin_seq = sin_table[:seq_len, :]
      q, k = apply_qwen3_rotary_pos_emb(q, k, cos_seq, sin_seq)

    if self.num_kv_heads != self.num_heads:
      num_repeats = self.num_heads // self.num_kv_heads
      k = jnp.repeat(k, num_repeats, axis=2)
      v = jnp.repeat(v, num_repeats, axis=2)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    scale = 1.0 / math.sqrt(self.head_dim)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    if attention_mask is not None:
      scores = scores + attention_mask

    attn_probs = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_probs, v)
    output = jnp.transpose(output, (0, 2, 1, 3))
    output = output.reshape(batch_size, seq_len, -1)
    output = self.o_proj(output)
    return output


class NNXFlaxQwen3DecoderLayer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3Config):
    self.config = config
    self.input_layernorm = NNXFlaxQwen3RMSNorm(
        rngs=rngs, dim=config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype
    )
    self.self_attn = NNXFlaxQwen3Attention(rngs=rngs, config=config)
    self.post_attention_layernorm = NNXFlaxQwen3RMSNorm(
        rngs=rngs, dim=config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype
    )
    self.mlp = NNXFlaxQwen3MLP(rngs=rngs, config=config)

  def __call__(
      self,
      x: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
      cos_table: Optional[jnp.ndarray] = None,
      sin_table: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    residual = x
    normed_x = self.input_layernorm(x)
    attn_out = self.self_attn(
        normed_x,
        attention_mask=attention_mask,
        cos_table=cos_table,
        sin_table=sin_table,
    )
    x = residual + attn_out

    mlp_out = self.mlp(self.post_attention_layernorm(x))
    x = x + mlp_out
    return x


class NNXFlaxQwen3Model(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3Config):
    self.config = config
    self.embed_tokens = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=config.hidden_size**-0.5), ("vocab", "embed")),
        dtype=config.dtype,
        rngs=rngs,
    )
    self.layers = nnx.List([NNXFlaxQwen3DecoderLayer(rngs=rngs, config=config) for _ in range(config.num_hidden_layers)])
    self.norm = NNXFlaxQwen3RMSNorm(rngs=rngs, dim=config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype)

  def __call__(
      self,
      input_ids: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    hidden_states = self.embed_tokens(input_ids)
    all_hidden_states = [hidden_states]

    cos_table, sin_table = precompute_qwen3_freqs_cis(
        head_dim=self.config.head_dim,
        max_seq_len=self.config.max_position_embeddings,
        theta=self.config.rope_theta,
    )

    for layer in self.layers:
      hidden_states = layer(
          hidden_states,
          attention_mask=attention_mask,
          cos_table=cos_table,
          sin_table=sin_table,
      )
      all_hidden_states.append(hidden_states)

    hidden_states = self.norm(hidden_states)
    return hidden_states, all_hidden_states


# -----------------------------------------------------------------------------
# Weight Mapping & Conversion Utilities
# -----------------------------------------------------------------------------


# Bump KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION (models/krea2/weight_cache.py) when this changes the produced values.
def load_and_convert_qwen3_weights(
    safetensors_path: str, jax_params: dict, config: FlaxQwen3Config, key_prefix: str = "model."
) -> dict:
  """
  Loads weights from safetensors via zero-copy safetensors.numpy and converts them to JAX parameter dictionary.

  Args:
      key_prefix: prefix of the decoder weights inside the checkpoint. Plain Qwen3
          checkpoints use "model." (default). Qwen3-VL checkpoints (e.g. the Krea 2
          text encoder) store the text tower under "model.language_model." or
          "language_model."; vision-tower weights are ignored. If the given prefix
          is not found, common alternatives are auto-detected.
  """
  reader = SafetensorsShardReader(safetensors_path)
  max_logging.log(
      f"Streaming Qwen3 weights from {safetensors_path} "
      f"({len(reader.files)} safetensors file{'s' if len(reader.files) != 1 else ''}); "
      "unused vision-tower tensors will not be read."
  )
  with reader:
    return _convert_qwen3_weights(reader, jax_params, config, key_prefix)


def _detect_qwen3_key_prefix(torch_weights, key_prefix: str) -> str:
  """Returns `key_prefix`, or a detected alternative if its embedding key is absent."""
  if f"{key_prefix}embed_tokens.weight" not in torch_weights:
    for candidate in ("model.", "model.language_model.", "language_model.", ""):
      if f"{candidate}embed_tokens.weight" in torch_weights:
        max_logging.log(f"Qwen3 key prefix '{key_prefix}' not found; using detected prefix '{candidate}'.")
        return candidate
  return key_prefix


# Bump KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION (models/krea2/weight_cache.py) when this changes the produced values.
def load_qwen3_embedding_table(safetensors_path: str, key_prefix: str = "model.") -> np.ndarray:
  """Reads only the token-embedding table `(vocab_size, hidden_size)` as a bf16 host array.

  Used when the embedding lookup runs on the host (`FlaxQwen3Model(inputs_embeds=...)`),
  so the table never occupies device memory. Key-prefix detection matches
  `load_and_convert_qwen3_weights`.
  """
  reader = SafetensorsShardReader(safetensors_path)
  with reader:
    key_prefix = _detect_qwen3_key_prefix(reader, key_prefix)
    table = reader.get_tensor(f"{key_prefix}embed_tokens.weight")
  table = np.asarray(table)
  if np.dtype(table.dtype) != np.dtype(jnp.bfloat16):
    table = table.astype(jnp.bfloat16)
  max_logging.log(f"Loaded Qwen3 embedding table {table.shape} ({table.nbytes / 1024**3:.2f} GiB) for host-side lookup.")
  return table


def _convert_qwen3_weights(torch_weights, jax_params: dict, config: FlaxQwen3Config, key_prefix: str) -> dict:
  """Maps only the Qwen language-tower tensors requested by the Flax tree."""
  del config
  max_logging.log("Starting lazy Qwen3 JAX parameter mapping...")

  key_prefix = _detect_qwen3_key_prefix(torch_weights, key_prefix)

  # Helper to transpose and cast weight
  def get_w(name: str, transpose: bool = True) -> np.ndarray:
    if name not in torch_weights:
      raise KeyError(f"Weight '{name}' not found in safetensors!")
    t = torch_weights.get_tensor(name)
    if len(t.shape) == 2 and transpose:
      t = t.T
    return t

  # Create mutable copy of JAX params to populate
  import flax

  flat_params = flax.traverse_util.flatten_dict(jax_params)
  converted_flat = {}

  for k, v in flat_params.items():
    # Reconstruct path string for debugging/matching
    path_str = ".".join(k)

    # 1. Token Embeddings
    if k[0] == "embed_tokens" and k[1] == "embedding":
      converted_flat[k] = get_w(f"{key_prefix}embed_tokens.weight", transpose=False)

    # 2. Decoder Layer Normalizations (RMSNorm)
    elif "input_layernorm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"{key_prefix}layers.{layer_idx}.input_layernorm.weight")

    elif "post_attention_layernorm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"{key_prefix}layers.{layer_idx}.post_attention_layernorm.weight")

    # 3. Attention Projections & QK-Norm
    elif "self_attn" in path_str and k[-1] == "kernel":
      layer_idx = k[0].split("_")[1]
      proj_name = k[2]  # q_proj, k_proj, v_proj, o_proj
      converted_flat[k] = get_w(f"{key_prefix}layers.{layer_idx}.self_attn.{proj_name}.weight")

    elif "self_attn" in path_str and "q_norm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"{key_prefix}layers.{layer_idx}.self_attn.q_norm.weight")

    elif "self_attn" in path_str and "k_norm" in path_str and k[-1] == "weight":
      layer_idx = k[0].split("_")[1]
      converted_flat[k] = get_w(f"{key_prefix}layers.{layer_idx}.self_attn.k_norm.weight")

    # 4. MLP Block
    elif "mlp" in path_str and k[-1] == "kernel":
      layer_idx = k[0].split("_")[1]
      proj_name = k[2]  # gate_proj, up_proj, down_proj
      converted_flat[k] = get_w(f"{key_prefix}layers.{layer_idx}.mlp.{proj_name}.weight")

    # 5. Final RMSNorm
    elif k[0] == "norm" and k[1] == "weight":
      converted_flat[k] = get_w(f"{key_prefix}norm.weight")

    else:
      max_logging.log(f"WARNING: JAX parameter '{path_str}' did not match any PyTorch weights!")
      converted_flat[k] = np.zeros(v.shape, dtype=np.float32) if hasattr(v, "shape") and not isinstance(v, np.ndarray) else v

  # Clean up PyTorch memory immediately
  del torch_weights
  import gc

  gc.collect()

  res = flax.traverse_util.unflatten_dict(converted_flat)
  return jax.tree_util.tree_map(
      lambda leaf: np.zeros(leaf.shape, dtype=leaf.dtype) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, res
  )
