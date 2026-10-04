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

"""Pallas TPU kernel: Krea 2 q/k post-processing in one HBM pass.

Between the q (or k) projection and the attention kernel the Krea 2 DiT block
applies a per-head zero-centered RMSNorm, rotate-half RoPE, the (B, L, H, D)
-> (B, H, L, D) transpose, the softmax scale (q only) and the padding of the
sequence to the attention kernel's block size. Written in jnp, XLA on TPU
splits this into three HBM-bound fusions per tensor (a float32 normed
intermediate, two D/2-wide RoPE halves in half-empty 128-lane tiles, a
concatenate), and a pure-jnp rewrite with full-width tables still materializes
the normed values and the two halves (the D/2 lane swap does not fuse). This
kernel reads the bf16 projection once and writes the bf16 attention operand
once:

  * input: the projection in head-major (B, H, L, D) form. The caller's
    transpose of the (B, L, H, D) projection is free: XLA writes the W8A8
    matmul output head-major (its (L, H, D) result with layout {2,0,1} is the
    same bytes as row-major (H, L, D) under the (8, 128) tiling). Reading
    (B, L, H*D) instead costs two relayout copies, since (L, H, D) -> (L, H*D)
    is not a bitcast under TPU tiling;
  * blocks (heads_per_block, block_rows, D) at (b, h, i, 0) in and out; the
    output is (B, H, padded_len, D) and rows >= L are zero-filled (an iota
    mask), so the padding to the attention block size costs nothing extra.
    block_rows must divide padded_len; input blocks past the end of L are
    clamped to the last one (fully masked) and the last partial input block
    overhangs L (unspecified rows, masked);
  * per head, in float32: `normed = x * rsqrt(mean(x^2) + eps) * (1 + w)`,
    `out = normed * cos2 + roll(normed, D/2) * sin2` with the full-width tables
    of `rotate_half_full_tables`, optionally `out * scale`, one cast at the end.
    That is the arithmetic of `Krea2RMSNorm` + `apply_rope_rotate_half`
    (`x1*cos - x2*sin == x1*cos + x2*(-sin)` exactly) without their
    intermediate cast to bf16 after the norm, and with the scale applied in
    float32 before the cast instead of in bf16 after it.

The grid is (B, row blocks, head blocks) with heads innermost, so the
(block_rows, D) cos/sin blocks stay in VMEM across the heads of a row block.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per grid step at most; the row block must also divide the padded length.
MAX_BLOCK_ROWS = 1024
# Heads per grid step at most (contiguous heads*D lanes per input row DMA).
MAX_HEADS_PER_BLOCK = 4


def rotate_half_full_tables(cos, sin):
  """Full-width `(L, D)` float32 tables `([cos, cos], [-sin, sin])` from the `(L, D/2)` rotate_half tables.

  With them rotate-half RoPE is one full-width expression,
  `x * cos2 + roll(x, D/2) * sin2`.
  """
  cos = cos.astype(jnp.float32)
  sin = sin.astype(jnp.float32)
  return jnp.concatenate([cos, cos], axis=-1), jnp.concatenate([-sin, sin], axis=-1)


def qk_prep_reference(x, weight, eps, cos, sin, scale=None, padded_len=None):
  """jnp reference of `krea2_qk_prep` on `(B, L, H, D)` with the `(L, D/2)` rotate_half tables.

  Same float32 arithmetic as the kernel (no intermediate cast); returns
  `(B, H, padded_len or L, D)` in `x.dtype`, zero rows past L.
  """
  cos2, sin2 = rotate_half_full_tables(cos, sin)
  half = x.shape[-1] // 2
  x_f32 = x.astype(jnp.float32)
  variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  normed = x_f32 * lax.rsqrt(variance + eps) * (1.0 + weight.astype(jnp.float32))
  rotated = jnp.concatenate([normed[..., half:], normed[..., :half]], axis=-1)
  out = normed * cos2[None, :, None, :] + rotated * sin2[None, :, None, :]
  if scale is not None:
    out = out * jnp.float32(scale)
  out = jnp.transpose(out.astype(x.dtype), (0, 2, 1, 3))
  seq_len = x.shape[1]
  if padded_len is not None and padded_len > seq_len:
    out = jnp.pad(out, ((0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)))
  return out


def pick_block_rows(padded_len: int, max_rows: int = MAX_BLOCK_ROWS) -> int:
  """Rows per grid step for a `padded_len`-row sequence, never more than `max_rows` unless forced to.

  Order of preference: the largest multiple of 16 that divides `padded_len` and is at most `max_rows`; else the
  largest such multiple of 8; else `padded_len` itself (one block) when `padded_len <= max_rows`.

  Raises:
    ValueError: no multiple of 8 at most `max_rows` divides `padded_len` and `padded_len > max_rows` (a single
      block would exceed the VMEM budget); pass `block_rows` explicitly or pad to a multiple of 8.
  """
  for step in (16, 8):
    for rows in range(min(max_rows, padded_len) // step * step, 0, -step):
      if padded_len % rows == 0:
        return rows
  if padded_len <= max_rows:
    return padded_len
  raise ValueError(f"pick_block_rows: no multiple of 8 that is at most max_rows={max_rows} divides "
                   f"padded_len={padded_len}, and a single {padded_len}-row block exceeds max_rows; pass "
                   "block_rows explicitly or pad the sequence to a multiple of 8.")


def pick_heads_per_block(num_heads: int, max_heads: int = MAX_HEADS_PER_BLOCK) -> int:
  """Largest divisor of `num_heads` that is at most `max_heads`."""
  for heads in range(min(max_heads, num_heads), 0, -1):
    if num_heads % heads == 0:
      return heads
  return 1


def _qk_prep_kernel(w_ref, cos_ref, sin_ref, x_ref, o_ref, *, heads_per_block, head_dim, block_rows, seq_len, eps,
                    scale, mask_tail):
  multiplier = 1.0 + w_ref[...].astype(jnp.float32)  # (1, D)
  cos2 = cos_ref[...]
  sin2 = sin_ref[...]
  if mask_tail:
    rows = pl.program_id(1) * block_rows + lax.broadcasted_iota(jnp.int32, (block_rows, head_dim), 0)
    valid = rows < seq_len
  for j in range(heads_per_block):
    x = x_ref[j].astype(jnp.float32)
    variance = jnp.mean(x * x, axis=-1, keepdims=True)
    normed = x * lax.rsqrt(variance + eps) * multiplier
    # roll by D/2 swaps the two halves: the rotate-half partner of every lane.
    out = normed * cos2 + pltpu.roll(normed, head_dim // 2, 1) * sin2
    if scale is not None:
      out = out * scale
    if mask_tail:
      out = jnp.where(valid, out, 0.0)
    o_ref[j] = out.astype(o_ref.dtype)


def krea2_qk_prep(x, weight, cos2, sin2, *, eps, padded_len, scale=None, block_rows=None, heads_per_block=None,
                  interpret=False):
  """Norm + rotate-half RoPE (+ scale) + zero padding of a head-major Krea 2 q/k projection.

  Args:
    x: `(B, H, L, D)` raw projection (bf16 on TPU), i.e. the transposed `(B, L, H, D)` projection.
    weight: `(D,)` zero-centered RMSNorm weight (multiplier `1 + weight`).
    cos2, sin2: `(L, D)` float32 tables of `rotate_half_full_tables`.
    eps: RMSNorm epsilon.
    padded_len: output sequence length (>= L), e.g. L padded to the attention block size.
    scale: optional float multiplier applied in float32 before the cast (the q softmax scale).
    block_rows: rows per grid step, must divide `padded_len` (default `pick_block_rows`).
    heads_per_block: heads per grid step, must divide H (default `pick_heads_per_block`).
    interpret: Pallas interpret mode (CPU tests).

  Returns:
    `(B, H, padded_len, D)` in `x.dtype`; rows >= L are zero.
  """
  batch, num_heads, seq_len, head_dim = x.shape
  if head_dim % 2:
    raise ValueError(f"head_dim={head_dim} must be even for rotate-half RoPE")
  if padded_len < seq_len:
    raise ValueError(f"padded_len={padded_len} is shorter than the sequence ({seq_len})")
  if cos2.shape != (seq_len, head_dim) or sin2.shape != (seq_len, head_dim):
    raise ValueError(f"cos2/sin2 must have shape {(seq_len, head_dim)}, got {cos2.shape} / {sin2.shape}")
  block_rows = block_rows or pick_block_rows(padded_len)
  if padded_len % block_rows:
    raise ValueError(f"block_rows={block_rows} must divide padded_len={padded_len}")
  heads_per_block = heads_per_block or pick_heads_per_block(num_heads)
  if num_heads % heads_per_block:
    raise ValueError(f"heads_per_block={heads_per_block} must divide num_heads={num_heads}")

  row_blocks = padded_len // block_rows
  # Input row blocks that start inside L; later output blocks re-read the last one (fully masked).
  last_in_block = math.ceil(seq_len / block_rows) - 1
  mask_tail = padded_len != seq_len or seq_len % block_rows != 0

  def x_index_map(b, i, h):
    return (b, h, jnp.minimum(i, last_in_block), 0)

  def table_index_map(b, i, h):
    del b, h
    return (jnp.minimum(i, last_in_block), 0)

  def weight_index_map(b, i, h):
    del b, i, h
    return (0, 0)

  def out_index_map(b, i, h):
    return (b, h, i, 0)

  in_specs = [
      pl.BlockSpec((1, head_dim), weight_index_map),
      pl.BlockSpec((block_rows, head_dim), table_index_map),
      pl.BlockSpec((block_rows, head_dim), table_index_map),
      pl.BlockSpec((None, heads_per_block, block_rows, head_dim), x_index_map),
  ]
  out_specs = pl.BlockSpec((None, heads_per_block, block_rows, head_dim), out_index_map)
  kernel = functools.partial(
      _qk_prep_kernel,
      heads_per_block=heads_per_block,
      head_dim=head_dim,
      block_rows=block_rows,
      seq_len=seq_len,
      eps=float(eps),
      scale=None if scale is None else float(scale),
      mask_tail=mask_tail,
  )
  return pl.pallas_call(
      kernel,
      grid=(batch, row_blocks, num_heads // heads_per_block),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=jax.ShapeDtypeStruct((batch, num_heads, padded_len, head_dim), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          disable_bounds_checks=True,
      ),
      interpret=interpret,
  )(weight.astype(jnp.float32).reshape(1, head_dim), cos2.astype(jnp.float32), sin2.astype(jnp.float32), x)
