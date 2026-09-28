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

"""Pallas TPU flash attention kernel for Krea 2 batch-1 inference.

Forward-only GQA flash attention built on the structure of the tuned custom
splash kernel (`custom_splash_attention.py`):

  * transposed layout: qk = k . q^T has shape (bkv_compute, bq), so the per-query
    softmax stats m, l are (1, bq) lane vectors and the o accumulator is
    (head_dim, bq);
  * exp2 with q pre-scaled by softmax_scale * log2(e) (done by the caller);
  * bf16 P . V with f32 accumulation and VPU register tiling (bkv_compute_in);
  * native GQA through the k/v index map (h // q_heads_per_kv_head);
  * static ragged kv tail (kv_seq_len % block_kv) handled in the last kv block.

On top of that it supports a dynamic per-batch PREFIX key mask: kv position p of
batch element b is valid iff p < valid_kv_len[b]. The lengths are delivered via
scalar prefetch and batch is a grid axis (not a vmap) so the kernel can read
them. Only kv blocks that can contain invalid positions pay for the mask.

Every q block re-reads all k/v blocks from HBM, so fewer, larger q blocks are
faster. The automatic block_q minimizes padded query waste over multiples of 128
in [512, max], where max is 2048 unless the chip has a calibrated VMEM budget
(`_VMEM_BUDGET_BYTES`, keyed by device_kind, calibrated for bfloat16 q/k/v):
there it grows to the largest block_q whose `_estimated_vmem_bytes` fits the
budget (up to 8192), since the compile fails once the kernel exceeds Mosaic's
default scoped VMEM limit. Other operand dtypes keep max at 2048.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

# Registry-level switch: the `flash_custom` attention wrapper passes this as
# `interpret` so CPU tests can run the kernel in Pallas interpret mode.
INTERPRET = False

_DEFAULT_BLOCK_KV = 1024
_DEFAULT_BLOCK_KV_COMPUTE = 512
_DEFAULT_BLOCK_KV_COMPUTE_IN = 256
_BLOCK_Q_MIN = 512
_BLOCK_Q_BASE_MAX = 2048  # always allowed: the range before the VMEM budget
_BLOCK_Q_ABS_MAX = 8192  # never picked automatically above this

# Bumped whenever the automatic block-size choice changes, so AOT caches keyed
# on it miss executables compiled with the previous choice.
KREA2_BLOCK_SELECTION_REVISION = 2

# Budget for `_estimated_vmem_bytes` per device kind, ~8 % under the smallest
# estimate that failed to compile. Compile-only calibration with bfloat16
# q/k/v (jax 0.11.2, libtpu 0.0.48, no vmem_limit_bytes); wider operands need
# more VMEM, so other dtypes keep the base maximum. Largest OK / smallest
# failing block_q for block_kv/block_kv_compute/block_kv_compute_in:
#   v6e: 1024/512/256 8832/9088, 2048/512/256 8320/8832, 1024/1024/256 5632/5888,
#        2048/1024/256 5504/5632, 4096/1024/256 4992/5120, 2048/2048/256 3328/3712,
#        4096/2048/256 2688/3328, 2048/1024/128 5120/5504;
#   v5e: 1024/512/256 4224/4608, 2048/1024/256 2560/3072, 4096/1024/256 2048/3072.
# Every failing point estimates >= 39.39e6 bytes on v6e and >= 21.76e6 on v5e.
_VMEM_BUDGET_BYTES = {"TPU v6 lite": 36_000_000, "TPU v5 lite": 18_500_000}


def padded_len(n: int, multiple: int) -> int:
  """Rounds `n` up to a multiple of `multiple`."""
  return ((n + multiple - 1) // multiple) * multiple


@dataclasses.dataclass(frozen=True)
class Krea2BlockSizes:
  """Block sizes for the Krea 2 attention kernel.

  block_q: query rows per grid step (lane dimension of the stats/accumulator).
  block_kv: kv rows DMA'd per grid step.
  block_kv_compute: kv rows per QK^T matmul inside a kv block.
  block_kv_compute_in: kv rows per online-softmax VPU register tile.
  """

  block_q: int
  block_kv: int = _DEFAULT_BLOCK_KV
  block_kv_compute: int = _DEFAULT_BLOCK_KV_COMPUTE
  block_kv_compute_in: int = _DEFAULT_BLOCK_KV_COMPUTE_IN

  def __post_init__(self):
    for name in ("block_q", "block_kv", "block_kv_compute", "block_kv_compute_in"):
      value = getattr(self, name)
      if not isinstance(value, (int, np.integer)) or value <= 0 or value % NUM_LANES != 0:
        raise ValueError(f"{name}={value!r} must be a positive multiple of {NUM_LANES}.")
    if self.block_kv % self.block_kv_compute != 0:
      raise ValueError(f"block_kv={self.block_kv} must be a multiple of block_kv_compute={self.block_kv_compute}.")
    if self.block_kv_compute % self.block_kv_compute_in != 0:
      raise ValueError(
          f"block_kv_compute={self.block_kv_compute} must be a multiple of "
          f"block_kv_compute_in={self.block_kv_compute_in}."
      )


def _user_block_value(user, name):
  if user is None:
    return None
  if isinstance(user, dict):
    return user.get(name, None)
  return getattr(user, name, None)


def _estimated_vmem_bytes(block_q: int, block_kv: int, block_kv_compute: int) -> int:
  """Empirical VMEM estimate: f32 (bkv_compute, bq) scores, per-bq temporaries, k/v buffers."""
  return block_q * (4 * block_kv_compute + 2304) + 1664 * block_kv


def max_auto_block_q(
    device_kind,
    block_kv: int,
    block_kv_compute: int,
    block_kv_compute_in: int,
    vmem_limit_bytes=None,
    dtype=None,
) -> int:
  """Largest block_q the automatic choice may use for these kv block sizes.

  Returns `_BLOCK_Q_BASE_MAX` for chips without a calibrated budget, when the
  user sets `vmem_limit_bytes`, when block_kv_compute_in < 256 or when the q/k/v
  `dtype` is not bfloat16 (not calibrated; None means bfloat16). Otherwise the
  largest multiple of 128 in [_BLOCK_Q_BASE_MAX, _BLOCK_Q_ABS_MAX] whose
  estimate fits the chip's budget, never below the base.
  """
  budget = _VMEM_BUDGET_BYTES.get(device_kind)
  bf16 = dtype is None or jnp.dtype(dtype) == jnp.bfloat16
  if budget is None or vmem_limit_bytes is not None or block_kv_compute_in < 256 or not bf16:
    return _BLOCK_Q_BASE_MAX
  block_q = _BLOCK_Q_BASE_MAX
  while (
      block_q + NUM_LANES <= _BLOCK_Q_ABS_MAX
      and _estimated_vmem_bytes(block_q + NUM_LANES, block_kv, block_kv_compute) <= budget
  ):
    block_q += NUM_LANES
  return block_q


def _default_block_q(seq_len: int, max_block_q: int = _BLOCK_Q_BASE_MAX) -> int:
  """Picks block_q minimizing padded query waste, preferring larger blocks on ties."""
  cap = padded_len(max(seq_len, 1), NUM_LANES)
  candidates = [bq for bq in range(_BLOCK_Q_MIN, max_block_q + 1, NUM_LANES) if bq <= cap] or [cap]
  best = None
  for bq in candidates:
    waste = padded_len(seq_len, bq) - seq_len
    key = (waste, -bq)
    if best is None or key < best[0]:
      best = (key, bq)
  return best[1]


def select_krea2_block_sizes(seq_len: int, user=None, device_kind=None, dtype=None) -> Krea2BlockSizes:
  """Selects kernel block sizes for a sequence of length `seq_len`.

  Args:
    seq_len: static (unpadded) sequence length.
    user: optional CustomFlashBlockSizes-like object or dict. Its non-None
      block_q / block_kv / block_kv_compute / block_kv_compute_in override the
      defaults; missing or None fields fall back to the defaults. A non-None
      vmem_limit_bytes keeps the automatic block_q at most 2048.
    device_kind: `device_kind` of the chip the kernel runs on (e.g.
      'TPU v6 lite'); None or an uncalibrated chip keeps block_q at most 2048.
    dtype: dtype of q/k/v. None means bfloat16, the dtype the VMEM budget is
      calibrated for; any other dtype keeps block_q at most 2048.

  Returns:
    A validated Krea2BlockSizes. Default block_q is the multiple of 128 in
    [512, max_auto_block_q(...)] (capped at seq_len rounded up to 128)
    minimizing the padded waste ceil(seq_len / bq) * bq - seq_len, larger bq
    winning ties.
  """
  block_kv = _user_block_value(user, "block_kv") or _DEFAULT_BLOCK_KV
  block_kv_compute = _user_block_value(user, "block_kv_compute")
  block_kv_compute_in = _user_block_value(user, "block_kv_compute_in")
  if block_kv_compute is None:
    block_kv_compute = min(_DEFAULT_BLOCK_KV_COMPUTE, block_kv)
  if block_kv_compute_in is None:
    block_kv_compute_in = min(_DEFAULT_BLOCK_KV_COMPUTE_IN, block_kv_compute)
  block_q = _user_block_value(user, "block_q")
  if not block_q:
    max_block_q = max_auto_block_q(
        device_kind,
        block_kv,
        block_kv_compute,
        block_kv_compute_in,
        _user_block_value(user, "vmem_limit_bytes"),
        dtype=dtype,
    )
    block_q = _default_block_q(seq_len, max_block_q)
  return Krea2BlockSizes(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_kv_compute_in=block_kv_compute_in,
  )


def _krea2_attention_kernel(
    valid_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    kv_seq_len: int,
    use_base2_exp: bool,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

  b, j = pl.program_id(0), pl.program_id(3)
  exp = jnp.exp2 if use_base2_exp else jnp.exp
  sv_dims = (((0,), (0,)), ((), ()))
  valid_len = valid_ref[b]

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def _online_inner(qk, v_chunk, m_prev, l_prev, o_prev):
    # Standard online-softmax tiling over the VPU register block.
    step = bkv_compute_in
    for i in range(0, qk.shape[0], step):
      qk_slice = qk[i : i + step]

      m_curr = qk_slice.max(axis=0)[None, :]
      m_next = jnp.maximum(m_prev, m_curr)
      s_curr = exp(qk_slice - m_next[0:1])
      l_curr = s_curr.sum(axis=0, keepdims=True)

      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev

      o_curr = lax.dot_general(
          v_chunk[i : i + step],
          s_curr.astype(q_ref.dtype),
          sv_dims,
          preferred_element_type=float32,
      )
      o_prev = alpha[0:1, ...] * o_prev + o_curr
      m_prev, l_prev = m_next, l_next
    return m_prev, l_prev, o_prev

  def _compute(kv_compute_index, length: int, masked: bool):
    q = q_ref[...]
    slice_k = pl.ds(kv_compute_index * bkv_compute, length)
    qk = lax.dot_general(k_ref[slice_k, :], q, NT_DIM_NUMBERS, preferred_element_type=float32)
    if masked:
      # Transposed layout: kv positions live on rows (sublanes).
      kv_pos = j * bkv + kv_compute_index * bkv_compute + lax.broadcasted_iota(jnp.int32, (length, 1), 0)
      qk = jnp.where(kv_pos < valid_len, qk, mask_value)
    v_chunk = v_ref[slice_k, :]
    m_prev, l_prev, o_prev = _online_inner(qk, v_chunk, m_scratch_ref[...], l_scratch_ref[...], o_scratch_ref[...])
    m_scratch_ref[...], l_scratch_ref[...] = m_prev, l_prev
    o_scratch_ref[...] = o_prev

  def _run_block(block_len: int, masked: bool):
    """Processes `block_len` (static) kv rows of the current kv block."""
    full_iters, tail = divmod(block_len, bkv_compute)
    if full_iters:
      lax.fori_loop(
          0,
          full_iters,
          lambda idx, carry: _compute(idx, bkv_compute, masked),
          None,
          unroll=True,
      )
    if tail:
      _compute(full_iters, tail, masked)

  # Only blocks that reach past the valid prefix pay for the mask. Positions in
  # [kv_seq_len, padded) are never computed (static ragged tail), so a block is
  # clean iff its end, clipped to kv_seq_len, is <= valid_len.
  block_end = jnp.minimum((j + 1) * bkv, kv_seq_len)
  needs_mask = block_end > valid_len
  last_block_len = kv_seq_len - (grid_width - 1) * bkv
  is_last = j == grid_width - 1

  if grid_width > 1:

    @pl.when(jnp.logical_not(is_last) & jnp.logical_not(needs_mask))
    def _body():
      _run_block(bkv, masked=False)

    @pl.when(jnp.logical_not(is_last) & needs_mask)
    def _body_masked():
      _run_block(bkv, masked=True)

  @pl.when(is_last & jnp.logical_not(needs_mask))
  def _last_body():
    _run_block(last_block_len, masked=False)

  @pl.when(is_last & needs_mask)
  def _last_body_masked():
    _run_block(last_block_len, masked=True)

  @pl.when(is_last)
  def end():
    l = l_scratch_ref[...]
    l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
    o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


def _krea2_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    valid_kv_len: jax.Array,
    block_sizes: Krea2BlockSizes,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    use_base2_exp: bool,
    vmem_limit_bytes: int | None,
    interpret: bool,
) -> jax.Array:
  batch, num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  _, num_kv_heads, padded_kv_seq_len, _ = k.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bkv_compute_in = block_sizes.block_kv_compute_in

  assert head_dim_qk == 128 and head_dim_v == 128, f"head_dim must be 128, got {head_dim_qk=} {head_dim_v=}"
  assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
  assert k.shape[0] == batch, f"q/k batch mismatch: {q.shape} vs {k.shape}"
  assert num_q_heads % num_kv_heads == 0, f"{num_q_heads=} must be a multiple of {num_kv_heads=}"
  assert padded_q_seq_len % bq == 0, f"q length {padded_q_seq_len} must be padded to a multiple of block_q={bq}"
  assert padded_kv_seq_len % bkv == 0, f"kv length {padded_kv_seq_len} must be padded to a multiple of block_kv={bkv}"
  assert 0 < q_seq_len <= padded_q_seq_len, f"{q_seq_len=} out of range for padded length {padded_q_seq_len}"
  assert 0 < kv_seq_len <= padded_kv_seq_len, f"{kv_seq_len=} out of range for padded length {padded_kv_seq_len}"
  assert valid_kv_len.shape == (batch,), f"valid_kv_len must have shape ({batch},), got {valid_kv_len.shape}"

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  def q_index_map(b, h, i, j, *_):
    return (b, h, i, 0)

  def kv_index_map(b, h, i, j, *_):
    return (b, h // q_heads_per_kv_head, j, 0)

  def out_index_map(b, h, i, j, *_):
    return (b, h, 0, i)

  in_specs = [
      pl.BlockSpec((None, None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((None, None, bkv, head_dim_qk), kv_index_map),
      pl.BlockSpec((None, None, bkv, head_dim_v), kv_index_map),
  ]
  out_specs = pl.BlockSpec((None, None, head_dim_v, bq), out_index_map)
  out_shape = jax.ShapeDtypeStruct((batch, num_q_heads, head_dim_v, q_seq_len), q.dtype)
  scratch_shapes = [
      pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),  # m
      pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),  # l
      pltpu.VMEM((head_dim_v, bq), jnp.float32),  # o
  ]
  grid_width = (kv_seq_len + bkv - 1) // bkv
  grid_height = (q_seq_len + bq - 1) // bq
  grid = (batch, num_q_heads, grid_height, grid_width)

  return pl.pallas_call(
      functools.partial(
          _krea2_attention_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bkv=bkv,
          bkv_compute=bkv_compute,
          bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v,
          kv_seq_len=kv_seq_len,
          use_base2_exp=use_base2_exp,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary"),
          disable_bounds_checks=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=out_shape,
      interpret=interpret,
  )(valid_kv_len.astype(jnp.int32), q, k, v)


def make_krea2_attention(
    block_sizes: Krea2BlockSizes,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    use_base2_exp: bool = True,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Builds the Krea 2 prefix-masked GQA flash attention function.

  The returned `fn(q, k, v, valid_kv_len)` takes
    q: (B, Hq, Lq_pad, 128), already scaled by softmax_scale (times log2(e)
      when `use_base2_exp`), Lq_pad a multiple of block_q;
    k, v: (B, Hkv, Lkv_pad, 128), Hq % Hkv == 0, Lkv_pad a multiple of block_kv;
    valid_kv_len: (B,) int32, kv position p of batch b is attended iff
      p < valid_kv_len[b] (a prefix mask; must be >= 1);
  and returns (B, Hq, 128, q_seq_len) in q.dtype (head_dim-major, transposed
  like the custom splash kernel; the caller swaps the last two axes).
  `q_seq_len` / `kv_seq_len` are the static unpadded lengths.
  """

  def _fn(q, k, v, valid_kv_len):
    return _krea2_attention_forward(
        q,
        k,
        v,
        valid_kv_len,
        block_sizes,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        use_base2_exp=use_base2_exp,
        vmem_limit_bytes=vmem_limit_bytes,
        interpret=interpret,
    )

  return _fn
