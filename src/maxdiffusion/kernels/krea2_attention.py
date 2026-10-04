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
faster. The automatic block_q is the multiple of 128 in [512, 2048] minimizing
padded query rows plus ~200 rows of per-block overhead. A VMEM-budget
extension (block_q up to 8192, bounded by the calibrated per-chip budgets in
`_VMEM_BUDGET_BYTES` for bfloat16 q/k/v) exists behind
`AUTO_BLOCK_Q_BUDGET_EXTENSION` and is off: with block_kv 2048 it picks
block_q 4096 at the aspect presets' sequence lengths (4016, 16256, 16352),
which is about 3x slower in the kernel on a TPU v6e-1.

Two kernel variants share the wrapper, grid, BlockSpecs and inputs/outputs
(`Krea2BlockSizes.variant`):

  * "flash": the kernel described above. Per kv chunk of block_kv_compute rows
    it updates m / l / o for the whole (bq) lane range per block_kv_compute_in
    rows.
  * "hybrid": the same math restructured for the TPU's VLIW issue slots and
    VMEM spills (both measured as the limit of "flash" on a v6e-1):
      1. one (block_kv_compute, bq) f32 QK^T matmul per kv chunk, written to an
         explicit VMEM scratch;
      2. online softmax + P.V per lane strip of block_q_strip lanes (the last
         strip may be narrower), with the strip's running max m and
         accumulator o held in registers over the whole chunk;
      3. the softmax denominator l on the MXU: v^T of the kv block is written
         once per grid step into a (136, block_kv) scratch whose row 128 is
         ones (rows 129..135 zeros, written at j == 0), so the P.V matmul
         o_ext = v^T_ext . P carries l in row 128 and finalize divides by it;
      4. the running max is updated once per block_kv_compute_in rows, exp2 +
         P.V run per block_kv_pv rows (one v6e MXU contraction tile).
    On a TPU v6e-1 at the Krea 2 1024x1024 / 2048x2048 lengths (seq 4224 /
    16512, block_q 1408 / 1664) it takes 1.444 / 16.869 ms per call against
    1.674 / 20.558 ms for "flash" (same wrapper, same block_q), with the same
    accuracy. Its QK^T scratch makes it VMEM-hungry: block_q is capped by
    `_estimated_hybrid_vmem_bytes` against `_HYBRID_VMEM_BUDGET_BYTES`.

Selection (`select_krea2_block_sizes`): the user carrier's `kernel` field
("auto" / "flash" / "hybrid", `krea2_attention_kernel` in the Krea 2 configs)
picks the variant; "auto" is "hybrid" on a TPU v6e ("TPU v6 lite") and "flash"
everywhere else (`resolve_kernel_variant`). kv block sizes default per
(variant, chip) from `_DEFAULT_KV_BLOCKS`, each field overridable by the user.
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
_HEAD_DIM = 128
# Hybrid v^T_ext / o_ext rows: 128 v channels + a ones row (l) + 7 zero rows.
_EXT_ROWS = _HEAD_DIM + NUM_SUBLANES

# Registry-level switch: the `flash_custom` attention wrapper passes this as
# `interpret` so CPU tests can run the kernel in Pallas interpret mode.
INTERPRET = False

KERNEL_VARIANTS = ("flash", "hybrid")
KERNEL_CHOICES = ("auto",) + KERNEL_VARIANTS
# Chips on which kernel choice "auto" runs the hybrid variant (measured faster
# on a v6e-1; it does not fit the default v5e VMEM limit at the same sizes).
_AUTO_HYBRID_DEVICE_KINDS = ("TPU v6 lite",)

# Default kv block sizes per (variant, device_kind), the None chip being the
# fallback of a variant: (block_kv, block_kv_compute, block_kv_compute_in,
# block_kv_pv, block_q_strip). flash/v6e are the sizes tuned on a v6e-1
# (2026-09-28: 1.512 -> 1.348 ms per call at 1024x1024 against flash/None);
# hybrid are the probe winner's sizes (2026-10-03).
_DEFAULT_KV_BLOCKS = {
    ("flash", None): (1024, 512, 256, None, None),
    ("flash", "TPU v6 lite"): (2048, 1024, 256, None, None),
    ("hybrid", None): (2048, 2048, 1024, 256, 256),
    ("hybrid", "TPU v6 lite"): (2048, 2048, 1024, 256, 256),
}
_DEFAULT_BLOCK_KV, _DEFAULT_BLOCK_KV_COMPUTE, _DEFAULT_BLOCK_KV_COMPUTE_IN = _DEFAULT_KV_BLOCKS[("flash", None)][:3]
_BLOCK_Q_MIN = 512
# Per-q-block cost of the automatic block_q choice, in query rows. Calibrated on
# v6e-1 (kv 2048/1024/256): one fewer q block saved ~250 rows at seq 4224, ~180
# at 16512 and ~295 at 4608, i.e. roughly 200-300 rows per block; 200 is the
# conservative end.
_BLOCK_Q_OVERHEAD_ROWS = 200
_BLOCK_Q_BASE_MAX = 2048  # always allowed: the range before the VMEM budget
_BLOCK_Q_ABS_MAX = 8192  # never picked automatically above this

# The VMEM-budget extension of the automatic block_q is off since 2026-10-01
# (block selection revision 3): with block_kv 2048 it picks block_q values that
# are 3x slower at the aspect presets' sequence lengths (seq 4016 / 16256 /
# 16352 -> 4096; measured on v6e-1). The budget code stays for an explicit
# opt-in (`max_auto_block_q(..., budget_extension=True)`).
AUTO_BLOCK_Q_BUDGET_EXTENSION = False

# Bumped whenever the automatic block-size choice changes, so AOT caches keyed
# on it miss executables compiled with the previous choice (the AOT meta appends
# "+budget" when AUTO_BLOCK_Q_BUDGET_EXTENSION is on).
# 3: budget extension off, per-block overhead term in the automatic choice.
# 4: per-chip kernel choice (auto -> hybrid on v6e), per-variant per-chip default block sizes
KREA2_BLOCK_SELECTION_REVISION = 4

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


# Hybrid variant: budget for `_estimated_hybrid_vmem_bytes` per device kind,
# `_HYBRID_VMEM_SAFETY` times the chip's default scoped VMEM limit (a user
# vmem_limit_bytes replaces the limit, same factor). Compile-only calibration
# (jax 0.11.2, libtpu 0.0.48, bf16 q/k/v, block_kv 2048, block_kv_compute_in
# min(1024, block_kv_compute), block_kv_pv 256, block_q_strip 256); largest OK
# / smallest failing block_q per block_kv_compute, with the estimate / limit:
#   v6e (32 MiB): 2048: 2176 / 2304 (0.848 / 0.893), 1024: 3328 / 3456 (0.848 / 0.878);
#   v5e (16 MiB): 2048: 1024 / 1152 (0.883 / 0.973), 1024: 1536 / 1664 (0.869 / 0.928);
#   v5e with vmem_limit_bytes 32 MiB: 2048: 2176 / 2304 (the v6e default limit:
#   same points); v6e with 64 MiB: 2048: 4096 OK (0.763).
# Every failing point estimates >= 0.878 x its limit; 0.86 keeps the v6e
# defaults at block_q 2048 (0.803) and costs v5e one step (896 instead of
# 1024 with block_kv_compute 2048, 1408 instead of 1536 with 1024).
_SCOPED_VMEM_LIMIT_BYTES = {"TPU v6 lite": 32 * 1024 * 1024, "TPU v5 lite": 16 * 1024 * 1024}
_HYBRID_VMEM_SAFETY = 0.86
_HYBRID_TEMP_BYTES_PER_LANE = 2048
_HYBRID_VMEM_BUDGET_BYTES = {kind: int(limit * _HYBRID_VMEM_SAFETY) for kind, limit in _SCOPED_VMEM_LIMIT_BYTES.items()}


def padded_len(n: int, multiple: int) -> int:
  """Rounds `n` up to a multiple of `multiple`."""
  return ((n + multiple - 1) // multiple) * multiple


def parse_kernel_choice(value) -> str:
  """Normalizes a kernel choice: "auto", "flash" or "hybrid".

  None, an empty or whitespace-only string and a quoted empty string ("''",
  '""', as a command-line override `krea2_attention_kernel=''` arrives) mean
  "auto". Surrounding whitespace and quotes are stripped and the value is
  lower-cased; anything else raises ValueError naming the allowed values.
  """
  if value is None:
    return "auto"
  if not isinstance(value, str):
    raise ValueError(f"kernel choice must be a string, one of {list(KERNEL_CHOICES)}; got {value!r}")
  text = value.strip()
  while len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
    text = text[1:-1].strip()
  text = text.lower()
  if not text:
    return "auto"
  if text not in KERNEL_CHOICES:
    raise ValueError(f"kernel choice must be one of {list(KERNEL_CHOICES)} (empty means 'auto'); got {value!r}")
  return text


def resolve_kernel_variant(choice, device_kind) -> str:
  """Kernel variant for a (parsed or raw) kernel choice on `device_kind`.

  "auto" is "hybrid" on a TPU v6e ("TPU v6 lite") and "flash" on every other
  chip, including an unknown one (None) and "cpu"; "flash" / "hybrid" are
  returned as they are.
  """
  choice = parse_kernel_choice(choice)
  if choice != "auto":
    return choice
  return "hybrid" if device_kind in _AUTO_HYBRID_DEVICE_KINDS else "flash"


@dataclasses.dataclass(frozen=True)
class Krea2BlockSizes:
  """Block sizes for the Krea 2 attention kernel.

  block_q: query rows per grid step (lane dimension of the stats/accumulator).
  block_kv: kv rows DMA'd per grid step.
  block_kv_compute: kv rows per QK^T matmul inside a kv block.
  block_kv_compute_in: kv rows per online-softmax VPU register tile ("flash")
    or per running-max update ("hybrid").
  variant: "flash" or "hybrid" (see the module docstring).
  block_kv_pv: "hybrid" only: kv rows per exp2 + P.V matmul.
  block_q_strip: "hybrid" only: q lanes per online-softmax strip (the last
    strip of a block may be narrower).
  """

  block_q: int
  block_kv: int = _DEFAULT_BLOCK_KV
  block_kv_compute: int = _DEFAULT_BLOCK_KV_COMPUTE
  block_kv_compute_in: int = _DEFAULT_BLOCK_KV_COMPUTE_IN
  variant: str = "flash"
  block_kv_pv: int | None = None
  block_q_strip: int | None = None

  def __post_init__(self):
    if self.variant not in KERNEL_VARIANTS:
      raise ValueError(f"variant={self.variant!r} must be one of {list(KERNEL_VARIANTS)}.")
    names = ["block_q", "block_kv", "block_kv_compute", "block_kv_compute_in"]
    if self.variant == "flash":
      for name in ("block_kv_pv", "block_q_strip"):
        if getattr(self, name) is not None:
          raise ValueError(f"{name}={getattr(self, name)!r} only applies to the hybrid variant; leave it None for flash.")
    else:
      names += ["block_kv_pv", "block_q_strip"]
    for name in names:
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0 or value % NUM_LANES != 0:
        raise ValueError(f"{name}={value!r} must be a positive multiple of {NUM_LANES}.")
    if self.block_kv % self.block_kv_compute != 0:
      raise ValueError(f"block_kv={self.block_kv} must be a multiple of block_kv_compute={self.block_kv_compute}.")
    if self.block_kv_compute % self.block_kv_compute_in != 0:
      raise ValueError(
          f"block_kv_compute={self.block_kv_compute} must be a multiple of "
          f"block_kv_compute_in={self.block_kv_compute_in}."
      )
    if self.variant == "hybrid" and self.block_kv_compute_in % self.block_kv_pv != 0:
      raise ValueError(
          f"block_kv_compute_in={self.block_kv_compute_in} must be a multiple of block_kv_pv={self.block_kv_pv}."
      )


def _user_block_value(user, name):
  if user is None:
    return None
  if isinstance(user, dict):
    return user.get(name, None)
  return getattr(user, name, None)


def default_kv_block_sizes(variant: str, device_kind=None) -> dict:
  """Default kv block sizes of `variant` on `device_kind` (the variant's None-chip entry for other chips)."""
  if variant not in KERNEL_VARIANTS:
    raise ValueError(f"variant={variant!r} must be one of {list(KERNEL_VARIANTS)}.")
  sizes = _DEFAULT_KV_BLOCKS.get((variant, device_kind)) or _DEFAULT_KV_BLOCKS[(variant, None)]
  return dict(zip(("block_kv", "block_kv_compute", "block_kv_compute_in", "block_kv_pv", "block_q_strip"), sizes))


def _estimated_vmem_bytes(block_q: int, block_kv: int, block_kv_compute: int) -> int:
  """Empirical VMEM estimate: f32 (bkv_compute, bq) scores, per-bq temporaries, k/v buffers."""
  return block_q * (4 * block_kv_compute + 2304) + 1664 * block_kv


def _estimated_hybrid_vmem_bytes(block_q: int, block_kv: int, block_kv_compute: int, itemsize: int = 2) -> int:
  """VMEM estimate of the hybrid variant (bytes).

  Explicit scratch: the f32 (block_kv_compute, bq) QK^T tile, f32 m (8, bq)
  and o_ext (136, bq), the (136 -> 144, block_kv) v^T_ext tile; double-buffered
  q (bq, 128) / out (128, bq) / k, v (block_kv, 128) blocks of `itemsize`
  bytes; plus `_HYBRID_TEMP_BYTES_PER_LANE` compiler temporaries per q lane
  (fitted to compiles, see `_HYBRID_VMEM_BUDGET_BYTES`).
  """
  scratch = 4 * block_kv_compute * block_q + 4 * (NUM_SUBLANES + _EXT_ROWS) * block_q + itemsize * 144 * block_kv
  buffers = 2 * itemsize * _HEAD_DIM * (2 * block_q + 2 * block_kv)
  return scratch + buffers + _HYBRID_TEMP_BYTES_PER_LANE * block_q


def _hybrid_vmem_budget(device_kind, vmem_limit_bytes):
  """Budget for `_estimated_hybrid_vmem_bytes`; None when unknown (no cap)."""
  if vmem_limit_bytes:
    return int(vmem_limit_bytes * _HYBRID_VMEM_SAFETY)
  return _HYBRID_VMEM_BUDGET_BYTES.get(device_kind)


def max_hybrid_block_q(device_kind, block_kv: int, block_kv_compute: int, vmem_limit_bytes=None, dtype=None):
  """Largest block_q (multiple of 128, at most 2048) whose hybrid VMEM estimate fits the budget.

  Returns `_BLOCK_Q_BASE_MAX` (2048) without a budget (a chip without a
  calibrated limit and no user vmem_limit_bytes). May return less than
  `_BLOCK_Q_MIN` (even 0) when nothing fits; the caller rejects that.
  """
  budget = _hybrid_vmem_budget(device_kind, vmem_limit_bytes)
  if budget is None:
    return _BLOCK_Q_BASE_MAX
  itemsize = 2 if dtype is None else jnp.dtype(dtype).itemsize
  block_q = _BLOCK_Q_BASE_MAX
  while block_q > 0 and _estimated_hybrid_vmem_bytes(block_q, block_kv, block_kv_compute, itemsize) > budget:
    block_q -= NUM_LANES
  return block_q


def max_auto_block_q(
    device_kind,
    block_kv: int,
    block_kv_compute: int,
    block_kv_compute_in: int,
    vmem_limit_bytes=None,
    dtype=None,
    *,
    budget_extension: bool | None = None,
) -> int:
  """Largest block_q the automatic choice may use for these flash kv block sizes.

  Returns `_BLOCK_Q_BASE_MAX` (2048) unless the VMEM-budget extension is on:
  `budget_extension` True / False switches it for this call, None follows
  `AUTO_BLOCK_Q_BUDGET_EXTENSION` (off). With the extension on it still returns
  `_BLOCK_Q_BASE_MAX` for chips without a calibrated budget, when the user sets
  `vmem_limit_bytes`, when block_kv_compute_in < 256 or when the q/k/v `dtype`
  is not bfloat16 (not calibrated; None means bfloat16). Otherwise the largest
  multiple of 128 in [_BLOCK_Q_BASE_MAX, _BLOCK_Q_ABS_MAX] whose estimate fits
  the chip's budget, never below the base. The hybrid variant uses
  `max_hybrid_block_q` instead.
  """
  if budget_extension is None:
    budget_extension = AUTO_BLOCK_Q_BUDGET_EXTENSION
  if not budget_extension:
    return _BLOCK_Q_BASE_MAX
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
  """Picks block_q minimizing padded query rows plus a per-block overhead, preferring larger blocks on ties."""
  cap = padded_len(max(seq_len, 1), NUM_LANES)
  candidates = [bq for bq in range(_BLOCK_Q_MIN, max_block_q + 1, NUM_LANES) if bq <= cap] or [cap]
  best = None
  for bq in candidates:
    num_blocks = (seq_len + bq - 1) // bq
    cost = padded_len(seq_len, bq) + _BLOCK_Q_OVERHEAD_ROWS * num_blocks
    key = (cost, -bq)
    if best is None or key < best[0]:
      best = (key, bq)
  return best[1]


def select_krea2_block_sizes(seq_len: int, user=None, device_kind=None, dtype=None) -> Krea2BlockSizes:
  """Selects the kernel variant and block sizes for a sequence of length `seq_len`.

  Args:
    seq_len: static (unpadded) sequence length.
    user: optional CustomFlashBlockSizes-like object or dict. Its `kernel`
      ("auto" / "flash" / "hybrid", see `parse_kernel_choice`; missing or None
      means "auto") picks the variant via `resolve_kernel_variant`. Its non-None
      block_q / block_kv / block_kv_compute / block_kv_compute_in /
      block_kv_pv / block_q_strip override the variant's defaults for the chip
      (`default_kv_block_sizes`); block_kv_pv / block_q_strip with the flash
      variant raise ValueError. A non-None vmem_limit_bytes keeps the flash
      automatic block_q at most 2048 and sets the hybrid VMEM budget.
    device_kind: `device_kind` of the chip the kernel runs on (e.g.
      'TPU v6 lite'): picks the "auto" variant, the default kv sizes and the
      VMEM budgets; None means unknown (flash, fallback sizes, no budget).
    dtype: dtype of q/k/v. None means bfloat16, the dtype the VMEM budgets are
      calibrated for.

  Returns:
    A validated Krea2BlockSizes. A missing block_q is automatic: the multiple
    of 128 in [512, max] (capped at seq_len rounded up to 128) minimizing the
    padded rows ceil(seq_len / bq) * bq plus `_BLOCK_Q_OVERHEAD_ROWS` (200)
    rows per q block, larger bq winning ties. For flash the max is 2048 unless
    the VMEM-budget extension (`AUTO_BLOCK_Q_BUDGET_EXTENSION`, off by default)
    is switched on; for hybrid it is 2048 capped by `max_hybrid_block_q`, and a
    ValueError names the sizes to lower when not even block_q 512 fits. A user
    block_q is never capped (the compile decides).
  """
  variant = resolve_kernel_variant(_user_block_value(user, "kernel"), device_kind)
  defaults = default_kv_block_sizes(variant, device_kind)
  if variant == "flash":
    for name in ("block_kv_pv", "block_q_strip"):
      if _user_block_value(user, name) is not None:
        raise ValueError(
            f"{name}={_user_block_value(user, name)!r} only applies to the hybrid kernel variant, but the flash variant "
            f"was selected (kernel={_user_block_value(user, 'kernel')!r}, device_kind={device_kind!r})."
        )
  block_kv = _user_block_value(user, "block_kv") or defaults["block_kv"]
  block_kv_compute = _user_block_value(user, "block_kv_compute")
  block_kv_compute_in = _user_block_value(user, "block_kv_compute_in")
  if block_kv_compute is None:
    block_kv_compute = min(defaults["block_kv_compute"], block_kv)
  if block_kv_compute_in is None:
    block_kv_compute_in = min(defaults["block_kv_compute_in"], block_kv_compute)
  hybrid = {}
  if variant == "hybrid":
    block_kv_pv = _user_block_value(user, "block_kv_pv")
    if block_kv_pv is None:
      block_kv_pv = min(defaults["block_kv_pv"], block_kv_compute_in)
    hybrid = {
        "block_kv_pv": block_kv_pv,
        "block_q_strip": _user_block_value(user, "block_q_strip") or defaults["block_q_strip"],
    }
  block_q = _user_block_value(user, "block_q")
  vmem_limit_bytes = _user_block_value(user, "vmem_limit_bytes")
  if not block_q:
    if variant == "flash":
      max_block_q = max_auto_block_q(
          device_kind, block_kv, block_kv_compute, block_kv_compute_in, vmem_limit_bytes, dtype=dtype
      )
    else:
      max_block_q = max_hybrid_block_q(device_kind, block_kv, block_kv_compute, vmem_limit_bytes, dtype=dtype)
      if max_block_q < _BLOCK_Q_MIN:
        budget = _hybrid_vmem_budget(device_kind, vmem_limit_bytes)
        raise ValueError(
            f"The hybrid Krea 2 attention kernel does not fit block_q {_BLOCK_Q_MIN} into the VMEM budget of "
            f"{budget} bytes on {device_kind!r} with block_kv={block_kv}, block_kv_compute={block_kv_compute} "
            f"(estimate {_estimated_hybrid_vmem_bytes(_BLOCK_Q_MIN, block_kv, block_kv_compute)} bytes): lower "
            "block_kv_compute (its f32 (block_kv_compute, block_q) QK^T scratch dominates) and/or block_kv, raise "
            "vmem_limit_bytes, set block_q explicitly, or use krea2_attention_kernel=flash."
        )
    block_q = _default_block_q(seq_len, max_block_q)
  return Krea2BlockSizes(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_kv_compute_in=block_kv_compute_in,
      variant=variant,
      **hybrid,
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


def _krea2_hybrid_kernel(
    valid_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_scratch_ref,
    o_scratch_ref,
    vt_scratch_ref,
    qk_scratch_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    bkv_pv: int,
    strip: int,
    kv_seq_len: int,
    use_base2_exp: bool,
    interpret: bool,
):
  """Hybrid variant: explicit QK^T scratch, lane strips, l-sum on the MXU (see the module docstring)."""
  float32, p_dtype = jnp.float32, q_ref.dtype
  b, j = pl.program_id(0), pl.program_id(3)
  valid_len = valid_ref[b]
  exp = jnp.exp2 if use_base2_exp else jnp.exp

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    # Row 128 of v^T_ext is ones (its P.V row is l), rows 129..135 zeros. The
    # scratch persists over the sequential kv grid axis and only rows 0..127
    # are rewritten per kv block.
    row = lax.broadcasted_iota(jnp.int32, (_EXT_ROWS - _HEAD_DIM, bkv), 0)
    vt_scratch_ref[_HEAD_DIM:_EXT_ROWS, :] = jnp.where(row == 0, 1.0, 0.0).astype(vt_scratch_ref.dtype)

  strips = [(s0, min(strip, bq - s0)) for s0 in range(0, bq, strip)]

  def pv(kv_start, length, p):
    vt = vt_scratch_ref[:, pl.ds(kv_start, length)]
    if interpret:  # XLA:CPU has no bf16 x bf16 -> f32 NN dot; bf16 products are exact in f32.
      vt, p = vt.astype(float32), p.astype(float32)
    return jnp.dot(vt, p, preferred_element_type=float32)  # (136, w): rows 0..127 P.V, row 128 sum(P)

  def online(qk, kv_start, m_prev, o_prev):
    """qk (rows, w) f32 (masked): one running-max update, exp + P.V per bkv_pv rows."""
    m_next = jnp.maximum(m_prev, qk.max(axis=0)[None, :])
    alpha = exp(m_prev - m_next)
    o_next = alpha[0:1] * o_prev
    for p0 in range(0, qk.shape[0], bkv_pv):
      length = min(bkv_pv, qk.shape[0] - p0)
      p = exp(qk[p0 : p0 + length] - m_next[0:1]).astype(p_dtype)
      o_next = o_next + pv(kv_start + p0, length, p)
    return m_next, o_next

  def compute(c0, length, masked):
    qk = lax.dot_general(k_ref[pl.ds(c0, length), :], q_ref[...], NT_DIM_NUMBERS, preferred_element_type=float32)
    if masked:
      # Transposed layout: kv positions live on rows (sublanes).
      kv_pos = j * bkv + c0 + lax.broadcasted_iota(jnp.int32, (length, 1), 0)
      qk = jnp.where(kv_pos < valid_len, qk, mask_value)
    qk_scratch_ref[pl.ds(0, length), :] = qk
    for s0, w in strips:
      lanes = pl.ds(s0, w)
      m, o = m_scratch_ref[:, lanes], o_scratch_ref[:, lanes]
      for i in range(0, length, bkv_compute_in):
        rows = min(bkv_compute_in, length - i)
        m, o = online(qk_scratch_ref[pl.ds(i, rows), lanes], c0 + i, m, o)
      m_scratch_ref[:, lanes], o_scratch_ref[:, lanes] = m, o

  def run_block(block_len: int, masked: bool):
    """Processes `block_len` (static) kv rows of the current kv block."""
    vt_scratch_ref[0:_HEAD_DIM, pl.ds(0, block_len)] = v_ref[pl.ds(0, block_len), :].T
    for c0 in range(0, block_len, bkv_compute):
      compute(c0, min(bkv_compute, block_len - c0), masked)

  # Same block classification as the flash variant.
  block_end = jnp.minimum((j + 1) * bkv, kv_seq_len)
  needs_mask = block_end > valid_len
  last_block_len = kv_seq_len - (grid_width - 1) * bkv
  is_last = j == grid_width - 1

  if grid_width > 1:

    @pl.when(jnp.logical_not(is_last) & jnp.logical_not(needs_mask))
    def _body():
      run_block(bkv, masked=False)

    @pl.when(jnp.logical_not(is_last) & needs_mask)
    def _body_masked():
      run_block(bkv, masked=True)

  @pl.when(is_last & jnp.logical_not(needs_mask))
  def _last_body():
    run_block(last_block_len, masked=False)

  @pl.when(is_last & needs_mask)
  def _last_body_masked():
    run_block(last_block_len, masked=True)

  @pl.when(is_last)
  def end():
    l_inv = 1.0 / o_scratch_ref[_HEAD_DIM : _HEAD_DIM + 1, :]
    o_ref[...] = (o_scratch_ref[0:_HEAD_DIM, :] * l_inv).astype(o_ref.dtype)


def _krea2_hybrid_forward(
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

  assert block_sizes.variant == "hybrid", f"{block_sizes.variant=}"
  assert head_dim_qk == _HEAD_DIM and head_dim_v == _HEAD_DIM, f"head_dim must be 128, got {head_dim_qk=} {head_dim_v=}"
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
      pltpu.VMEM((_EXT_ROWS, bq), jnp.float32),  # o_ext: rows 0..127 o, row 128 l
      pltpu.VMEM((_EXT_ROWS, bkv), v.dtype),  # v^T of the kv block + ones row
      pltpu.VMEM((block_sizes.block_kv_compute, bq), jnp.float32),  # QK^T of one chunk
  ]
  grid_width = (kv_seq_len + bkv - 1) // bkv
  grid_height = (q_seq_len + bq - 1) // bq
  grid = (batch, num_q_heads, grid_height, grid_width)

  return pl.pallas_call(
      functools.partial(
          _krea2_hybrid_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bq=bq,
          bkv=bkv,
          bkv_compute=block_sizes.block_kv_compute,
          bkv_compute_in=block_sizes.block_kv_compute_in,
          bkv_pv=block_sizes.block_kv_pv,
          strip=block_sizes.block_q_strip,
          kv_seq_len=kv_seq_len,
          use_base2_exp=use_base2_exp,
          interpret=interpret,
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
  `block_sizes.variant` picks the kernel ("flash" or "hybrid"); both take and
  return the same arrays.
  """
  forward = _krea2_hybrid_forward if block_sizes.variant == "hybrid" else _krea2_attention_forward

  def _fn(q, k, v, valid_kv_len):
    return forward(
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
