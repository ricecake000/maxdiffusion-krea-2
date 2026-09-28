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

# int8 W8A8 matmuls for the Krea 2 transformer's DiT block projections.
#
# Weights are quantized once on the host at load time (symmetric absmax, one
# scale per output column of the flax `(in, out)` kernel); activations are
# quantized inside the jitted step (symmetric absmax, one scale per token).
# The matmul runs as int8 x int8 -> int32 and is rescaled by both scales.
# Only `blocks_*` projections are ever quantized; everything else stays float.
# This module must not import `transformer_krea2_flax` (which imports it).

import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from maxdiffusion import max_logging
from .text_encoder_quant import tree_nbytes

KREA2_TRANSFORMER_QUANTIZATION_MODES = ("", "w8a8")
KREA2_QUANT_ATTN_TARGETS = ("to_q", "to_k", "to_v", "to_gate", "to_out")
KREA2_QUANT_FF_TARGETS = ("gate_proj", "up_proj", "down_proj")
KREA2_QUANT_TARGETS = KREA2_QUANT_ATTN_TARGETS + KREA2_QUANT_FF_TARGETS
KREA2_DEFAULT_QUANT_TARGETS = ("to_q", "to_gate", "to_out", "gate_proj", "up_proj", "down_proj")
# Bump when the traced W8A8 graph changes; it is part of the AOT cache key, so this invalidates cached executables.
KREA2_TRANSFORMER_QUANT_REVISION = 2

_BLOCK_KEY = re.compile(r"blocks_\d+")
_GIB = 1024**3


def normalize_quant_targets(targets) -> Tuple[str, ...]:
  """Returns the de-duplicated target names in the canonical `KREA2_QUANT_TARGETS` order.

  Accepts None, a list/tuple of names, or a comma-separated string (whitespace,
  quotes and brackets are stripped), e.g. for configs built in code. A pyconfig
  command-line override of the list-valued yml key goes through `ast.literal_eval`
  first and must be a Python list literal:
  `"krea2_transformer_quant_targets=['to_q','gate_proj']"` (a bare `to_q,gate_proj` raises there).
  """
  if targets is None:
    return ()
  if isinstance(targets, str):
    targets = targets.strip().strip("[]()").split(",")
  names = [str(name).strip().strip("'\"").strip() for name in targets]
  names = [name for name in names if name]
  unknown = [name for name in names if name not in KREA2_QUANT_TARGETS]
  if unknown:
    raise ValueError(
        f"Unknown Krea 2 transformer quantization target(s) {unknown}; allowed: {list(KREA2_QUANT_TARGETS)}."
    )
  return tuple(name for name in KREA2_QUANT_TARGETS if name in names)


def resolve_transformer_quantization(config):
  """Returns `(mode, targets)` from a pyconfig; an unset mode means off (`("", ())`)."""
  # Strip quotes so a command-line override `krea2_transformer_quantization=''` means off.
  mode = str(getattr(config, "krea2_transformer_quantization", "") or "").strip().strip("'\"").lower()
  if mode in ("none", "bf16", "bfloat16"):
    mode = ""
  if mode not in KREA2_TRANSFORMER_QUANTIZATION_MODES:
    raise ValueError(
        f"krea2_transformer_quantization={mode!r} is not supported; "
        f"allowed: {list(KREA2_TRANSFORMER_QUANTIZATION_MODES)}."
    )
  if not mode:
    return "", ()
  raw_targets = getattr(config, "krea2_transformer_quant_targets", None)
  if raw_targets is None:
    return mode, KREA2_DEFAULT_QUANT_TARGETS
  targets = normalize_quant_targets(raw_targets)
  if not targets:
    raise ValueError(
        f"krea2_transformer_quantization={mode!r} needs at least one krea2_transformer_quant_targets entry; "
        f"allowed: {list(KREA2_QUANT_TARGETS)}."
    )
  return mode, targets


def describe_transformer_quantization(mode, targets) -> str:
  if mode == "w8a8" and targets:
    return f"transformer: int8 W8A8 matmuls ({', '.join(targets)}); other projections unquantized"
  return "transformer: unquantized"


def quantize_activation(x, scale_dtype):
  """Symmetric absmax int8 quantization with one scale per token (last axis).

  Returns `(x_q, scale)`: int8 values in [-127, 127] and a `(..., 1)` scale in
  `scale_dtype`. The division uses the scale as rounded to `scale_dtype`, so
  `x_q * scale` dequantizes consistently; an all-zero token gets scale 1.
  """
  x_f32 = x.astype(jnp.float32)
  absmax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
  scale = (absmax / 127.0).astype(scale_dtype)
  # After the cast: a tiny absmax can round to zero in bf16.
  scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
  x_q = jnp.clip(jnp.round(x_f32 / scale.astype(jnp.float32)), -127, 127).astype(jnp.int8)
  return x_q, scale


class Krea2QuantDense(nn.Module):
  """Bias-free W8A8 linear projection: int8 `kernel` `(in, features)` with a
  float `kernel_scale` per output column, int8 x int8 -> int32 matmul. The
  int32 result is rescaled in `dtype`, or in float32 when `dtype` cannot hold
  the largest accumulator (float16), then cast to `dtype`.

  `features`, `dtype`, `param_dtype` and `precision` mirror `nn.Dense` so the
  Krea 2 LoRA interceptor can build its layer from this module; `param_dtype`
  is the float dtype of `kernel_scale` (the kernel itself is always int8) and
  `precision` only matters to that LoRA layer (the integer matmul is exact).

  `unflatten` is the feature layout the caller reshapes the output to (e.g.
  `(num_heads, head_dim)`); the rescale then runs in that layout so the matmul
  can emit it directly. The output is still `(..., features)`.
  """

  features: int
  kernel_axes: Tuple[str, str]
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  unflatten: Optional[Tuple[int, ...]] = None

  @nn.compact
  def __call__(self, inputs, quantized_inputs=None):
    """`quantized_inputs` is the `(x_q, x_scale)` pair of `quantize_activation(inputs, dtype)`,
    passed in when several projections share one activation; None quantizes `inputs` here."""
    kernel = self.param(
        "kernel",
        nn.with_logical_partitioning(nn.initializers.zeros_init(), self.kernel_axes),
        (inputs.shape[-1], self.features),
        jnp.int8,
    )
    # Unboxed, so the per-column scale is replicated.
    kernel_scale = self.param("kernel_scale", nn.initializers.ones, (self.features,), self.param_dtype)
    if quantized_inputs is None:
      quantized_inputs = quantize_activation(inputs, self.dtype)
    x_q, x_scale = quantized_inputs
    acc = jax.lax.dot_general(
        x_q, kernel, (((x_q.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.int32
    )
    # Rescale in the activation dtype (fuses into the matmul on TPU; the benchmarked bf16 form),
    # unless it cannot hold the largest int32 accumulator (float16: max 65504), then in float32.
    # Trace-time decision on static shapes/dtypes.
    max_acc = 127 * 127 * inputs.shape[-1]
    rescale_dtype = self.dtype if float(jnp.finfo(self.dtype).max) > max_acc else jnp.float32
    if not self.unflatten:  # None or () (no layout)
      out =acc.astype(rescale_dtype) * x_scale.astype(rescale_dtype) * kernel_scale.astype(rescale_dtype)
      return out.astype(self.dtype)
    # Same arithmetic in the caller's layout, so its reshape folds into the matmul instead of
    # following a flat rescaled tensor; the final reshape cancels against the caller's.
    layout = tuple(self.unflatten)
    if math.prod(layout) != self.features:
      raise ValueError(f"unflatten={layout} has {math.prod(layout)} elements, but features={self.features}.")
    acc = acc.reshape(acc.shape[:-1] + layout)
    x_scale = x_scale.reshape(x_scale.shape + (1,) * (len(layout) - 1))
    kernel_scale = kernel_scale.reshape(layout)
    out = acc.astype(rescale_dtype) * x_scale.astype(rescale_dtype) * kernel_scale.astype(rescale_dtype)
    return out.astype(self.dtype).reshape(out.shape[: -len(layout)] + (self.features,))


def quantize_kernel(kernel, scale_dtype):
  """Quantizes a 2-D float `(in, out)` kernel to int8 with one absmax scale per output column.

  Host numpy only. Returns `(int8 kernel, scale)` with `scale.shape == (out,)`
  and `scale.dtype == scale_dtype`; a zero column gets scale 1.
  """
  kernel = np.asarray(kernel)
  if kernel.ndim != 2 or not jnp.issubdtype(kernel.dtype, jnp.floating):
    raise ValueError(f"Expected a 2-D float kernel, got shape {kernel.shape} and dtype {kernel.dtype}.")
  scale_dtype = jnp.dtype(scale_dtype)
  w = kernel.astype(np.float32)
  absmax = np.abs(w).max(axis=0)
  if not np.all(np.isfinite(absmax)):
    raise ValueError("Cannot quantize a kernel with non-finite values.")
  scale = (absmax / 127).astype(scale_dtype)
  scale[scale == 0] = 1
  w /= scale.astype(np.float32)
  np.rint(w, out=w)
  np.clip(w, -127, 127, out=w)
  return w.astype(np.int8), scale


def _plain_dict(tree):
  """Copies the dict structure of a nested dict / FrozenDict; leaves are shared."""
  if isinstance(tree, (dict, flax.core.FrozenDict)):
    return {key: _plain_dict(value) for key, value in tree.items()}
  return tree


def quantize_transformer_params(host_params, targets, scale_dtype, num_workers=None):
  """Quantizes the target projection kernels of every `blocks_*` group to int8.

  Args:
    host_params: the loaded float transformer tree (nested dict or FrozenDict,
      numpy or jax leaves), including any `lora-*` subtrees and the rotate-half
      permutation.
    targets: projection names (see `normalize_quant_targets`); attention
      targets live under `blocks_i/attn`, feed-forward targets under `blocks_i/ff`.
    scale_dtype: dtype of the per-column `kernel_scale` (the model's weights dtype).
    num_workers: quantization threads (default `min(8, os.cpu_count())`).

  Returns:
    A new plain nested dict in which each target's `kernel` is int8 and a
    `kernel_scale` is added; every other leaf (incl. `lora-*` subtrees) is
    shared with the input, which is not modified.

  Raises:
    ValueError: no `blocks_*` group, or a missing projection/kernel, or a
      kernel that is already integer.
  """
  targets = normalize_quant_targets(targets)
  result = _plain_dict(host_params)
  if not targets:
    return result
  block_keys = sorted((key for key in result if _BLOCK_KEY.fullmatch(str(key))), key=lambda k: int(k.split("_")[1]))
  if not block_keys:
    raise ValueError("Transformer params have no blocks_* groups to quantize.")

  projections = []
  for block_key in block_keys:
    for name in targets:
      group = "attn" if name in KREA2_QUANT_ATTN_TARGETS else "ff"
      path = f"{block_key}/{group}/{name}"
      proj = result[block_key].get(group, {}).get(name)
      if not isinstance(proj, dict) or "kernel" not in proj:
        raise ValueError(f"Transformer params have no kernel at {path}/kernel to quantize.")
      if not jnp.issubdtype(proj["kernel"].dtype, jnp.floating):
        raise ValueError(f"{path}/kernel has dtype {proj['kernel'].dtype}; it is already quantized or not a float.")
      projections.append(proj)

  bytes_before = tree_nbytes(host_params)
  start = time.time()
  num_workers = num_workers or min(8, os.cpu_count() or 1)
  # numpy releases the GIL in the elementwise ops, so threads quantize in parallel.
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    quantized = list(executor.map(lambda proj: quantize_kernel(proj["kernel"], scale_dtype), projections))
  for proj, (kernel, scale) in zip(projections, quantized):
    proj["kernel"] = kernel
    proj["kernel_scale"] = scale
  bytes_after = tree_nbytes(result)
  max_logging.log(
      f"Quantized {len(projections)} transformer projection kernels to int8 ({', '.join(targets)}): "
      f"{bytes_before / _GIB:.2f} GiB -> {bytes_after / _GIB:.2f} GiB in {time.time() - start:.1f} s"
  )
  return result


def check_transformer_param_tree(host_params, abstract_params) -> None:
  """Raises ValueError unless the host tree has exactly the abstract tree's paths, shapes and dtypes.

  `abstract_params` is the runtime model's `jax.eval_shape(init)["params"]`
  (boxed or unboxed, dict or FrozenDict).
  """
  host = flax.traverse_util.flatten_dict(_plain_dict(nn.unbox(host_params)))
  abstract = flax.traverse_util.flatten_dict(_plain_dict(nn.unbox(abstract_params)))

  def fmt_path(path):
    return "/".join(str(key) for key in path)

  missing = sorted(set(abstract) - set(host))
  unexpected = sorted(set(host) - set(abstract))
  shape_errors, dtype_errors = [], []
  for path in sorted(set(host) & set(abstract)):
    expected, actual = abstract[path], host[path]
    if tuple(expected.shape) != tuple(np.shape(actual)):
      shape_errors.append(f"{fmt_path(path)} (expected {tuple(expected.shape)}, got {tuple(np.shape(actual))})")
    elif jnp.dtype(expected.dtype) != jnp.dtype(actual.dtype):
      dtype_errors.append(f"{fmt_path(path)} (expected {jnp.dtype(expected.dtype)}, got {jnp.dtype(actual.dtype)})")

  def fmt(items):
    shown = ", ".join(items[:10])
    return f"{shown}, ... ({len(items)} total)" if len(items) > 10 else shown

  lines = []
  if missing:
    lines.append(f"Missing from the params: {fmt([fmt_path(path) for path in missing])}.")
  if unexpected:
    lines.append(f"Not in the model's parameter tree: {fmt([fmt_path(path) for path in unexpected])}.")
  if shape_errors:
    lines.append(f"Wrong shape: {fmt(shape_errors)}.")
  if dtype_errors:
    lines.append(f"Wrong dtype: {fmt(dtype_errors)}.")
  if lines:
    raise ValueError(" ".join(["Transformer params do not match the model's parameter tree."] + lines))
