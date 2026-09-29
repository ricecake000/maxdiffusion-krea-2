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

# int8 weight-only residency for the Krea 2 text encoder (Qwen3-VL text tower).
#
# Only the seven projection kernels of each decoder layer are quantized (qwix
# PTQ, int8 values with one scale per `tile_size` input rows and output
# column); RMSNorm scales stay in their loaded dtype and the attention einsums
# are activation-only. Dequantization happens inside the jitted forward right
# before each matmul, so XLA fuses it into the dot prologue.

import math
from typing import Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging

TEXT_ENCODER_QUANT_MODULE_PATH = r"layers_\d+/(self_attn/(q_proj|k_proj|v_proj|o_proj)|mlp/(gate_proj|up_proj|down_proj))"
TEXT_ENCODER_QUANTIZATION_MODES = ("", "int8")
# Bump when `quantize_text_encoder_params` changes the stored values; it is part of the weight cache key
# (krea2_weight_cache_dir), so this invalidates cached quantized trees.
KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION = 1

_GIB = 1024**3


def resolve_text_encoder_quantization(config):
  """Returns `(mode, tile_size, embed_on_host)` from a pyconfig; unset keys mean off."""
  # Strip quotes so a command-line override `krea2_text_encoder_quantization=''` means off.
  mode = str(getattr(config, "krea2_text_encoder_quantization", "") or "").strip().strip("'\"").lower()
  if mode in ("none", "bf16", "bfloat16"):
    mode = ""
  if mode not in TEXT_ENCODER_QUANTIZATION_MODES:
    raise ValueError(
        f"krea2_text_encoder_quantization={mode!r} is not supported; allowed: {list(TEXT_ENCODER_QUANTIZATION_MODES)}."
    )
  tile_size = int(getattr(config, "krea2_text_encoder_quant_tile_size", 128) or 128)
  embed_on_host = bool(getattr(config, "krea2_text_embed_on_host", False))
  return mode, tile_size, embed_on_host


def describe_text_encoder_residency(mode, tile_size, embed_on_host):
  weights = f"int8 weight-only (tile {tile_size})" if mode == "int8" else "unquantized"
  embedding = "embedding on host" if embed_on_host else "embedding on device"
  return f"text encoder: {weights}, {embedding}"


def build_text_encoder_rules(tile_size: int = 128):
  import qwix

  return [
      qwix.QtRule(
          module_path=TEXT_ENCODER_QUANT_MODULE_PATH,
          weight_qtype=jnp.int8,
          act_qtype=None,
          op_names=("dot_general",),
          tile_size=tile_size,
      )
  ]


def quantize_text_encoder_model(model: nn.Module, tile_size: int = 128) -> nn.Module:
  """Wraps a linen FlaxQwen3Model so its projection matmuls take int8 PTQ weights.

  `init`/`eval_shape` of the returned module yield the quantized parameter
  structure (qwix `WithAux(QArray(qvalue, scale))` leaves); `apply` expects a
  tree produced by `quantize_text_encoder_params`.
  """
  import qwix

  return qwix.quantize_model(model, qwix.PtqProvider(build_text_encoder_rules(tile_size)))


def tree_nbytes(tree) -> int:
  return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def _is_quantized_leaf(abstract_leaf) -> bool:
  from qwix._src.providers.ptq import WithAux

  return isinstance(abstract_leaf, WithAux)


def _abstract_at(tree, path):
  for key in path:
    if not isinstance(tree, (dict, flax.core.FrozenDict)) or key not in tree:
      return None
    tree = tree[key]
  return tree


def _param_paths(tree):
  """Parameter paths of a nested dict tree. A path ends at the first non-dict
  node: an array, a ShapeDtypeStruct, a qwix `WithAux` or a flax box."""
  return set(flax.traverse_util.flatten_dict(flax.core.unfreeze(tree)).keys())


def _check_param_paths(host_params, abstract):
  """Raises ValueError if the host tree and the abstract tree hold different parameters."""
  host_paths = _param_paths(nn.unbox(host_params))
  abstract_paths = _param_paths(abstract)
  missing = sorted(abstract_paths - host_paths)
  unexpected = sorted(host_paths - abstract_paths)
  if not missing and not unexpected:
    return

  def fmt(paths):
    shown = ", ".join("/".join(str(k) for k in path) for path in paths[:10])
    return f"{shown}, ... ({len(paths)} total)" if len(paths) > 10 else shown

  lines = ["Text encoder params do not match the quantized model's parameter tree."]
  if missing:
    lines.append(f"Missing from the loaded params: {fmt(missing)}.")
  if unexpected:
    lines.append(f"Not in the quantized model's parameter tree: {fmt(unexpected)}.")
  if any(path[0] == "embed_tokens" for path in unexpected):
    lines.append(
        "The params include `embed_tokens` but the model was initialized from embeddings "
        "(krea2_text_embed_on_host=True): drop the table from the params or build the model with "
        "krea2_text_embed_on_host=False."
    )
  if any(path[0] == "embed_tokens" for path in missing):
    lines.append(
        "The model expects `embed_tokens` (krea2_text_embed_on_host=False) but the params do not have it: load the "
        "table or build the model with krea2_text_embed_on_host=True."
    )
  raise ValueError(" ".join(lines))


def quantize_text_encoder_params(
    host_params, abstract_quantized_params, scale_dtype: Optional[jnp.dtype] = None, device=None
):
  """Quantizes a loaded (unquantized) host tree into the qwix PTQ structure.

  Works one top-level child (`layers_i`, `norm`, ...) at a time: the child's
  projection kernels are cast to float32 on the host CPU device and quantized
  with `qwix.quantize_params`, which bounds the transient float32 copy to one
  decoder layer. Leaves that are not quantized (RMSNorm scales, and the
  embedding table if present) are kept exactly as loaded.

  Args:
    host_params: unquantized params (numpy or jax leaves), structured like the
      unquantized FlaxQwen3Model tree.
    abstract_quantized_params: `eval_shape(quantized_model.init)["params"]`
      (boxed or unboxed).
    scale_dtype: dtype to store the per-tile scales in. The scale dtype is the
      dequantized weight's dtype, so pass the model's compute dtype (bf16 in
      production) to keep the matmuls and activations in that dtype; None keeps
      the float32 scales.
    device: device used for the quantization math (default: first CPU device).

  Returns:
    The quantized host tree (numpy leaves) matching `abstract_quantized_params`.

  Raises:
    ValueError: if the two trees do not hold the same parameter paths.
  """
  import qwix

  if device is None:
    device = jax.local_devices(backend="cpu")[0]
  abstract = flax.core.unfreeze(abstract_quantized_params)
  host_params = flax.core.unfreeze(host_params)
  _check_param_paths(host_params, abstract)
  bytes_before = tree_nbytes(host_params)
  num_quantized = 0
  result = {}
  for name, child in host_params.items():
    if name not in abstract:
      raise ValueError(f"Text encoder param '{name}' is not in the quantized model's parameter tree.")
    if not isinstance(child, dict):
      result[name] = child
      continue
    flat = flax.traverse_util.flatten_dict(child)
    to_quantize, kept = {}, {}
    for path, leaf in flat.items():
      if _is_quantized_leaf(_abstract_at(abstract, (name, *path))):
        to_quantize[(name, *path)] = jax.device_put(np.asarray(leaf, dtype=np.float32), device)
      else:
        kept[path] = leaf
    quantized_flat = {}
    if to_quantize:
      quantized = qwix.quantize_params(flax.traverse_util.unflatten_dict(to_quantize), abstract)
      quantized = jax.tree_util.tree_map(np.asarray, quantized)
      if scale_dtype is not None:
        quantized = _cast_scales(quantized, scale_dtype)
      num_quantized += len(to_quantize)
      # flatten_dict stops at the (non-dict) WithAux nodes.
      quantized_flat = {path[1:]: leaf for path, leaf in flax.traverse_util.flatten_dict(quantized).items()}
    del to_quantize
    result[name] = flax.traverse_util.unflatten_dict({**kept, **quantized_flat})
  bytes_after = tree_nbytes(result)
  max_logging.log(
      f"Quantized {num_quantized} text encoder projection kernels to int8: {bytes_before / _GIB:.2f} GiB -> "
      f"{bytes_after / _GIB:.2f} GiB ({bytes_after / max(bytes_before, 1):.3f}x)"
  )
  return result


def _cast_scales(quantized_tree, scale_dtype):
  from qwix._src.providers.ptq import WithAux

  def cast(leaf):
    if not isinstance(leaf, WithAux):
      return leaf
    qarr = leaf.array
    return leaf.replace(array=qarr.replace(scale=np.asarray(qarr.scale).astype(scale_dtype)))

  return jax.tree_util.tree_map(cast, quantized_tree, is_leaf=lambda x: isinstance(x, WithAux))


def safe_param_shardings(abstract_params, shardings, mesh):
  """Replaces shardings that cannot hold their leaf (non-NamedSharding, or a dim
  not divisible by its mesh axes) with a replicated `NamedSharding(mesh, P())`.

  Both trees must be unboxed. Needed for qwix scale leaves, whose tiled input
  dim (in_features / tile_size) inherits the kernel's logical axes.
  """
  replicated = NamedSharding(mesh, P())
  fallbacks = []

  def fix(path, leaf, sharding):
    if not isinstance(sharding, NamedSharding):
      fallbacks.append(jax.tree_util.keystr(path))
      return replicated
    for dim, axes in zip(leaf.shape, sharding.spec):
      if axes is None:
        continue
      axes = axes if isinstance(axes, tuple) else (axes,)
      if dim % math.prod(mesh.shape[a] for a in axes) != 0:
        fallbacks.append(jax.tree_util.keystr(path))
        return replicated
    return sharding

  fixed = jax.tree_util.tree_map_with_path(fix, abstract_params, shardings)
  if fallbacks:
    max_logging.log(f"Text encoder: {len(fallbacks)} param leaves fall back to replicated sharding (e.g. {fallbacks[0]}).")
  return fixed
