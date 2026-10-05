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

# Utilities for Krea 2 (K2): HuggingFace diffusers checkpoint conversion,
# timestep-shift computation and rotary position-id helpers.

import gc
import json
import os
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import max_logging
from maxdiffusion.safetensors_utils import SafetensorsShardReader
from ..flux.util import validate_flax_state_dict

# Default hidden-state taps into the Qwen3-VL-4B text encoder (0 is the
# embedding output), matching the Krea 2 reference pipeline.
KREA2_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Qwen-Image chat template used by Krea 2 for prompt conditioning. Prompts are
# tokenized as a fixed-length block `[prefix | prompt | PAD | suffix]` and the
# first `KREA2_PROMPT_TEMPLATE_START_IDX` (system prefix) tokens are dropped
# from the encoder outputs.
KREA2_PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
KREA2_PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
KREA2_PROMPT_TEMPLATE_START_IDX = 34
KREA2_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS = 5


def _is_extra_special_tokens_error(err: BaseException) -> bool:
  """True only for the known transformers 4.x incompatibility with the Krea 2
  tokenizer_config.json: its `extra_special_tokens` is a list, which
  `_set_model_specific_special_tokens` treats as a dict and fails with
  `AttributeError: 'list' object has no attribute 'keys'`.

  Requires both the AttributeError message and a traceback frame whose name
  contains `special_tokens`, so genuine special-token validation errors (e.g.
  transformers' "Special token ... has to be either str or AddedToken"
  TypeError) are not mistaken for it and still propagate.
  """
  if not isinstance(err, AttributeError) or "'list' object has no attribute" not in str(err):
    return False
  return any("special_tokens" in frame.name for frame in traceback.extract_tb(err.__traceback__))


def _config_has_list_extra_special_tokens(*tokenizer_dirs) -> bool:
  """Whether the first readable `tokenizer_config.json` among `tokenizer_dirs` stores `extra_special_tokens` as a list.

  False when none can be read or parsed: the load then runs as before (with the
  retry on the known error).
  """
  for directory in tokenizer_dirs:
    if not directory:
      continue
    try:
      with open(os.path.join(directory, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    except (OSError, ValueError):
      continue
    return isinstance(tokenizer_config, dict) and isinstance(tokenizer_config.get("extra_special_tokens"), list)
  return False


def _transformers_major_version(transformers_module) -> int:
  try:
    return int(str(transformers_module.__version__).split(".", 1)[0])
  except (AttributeError, ValueError):
    return 0


def load_krea2_tokenizer(tokenizer_path: str, snapshot_dir: str = None):
  """Loads the Krea 2 (Qwen) tokenizer from `tokenizer_path`, falling back to
  `snapshot_dir`'s `tokenizer` subfolder.

  transformers < 5 expects a dict for `extra_special_tokens`; the Krea 2
  tokenizer_config.json stores a list, which fails with
  `AttributeError: 'list' object has no attribute 'keys'` inside
  `_set_model_specific_special_tokens`. Those tokens are already special added
  tokens in tokenizer.json, so on exactly that error the load is retried once
  with `extra_special_tokens={}`, which does not change tokenization. Any other
  error propagates (after the `snapshot_dir` subfolder fallback, if given).

  The tokenizer is built once: when tokenizer_config.json (read first) has a
  list there and transformers is older than 5, the first load already passes
  `extra_special_tokens={}`; the retry stays for a config that could not be read.
  """
  import transformers  # pylint: disable=import-outside-toplevel
  from transformers import AutoTokenizer

  def load(**kwargs):
    try:
      return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, **kwargs)
    except Exception as err:  # pylint: disable=broad-except
      if snapshot_dir is None or _is_extra_special_tokens_error(err):
        raise
      return AutoTokenizer.from_pretrained(snapshot_dir, subfolder="tokenizer", local_files_only=True, **kwargs)

  snapshot_tokenizer_dir = os.path.join(snapshot_dir, "tokenizer") if snapshot_dir is not None else None
  initial_kwargs = {}
  if _transformers_major_version(transformers) < 5 and _config_has_list_extra_special_tokens(
      tokenizer_path, snapshot_tokenizer_dir
  ):
    initial_kwargs = {"extra_special_tokens": {}}
  try:
    return load(**initial_kwargs)
  except AttributeError as err:
    if initial_kwargs or not _is_extra_special_tokens_error(err):
      raise
    max_logging.log(
        "Warning: this transformers version expects a dict for the tokenizer's `extra_special_tokens` but "
        f"tokenizer_config.json has a list ({err}); retrying with extra_special_tokens={{}} "
        "(those tokens are already special tokens in tokenizer.json, so tokenization is unchanged)."
    )
    return load(extra_special_tokens={})


def round_up_to_multiple(value: int, multiple: int) -> int:
  """Rounds `value` up to the nearest multiple of `multiple`."""
  return ((value + multiple - 1) // multiple) * multiple


def calculate_krea2_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 6400,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
  """Resolution-aware exponential time-shift parameter (mu) for the Krea 2 base
  (midtrain) checkpoint. The distilled (Turbo) checkpoint uses a fixed mu=1.15."""
  m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
  b = base_shift - m * base_seq_len
  return float(image_seq_len * m + b)


def prepare_krea2_text_ids(batch_size: int, seq_len: int):
  """Text tokens sit at the rotary origin: (batch, seq_len, 3) of zeros."""
  text_ids = jnp.zeros((seq_len, 3), dtype=jnp.float32)
  return jnp.tile(text_ids[None, ...], (batch_size, 1, 1))


def prepare_krea2_image_ids(batch_size: int, grid_height: int, grid_width: int):
  """Image tokens carry their `(0, h, w)` latent-grid coordinates: (batch, h*w, 3)."""
  grid = jnp.zeros((grid_height, grid_width, 3), dtype=jnp.float32)
  grid = grid.at[..., 1].set(jnp.arange(grid_height)[:, None])
  grid = grid.at[..., 2].set(jnp.arange(grid_width)[None, :])
  image_ids = grid.reshape(-1, 3)
  return jnp.tile(image_ids[None, ...], (batch_size, 1, 1))


# Bump when `permute_rope_weights_to_rotate_half` changes the permuted values; it is part of the weight
# cache key (krea2_weight_cache_dir), so this invalidates cached permuted trees.
KREA2_ROPE_PERMUTATION_REVISION = 1


def rotate_half_permutation(head_dim: int) -> np.ndarray:
  """Per-head index map `new = old[perm]` from interleaved to rotate-half order:
  `new[i] = old[2i]` and `new[i + head_dim/2] = old[2i + 1]` for i in [0, head_dim/2)."""
  if head_dim % 2:
    raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
  return np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])


def permute_rope_weights_to_rotate_half(params, num_heads: int, num_kv_heads: int, head_dim: int):
  """Reorders q/k head dimensions so rotate-half RoPE reproduces interleaved RoPE.

  For every `blocks_*/attn` (the only attention with RoPE; text-fusion
  attention is left alone) this returns a new tree in which, within every head,
  `new[i] = old[2i]` and `new[i + D/2] = old[2i + 1]` is applied to:

  - the output columns of `to_q/kernel` `(in, H*D)` and `to_k/kernel` `(in, Hkv*D)`,
  - `norm_q/weight` and `norm_k/weight` `(D,)`,
  - the output columns of every LoRA up kernel `to_q/lora-*/up/kernel` and
    `to_k/lora-*/up/kernel` `(rank, H*D)` (so LoRA updates, like the base
    projection, come out permuted). LoRA `down` kernels, `to_v`, `to_gate`,
    `to_out`, biases and all other leaves are shared with the input tree.

  Why this is exact:

  - Interleaved RoPE rotates the pair `(x[2i], x[2i+1])` by angle `theta_i`.
    With `y = P(x)` (i.e. `y[i] = x[2i]`, `y[i + D/2] = x[2i+1]`), rotate-half
    RoPE with the un-repeated `theta_i` table maps `y[i]` to
    `y[i] cos - y[i+D/2] sin = x[2i] cos - x[2i+1] sin` and `y[i + D/2]` to
    `x[2i+1] cos + x[2i] sin`: exactly `P` applied to the interleaved result.
  - The per-head RMSNorm over head_dim is permutation-invariant (its mean of
    squares does not depend on order) once its weight is permuted alongside.
  - q and k receive the same permutation, so every dot product `q . k` (and
    therefore the attention weights) is unchanged; v and the output projection
    never see the permuted axis.

  Works on host numpy or jax arrays; the input tree is not modified.
  """
  if params is None:
    return params
  perm = rotate_half_permutation(head_dim)

  def head_index(heads):
    return (np.arange(heads)[:, None] * head_dim + perm[None, :]).reshape(-1)

  q_index = head_index(num_heads)
  k_index = head_index(num_kv_heads)

  def permute_projection(proj, index):
    new_proj = dict(proj)
    if "kernel" in proj:
      kernel = proj["kernel"]
      if kernel.shape[-1] != index.shape[0]:
        raise ValueError(f"Projection kernel output width {kernel.shape[-1]} != heads*head_dim {index.shape[0]}.")
      new_proj["kernel"] = kernel[..., index]
    for name, sub in proj.items():
      if isinstance(name, str) and name.startswith("lora-"):
        new_sub = dict(sub)
        if "up" in sub:
          new_up = dict(sub["up"])
          new_up["kernel"] = sub["up"]["kernel"][..., index]
          new_sub["up"] = new_up
        new_proj[name] = new_sub
    return new_proj

  def permute_norm(norm):
    new_norm = dict(norm)
    new_norm["weight"] = norm["weight"][..., perm]
    return new_norm

  new_params = dict(params)
  num_blocks = 0
  for name, block in params.items():
    if not (isinstance(name, str) and name.startswith("blocks_")) or "attn" not in block:
      continue
    attn = dict(block["attn"])
    attn["to_q"] = permute_projection(attn["to_q"], q_index)
    attn["to_k"] = permute_projection(attn["to_k"], k_index)
    attn["norm_q"] = permute_norm(attn["norm_q"])
    attn["norm_k"] = permute_norm(attn["norm_k"])
    new_block = dict(block)
    new_block["attn"] = attn
    new_params[name] = new_block
    num_blocks += 1
  max_logging.log(f"Permuted q/k head dims to rotate-half RoPE order in {num_blocks} Krea 2 blocks.")
  return new_params


def _pop_weight(pt_state_dict, *candidate_keys):
  for key in candidate_keys:
    if key in pt_state_dict:
      return pt_state_dict.pop(key)
  raise KeyError(f"None of the candidate keys {candidate_keys} found in the Krea 2 checkpoint.")


# Bump KREA2_TRANSFORMER_WEIGHT_BUILD_REVISION (weight_cache.py) when this changes the produced values.
def load_and_convert_krea2_weights(safetensors_path: str, params: dict, num_layers: int) -> dict:
  """Loads Krea 2 transformer weights from a diffusers-format (sharded) safetensors
  directory and maps them into the Flax `Krea2Transformer2DModel` parameter tree.

  Norm weights (zero-centered) and scale_shift_tables are loaded verbatim in
  float32; 2-D matmul weights are transposed to Flax kernel layout.
  """
  reader = SafetensorsShardReader(safetensors_path)
  max_logging.log(
      f"Streaming Krea 2 weights from {safetensors_path} "
      f"({len(reader.files)} safetensors file{'s' if len(reader.files) != 1 else ''})..."
  )
  with reader:
    return _convert_krea2_weights(reader, params, num_layers)


def _convert_krea2_weights(pt_state_dict, params: dict, num_layers: int) -> dict:
  """Maps lazily-read Diffusers tensors into the Flax parameter tree."""

  max_logging.log("Mapping Krea 2 weights to JAX parameters...")
  expected_pytree = jax.tree_util.tree_map(lambda leaf: leaf, params)

  # Matmul weights follow the model's weights dtype; norms and modulation tables
  # are explicitly kept in float32 below.
  target_dtype = np.dtype(params["img_in"]["kernel"].dtype)

  def as_dtype(tensor, dtype):
    array = np.asarray(tensor)
    dtype = np.dtype(dtype)
    if array.dtype == dtype:
      return array
    return array.astype(dtype, copy=False)

  def as_kernel(tensor):
    # PyTorch Linear weight (out, in) -> Flax kernel (in, out).
    return as_dtype(np.asarray(tensor).T, target_dtype)

  def as_is(tensor, dtype=None):
    return as_dtype(tensor, dtype or target_dtype)

  def as_fp32(tensor):
    return as_dtype(tensor, np.float32)

  def convert_attention(jax_attn, pt_prefix):
    jax_attn["to_q"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, pt_prefix + "to_q.weight"))
    jax_attn["to_k"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, pt_prefix + "to_k.weight"))
    jax_attn["to_v"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, pt_prefix + "to_v.weight"))
    jax_attn["to_gate"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, pt_prefix + "to_gate.weight"))
    jax_attn["to_out"]["kernel"] = as_kernel(
        _pop_weight(pt_state_dict, pt_prefix + "to_out.0.weight", pt_prefix + "to_out.weight")
    )
    jax_attn["norm_q"]["weight"] = as_fp32(_pop_weight(pt_state_dict, pt_prefix + "norm_q.weight"))
    jax_attn["norm_k"]["weight"] = as_fp32(_pop_weight(pt_state_dict, pt_prefix + "norm_k.weight"))

  def convert_swiglu(jax_ff, pt_prefix):
    jax_ff["gate_proj"]["kernel"] = as_kernel(
        _pop_weight(pt_state_dict, pt_prefix + "gate.weight", pt_prefix + "gate_proj.weight")
    )
    jax_ff["up_proj"]["kernel"] = as_kernel(
        _pop_weight(pt_state_dict, pt_prefix + "up.weight", pt_prefix + "up_proj.weight")
    )
    jax_ff["down_proj"]["kernel"] = as_kernel(
        _pop_weight(pt_state_dict, pt_prefix + "down.weight", pt_prefix + "down_proj.weight")
    )

  def convert_fusion_block(jax_block, pt_prefix):
    jax_block["norm1"]["weight"] = as_fp32(_pop_weight(pt_state_dict, pt_prefix + "norm1.weight"))
    jax_block["norm2"]["weight"] = as_fp32(_pop_weight(pt_state_dict, pt_prefix + "norm2.weight"))
    convert_attention(jax_block["attn"], pt_prefix + "attn.")
    convert_swiglu(jax_block["ff"], pt_prefix + "ff.")

  # Input projections
  params["img_in"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, "img_in.weight"))
  params["img_in"]["bias"] = as_is(_pop_weight(pt_state_dict, "img_in.bias"))

  # Timestep embedding + shared modulation projection
  for name in ("linear_1", "linear_2"):
    params["time_embed"][name]["kernel"] = as_kernel(_pop_weight(pt_state_dict, f"time_embed.{name}.weight"))
    params["time_embed"][name]["bias"] = as_is(_pop_weight(pt_state_dict, f"time_embed.{name}.bias"))
  params["time_mod_proj"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, "time_mod_proj.weight"))
  params["time_mod_proj"]["bias"] = as_is(_pop_weight(pt_state_dict, "time_mod_proj.bias"))

  # Text fusion stage
  fusion = params["text_fusion"]
  num_layerwise = len([k for k in fusion.keys() if k.startswith("layerwise_blocks_")])
  num_refiner = len([k for k in fusion.keys() if k.startswith("refiner_blocks_")])
  for i in range(num_layerwise):
    convert_fusion_block(fusion[f"layerwise_blocks_{i}"], f"text_fusion.layerwise_blocks.{i}.")
  fusion["projector"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, "text_fusion.projector.weight"))
  for i in range(num_refiner):
    convert_fusion_block(fusion[f"refiner_blocks_{i}"], f"text_fusion.refiner_blocks.{i}.")

  # Text projection into the transformer width
  params["txt_in"]["norm"]["weight"] = as_fp32(_pop_weight(pt_state_dict, "txt_in.norm.weight"))
  for name in ("linear_1", "linear_2"):
    params["txt_in"][name]["kernel"] = as_kernel(_pop_weight(pt_state_dict, f"txt_in.{name}.weight"))
    params["txt_in"][name]["bias"] = as_is(_pop_weight(pt_state_dict, f"txt_in.{name}.bias"))

  # Transformer blocks
  max_logging.log(f"Mapping {num_layers} Krea 2 transformer blocks...")
  for i in range(num_layers):
    jax_block = params[f"blocks_{i}"]
    prefix = f"transformer_blocks.{i}."
    jax_block["scale_shift_table"] = as_fp32(_pop_weight(pt_state_dict, prefix + "scale_shift_table"))
    jax_block["norm1"]["weight"] = as_fp32(_pop_weight(pt_state_dict, prefix + "norm1.weight"))
    jax_block["norm2"]["weight"] = as_fp32(_pop_weight(pt_state_dict, prefix + "norm2.weight"))
    convert_attention(jax_block["attn"], prefix + "attn.")
    convert_swiglu(jax_block["ff"], prefix + "ff.")

  # Final layer
  params["final_layer"]["scale_shift_table"] = as_fp32(_pop_weight(pt_state_dict, "final_layer.scale_shift_table"))
  params["final_layer"]["norm"]["weight"] = as_fp32(_pop_weight(pt_state_dict, "final_layer.norm.weight"))
  params["final_layer"]["linear"]["kernel"] = as_kernel(_pop_weight(pt_state_dict, "final_layer.linear.weight"))
  params["final_layer"]["linear"]["bias"] = as_is(_pop_weight(pt_state_dict, "final_layer.linear.bias"))

  if pt_state_dict:
    max_logging.log(f"WARNING: {len(pt_state_dict)} unconsumed Krea 2 checkpoint keys: {sorted(pt_state_dict.keys())[:20]}")

  params = jax.tree_util.tree_map(
      lambda leaf: np.zeros(leaf.shape, dtype=leaf.dtype) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, params
  )
  del pt_state_dict
  gc.collect()
  max_logging.log("Validating converted Krea 2 Flax pytree...")
  validate_flax_state_dict(expected_pytree, params)
  max_logging.log("Krea 2 weight conversion complete & verified!")
  return params
