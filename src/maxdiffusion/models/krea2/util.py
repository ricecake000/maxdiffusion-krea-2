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


def _pop_weight(pt_state_dict, *candidate_keys):
  for key in candidate_keys:
    if key in pt_state_dict:
      return pt_state_dict.pop(key)
  raise KeyError(f"None of the candidate keys {candidate_keys} found in the Krea 2 checkpoint.")


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
