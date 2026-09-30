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

# End-to-end inference entry point for Krea 2 (K2) Raw and Turbo on JAX.
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2.yml \
#     run_name=krea2_raw output_dir=output/ prompt="a fox in the snow"
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo.yml \
#     run_name=krea2_turbo output_dir=output/ prompt="a fox in the snow"
#
# Named resolutions (krea2_aspect_ratio / krea2_image_size override height/width). Precompile every preset into
# the AOT cache once (no image is generated), then generate with a ratio; the run loads only the executables it
# calls when aot_cache_lazy_load is on (the v6e-1 preset default):
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
#     aot_cache_dir=/path/to/aot krea2_precompile=all "krea2_precompile_text_tokens=[128]"
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
#     aot_cache_dir=/path/to/aot krea2_aspect_ratio=16:9 krea2_image_size=2k prompt="a fox in the snow"
#
# Quantized-weight cache: with krea2_weight_cache_dir set, a run that misses saves the final quantized
# transformer / text encoder host trees there, and later runs read them instead of reading and re-quantizing
# the checkpoint (only quantized components; any LoRA adapter bypasses the transformer cache):
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
#     krea2_weight_cache_dir=/path/to/weights aot_cache_dir=/path/to/aot prompt="a fox in the snow"
#
# Build the weight cache only (krea2_weight_cache_build_only): reads and quantizes the checkpoint on the host,
# saves the missing components and exits before device placement, so it needs no accelerator (e.g. a CPU VM).
# A machine without a TPU has no JAX distributed coordinator, hence skip_jax_distributed_system=True:
#
#   JAX_PLATFORMS=cpu python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
#     skip_jax_distributed_system=True krea2_weight_cache_dir=/path/to/weights krea2_weight_cache_build_only=True

import gc
import inspect
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
from contextlib import ExitStack
from typing import List, Tuple

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion import aot_cache, max_logging, max_utils, pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.krea2.resolution_presets import (
    KREA2_DEFAULT_ASPECT_RATIO,
    KREA2_DEFAULT_IMAGE_SIZE,
    Krea2Resolution,
    parse_krea2_precompile,
    resolve_krea2_resolution,
)


def partition_prompts(prompt_str: str, batch_size: int) -> List[str]:
  """Splits a prompt string by '||' and replicates/truncates to fill the batch_size."""
  raw_prompts = [p.strip() for p in prompt_str.split("||") if p.strip()]
  if not raw_prompts:
    raw_prompts = ["a fox in the snow"]

  num_prompts = len(raw_prompts)
  if num_prompts == 1:
    return raw_prompts * batch_size
  elif num_prompts <= batch_size:
    reps = batch_size // num_prompts
    active = []
    for p in raw_prompts:
      active.extend([p] * reps)
    if len(active) < batch_size:
      active.extend([raw_prompts[-1]] * (batch_size - len(active)))
    return active
  else:
    max_logging.log(
        f"Warning: Found {num_prompts} prompts, but batch_size is {batch_size}. Truncating to the first {batch_size}."
    )
    return raw_prompts[:batch_size]


def resolve_prompts(prompt_str: str, batch_size: int, prompt_file: str = "") -> List[str]:
  """Returns one prompt per batch element, optionally from a line-oriented file."""
  if not prompt_file:
    return partition_prompts(prompt_str, batch_size)

  with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]
  if len(prompts) != batch_size:
    raise ValueError(
        f"prompt_file must contain exactly batch_size={batch_size} non-empty lines; "
        f"found {len(prompts)} in {prompt_file}."
    )
  return prompts


def load_qwen_image_vae(snapshot_dir, config, vae_mesh, rngs):
  """Loads the Qwen-Image VAE (Wan 2.1 architecture) from the Krea 2 snapshot."""
  from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
  from maxdiffusion.models.wan.wan_utils import load_wan_vae
  from functools import partial

  def create_model(rngs):
    return AutoencoderKLWan.from_config(
        snapshot_dir,
        subfolder="vae",
        rngs=rngs,
        mesh=vae_mesh,
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
        vae_decode_chunk=1,
        vae_encode_chunk=4,
    )

  wan_vae = nnx.eval_shape(partial(create_model), rngs=rngs)
  graphdef, state = nnx.split(wan_vae, nnx.Param)
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  params = load_wan_vae(snapshot_dir, params, "cpu")
  target_dtype = np.dtype(config.weights_dtype)
  params = jax.tree_util.tree_map(
      lambda x: x if np.dtype(x.dtype) == target_dtype else x.astype(config.weights_dtype),
      params,
  )
  # The VAE is small; replicate it across the VAE mesh.
  replicated_sharding = NamedSharding(vae_mesh, P())
  for path, val in flax.traverse_util.flatten_dict(params).items():
    state[path].value = max_utils.device_put_replicated(val, replicated_sharding)
  state = nnx.from_flat_state(state)

  wan_vae = nnx.merge(graphdef, state)
  vae_cache = AutoencoderKLWanCache(wan_vae)
  return wan_vae, vae_cache


def build_qwen3_config(te_config, config):
  """Builds the Qwen3 text-tower config from a Qwen3-VL `text_encoder/config.json` dict."""
  from maxdiffusion.models.qwen3_flax import FlaxQwen3Config

  text_config = te_config.get("text_config", te_config)
  rope_parameters = text_config.get("rope_parameters", {})
  rope_theta = rope_parameters.get("rope_theta", text_config.get("rope_theta", 5000000.0))

  return FlaxQwen3Config(
      vocab_size=text_config["vocab_size"],
      hidden_size=text_config["hidden_size"],
      intermediate_size=text_config["intermediate_size"],
      num_hidden_layers=text_config["num_hidden_layers"],
      num_attention_heads=text_config["num_attention_heads"],
      num_key_value_heads=text_config["num_key_value_heads"],
      head_dim=text_config.get("head_dim", 128),
      max_position_embeddings=text_config.get("max_position_embeddings", 262144),
      rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
      rope_theta=rope_theta,
      dtype=config.weights_dtype,
  )


def build_krea2_transformer(transformer_cfg, config, mesh, quant_targets=None):
  """Builds the Krea 2 transformer module from a `transformer/config.json` dict (defaults if empty).

  `quant_targets` selects the W8A8 block projections; None takes them from the
  config (`krea2_transformer_quantization` / `krea2_transformer_quant_targets`),
  `()` builds the unquantized model.
  """
  from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
  from maxdiffusion.models.krea2.transformer_quant import normalize_quant_targets, resolve_transformer_quantization

  if quant_targets is None:
    _, quant_targets = resolve_transformer_quantization(config)

  return Krea2Transformer2DModel(
      in_channels=transformer_cfg.get("in_channels", 64),
      num_layers=transformer_cfg.get("num_layers", 28),
      attention_head_dim=transformer_cfg.get("attention_head_dim", 128),
      num_attention_heads=transformer_cfg.get("num_attention_heads", 48),
      num_key_value_heads=transformer_cfg.get("num_key_value_heads", 12),
      intermediate_size=transformer_cfg.get("intermediate_size", 16384),
      timestep_embed_dim=transformer_cfg.get("timestep_embed_dim", 256),
      text_hidden_dim=transformer_cfg.get("text_hidden_dim", 2560),
      num_text_layers=transformer_cfg.get("num_text_layers", 12),
      text_num_attention_heads=transformer_cfg.get("text_num_attention_heads", 20),
      text_num_key_value_heads=transformer_cfg.get("text_num_key_value_heads", 20),
      text_intermediate_size=transformer_cfg.get("text_intermediate_size", 6912),
      num_layerwise_text_blocks=transformer_cfg.get("num_layerwise_text_blocks", 2),
      num_refiner_text_blocks=transformer_cfg.get("num_refiner_text_blocks", 2),
      axes_dims_rope=tuple(transformer_cfg.get("axes_dims_rope", (32, 48, 48))),
      rope_theta=transformer_cfg.get("rope_theta", 1000.0),
      norm_eps=transformer_cfg.get("norm_eps", 1e-5),
      attention_kernel=config.attention,
      flash_block_sizes=max_utils.get_flash_block_sizes(config),
      mask_padding_tokens=config.mask_padding_tokens,
      rope_layout=getattr(config, "krea2_rope_layout", "interleaved"),
      mesh=mesh,
      dtype=config.activations_dtype,
      weights_dtype=config.weights_dtype,
      quant_targets=normalize_quant_targets(quant_targets),
  )


def transformer_quantization_aot_meta(mode, targets) -> dict:
  """AOT cache meta entry for transformer quantization; empty when it is off.

  The key is present only when quantization is on, so the fingerprint of an
  unquantized setup (and its existing cached executables) stays unchanged.
  The value carries `KREA2_TRANSFORMER_QUANT_REVISION`, so a changed W8A8 graph
  misses executables cached for the previous one.
  """
  from maxdiffusion.models.krea2.transformer_quant import KREA2_TRANSFORMER_QUANT_REVISION

  if not mode:
    return {}
  return {"krea2_transformer_quantization": f"{mode}:{','.join(targets)}:r{KREA2_TRANSFORMER_QUANT_REVISION}"}


def flash_custom_block_selection_aot_meta(attention) -> dict:
  """AOT cache meta entry for the flash_custom automatic block sizes; empty for other kernels.

  `flash_block_sizes` in the meta only records the configured sizes, not the
  automatic choice, so the value carries `KREA2_BLOCK_SELECTION_REVISION` and a
  changed choice misses executables cached for the previous one. Other kernels
  keep their fingerprint.
  """
  from maxdiffusion.kernels.krea2_attention import KREA2_BLOCK_SELECTION_REVISION

  if attention != "flash_custom":
    return {}
  return {"krea2_block_selection": f"r{KREA2_BLOCK_SELECTION_REVISION}"}


def _weight_cache_dir(config) -> str:
  # Strip quotes so a command-line override `krea2_weight_cache_dir=''` means off.
  return str(getattr(config, "krea2_weight_cache_dir", "") or "").strip().strip("'\"")


def _weight_cache_bypass_reasons(transformer_quantization, te_quantization, lora_configured) -> dict:
  """Maps each component to why it does not use the weight cache, '' when it does."""
  transformer_reason = ""
  if not transformer_quantization:
    transformer_reason = "unquantized (krea2_transformer_quantization off), nothing to cache."
  elif lora_configured:
    transformer_reason = "bypassed, LoRA adapters are merged into its tree but not in its key."
  text_encoder_reason = ""
  if te_quantization != "int8":
    text_encoder_reason = "unquantized (krea2_text_encoder_quantization off), nothing to cache."
  return {"transformer": transformer_reason, "text_encoder": text_encoder_reason}


def resolve_weight_cache_dirs(config, transformer_quantization, te_quantization, lora_compile_spec) -> Tuple[str, str]:
  """Returns `(transformer_dir, text_encoder_dir)` of `krea2_weight_cache_dir`; '' means no cache for that component.

  Only quantized components are cached: the cache saves the re-quantization,
  an unquantized tree would only copy the checkpoint. Any LoRA adapter bypasses
  the transformer cache (its weights are merged into the tree but are not part
  of the fingerprint); the text encoder cache does not depend on LoRA. Logs one
  line per component that does not use the cache; nothing when the cache is off.
  """
  cache_dir = _weight_cache_dir(config)
  if not cache_dir:
    return "", ""
  reasons = _weight_cache_bypass_reasons(transformer_quantization, te_quantization, bool(lora_compile_spec))
  for component, reason in reasons.items():
    if reason:
      max_logging.log(f"[weight cache] {component}: {reason}")
  return tuple("" if reasons[component] else cache_dir for component in ("transformer", "text_encoder"))


def load_or_build_host_params(cache, abstract_params, build, load_trace, trace_key, extra_names=(), check_dtypes=True):
  """Returns `(tree, extras, hit)`: the cached host tree, or `build()`'s `(tree, extras)` on a miss.

  `cache` is a `WeightCacheSpec` or None (no cache: `build` runs, nothing is
  read). On a hit `build` is not called. `load_trace[trace_key]` gets the time
  of the cache read whenever one was attempted.
  """
  from maxdiffusion.models.krea2.weight_cache import load_component

  if cache is not None:
    t0 = time.perf_counter()
    cached = load_component(
        cache.cache_dir,
        cache.component,
        cache.meta,
        abstract_params,
        extra_names=extra_names,
        source_files=cache.source_files,
        check_dtypes=check_dtypes,
    )
    load_trace[trace_key] = time.perf_counter() - t0
    if cached is not None:
      tree, extras = cached
      return tree, extras, True
  tree, extras = build()
  return tree, extras, False


def save_host_params(cache, tree, load_trace, trace_key, extras=None):
  """Saves a freshly built host tree to the cache; a no-op for `cache` None. Returns the directory or None.

  `load_trace[trace_key]` gets the time of the write whenever one was attempted.
  """
  from maxdiffusion.models.krea2.weight_cache import save_component

  if cache is None:
    return None
  t0 = time.perf_counter()
  saved = save_component(cache.cache_dir, cache.component, cache.meta, tree, extras=extras, source_files=cache.source_files)
  load_trace[trace_key] = time.perf_counter() - t0
  return saved


# Build-only mode's load_trace stages per component: the validity check, or the build and the write
# (qwen_host includes the text encoder's quantization, embedding table and write).
_WEIGHT_CACHE_CHECK_STAGES = {"transformer": "transformer_cache_check", "text_encoder": "qwen_cache_check"}
_WEIGHT_CACHE_BUILD_STAGES = {
    "transformer": ("transformer_host", "lora_apply", "rope_permute", "transformer_quantize", "transformer_cache_write"),
    "text_encoder": ("qwen_host",),
}


def resolve_weight_cache_build_only(config) -> bool:
  """Whether `krea2_weight_cache_build_only` is on; validates it before anything is loaded.

  Build-only mode saves the missing weight cache components and exits before
  device placement. Raises ValueError without `krea2_weight_cache_dir`,
  together with `krea2_precompile` (the two modes exclude each other) or when
  no component would use the cache (decided from the config alone, as
  `resolve_weight_cache_dirs` decides it later, so nothing is downloaded or
  built first). Logs one line when on, nothing when off.
  """
  if not getattr(config, "krea2_weight_cache_build_only", False):
    return False
  if not _weight_cache_dir(config):
    raise ValueError("krea2_weight_cache_build_only needs krea2_weight_cache_dir: the built trees are only kept there.")
  spec = getattr(config, "krea2_precompile", "")
  if parse_krea2_precompile("" if spec is None else spec):
    raise ValueError(
        "krea2_weight_cache_build_only and krea2_precompile exclude each other: build-only mode exits before "
        "anything is compiled."
    )
  from maxdiffusion.models.krea2.text_encoder_quant import resolve_text_encoder_quantization
  from maxdiffusion.models.krea2.transformer_quant import resolve_transformer_quantization

  # Configured adapters, as maybe_load_krea2_lora reads them (it returns a compile spec per adapter).
  lora_config = getattr(config, "lora_config", None) or {}
  lora_configured = len(lora_config.get("lora_model_name_or_path") or ()) > 0
  reasons = _weight_cache_bypass_reasons(
      resolve_transformer_quantization(config)[0], resolve_text_encoder_quantization(config)[0], lora_configured
  )
  if all(reasons.values()):
    raise ValueError(
        "krea2_weight_cache_build_only: nothing to build, no component would use the weight cache: "
        + "; ".join(f"{component}: {reason.rstrip('.')}" for component, reason in reasons.items())
        + "."
    )
  max_logging.log(
      f"Weight cache build-only mode: saving the missing components to {_weight_cache_dir(config)}; "
      "no image is generated."
  )
  return True


def select_weight_cache_builds(checks, load_trace) -> Tuple[str, ...]:
  """Build-only mode: the components whose cache must be built, in `checks` order.

  `checks` maps a component to `(cache, abstract_params, extra_names,
  check_dtypes)` as main reads it. A component with `cache` None (not cached,
  see `resolve_weight_cache_dirs`) is skipped, one whose cache is already
  complete (`component_is_valid`, which never reads the arrays) too;
  `load_trace` gets the time of each check. Raises ValueError when no
  component has a cache.
  """
  from maxdiffusion.models.krea2.weight_cache import component_is_valid

  if all(cache is None for cache, _, _, _ in checks.values()):
    raise ValueError(
        "krea2_weight_cache_build_only: nothing to build, no component uses the weight cache "
        "(see the [weight cache] lines above)."
    )
  builds = []
  for component, (cache, abstract_params, extra_names, check_dtypes) in checks.items():
    if cache is None:
      continue
    t0 = time.perf_counter()
    valid = component_is_valid(
        cache.cache_dir,
        cache.component,
        cache.meta,
        abstract_params,
        extra_names=extra_names,
        source_files=cache.source_files,
        check_dtypes=check_dtypes,
    )
    load_trace[_WEIGHT_CACHE_CHECK_STAGES[component]] = time.perf_counter() - t0
    if not valid:
      builds.append(component)
  return tuple(builds)


def check_weight_cache_saves(builds, saved) -> None:
  """Build-only mode: raises RuntimeError naming each built component whose save failed.

  `saved` maps a component to `save_host_params`' result (None when nothing
  was written). In a normal run a failed save stays a warning.
  """
  failed = [component for component in builds if not saved.get(component)]
  if failed:
    raise RuntimeError(
        f"krea2_weight_cache_build_only: could not save {', '.join(failed)} to the weight cache "
        "(see the [weight cache] warnings above)."
    )


def log_weight_cache_build_summary(caches, builds, saved, load_trace, load_time) -> None:
  """Logs one line per cached component (built and saved / already complete) plus the load timing."""
  from maxdiffusion.models.krea2.weight_cache import component_dir, weights_nbytes

  max_logging.log("=" * 80)
  max_logging.log("KREA 2 WEIGHT CACHE BUILD SUMMARY")
  max_logging.log("=" * 80)
  for component, cache in caches.items():
    if cache is None:
      continue
    if component in builds:
      status, directory = "built and saved", saved[component]
      stages = _WEIGHT_CACHE_BUILD_STAGES[component]
    else:
      status, directory = "already complete", component_dir(cache.cache_dir, cache.component, cache.meta)
      stages = (_WEIGHT_CACHE_CHECK_STAGES[component],)
    nbytes = weights_nbytes(directory)
    size = f"{nbytes / 1024**3:.2f} GiB" if nbytes is not None else "? GiB"
    seconds = sum(load_trace.get(stage, 0.0) for stage in stages)
    max_logging.log(f"{component:<12} {status:<16} {size:>10} {seconds:>7.1f} s  {directory}")
  max_logging.log(f" -> [TIMING] Total Weight Cache Build: {load_time:.2f} seconds")
  max_logging.log(
      " -> [TIMING] Load breakdown: "
      + ", ".join(f"{stage}={seconds:.2f}s" for stage, seconds in load_trace.items())
  )
  max_logging.log("=" * 80)
  max_logging.log(
      f"SUCCESS! Weight cache build complete: {len(builds)} component(s) built, "
      f"{sum(cache is not None for cache in caches.values()) - len(builds)} already complete!"
  )


# Precompile mode's prompt: short enough for the smallest text bucket, so each
# plan entry's forced bucket decides the compiled text length.
KREA2_PRECOMPILE_PROMPT = "a fox in the snow"


def _preset_key_unset(value) -> bool:
  return value is None or (isinstance(value, str) and not value.strip())


def resolve_generation_size(config) -> Tuple[int, int, str]:
  """Returns `(height, width, description)` of the image to generate.

  `krea2_aspect_ratio` / `krea2_image_size` win over `height` / `width`: when
  either is set, the other takes its default (1:1 / 1k) and the size comes from
  the preset table. Otherwise `height` / `width` are rounded up to multiples of
  16 (VAE 8x downsampling x 2x2 latent patches), as the pipeline would.
  """
  from maxdiffusion.models.krea2.util import round_up_to_multiple

  aspect_ratio = getattr(config, "krea2_aspect_ratio", "")
  image_size = getattr(config, "krea2_image_size", "")
  if _preset_key_unset(aspect_ratio) and _preset_key_unset(image_size):
    height = round_up_to_multiple(config.height, 16)
    width = round_up_to_multiple(config.width, 16)
    if (height, width) != (config.height, config.width):
      max_logging.log(
          f"Warning: height and width must be multiples of 16; rounding up from "
          f"{config.height}x{config.width} to {height}x{width}."
      )
    return height, width, f"{width}x{height} (height/width)"

  resolution = resolve_krea2_resolution(
      KREA2_DEFAULT_ASPECT_RATIO if _preset_key_unset(aspect_ratio) else aspect_ratio,
      KREA2_DEFAULT_IMAGE_SIZE if _preset_key_unset(image_size) else image_size,
  )
  description = f"preset {resolution.label} -> {resolution.width}x{resolution.height}"
  max_logging.log(f"Output resolution: {description}")
  config_height, config_width = getattr(config, "height", None), getattr(config, "width", None)
  if (config_height, config_width) != (resolution.height, resolution.width):
    max_logging.log(
        f"Ignoring height={config_height} width={config_width} from the config: "
        "krea2_aspect_ratio / krea2_image_size set the resolution."
    )
  return resolution.height, resolution.width, description


def resolve_precompile_plan(config, text_compaction_multiple) -> List[Tuple[Krea2Resolution, int]]:
  """Every (resolution, text bucket) to compile, in order: resolutions in spec order, buckets ascending.

  Buckets come from `krea2_precompile_text_tokens` (positive ints, [] means
  [128]), rounded up to `text_compaction_multiple` and clipped to
  `max_sequence_length` like the pipeline's compaction. Without compaction
  (`text_compaction_multiple <= 0`) the text always has the full length, the
  only bucket. Raises ValueError on a bad spec, a bad bucket, or a non-empty
  spec without `aot_cache_dir` (nothing could be saved).
  """
  from maxdiffusion.models.krea2.util import round_up_to_multiple

  spec = getattr(config, "krea2_precompile", "")
  resolutions = parse_krea2_precompile("" if spec is None else spec)
  if not resolutions:
    return []
  if not getattr(config, "aot_cache_dir", ""):
    raise ValueError("krea2_precompile needs aot_cache_dir: the compiled executables are only kept in the AOT cache.")

  raw_tokens = getattr(config, "krea2_precompile_text_tokens", None)
  if raw_tokens is None:
    raw_tokens = []
  tokens = list(raw_tokens) if isinstance(raw_tokens, (list, tuple)) else [raw_tokens]
  for value in tokens:
    # bool is an int subclass; True would silently mean a 1-token bucket.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
      raise ValueError(f"krea2_precompile_text_tokens entries must be positive ints, got {value!r} in {raw_tokens!r}.")
  if not tokens:
    tokens = [128]

  max_length = int(config.max_sequence_length)
  if text_compaction_multiple <= 0:
    buckets = [max_length]
  else:
    buckets = sorted({min(round_up_to_multiple(value, text_compaction_multiple), max_length) for value in tokens})
  return [(resolution, bucket) for resolution in resolutions for bucket in buckets]


def run_precompile(pipeline, plan, call_kwargs, prompts) -> List[dict]:
  """Compiles every plan entry in warmup mode and saves after each one. Returns one record per entry.

  Each entry is a zero-execution warmup call at the entry's resolution with the
  text bucket forced to at least its size. The cache is saved after every
  entry, outside the warmup context, so a preempted VM keeps what is already
  compiled. The forced size is only a lower bound (a longer prompt or negative
  prompt gets a larger bucket), so the buckets the pipeline reports in its trace
  are checked against the requested one: a mismatch, or a trace without
  `text_tokens`, raises RuntimeError after that entry was saved and before the
  next one starts. With classifier-free guidance (decided from `call_kwargs`
  exactly like the pipeline does) a trace without `negative_text_tokens` raises
  too. Records carry the actual `text_tokens` (and `negative_text_tokens` with
  guidance) next to `requested_text_tokens`.
  """
  from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline, is_classifier_free_guidance_enabled

  # Same decision as FlaxKrea2Pipeline.__call__, with its defaults for absent keys.
  call_defaults = inspect.signature(FlaxKrea2Pipeline.__call__).parameters
  guidance = is_classifier_free_guidance_enabled(
      call_kwargs.get("guidance_scale", call_defaults["guidance_scale"].default),
      call_kwargs.get("do_classifier_free_guidance", call_defaults["do_classifier_free_guidance"].default),
  )
  records = []
  for index, (resolution, requested) in enumerate(plan, start=1):
    entry = f"[precompile {index}/{len(plan)}] {resolution.label} {resolution.width}x{resolution.height}"
    # Custom latents fit only one resolution; every entry draws its own noise.
    kwargs = {**call_kwargs, "height": resolution.height, "width": resolution.width, "latents": None}
    t0 = time.perf_counter()
    with aot_cache.warmup_mode():
      _, trace = pipeline(
          prompt=prompts,
          output_name="krea2_precompile.png",
          save_outputs=False,
          min_text_tokens=requested,
          **kwargs,
      )
    # Save before any check below can raise, so the compiled work is kept.
    saved = aot_cache.save_pending()
    seconds = time.perf_counter() - t0
    if "text_tokens" not in trace:
      raise RuntimeError(
          f"{entry}: the pipeline did not report its text bucket (no 'text_tokens' in its trace), so the "
          f"requested bucket {requested} cannot be confirmed; {saved} executable(s) were saved."
      )
    if guidance and "negative_text_tokens" not in trace:
      raise RuntimeError(
          f"{entry}: classifier-free guidance is on but the pipeline did not report the negative prompt's text "
          f"bucket (no 'negative_text_tokens' in its trace), so the requested bucket {requested} cannot be "
          f"confirmed; {saved} executable(s) were saved."
      )
    record = {
        "label": resolution.label,
        "height": resolution.height,
        "width": resolution.width,
        "image_tokens": resolution.image_tokens,
        "text_tokens": trace["text_tokens"],
        "requested_text_tokens": requested,
        "seconds": seconds,
        "saved": saved,
    }
    buckets = f"text {record['text_tokens']}"
    if "negative_text_tokens" in trace:
      record["negative_text_tokens"] = trace["negative_text_tokens"]
      buckets += f" (negative {record['negative_text_tokens']})"
    records.append(record)
    max_logging.log(f"{entry} {buckets}: {seconds:.1f} s, {saved} executable(s) saved")
    actual = {"prompt": record["text_tokens"]}
    if "negative_text_tokens" in record:
      # Checked whenever reported, also if guidance is off.
      actual["negative prompt"] = record["negative_text_tokens"]
    mismatched = {kind: tokens for kind, tokens in actual.items() if tokens != requested}
    if mismatched:
      found = ", ".join(f"{kind} {tokens}" for kind, tokens in mismatched.items())
      raise RuntimeError(
          f"{entry}: requested text bucket {requested}, but the pipeline compiled {found} (a prompt with more "
          f"valid tokens than the bucket); {saved} executable(s) were saved, the requested bucket was not compiled."
      )
  return records


def log_precompile_summary(records) -> None:
  """Logs one line per precompile record plus the totals."""
  max_logging.log("=" * 80)
  max_logging.log("KREA 2 PRECOMPILE SUMMARY")
  max_logging.log("=" * 80)
  max_logging.log(f"{'resolution':<10} {'size':>9} {'image tok':>9} {'text tok':>8} {'seconds':>8} {'saved':>5}")
  for record in records:
    max_logging.log(
        f"{record['label']:<10} {record['width']:>4}x{record['height']:<4} {record['image_tokens']:>9} "
        f"{record['text_tokens']:>8} {record['seconds']:>8.1f} {record['saved']:>5}"
    )
  total_saved = sum(record["saved"] for record in records)
  total_seconds = sum(record["seconds"] for record in records)
  max_logging.log(f"Total: {len(records)} entries, {total_saved} executable(s) saved, {total_seconds:.1f} s")
  max_logging.log("=" * 80)


def main(argv):
  jax.config.update("jax_use_shardy_partitioner", True)

  # 1. Load configurations
  config_path = "src/maxdiffusion/configs/base_krea2.yml"
  custom_overrides = []
  if len(argv) > 1:
    if argv[1].endswith(".yml") or argv[1].endswith(".yaml"):
      config_path = argv[1]
      if len(argv) > 2:
        custom_overrides = argv[2:]
    else:
      custom_overrides = argv[1:]

  max_logging.log(f"Initializing pyconfig with config: {config_path}")
  default_args = [
      None,
      config_path,
      "run_name=krea2_generation",
      "output_dir=output/",
  ]
  default_args.extend(custom_overrides)
  pyconfig.initialize(default_args)

  # Import modules after jax.distributed.initialize() has run via pyconfig.initialize()
  from maxdiffusion.models.krea2.util import (
      KREA2_PROMPT_TEMPLATE_START_IDX,
      load_and_convert_krea2_weights,
      load_krea2_tokenizer,
      permute_rope_weights_to_rotate_half,
  )
  from maxdiffusion.models.qwen3_flax import (
      FlaxQwen3Model,
      load_and_convert_qwen3_weights,
      load_qwen3_embedding_table,
  )
  from maxdiffusion.models.krea2.text_encoder_quant import (
      describe_text_encoder_residency,
      quantize_text_encoder_model,
      quantize_text_encoder_params,
      resolve_text_encoder_quantization,
      safe_param_shardings,
  )
  from maxdiffusion.models.krea2.transformer_quant import (
      check_transformer_param_tree,
      describe_transformer_quantization,
      quantize_transformer_params,
      resolve_transformer_quantization,
  )
  from maxdiffusion.models.krea2.weight_cache import (
      WeightCacheSpec,
      list_source_files,
      text_encoder_weight_cache_meta,
      transformer_weight_cache_meta,
  )
  from maxdiffusion.models.flux.util import cast_dict_to_bfloat16_inplace
  from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
  from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline
  from maxdiffusion.loaders.krea2_lora_pipeline import (
      apply_diff_updates,
      insert_lora_params,
      maybe_load_krea2_lora,
  )

  config = pyconfig.config
  os.makedirs(config.output_dir, exist_ok=True)
  # The resolution and the precompile plan are resolved before anything else so
  # a bad preset name or precompile spec fails before the model loads.
  # Height/width are multiples of 16 (VAE 8x downsampling x 2x2 latent patches),
  # so the eval_shape dummies below match what the pipeline will actually run.
  height, width, _ = resolve_generation_size(config)
  # Mirrors FlaxKrea2Pipeline: flash_custom always compacts, with the full text
  # length as the bucket when krea2_text_compaction_multiple is off.
  text_compaction_multiple = int(getattr(config, "krea2_text_compaction_multiple", 0) or 0)
  if config.attention == "flash_custom" and text_compaction_multiple <= 0:
    text_compaction_multiple = int(config.max_sequence_length)
  # Before the precompile plan: its aot_cache_dir error would hide that the two modes exclude each other.
  build_only = resolve_weight_cache_build_only(config)
  precompile_plan = resolve_precompile_plan(config, text_compaction_multiple)
  if precompile_plan:
    max_logging.log(
        f"Precompile mode: {len(precompile_plan)} resolution/text combination(s) into {config.aot_cache_dir}; "
        "no image is generated."
    )
  # Offloaded components keep their host tree; the pipeline places them on
  # device only for their phase (validated by FlaxKrea2Pipeline).
  offload_components = tuple(getattr(config, "krea2_offload_components", None) or ())

  # A line-oriented file avoids shell quoting limits for heterogeneous
  # production batches.
  prompt_file = getattr(config, "prompt_file", "")
  active_prompts = resolve_prompts(config.prompt, config.batch_size, prompt_file)
  if prompt_file:
    max_logging.log(f"Loaded {len(active_prompts)} prompt(s) from {prompt_file}.")

  # 2. Setup device meshes
  # The ICI parallelism product must equal the number of devices PER SLICE
  # (see max_utils.create_device_mesh), not the global device count.
  all_devices = jax.devices()
  try:
    num_slices = 1 + max(d.slice_index for d in all_devices)
  except Exception:
    num_slices = 1
  devices_per_slice = len(all_devices) // num_slices
  if config.batch_size == 1 and config.ici_tensor_parallelism == 1 and devices_per_slice > 1:
    max_logging.log(
        f"Auto-configuring Tensor Parallelism: ici_tensor_parallelism={devices_per_slice}, "
        f"ici_fsdp_parallelism=1 for batch_size=1 on {devices_per_slice} devices per slice "
        f"({num_slices} slice(s))."
    )
    pyconfig._config.keys["ici_tensor_parallelism"] = devices_per_slice
    pyconfig._config.keys["ici_fsdp_parallelism"] = 1

  max_logging.log("Setting up JAX device mesh...")
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Dedicated mesh for the Wan-architecture VAE (axes: redundant, vae_spatial).
  vae_spatial = getattr(config, "vae_spatial", -1)
  total_devices = math.prod(devices_array.shape)
  if vae_spatial == -1:
    vae_spatial = total_devices
  assert (
      total_devices % vae_spatial == 0
  ), f"total devices ({total_devices}) must be a multiple of vae_spatial ({vae_spatial})"
  vae_devices_array = devices_array.flatten().reshape(total_devices // vae_spatial, vae_spatial)
  vae_mesh = Mesh(vae_devices_array, ("redundant", "vae_spatial"))
  vae_logical_axis_rules = getattr(config, "vae_logical_axis_rules", None)

  # 3. Resolve weights repository snapshot
  repo_id = config.pretrained_model_name_or_path
  max_logging.log(f"Target model: {repo_id}")
  if os.path.exists(repo_id):
    snapshot_dir = repo_id
    max_logging.log(f"Using local model directory: {snapshot_dir}")
  else:
    from huggingface_hub import snapshot_download

    max_logging.log(f"Resolving snapshot directory for model '{repo_id}' from HF Hub...")
    snapshot_dir = snapshot_download(repo_id=repo_id)

  max_logging.log(f"Host {jax.process_index()} using snapshot directory: {snapshot_dir}")
  transformer_path = os.path.join(snapshot_dir, "transformer")
  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
  tokenizer_path = os.path.join(snapshot_dir, "tokenizer")

  # 4. Text encoder config (Qwen3-VL: text tower lives under `text_config`)
  with open(os.path.join(text_encoder_path, "config.json"), "r") as f:
    te_config = json.load(f)
  qwen3_config = build_qwen3_config(te_config, config)
  # `qwen3_model` is the unquantized structure the checkpoint loader fills;
  # `qwen3_runtime_model` is what the pipeline applies (qwix-wrapped for int8).
  qwen3_model = FlaxQwen3Model(qwen3_config)
  te_quantization, te_quant_tile_size, te_embed_on_host = resolve_text_encoder_quantization(config)
  if te_quantization == "int8":
    qwen3_runtime_model = quantize_text_encoder_model(qwen3_model, te_quant_tile_size)
  else:
    qwen3_runtime_model = qwen3_model
  max_logging.log(describe_text_encoder_residency(te_quantization, te_quant_tile_size, te_embed_on_host))

  # 5. Transformer config
  transformer_cfg = {}
  transformer_config_json = os.path.join(transformer_path, "config.json")
  if os.path.exists(transformer_config_json):
    with open(transformer_config_json, "r") as f:
      transformer_cfg = json.load(f)

  num_layers = transformer_cfg.get("num_layers", 28)
  # `transformer_load_model` is the float structure the checkpoint loader fills;
  # `transformer` is what the pipeline applies (int8 block projections for w8a8).
  transformer_quantization, transformer_quant_targets = resolve_transformer_quantization(config)
  transformer = build_krea2_transformer(transformer_cfg, config, mesh, quant_targets=transformer_quant_targets)
  if transformer_quantization:
    transformer_load_model = build_krea2_transformer(transformer_cfg, config, mesh, quant_targets=())
  else:
    transformer_load_model = transformer
  max_logging.log(describe_transformer_quantization(transformer_quantization, transformer_quant_targets))

  # 5b. Optionally load LoRA adapters (kohya/ComfyUI/diffusers .safetensors).
  # The interceptors must be live around shape evaluation AND every pipeline
  # call so the abstract param tree (and the jit traces) include the lora-*
  # subtrees; with no adapters configured this is a single no-op interceptor.
  # Adapter arrays are materialized on host CPU like every other host-tree
  # leaf: a device-resident leaf would share a device with its placed copy,
  # so free_params would never release an offloaded transformer's adapters.
  cpu_device = jax.local_devices(backend="cpu")[0]
  with jax.default_device(cpu_device):
    lora_flat_params, lora_interceptors, lora_diff_updates, lora_compile_spec = maybe_load_krea2_lora(
        config, config.weights_dtype, return_compile_spec=True
    )

  # 6. Evaluate shapes & extract mesh shardings
  max_logging.log("Evaluating model shapes and shardings...")
  grid_h = height // 16
  grid_w = width // 16
  seq_len_img = grid_h * grid_w
  seq_len_txt = config.max_sequence_length
  # Total tokenized length before the system prefix is dropped.
  seq_len_txt_full = seq_len_txt + KREA2_PROMPT_TEMPLATE_START_IDX

  img_dummy = jnp.zeros((config.batch_size, seq_len_img, transformer_cfg.get("in_channels", 64)))
  txt_dummy = jnp.zeros(
      (
          config.batch_size,
          seq_len_txt,
          transformer_cfg.get("num_text_layers", 12),
          transformer_cfg.get("text_hidden_dim", 2560),
      )
  )
  t_dummy = jnp.zeros((config.batch_size,))
  img_ids_dummy = jnp.zeros((seq_len_img, 3))
  txt_ids_dummy = jnp.zeros((seq_len_txt, 3))
  text_mask_dummy = jnp.ones((config.batch_size, seq_len_txt), dtype=jnp.bool_)
  qwen_ids_dummy = jnp.zeros((config.batch_size, seq_len_txt_full), dtype=jnp.int32)
  qwen_mask_dummy = jnp.ones((config.batch_size, seq_len_txt_full), dtype=jnp.int32)

  key = jax.random.PRNGKey(config.seed if config.seed is not None else 0)
  key, qwen_key = jax.random.split(key)

  def transformer_init_fn(model):
    return model.init(
        key,
        hidden_states=img_dummy,
        encoder_hidden_states=txt_dummy,
        timestep=t_dummy,
        img_ids=img_ids_dummy,
        txt_ids=txt_ids_dummy,
        encoder_attention_mask=text_mask_dummy,
    )

  def qwen3_init_fn(model):
    # With the embedding lookup on the host the model is initialized from
    # embeddings, so its tree has no `embed_tokens` table.
    if te_embed_on_host:
      qwen_embeds_dummy = jnp.zeros(
          (config.batch_size, seq_len_txt_full, qwen3_config.hidden_size), dtype=qwen3_config.dtype
      )
      return model.init(qwen_key, None, qwen_mask_dummy, inputs_embeds=qwen_embeds_dummy)
    return model.init(qwen_key, qwen_ids_dummy, qwen_mask_dummy)

  with ExitStack() as stack:
    for interceptor in lora_interceptors:
      stack.enter_context(nn.intercept_methods(interceptor))
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      abstract_transformer_vars = jax.eval_shape(lambda: transformer_init_fn(transformer))
      # The loader fills the float structure; int8 block kernels are derived from it.
      abstract_transformer_load_vars = (
          jax.eval_shape(lambda: transformer_init_fn(transformer_load_model))
          if transformer_load_model is not transformer
          else abstract_transformer_vars
      )
      abstract_qwen3_vars = jax.eval_shape(lambda: qwen3_init_fn(qwen3_runtime_model))
      # The loader fills the unquantized structure; int8 params are derived from it.
      abstract_qwen3_load_vars = (
          jax.eval_shape(lambda: qwen3_init_fn(qwen3_model))
          if qwen3_runtime_model is not qwen3_model
          else abstract_qwen3_vars
      )

      logical_transformer_specs = nn.get_partition_spec(abstract_transformer_vars)
      logical_qwen3_specs = nn.get_partition_spec(abstract_qwen3_vars)

      transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
      qwen3_mesh_shardings = nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)

  transformer_shardings = flax.core.freeze(transformer_mesh_shardings["params"])
  qwen3_shardings = flax.core.freeze(qwen3_mesh_shardings["params"])
  if te_quantization:
    # qwix scale leaves inherit the kernel's logical axes on a tiled dim; fall
    # back to replication wherever the mesh does not divide a leaf.
    qwen3_shardings = flax.core.freeze(
        safe_param_shardings(
            flax.core.unfreeze(nn.unbox(abstract_qwen3_vars["params"])),
            flax.core.unfreeze(qwen3_shardings),
            mesh,
        )
    )

  # 6b. Quantized-weight cache (krea2_weight_cache_dir): a hit replaces the checkpoint
  # read, the rotate-half permutation and the quantization of that component.
  transformer_cache_dir, text_encoder_cache_dir = resolve_weight_cache_dirs(
      config, transformer_quantization, te_quantization, lora_compile_spec
  )
  transformer_cache = (
      WeightCacheSpec(
          transformer_cache_dir,
          "transformer",
          transformer_weight_cache_meta(
              config,
              snapshot_dir,
              transformer_quantization,
              transformer_quant_targets,
              transformer.rope_layout,
              transformer.num_attention_heads,
              transformer.num_key_value_heads,
              transformer.attention_head_dim,
          ),
          list_source_files(transformer_path),
      )
      if transformer_cache_dir
      else None
  )
  text_encoder_cache = (
      WeightCacheSpec(
          text_encoder_cache_dir,
          "text_encoder",
          text_encoder_weight_cache_meta(
              config, snapshot_dir, te_quantization, te_quant_tile_size, te_embed_on_host, qwen3_config.dtype
          ),
          list_source_files(text_encoder_path),
      )
      if text_encoder_cache_dir
      else None
  )

  # 7. Stream weights on host CPU, overlap the independent components, then
  # place the final trees directly into their target TPU shardings.
  max_logging.log("Streaming parameters from safetensors...")
  t_load_start = time.time()
  load_trace = {}
  parallel_loading = getattr(config, "parallel_component_loading", True)
  rngs = nnx.Rngs(jax.random.key(config.seed if config.seed is not None else 0))
  qwen_extra_names = ("embedding_table",) if te_embed_on_host else ()
  # Build-only mode loads only the components whose cache is missing: a complete
  # one is not read, one without a cache (nothing it could save) not loaded.
  weight_cache_builds = ()
  if build_only:
    weight_cache_builds = select_weight_cache_builds(
        {
            "transformer": (transformer_cache, abstract_transformer_vars["params"], (), True),
            "text_encoder": (text_encoder_cache, abstract_qwen3_vars["params"], qwen_extra_names, False),
        },
        load_trace,
    )
  load_transformer = not build_only or "transformer" in weight_cache_builds
  load_text_encoder = not build_only or "text_encoder" in weight_cache_builds
  # save_host_params' result per component; a failed save fails build-only mode.
  weight_cache_saved = {}

  def load_vae_timed():
    t0 = time.perf_counter()
    result = load_qwen_image_vae(snapshot_dir, config, vae_mesh, rngs)
    return result, time.perf_counter() - t0

  # VAE is small and independent. Hide it behind the much larger Transformer
  # and Qwen host reads, but wait before the main device transfer so PCIe/ICI
  # traffic does not contend.
  # Build-only mode never loads the VAE.
  common_executor = ThreadPoolExecutor(max_workers=1) if parallel_loading and not build_only else None
  vae_future = common_executor.submit(load_vae_timed) if common_executor is not None else None

  try:
    with jax.default_device(cpu_device):
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        import flax.linen.spmd as flax_spmd

        def unbox_fn(x):
          return x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x

        params = jax.tree_util.tree_map(
            unbox_fn,
            abstract_transformer_load_vars["params"],
            is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
        )
        params = flax.core.unfreeze(params)

        qwen3_params = jax.tree_util.tree_map(
            unbox_fn,
            abstract_qwen3_load_vars["params"],
            is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
        )
        qwen3_params = flax.core.unfreeze(qwen3_params)

        # Components read from the weight cache; their trees are final (permuted, quantized).
        weight_cache_hits = set()

        def load_transformer_timed():
          t0 = time.perf_counter()
          # Build-only mode builds without a read: the cache was just found incomplete.
          result, _, hit = load_or_build_host_params(
              None if build_only else transformer_cache,
              abstract_transformer_vars["params"],
              lambda: (load_and_convert_krea2_weights(transformer_path, params, num_layers), None),
              load_trace,
              "transformer_cache_read",
          )
          if hit:
            weight_cache_hits.add("transformer")
          return result, time.perf_counter() - t0

        # Host-side embedding table (krea2_text_embed_on_host), filled by load_qwen_timed.
        text_embedding = []

        def build_qwen_host_params():
          result = load_and_convert_qwen3_weights(
              text_encoder_path, qwen3_params, qwen3_config, key_prefix="model.language_model."
          )
          # Bump KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION (weight_cache.py) when this changes the values.
          if config.weights_dtype == jnp.bfloat16:
            max_logging.log("Normalizing Qwen3 dtypes (BF16 weights, FP32 norms; matching dtypes are reused)...")
            cast_dict_to_bfloat16_inplace(result, exclude_keywords=("norm",))
          if te_quantization == "int8":
            t_quant = time.perf_counter()
            # Scales in the compute dtype: dequantized weights (and so every
            # matmul and activation) keep the unquantized model's dtypes.
            result = quantize_text_encoder_params(
                result, abstract_qwen3_vars["params"], scale_dtype=qwen3_config.dtype, device=cpu_device
            )
            load_trace["qwen_quantize"] = time.perf_counter() - t_quant
          extras = None
          if te_embed_on_host:
            t_embed = time.perf_counter()
            table = load_qwen3_embedding_table(text_encoder_path, key_prefix="model.language_model.")
            extras = {"embedding_table": table}
            load_trace["qwen_embedding_table"] = time.perf_counter() - t_embed
          return result, extras

        def load_qwen_timed():
          t0 = time.perf_counter()
          # The cached scales have the compute dtype, the abstract tree's may not: floating
          # leaves only need a floating dtype, the others (int8 qvalues) their exact dtype.
          result, extras, hit = load_or_build_host_params(
              None if build_only else text_encoder_cache,
              abstract_qwen3_vars["params"],
              build_qwen_host_params,
              load_trace,
              "qwen_cache_read",
              extra_names=qwen_extra_names,
              check_dtypes=False,
          )
          if hit:
            weight_cache_hits.add("text_encoder")
          else:
            weight_cache_saved["text_encoder"] = save_host_params(
                text_encoder_cache, result, load_trace, "qwen_cache_write", extras=extras
            )
          if te_embed_on_host:
            text_embedding.append(extras["embedding_table"])
          return result, time.perf_counter() - t0

        if parallel_loading and load_transformer and load_text_encoder:
          with ThreadPoolExecutor(max_workers=2) as weight_executor:
            transformer_future = weight_executor.submit(load_transformer_timed)
            qwen_future = weight_executor.submit(load_qwen_timed)
            params, load_trace["transformer_host"] = transformer_future.result()
            qwen3_params, load_trace["qwen_host"] = qwen_future.result()
            # Drop the futures: they would pin the float trees (with W8A8 the ~24 GiB
            # float kernels that quantize_transformer_params replaces) until main returns.
            del transformer_future, qwen_future
        else:
          if load_transformer:
            params, load_trace["transformer_host"] = load_transformer_timed()
          if load_text_encoder:
            qwen3_params, load_trace["qwen_host"] = load_qwen_timed()

        if "transformer" in weight_cache_hits:
          # Already permuted and quantized (permuting again would corrupt it); no
          # LoRA (it bypasses the cache). Only check it against the runtime model.
          check_transformer_param_tree(params, abstract_transformer_vars["params"])
        elif load_transformer:
          # load_and_convert_krea2_weights zero-fills the lora-* leaves it
          # doesn't recognize, so write the real adapter tensors afterwards.
          t0 = time.perf_counter()
          params = insert_lora_params(params, lora_flat_params)
          params = apply_diff_updates(params, lora_diff_updates)
          load_trace["lora_apply"] = time.perf_counter() - t0

          # rotate_half RoPE needs q/k head dims reordered. Runs after the LoRA
          # up kernels and diff updates are in the tree so they are permuted too.
          if transformer.rope_layout == "rotate_half":
            t0 = time.perf_counter()
            params = permute_rope_weights_to_rotate_half(
                params,
                num_heads=transformer.num_attention_heads,
                num_kv_heads=transformer.num_key_value_heads,
                head_dim=transformer.attention_head_dim,
            )
            load_trace["rope_permute"] = time.perf_counter() - t0

          # W8A8: quantize the final float kernels (LoRA subtrees and the rotate-half
          # permutation included), then check the tree against the runtime model's.
          if transformer_quantization:
            t0 = time.perf_counter()
            params = quantize_transformer_params(params, transformer_quant_targets, scale_dtype=config.weights_dtype)
            check_transformer_param_tree(params, abstract_transformer_vars["params"])
            load_trace["transformer_quantize"] = time.perf_counter() - t0

          weight_cache_saved["transformer"] = save_host_params(
              transformer_cache, params, load_trace, "transformer_cache_write"
          )

        if build_only:
          # Both components were attempted; no tokenizer, pipeline, device placement or generation.
          check_weight_cache_saves(weight_cache_builds, weight_cache_saved)
          log_weight_cache_build_summary(
              {"transformer": transformer_cache, "text_encoder": text_encoder_cache},
              weight_cache_builds,
              weight_cache_saved,
              load_trace,
              time.time() - t_load_start,
          )
          return

        params = flax.core.freeze(params)
        qwen3_params = flax.core.freeze(qwen3_params)

        if vae_future is not None:
          (vae, vae_cache), load_trace["vae"] = vae_future.result()

        max_logging.log("Placing parameters into final TPU shardings...")
        if offload_components:
          max_logging.log(f"Keeping offloaded components on host until their phase: {', '.join(offload_components)}")
        t0 = time.perf_counter()
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          # Keep the established callback path: it supplies process-local
          # slices directly and was faster than staging a batched device_put
          # on the single-host v5e reference run.
          if "transformer" not in offload_components:
            params = jax.tree_util.tree_map(max_utils.device_put_replicated, params, transformer_shardings)
          if "text_encoder" not in offload_components:
            qwen3_params = jax.tree_util.tree_map(max_utils.device_put_replicated, qwen3_params, qwen3_shardings)
        load_trace["device_placement"] = time.perf_counter() - t0
        max_logging.log("All resident parameters placed on device HBM successfully!")
        gc.collect()
        jax.effects_barrier()
  finally:
    if common_executor is not None:
      common_executor.shutdown(wait=True)

  # 8. VAE (Qwen-Image / Wan 2.1 architecture, NNX). In sequential mode,
  # preserve the historical order and load it after the main parameter trees.
  if vae_future is None:
    max_logging.log("Loading Qwen-Image VAE...")
    (vae, vae_cache), load_trace["vae"] = load_vae_timed()

  load_time = time.time() - t_load_start
  max_logging.log(f" -> [TIMING] Total Model Loading & Device Placement: {load_time:.2f} seconds")
  max_logging.log(
      " -> [TIMING] Load breakdown: "
      + ", ".join(f"{stage}={seconds:.2f}s" for stage, seconds in load_trace.items())
  )

  # 9. Tokenizer
  tokenizer = load_krea2_tokenizer(tokenizer_path, snapshot_dir)

  # 10. FlowMatch scheduler (exponential dynamic shifting; mu is set per-call)
  scheduler = FlaxFlowMatchScheduler(
      num_train_timesteps=1000,
      shift=1.0,
      sigma_max=1.0,
      sigma_min=0.001,
      inverse_timesteps=False,
      extra_one_step=False,
      reverse_sigmas=False,
      use_dynamic_shifting=True,
      time_shift_type="exponential",
  )

  # 11. Pipeline
  max_logging.log("Instantiating FlaxKrea2Pipeline...")
  pipeline = FlaxKrea2Pipeline(
      transformer=transformer,
      vae=vae,
      vae_cache=vae_cache,
      text_encoder=qwen3_runtime_model,
      tokenizer=tokenizer,
      scheduler=scheduler,
      config=config,
      mesh=mesh,
      vae_mesh=vae_mesh,
      vae_logical_axis_rules=vae_logical_axis_rules,
      lora_compile_spec=lora_compile_spec,
      offload_components=offload_components,
      param_shardings={"transformer": transformer_shardings, "text_encoder": qwen3_shardings},
      text_embedding_table=text_embedding[0] if text_embedding else None,
  )
  # Register the Krea-specific jitted entry points before installing the
  # process-global AOT cache so existing per-shape executables can load.
  pipeline._setup_jit_functions()
  aot_cache.install(
      getattr(config, "aot_cache_dir", ""),
      meta={
          "model": config.pretrained_model_name_or_path,
          "attention": config.attention,
          "flash_block_sizes": str(config.flash_block_sizes),
          "mesh_shape": str(mesh.shape),
          "vae_mesh_shape": str(vae_mesh.shape),
          "weights_dtype": str(config.weights_dtype),
          "activations_dtype": str(config.activations_dtype),
          "max_sequence_length": str(config.max_sequence_length),
          "krea2_staged_transformer": str(config.krea2_staged_transformer),
          "krea2_staged_donate_hidden_states": str(getattr(config, "krea2_staged_donate_hidden_states", True)),
          "krea2_offload_components": str(offload_components),
          "krea2_text_encoder_quantization": f"{te_quantization}:{te_quant_tile_size}" if te_quantization else "",
          "krea2_text_embed_on_host": str(te_embed_on_host),
          "krea2_rope_layout": transformer.rope_layout,
          # Conditional key: with quantization off the meta (and fingerprint) matches older caches.
          **transformer_quantization_aot_meta(transformer_quantization, transformer_quant_targets),
          # Conditional key: only flash_custom picks block sizes automatically per chip.
          **flash_custom_block_selection_aot_meta(config.attention),
          "lora_compile_spec": lora_compile_spec,
          "jax": jax.__version__,
      },
      mesh=mesh,
      # With many cached shapes (e.g. after krea2_precompile) load only the ones this run calls.
      lazy_load=bool(getattr(config, "aot_cache_lazy_load", False)),
  )
  aot_cache.wait_for_loads()

  latents_to_use = None
  if getattr(config, "latents_path", ""):
    max_logging.log(f"Loading custom starting noise latents from: {config.latents_path}...")
    latents_to_use = np.load(config.latents_path)
    max_logging.log(f" -> Custom latents shape: {latents_to_use.shape}")

  call_kwargs = {
      "params": params,
      "qwen3_params": qwen3_params,
      "height": height,
      "width": width,
      "num_inference_steps": config.num_inference_steps,
      "guidance_scale": config.guidance_scale,
      "do_classifier_free_guidance": config.do_classifier_free_guidance,
      "negative_prompt": config.negative_prompt,
      "batch_size": config.batch_size,
      "latents": latents_to_use,
      "output_dir": config.output_dir,
  }

  # Swap-in entries are only present for offloaded components.
  timed_phases = (
      "text_encoder_swap_in",
      "prompt_encoding",
      "transformer_swap_in",
      "denoise_loop",
      "vae_decode",
  )

  def log_swap(trace, component, label):
    if f"{component}_swap_in" in trace:
      swap_gib = trace[f"{component}_swap_bytes"] / 1024**3
      max_logging.log(f"   - {label}: {trace[f'{component}_swap_in']:.2f}s ({swap_gib:.2f} GiB)")

  # One interceptor context spans both passes so every (re)trace of the jitted
  # transformer step sees the LoRA interceptors.
  with ExitStack() as stack:
    for interceptor in lora_interceptors:
      stack.enter_context(nn.intercept_methods(interceptor))

    if precompile_plan:
      # Compile and save only: no warmup/timed pass, no image, no profile. The
      # executables depend on the text bucket, not on the prompt text, so a
      # fixed short prompt and an empty negative prompt leave the bucket to the
      # plan entry alone (a long configured prompt would force a larger one).
      precompile_records = run_precompile(
          pipeline,
          precompile_plan,
          {**call_kwargs, "negative_prompt": ""},
          [KREA2_PRECOMPILE_PROMPT] * config.batch_size,
      )
      log_precompile_summary(precompile_records)
      max_logging.log(f"SUCCESS! Precompile complete for {len(precompile_records)} resolution/text combination(s)!")
      return

    max_logging.log("Running compile warmup (zero-execution when AOT cache is enabled)...")
    with aot_cache.warmup_mode():
      _, warmup_trace = pipeline(
          prompt=active_prompts,
          output_name="krea2_warmup.png",
          save_outputs=False,
          **call_kwargs,
      )
    # Persist newly-seen shape signatures synchronously. Saving in the
    # background competes with the first real request for CPU and disk I/O.
    aot_cache.save_pending()
    warmup_time = sum(warmup_trace.get(k, 0.0) for k in timed_phases)

    max_logging.log("Running timed pass at full device speed...")
    with max_utils.Profiler(config, session_name="krea2_timed"):
      with jax.profiler.StepTraceAnnotation("krea2_generate", step_num=0):
        _, main_trace = pipeline(prompt=active_prompts, output_name=config.output_name, **call_kwargs)
    main_time = sum(main_trace.get(k, 0.0) for k in timed_phases)

  if getattr(config, "enable_profiler", False) and jax.process_index() == 0:
    profile_dir = os.path.join(config.tensorboard_dir, "krea2_timed")
    os.makedirs(profile_dir, exist_ok=True)
    memory_profile_path = os.path.join(profile_dir, "device_memory_profile.pprof")
    try:
      jax.profiler.save_device_memory_profile(memory_profile_path)
      max_logging.log(f"Saved device memory profile: {memory_profile_path}")
    except (OSError, RuntimeError, ValueError) as exc:
      max_logging.log(f"Warning: unable to save device memory profile: {exc}")
    for device in jax.local_devices():
      memory_stats = device.memory_stats()
      if memory_stats:
        gib = 1024**3
        max_logging.log(
            f"[HBM] device={device.id} "
            f"in_use={memory_stats.get('bytes_in_use', 0) / gib:.2f} GiB "
            f"peak={memory_stats.get('peak_bytes_in_use', 0) / gib:.2f} GiB "
            f"limit={memory_stats.get('bytes_limit', 0) / gib:.2f} GiB"
        )

  max_logging.log("=" * 80)
  max_logging.log("KREA 2 LATENCY & TIMING BREAKDOWN")
  max_logging.log("=" * 80)
  max_logging.log(f"1) Total Model Loading & Placement Time:  {load_time:.2f} seconds")
  max_logging.log(f"2) Cold-Start / Warmup Pass (XLA Compilation): {warmup_time:.2f} seconds")
  log_swap(warmup_trace, "text_encoder", "Qwen3-VL Swap-in ")
  max_logging.log(f"   - Qwen3-VL Encoding: {warmup_trace.get('prompt_encoding', 0.0):.2f}s")
  log_swap(warmup_trace, "transformer", "Krea2 Swap-in    ")
  max_logging.log(f"   - Krea2 Denoising:   {warmup_trace.get('denoise_loop', 0.0):.2f}s")
  max_logging.log(f"   - VAE Decoding:      {warmup_trace.get('vae_decode', 0.0):.2f}s")
  max_logging.log(f"3) Main Warmed-Up Pass: {main_time:.2f} seconds")
  log_swap(main_trace, "text_encoder", "Qwen3-VL Swap-in ")
  max_logging.log(f"   - Qwen3-VL Encoding: {main_trace.get('prompt_encoding', 0.0):.2f}s")
  log_swap(main_trace, "transformer", "Krea2 Swap-in    ")
  max_logging.log(f"   - Krea2 Denoising:   {main_trace.get('denoise_loop', 0.0):.2f}s")
  max_logging.log(f"   - VAE Decoding:      {main_trace.get('vae_decode', 0.0):.2f}s")
  max_logging.log("=" * 80)
  max_logging.log(f"SUCCESS! Generation complete for {config.batch_size} image(s)!")


if __name__ == "__main__":
  app.run(main)
