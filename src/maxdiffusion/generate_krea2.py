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

import gc
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
from contextlib import ExitStack
from typing import List

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


def build_krea2_transformer(transformer_cfg, config, mesh):
  """Builds the Krea 2 transformer module from a `transformer/config.json` dict (defaults if empty)."""
  from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel

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
  )


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
      round_up_to_multiple,
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
  transformer = build_krea2_transformer(transformer_cfg, config, mesh)

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
  # Height/width must be multiples of 16 (VAE 8x downsampling x 2x2 latent
  # patches). Round up here so the eval_shape dummies match what the pipeline
  # (which applies the same rounding) will actually run.
  height = round_up_to_multiple(config.height, 16)
  width = round_up_to_multiple(config.width, 16)
  if (height, width) != (config.height, config.width):
    max_logging.log(
        f"Warning: height and width must be multiples of 16; rounding up from "
        f"{config.height}x{config.width} to {height}x{width}."
    )

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

  def transformer_init_fn():
    return transformer.init(
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
      abstract_transformer_vars = jax.eval_shape(transformer_init_fn)
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

  # 7. Stream weights on host CPU, overlap the independent components, then
  # place the final trees directly into their target TPU shardings.
  max_logging.log("Streaming parameters from safetensors...")
  t_load_start = time.time()
  load_trace = {}
  parallel_loading = getattr(config, "parallel_component_loading", True)
  rngs = nnx.Rngs(jax.random.key(config.seed if config.seed is not None else 0))

  def load_vae_timed():
    t0 = time.perf_counter()
    result = load_qwen_image_vae(snapshot_dir, config, vae_mesh, rngs)
    return result, time.perf_counter() - t0

  # VAE is small and independent. Hide it behind the much larger Transformer
  # and Qwen host reads, but wait before the main device transfer so PCIe/ICI
  # traffic does not contend.
  common_executor = ThreadPoolExecutor(max_workers=1) if parallel_loading else None
  vae_future = common_executor.submit(load_vae_timed) if common_executor is not None else None

  try:
    with jax.default_device(cpu_device):
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        import flax.linen.spmd as flax_spmd

        def unbox_fn(x):
          return x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x

        params = jax.tree_util.tree_map(
            unbox_fn,
            abstract_transformer_vars["params"],
            is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
        )
        params = flax.core.unfreeze(params)

        qwen3_params = jax.tree_util.tree_map(
            unbox_fn,
            abstract_qwen3_load_vars["params"],
            is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
        )
        qwen3_params = flax.core.unfreeze(qwen3_params)

        def load_transformer_timed():
          t0 = time.perf_counter()
          result = load_and_convert_krea2_weights(transformer_path, params, num_layers)
          return result, time.perf_counter() - t0

        # Host-side embedding table (krea2_text_embed_on_host), filled by load_qwen_timed.
        text_embedding = []

        def load_qwen_timed():
          t0 = time.perf_counter()
          result = load_and_convert_qwen3_weights(
              text_encoder_path, qwen3_params, qwen3_config, key_prefix="model.language_model."
          )
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
          if te_embed_on_host:
            t_embed = time.perf_counter()
            text_embedding.append(load_qwen3_embedding_table(text_encoder_path, key_prefix="model.language_model."))
            load_trace["qwen_embedding_table"] = time.perf_counter() - t_embed
          return result, time.perf_counter() - t0

        if parallel_loading:
          with ThreadPoolExecutor(max_workers=2) as weight_executor:
            transformer_future = weight_executor.submit(load_transformer_timed)
            qwen_future = weight_executor.submit(load_qwen_timed)
            params, load_trace["transformer_host"] = transformer_future.result()
            qwen3_params, load_trace["qwen_host"] = qwen_future.result()
        else:
          params, load_trace["transformer_host"] = load_transformer_timed()
          qwen3_params, load_trace["qwen_host"] = load_qwen_timed()

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
          "lora_compile_spec": lora_compile_spec,
          "jax": jax.__version__,
      },
      mesh=mesh,
  )
  aot_cache.wait_for_loads()

  latents_to_use = None
  if getattr(config, "latents_path", ""):
    max_logging.log(f"Loading custom starting noise latents from: {config.latents_path}...")
    latents_to_use = np.load(config.latents_path)
    max_logging.log(f" -> Custom latents shape: {latents_to_use.shape}")

  call_kwargs = dict(
      params=params,
      qwen3_params=qwen3_params,
      height=height,
      width=width,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=config.guidance_scale,
      do_classifier_free_guidance=config.do_classifier_free_guidance,
      negative_prompt=config.negative_prompt,
      batch_size=config.batch_size,
      latents=latents_to_use,
      output_dir=config.output_dir,
  )

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
