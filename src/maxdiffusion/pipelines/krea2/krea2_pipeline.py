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

# End-to-end JAX inference pipeline for Krea 2 (K2) Raw and Turbo.

import os
import time
from typing import Any, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from PIL import Image

from maxdiffusion import aot_cache, max_logging
from maxdiffusion.max_utils import device_put_replicated

from ...models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
from ...models.krea2.util import (
    KREA2_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
    KREA2_PROMPT_TEMPLATE_PREFIX,
    KREA2_PROMPT_TEMPLATE_START_IDX,
    KREA2_PROMPT_TEMPLATE_SUFFIX,
    KREA2_TEXT_ENCODER_SELECT_LAYERS,
    calculate_krea2_shift,
    mask_is_batch_uniform,
    prepare_krea2_image_ids,
    prepare_krea2_text_ids,
    round_up_to_multiple,
)
from ...models.flux.util import pack_latents, unpack_latents
from ...models.qwen3_flax import FlaxQwen3Model
from ...models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from ...schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler


@aot_cache.cached_jit
def vae_decode_pass(graphdef, state, rest_of_state, latents):
  """Decodes single-frame latents `(B, z_dim, 1, H/8, W/8)` to pixels in [-1, 1]."""
  wan_vae = nnx.merge(graphdef, state, rest_of_state)
  return wan_vae.decode(latents, AutoencoderKLWanCache(wan_vae), return_dict=False)[0]


class FlaxKrea2Pipeline:
  """
  Unified end-to-end inference pipeline for Krea 2 (Raw and Turbo) on JAX.

  Phase A encodes prompts with the Qwen3-VL text tower (Qwen-Image chat template
  with mid-sequence padding and cumulative-valid-token positions), Phase B runs
  the flow-matching denoise loop (with Krea-convention CFG when
  `guidance_scale > 0`), and Phase C decodes latents with the Qwen-Image (Wan
  architecture) VAE.
  """

  def __init__(
      self,
      transformer: Krea2Transformer2DModel,
      vae: AutoencoderKLWan,
      vae_cache: AutoencoderKLWanCache,
      text_encoder: FlaxQwen3Model,
      tokenizer,
      scheduler: FlaxFlowMatchScheduler,
      config,
      mesh,
      vae_mesh=None,
      vae_logical_axis_rules=None,
  ):
    self.transformer = transformer
    self.vae = vae
    self.vae_cache = vae_cache
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self.scheduler = scheduler
    self._config = config
    self.mesh = mesh
    self.vae_mesh = vae_mesh if vae_mesh is not None else mesh
    self.vae_logical_axis_rules = vae_logical_axis_rules if vae_logical_axis_rules is not None else config.logical_axis_rules

    self._jitted_qwen3_forward = None
    self._jitted_transformer_step = None
    self._jitted_vae_decode = None

  def _setup_jit_functions(self):
    if self._jitted_qwen3_forward is not None:
      return

    select_layers = tuple(KREA2_TEXT_ENCODER_SELECT_LAYERS)
    start_idx = KREA2_PROMPT_TEMPLATE_START_IDX

    @aot_cache.cached_jit
    def qwen3_forward(q_params, ids, mask, position_ids):
      _, all_hidden_states = self.text_encoder.apply(
          {"params": q_params}, input_ids=ids, attention_mask=mask, position_ids=position_ids
      )
      # Stack the tapped decoder layers per token: (B, S, num_text_layers, hidden)
      hidden = jnp.stack([all_hidden_states[i] for i in select_layers], axis=2)
      # Drop the system-prefix tokens.
      return hidden[:, start_idx:]

    @aot_cache.cached_jit
    def transformer_step(t_params, latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec):
      return self.transformer.apply(
          {"params": t_params},
          hidden_states=latents,
          encoder_hidden_states=prompt_embeds,
          timestep=t_vec,
          img_ids=img_ids,
          txt_ids=txt_ids,
          encoder_attention_mask=text_mask,
      ).sample

    self._jitted_qwen3_forward = qwen3_forward
    self._jitted_transformer_step = transformer_step
    self._jitted_vae_decode = vae_decode_pass

  def encode_prompt(self, prompts: List[str], qwen3_params):
    """Tokenizes prompts with the Qwen-Image fixed-length template
    `[prefix | prompt | PAD | suffix]` and taps the selected hidden states.

    Returns `(prompt_embeds, prompt_embeds_mask)` of shapes
    `(B, max_sequence_length, num_text_layers, text_hidden_dim)` and
    `(B, max_sequence_length)`.
    """
    max_sequence_length = self._config.max_sequence_length
    prefix_idx = KREA2_PROMPT_TEMPLATE_START_IDX
    num_suffix = KREA2_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS

    text = [KREA2_PROMPT_TEMPLATE_PREFIX + p for p in prompts]
    text_tokens = self.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length + prefix_idx - num_suffix,
        return_tensors="np",
    )
    suffix_tokens = self.tokenizer([KREA2_PROMPT_TEMPLATE_SUFFIX] * len(text), return_tensors="np")

    input_ids = np.concatenate([text_tokens["input_ids"], suffix_tokens["input_ids"]], axis=1)
    attention_mask = np.concatenate([text_tokens["attention_mask"], suffix_tokens["attention_mask"]], axis=1)

    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    attention_mask = jnp.array(attention_mask, dtype=jnp.int32)
    # Krea 2 pads mid-template, so rotary positions count only valid tokens
    # (padding does not consume a position).
    position_ids = jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, 0, None)

    prompt_embeds = self._jitted_qwen3_forward(qwen3_params, input_ids, attention_mask, position_ids)
    prompt_embeds_mask = attention_mask[:, prefix_idx:].astype(jnp.bool_)
    return prompt_embeds, prompt_embeds_mask

  def _prepare_latents(self, batch_size, height, width):
    num_channels_latents = 16
    latent_height = height // 8
    latent_width = width // 8
    latent_shape = (batch_size, num_channels_latents, latent_height, latent_width)

    seed_val = getattr(self._config, "seed", None)
    if seed_val is None:
      seed_val = int(time.time()) & 0x7FFFFFFF
    max_logging.log(f"Generating gaussian noise with seed: {seed_val} and unpacked shape: {latent_shape}...")
    np.random.seed(seed_val)
    latents_unpacked = np.random.randn(*latent_shape).astype(np.float32)
    # Pack 2x2 latent patches into the channel dim: (B, (H/16)*(W/16), 64)
    return pack_latents(latents_unpacked)

  def __call__(
      self,
      prompt: Union[str, List[str]],
      params,
      qwen3_params,
      height: int = 1024,
      width: int = 1024,
      num_inference_steps: int = 28,
      guidance_scale: float = 4.5,
      negative_prompt: Optional[Union[str, List[str]]] = None,
      batch_size: int = 1,
      latents: Optional[Any] = None,
      output_dir: str = "output/",
      output_name: str = "krea2_generated_image.png",
      save_outputs: bool = True,
  ):
    self._setup_jit_functions()

    if isinstance(prompt, str):
      prompts = [prompt] * batch_size
    else:
      prompts = prompt

    do_classifier_free_guidance = guidance_scale > 0.0
    if negative_prompt is None:
      negative_prompt = ""
    if isinstance(negative_prompt, str):
      negative_prompts = [negative_prompt] * batch_size
    else:
      negative_prompts = negative_prompt

    # The VAE downsamples 8x and latents are packed into 2x2 patches, so height
    # and width must be multiples of 16. Round up (with a warning) like the
    # diffusers reference pipeline instead of silently flooring.
    multiple = 16
    if height % multiple != 0 or width % multiple != 0:
      rounded_height = round_up_to_multiple(height, multiple)
      rounded_width = round_up_to_multiple(width, multiple)
      max_logging.log(
          f"Warning: height and width must be multiples of {multiple}; rounding up from "
          f"{height}x{width} to {rounded_height}x{rounded_width}."
      )
      height, width = rounded_height, rounded_width

    grid_height = height // 16
    grid_width = width // 16
    seq_len_img = grid_height * grid_width
    seq_len_txt = self._config.max_sequence_length

    # Latents (packed): (B, seq_len_img, 64)
    if latents is not None:
      latents_jax = jnp.array(latents)
      if latents_jax.ndim == 4:
        latents_jax = pack_latents(np.asarray(latents_jax))
    else:
      latents_jax = self._prepare_latents(batch_size, height, width)

    txt_ids_val = prepare_krea2_text_ids(batch_size, seq_len_txt)
    img_ids_val = prepare_krea2_image_ids(batch_size, grid_height, grid_width)

    # Scheduler: resolution-aware exponential shift for Raw, fixed mu for Turbo.
    if getattr(self._config, "is_distilled", False):
      mu = 1.15
    else:
      mu = calculate_krea2_shift(
          seq_len_img,
          base_seq_len=getattr(self._config, "base_image_seq_len", 256),
          max_seq_len=getattr(self._config, "max_image_seq_len", 6400),
          base_shift=getattr(self._config, "base_shift", 0.5),
          max_shift=getattr(self._config, "max_shift", 1.15),
      )
    scheduler_state = self.scheduler.create_state()
    sigmas_custom = jnp.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=jnp.float32)
    scheduler_state = self.scheduler.set_timesteps_ltx2(
        state=scheduler_state,
        num_inference_steps=num_inference_steps,
        shift=mu,
        sigmas=sigmas_custom,
    )

    trace = {}

    with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
      # -----------------------------------------------------------------
      # PHASE A: Encode prompts (Qwen3-VL text tower)
      # -----------------------------------------------------------------
      max_logging.log(f"[PHASE A] Encoding {len(prompts)} prompt(s) with the Qwen3-VL text encoder...")
      t0 = time.perf_counter()

      prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompts, qwen3_params)
      if do_classifier_free_guidance:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(negative_prompts, qwen3_params)
      prompt_embeds.block_until_ready()

      # The repo's flash-attention kernels share the key-padding mask of batch
      # element 0 across the whole batch (_build_padding_segment_ids). Refuse to
      # continue if that would silently miscompute the other batch elements.
      attention_kernel = getattr(self.transformer, "attention_kernel", "dot_product")
      if attention_kernel != "dot_product" and batch_size > 1:
        masks_uniform = mask_is_batch_uniform(prompt_embeds_mask)
        if do_classifier_free_guidance:
          masks_uniform = masks_uniform and mask_is_batch_uniform(negative_prompt_embeds_mask)
        if not masks_uniform:
          raise ValueError(
              f"attention='{attention_kernel}' shares the text padding mask of batch element 0 across the "
              "whole batch, but the prompts in this batch tokenize to different padding masks. "
              "Use identical prompts per batch, batch_size=1, or attention='dot_product'."
          )

      trace["prompt_encoding"] = time.perf_counter() - t0
      max_logging.log(f" -> [TIMING] Prompt Encoding (Qwen3-VL): {trace['prompt_encoding']:.4f} seconds")

      multihost_utils.sync_global_devices("krea2_phase_a_complete")

      # Shard batch inputs across the data axis.
      data_sharding = NamedSharding(self.mesh, P("data"))

      def put_data_on_devices(x, sharding):
        if isinstance(x, jax.Array) and hasattr(x, "sharding") and not x.sharding.is_fully_addressable:
          return x
        if hasattr(sharding, "is_fully_addressable") and sharding.is_fully_addressable:
          return jax.device_put(x, sharding)
        return device_put_replicated(x, sharding)

      latents_jax = put_data_on_devices(latents_jax, data_sharding)
      prompt_embeds = put_data_on_devices(prompt_embeds, data_sharding)
      prompt_embeds_mask = put_data_on_devices(prompt_embeds_mask, data_sharding)
      txt_ids_val = put_data_on_devices(txt_ids_val, data_sharding)
      img_ids_val = put_data_on_devices(img_ids_val, data_sharding)
      if do_classifier_free_guidance:
        negative_prompt_embeds = put_data_on_devices(negative_prompt_embeds, data_sharding)
        negative_prompt_embeds_mask = put_data_on_devices(negative_prompt_embeds_mask, data_sharding)

      multihost_utils.sync_global_devices("krea2_pre_phase_b_start")

      # -----------------------------------------------------------------
      # PHASE B: Denoising loop
      # -----------------------------------------------------------------
      cfg_note = f"CFG scale {guidance_scale}" if do_classifier_free_guidance else "no guidance"
      max_logging.log(f"[PHASE B] Running {num_inference_steps}-step denoise loop ({cfg_note})...")
      t0 = time.perf_counter()

      for step_idx in range(num_inference_steps):
        timestep = scheduler_state.timesteps[step_idx]
        t_vec = jnp.full((batch_size,), timestep / 1000.0, dtype=latents_jax.dtype)

        noise_pred = self._jitted_transformer_step(
            params, latents_jax, prompt_embeds, prompt_embeds_mask, img_ids_val, txt_ids_val, t_vec
        )
        if do_classifier_free_guidance:
          neg_noise_pred = self._jitted_transformer_step(
              params, latents_jax, negative_prompt_embeds, negative_prompt_embeds_mask, img_ids_val, txt_ids_val, t_vec
          )
          # Krea 2 guidance convention: cond + g * (cond - uncond); equals standard
          # CFG with scale (1 + g).
          noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

        prev_sample, _ = self.scheduler.step(
            state=scheduler_state,
            model_output=noise_pred.astype(latents_jax.dtype),
            timestep=timestep,
            sample=latents_jax,
            return_dict=False,
        )
        latents_jax = prev_sample

      latents_jax.block_until_ready()
      multihost_utils.sync_global_devices("krea2_phase_b_complete")

      trace["denoise_loop"] = time.perf_counter() - t0
      max_logging.log(f" -> [TIMING] Denoising Loop: {trace['denoise_loop']:.4f} seconds")

    # -----------------------------------------------------------------
    # PHASE C: Decode latents (Qwen-Image VAE, single frame)
    # -----------------------------------------------------------------
    max_logging.log("[PHASE C] Decoding final latents with the Qwen-Image VAE...")
    t0 = time.perf_counter()

    latents_unpacked = unpack_latents(np.asarray(latents_jax.astype(jnp.float32)), batch_size, 16, height, width)
    latents_mean = np.asarray(self.vae.latents_mean, dtype=np.float32).reshape(1, 16, 1, 1)
    latents_std = np.asarray(self.vae.latents_std, dtype=np.float32).reshape(1, 16, 1, 1)
    latents_unpacked = latents_unpacked * latents_std + latents_mean
    # Add a singleton frame dimension: (B, z_dim, 1, H/8, W/8)
    latents_5d = jnp.array(latents_unpacked[:, :, None, :, :], dtype=self._config.activations_dtype)

    with self.vae_mesh, nn_partitioning.axis_rules(self.vae_logical_axis_rules):
      graphdef, state, rest_of_state = nnx.split(self.vae, nnx.Param, ...)
      images = self._jitted_vae_decode(graphdef, state, rest_of_state, latents_5d)
      images.block_until_ready()

    trace["vae_decode"] = time.perf_counter() - t0
    max_logging.log(f" -> [TIMING] VAE Decoding: {trace['vae_decode']:.4f} seconds")

    # AOT warmup compiles against faithful shapes but substitutes zero-valued
    # device outputs. Skip host transfer, image conversion, and disk writes for
    # that deliberately-discarded pass.
    if not save_outputs:
      return [], trace

    # -----------------------------------------------------------------
    # POST-PROCESS: format and save
    # -----------------------------------------------------------------
    images = images[:, 0]  # (B, H, W, 3), values in [-1, 1]
    images = jnp.clip((images.astype(jnp.float32) + 1.0) / 2.0, 0.0, 1.0)
    if jax.process_count() > 1:
      images_numpy = multihost_utils.process_allgather(images, tiled=True)
    else:
      images_numpy = np.array(images)

    # Only process 0 writes files: on multihost every process holds the full
    # gathered batch, and concurrent writes to a shared filesystem would race.
    saved_paths = []
    if jax.process_index() == 0:
      for b_idx in range(batch_size):
        image_np = np.array(images_numpy[b_idx] * 255.0, dtype=np.uint8)
        if image_np.shape[0] == 3:
          image_np = image_np.transpose(1, 2, 0)
        img = Image.fromarray(image_np)

        if batch_size > 1:
          batch_output_name = output_name.replace(".png", f"_b{b_idx}.png")
        else:
          batch_output_name = output_name
        output_png_path = os.path.join(output_dir, batch_output_name)
        img.save(output_png_path)
        max_logging.log(f" -> Saved image: {output_png_path} | Prompt: '{prompts[b_idx]}'")
        saved_paths.append(output_png_path)

    return saved_paths, trace
