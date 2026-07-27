# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inference-time LoRA loading for the Krea 2 Flax transformer. Follows the Flux
# Linen interceptor pattern: LoRA tensors live under a `lora-{adapter}` subtree
# inside each target Dense's params, and an `nn.intercept_methods` interceptor
# adds the low-rank update to every intercepted `Dense.__call__`.

import glob
import inspect
import json
import os
import posixpath
from typing import Dict, Union

import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

from .. import max_logging
from ..models.krea2.lora_util import convert_krea2_lora_to_flax, parse_lora_metadata
from ..models.lora import BaseLoRALayer, LoRALinearLayer
from .lora_base import LoRABaseMixin

PEFT_ADAPTER_CONFIG_NAME = "adapter_config.json"
_HF_USER_AGENT = {"file_type": "attn_procs_weights", "framework": "pytorch"}


def _load_safetensors_with_metadata(file_path):
  """Loads tensors AND the file-level metadata (safetensors.torch.load_file
  discards the latter, which carries diffusers/PEFT/kohya scaling settings)."""
  from safetensors import safe_open

  state_dict = {}
  with safe_open(file_path, framework="pt", device="cpu") as f:
    metadata = f.metadata() or {}
    for key in f.keys():
      state_dict[key] = f.get_tensor(key)
  return state_dict, metadata


def _maybe_read_adapter_config(directory):
  """Reads a PEFT adapter_config.json if one sits next to the weights."""
  config_path = os.path.join(directory, PEFT_ADAPTER_CONFIG_NAME)
  if os.path.isfile(config_path):
    with open(config_path, "r") as f:
      return json.load(f)
  return None


class Krea2LoraLoaderMixin(LoRABaseMixin):
  _lora_lodable_modules = ["transformer"]

  @classmethod
  def _resolve_single_safetensors(cls, candidates, source):
    if len(candidates) == 1:
      return candidates[0]
    if not candidates:
      raise ValueError(f"No .safetensors file found in {source}.")
    raise ValueError(
        f"Multiple .safetensors files found in {source}: {sorted(candidates)}. "
        "Set lora_config.weight_name to pick one."
    )

  @classmethod
  def lora_state_dict(cls, pretrained_model_name_or_path_or_dict: Union[str, Dict], weight_name=None, **kwargs):
    """Resolves the adapter weights to a state dict plus its scaling metadata.

    Returns `(state_dict, sft_metadata, adapter_config)` where `sft_metadata`
    is the safetensors file-level metadata dict and `adapter_config` a parsed
    PEFT adapter_config.json (or None).
    """
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
      return pretrained_model_name_or_path_or_dict, {}, None

    path = pretrained_model_name_or_path_or_dict
    subfolder = kwargs.get("subfolder")
    if os.path.isfile(path):
      file_path = path
    elif os.path.isdir(path):
      search_dir = os.path.join(path, subfolder) if subfolder else path
      if weight_name:
        file_path = os.path.join(search_dir, weight_name)
        if not os.path.isfile(file_path):
          raise ValueError(f"LoRA weight file {weight_name} not found in directory {search_dir}.")
      else:
        file_path = cls._resolve_single_safetensors(
            glob.glob(os.path.join(search_dir, "*.safetensors")), f"directory {search_dir}"
        )
    else:
      # HF Hub repo id.
      from huggingface_hub import hf_hub_download
      from huggingface_hub.utils import EntryNotFoundError

      revision = kwargs.pop("revision", None)
      subfolder = kwargs.pop("subfolder", None)
      token = kwargs.pop("token", None)
      use_auth_token = kwargs.pop("use_auth_token", None)
      if token is None:
        token = use_auth_token
      download_kwargs = {
          "revision": revision,
          "cache_dir": kwargs.pop("cache_dir", None),
          "force_download": kwargs.pop("force_download", False),
          "local_files_only": kwargs.pop("local_files_only", False),
          "token": token,
          "user_agent": _HF_USER_AGENT,
      }
      # huggingface_hub 0.x accepts these legacy options, while newer
      # releases removed them. Forward them only when the installed version
      # still exposes the corresponding parameters.
      download_parameters = inspect.signature(hf_hub_download).parameters
      accepts_extra_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in download_parameters.values())
      for option in ("proxies", "resume_download"):
        if option in kwargs:
          value = kwargs.pop(option)
          if option in download_parameters or accepts_extra_kwargs:
            download_kwargs[option] = value

      if not weight_name:
        from huggingface_hub import HfApi

        hub_api = HfApi()
        if download_kwargs["local_files_only"]:
          snapshot_parameters = inspect.signature(hub_api.snapshot_download).parameters
          snapshot_kwargs = {
              key: value
              for key, value in download_kwargs.items()
              if key in snapshot_parameters and key != "user_agent"
          }
          snapshot_kwargs["allow_patterns"] = [
              "*.safetensors",
              "**/*.safetensors",
              "adapter_config.json",
              "**/adapter_config.json",
          ]
          snapshot_dir = hub_api.snapshot_download(repo_id=path, **snapshot_kwargs)
          search_dir = os.path.join(snapshot_dir, subfolder) if subfolder else snapshot_dir
          file_path = cls._resolve_single_safetensors(
              glob.glob(os.path.join(search_dir, "**", "*.safetensors"), recursive=True),
              f"cached HF Hub repo {path}" + (f" subfolder {subfolder}" if subfolder else ""),
          )
          state_dict, metadata = _load_safetensors_with_metadata(file_path)
          return state_dict, metadata, _maybe_read_adapter_config(os.path.dirname(os.path.abspath(file_path)))

        repo_files = hub_api.list_repo_files(path, revision=revision, token=token)
        prefix = f"{subfolder.strip('/')}/" if subfolder else ""
        weight_name = cls._resolve_single_safetensors(
            [f[len(prefix) :] for f in repo_files if f.startswith(prefix) and f.endswith(".safetensors")],
            f"HF Hub repo {path}" + (f" subfolder {subfolder}" if subfolder else ""),
        )
      file_path = hf_hub_download(repo_id=path, filename=weight_name, subfolder=subfolder, **download_kwargs)

      # PEFT stores adapter_config.json next to its weights. Try that location
      # first, then the requested subfolder and repository root for legacy
      # layouts.
      weight_dir = posixpath.dirname(posixpath.join(subfolder or "", weight_name)) or None
      requested_subfolder = subfolder.strip("/") if subfolder else None
      config_subfolders = []
      for candidate in (weight_dir, requested_subfolder, None):
        if candidate not in config_subfolders:
          config_subfolders.append(candidate)
      adapter_config = None
      for config_subfolder in config_subfolders:
        try:
          config_path = hf_hub_download(
              repo_id=path,
              filename=PEFT_ADAPTER_CONFIG_NAME,
              subfolder=config_subfolder,
              **download_kwargs,
          )
          with open(config_path, "r") as f:
            adapter_config = json.load(f)
          break
        except EntryNotFoundError:
          continue
      state_dict, metadata = _load_safetensors_with_metadata(file_path)
      return state_dict, metadata, adapter_config

    state_dict, metadata = _load_safetensors_with_metadata(file_path)
    return state_dict, metadata, _maybe_read_adapter_config(os.path.dirname(os.path.abspath(file_path)))

  @classmethod
  def load_lora_weights(cls, pretrained_model_name_or_path_or_dict, weight_name, adapter_name, weights_dtype, **kwargs):
    state_dict, sft_metadata, adapter_config = cls.lora_state_dict(
        pretrained_model_name_or_path_or_dict, weight_name=weight_name, **kwargs
    )
    lora_meta = parse_lora_metadata(sft_metadata, adapter_config)
    return convert_krea2_lora_to_flax(state_dict, adapter_name, weights_dtype, lora_meta=lora_meta)

  @classmethod
  def make_lora_interceptor(cls, ranks_by_path, network_alphas_by_path, adapter_name, scale=1.0):
    """Builds an `nn.intercept_methods` interceptor adding this adapter's LoRA
    update to every Dense whose module path appears in `ranks_by_path`."""
    lora_keys = frozenset(ranks_by_path.keys())

    def _intercept(next_fn, args, kwargs, context):
      mod = context.module
      while mod is not None:
        if isinstance(mod, BaseLoRALayer):
          return next_fn(*args, **kwargs)
        mod = mod.parent
      h = next_fn(*args, **kwargs)
      if context.method_name == "__call__":
        module_path = context.module.path
        if module_path in lora_keys:
          lora_layer = LoRALinearLayer(
              out_features=context.module.features,
              rank=ranks_by_path[module_path],
              network_alpha=network_alphas_by_path.get(module_path),
              dtype=context.module.dtype,
              weights_dtype=context.module.param_dtype,
              precision=context.module.precision,
              lora_scale=scale,
              name=f"lora-{adapter_name}",
          )
          return lora_layer(h, *args, **kwargs)
      return h

    return _intercept


def insert_lora_params(params, flat_lora_params):
  """Writes converted LoRA tensors into the (unfrozen) transformer param tree.

  Must run AFTER `load_and_convert_krea2_weights`, which zero-fills the lora-*
  leaves it doesn't recognize — writing earlier would silently zero the adapter.
  """
  if not flat_lora_params:
    return params
  flat = flatten_dict(params)
  for path, value in flat_lora_params.items():
    if path not in flat:
      raise ValueError(
          f"LoRA param {'.'.join(path)} is missing from the transformer param tree; "
          "the interceptors were not active while model shapes were evaluated."
      )
    if tuple(flat[path].shape) != tuple(value.shape):
      raise ValueError(
          f"LoRA param {'.'.join(path)} has shape {tuple(value.shape)} but the model "
          f"expects {tuple(flat[path].shape)}; this LoRA was likely trained for a different model."
      )
    flat[path] = value.astype(flat[path].dtype)
  return unflatten_dict(flat)


def apply_diff_updates(params, diff_updates, scale=1.0):
  """Additively merges `.diff` / `.diff_b` / norm-weight deltas into base params.
  Deltas are already in Flax layout (kernels transposed by the converter)."""
  if not diff_updates:
    return params
  flat = flatten_dict(params)
  applied_updates = 0
  for path, delta in diff_updates.items():
    if path not in flat:
      max_logging.log(f"WARNING: LoRA diff target {'.'.join(path)} not found in Krea 2 params; skipping.")
      continue
    leaf = flat[path]
    delta = jnp.asarray(delta)
    if tuple(delta.shape) != tuple(leaf.shape):
      raise ValueError(
          f"LoRA diff {'.'.join(path)} has shape {tuple(delta.shape)} but the model "
          f"expects {tuple(leaf.shape)}; this LoRA was likely trained for a different model."
      )
    flat[path] = (leaf + (delta * scale).astype(leaf.dtype)).astype(leaf.dtype)
    applied_updates += 1
  if applied_updates == 0:
    raise ValueError(
        "None of the LoRA diff targets exist in the Krea 2 parameter tree; "
        "the adapter was likely trained for a different model."
    )
  return unflatten_dict(flat)


def make_lora_compile_spec(adapter_name, scale, ranks, alphas, diffs):
  """Returns the immutable LoRA metadata that can change a compiled graph.

  Adapter tensor values remain runtime parameters and intentionally do not
  participate in this spec. Interceptor scale, rank and resolved alpha are
  captured while tracing, so the AOT executable fingerprint must include them.
  """
  modules = tuple(
      sorted(
          (
              ".".join(path),
              int(rank),
              None if alphas.get(path) is None else float(alphas[path]),
          )
          for path, rank in ranks.items()
      )
  )
  diff_targets = tuple(sorted(".".join(path) for path in diffs))
  return (str(adapter_name), float(scale), modules, diff_targets)


def maybe_load_krea2_lora(config, weights_dtype=jnp.bfloat16, return_compile_spec=False):
  """Loads every adapter listed in `config.lora_config`.

  Returns `(flat_lora_params, interceptors, diff_updates)`, plus `compile_spec`
  when `return_compile_spec=True`. With no adapters configured, the
  parameter/update dicts and compile spec are empty, while `interceptors`
  holds a single no-op so callers can wrap unconditionally (Flux convention).
  """

  def _noop_interceptor(next_fn, args, kwargs, context):
    return next_fn(*args, **kwargs)

  lora_config = config.lora_config
  model_paths = lora_config["lora_model_name_or_path"]
  if len(model_paths) == 0:
    result = ({}, [_noop_interceptor], {})
    return (*result, ()) if return_compile_spec else result

  def _entry(key, i, default=None):
    values = lora_config.get(key, [])
    return values[i] if i < len(values) else default

  adapter_names = [_entry("adapter_name", i) or f"adapter_{i}" for i in range(len(model_paths))]
  assert len(set(adapter_names)) == len(adapter_names), f"Duplicate LoRA adapter names: {adapter_names}"

  flat_lora_params = {}
  interceptors = []
  diff_updates = {}
  compile_spec = []
  for i, model_path in enumerate(model_paths):
    adapter_name = adapter_names[i]
    scale = float(_entry("scale", i, 1.0))
    max_logging.log(f"Loading Krea 2 LoRA '{adapter_name}' from {model_path} (scale={scale})...")
    lora_params, ranks, alphas, diffs = Krea2LoraLoaderMixin.load_lora_weights(
        model_path,
        weight_name=_entry("weight_name", i),
        adapter_name=adapter_name,
        weights_dtype=weights_dtype,
    )
    if not ranks and not diffs:
      raise ValueError(
          f"LoRA adapter '{adapter_name}' from {model_path} contains no keys applicable to the "
          "Krea 2 transformer (see the skipped/unmatched key warnings above). The file is likely "
          "in an unsupported format or was trained for a different model."
      )
    flat_lora_params.update(lora_params)
    for path, delta in diffs.items():
      scaled = jnp.asarray(delta) * scale
      diff_updates[path] = diff_updates[path] + scaled if path in diff_updates else scaled
    interceptors.append(Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, adapter_name, scale))
    compile_spec.append(make_lora_compile_spec(adapter_name, scale, ranks, alphas, diffs))
    max_logging.log(
        f"Krea 2 LoRA '{adapter_name}': {len(ranks)} LoRA module(s), {len(diffs)} diff update(s) loaded."
    )
  result = (flat_lora_params, interceptors, diff_updates)
  return (*result, tuple(compile_spec)) if return_compile_spec else result
