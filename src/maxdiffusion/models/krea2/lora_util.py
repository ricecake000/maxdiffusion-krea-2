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

# Pure key-conversion logic for applying externally-trained LoRA adapters to the
# Krea 2 Flax transformer at inference. Accepts kohya (`lora_unet_*` underscore
# keys), ComfyUI (`diffusion_model.*`) and diffusers/PEFT (`transformer.*`,
# `lora_A`/`lora_B`) state-dict styles, including `.alpha`, `.diff` and
# `.diff_b` entries.

import json
import math
import re

import jax.numpy as jnp
import numpy as np

from maxdiffusion import max_logging

# Key diffusers uses to embed the PEFT LoraConfig in safetensors metadata
# (diffusers loaders/lora_base.py, LORA_ADAPTER_METADATA_KEY). Values inside the
# JSON blob are prefixed per component, e.g. "transformer.lora_alpha".
DIFFUSERS_LORA_METADATA_KEY = "lora_adapter_metadata"

# Suffix -> canonical entry kind. `.diff_b` must be tested before `.diff`.
_SUFFIX_PATTERNS = (
    (re.compile(r"^(?P<base>.+?)\.alpha$"), "alpha"),
    (re.compile(r"^(?P<base>.+?)\.diff_b$"), "diff_b"),
    (re.compile(r"^(?P<base>.+?)\.diff$"), "diff"),
    (re.compile(r"^(?P<base>.+?)\.lora_down\.weight$"), "down"),
    (re.compile(r"^(?P<base>.+?)\.lora_up\.weight$"), "up"),
    (re.compile(r"^(?P<base>.+?)\.lora_A\.weight$"), "down"),
    (re.compile(r"^(?P<base>.+?)\.lora_B\.weight$"), "up"),
    (re.compile(r"^(?P<base>.+?)\.lora\.down\.weight$"), "down"),
    (re.compile(r"^(?P<base>.+?)\.lora\.up\.weight$"), "up"),
)

_TEXT_ENCODER_PREFIXES = ("lora_te", "text_encoder.", "te.")
_STRIP_PREFIXES = ("transformer.", "diffusion_model.")
_KOHYA_UNET_PREFIX = "lora_unet_"

_ATTN_PROJ = r"to_q|to_k|to_v|to_gate|to_out_0|to_out|norm_q|norm_k"
_FF_PROJ = r"gate_proj|up_proj|down_proj|gate|up|down"

# Kohya flattens module paths with underscores, which is ambiguous to invert, so
# each Krea 2 module family is matched explicitly.
_KOHYA_UNDERSCORE_PATTERNS = (
    (
        re.compile(rf"^transformer_blocks_(\d+)_attn_({_ATTN_PROJ})$"),
        lambda m: f"transformer_blocks.{m.group(1)}.attn.{m.group(2)}",
    ),
    (
        re.compile(rf"^transformer_blocks_(\d+)_ff_({_FF_PROJ})$"),
        lambda m: f"transformer_blocks.{m.group(1)}.ff.{m.group(2)}",
    ),
    (
        re.compile(r"^transformer_blocks_(\d+)_(norm1|norm2|scale_shift_table)$"),
        lambda m: f"transformer_blocks.{m.group(1)}.{m.group(2)}",
    ),
    (
        re.compile(rf"^text_fusion_(layerwise|refiner)_blocks_(\d+)_attn_({_ATTN_PROJ})$"),
        lambda m: f"text_fusion.{m.group(1)}_blocks.{m.group(2)}.attn.{m.group(3)}",
    ),
    (
        re.compile(rf"^text_fusion_(layerwise|refiner)_blocks_(\d+)_ff_({_FF_PROJ})$"),
        lambda m: f"text_fusion.{m.group(1)}_blocks.{m.group(2)}.ff.{m.group(3)}",
    ),
    (
        re.compile(r"^text_fusion_(layerwise|refiner)_blocks_(\d+)_(norm1|norm2)$"),
        lambda m: f"text_fusion.{m.group(1)}_blocks.{m.group(2)}.{m.group(3)}",
    ),
)

_KOHYA_LITERALS = {
    "img_in": "img_in",
    "time_embed_linear_1": "time_embed.linear_1",
    "time_embed_linear_2": "time_embed.linear_2",
    "time_mod_proj": "time_mod_proj",
    "txt_in_norm": "txt_in.norm",
    "txt_in_linear_1": "txt_in.linear_1",
    "txt_in_linear_2": "txt_in.linear_2",
    "text_fusion_projector": "text_fusion.projector",
    "final_layer_norm": "final_layer.norm",
    "final_layer_linear": "final_layer.linear",
    "final_layer_scale_shift_table": "final_layer.scale_shift_table",
}

# The diffusers checkpoint uses `ff.gate` while some exporters use `ff.gate_proj`;
# the Flax modules are always the `_proj` names.
_FF_RENAME = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
}

_ATTN_LEAVES = ("to_q", "to_k", "to_v", "to_gate", "to_out", "norm_q", "norm_k")
_NORM_LEAVES = ("norm1", "norm2", "norm_q", "norm_k", "norm")


def _to_numpy(value):
  """Converts a torch tensor (incl. bf16) or array-like to a float numpy array."""
  if isinstance(value, np.ndarray):
    return value
  try:
    import torch

    if isinstance(value, torch.Tensor):
      return value.detach().to(torch.float32).cpu().numpy()
  except ImportError:
    pass
  return np.asarray(value)


def krea2_torch_path_to_flax_path(torch_path: str):
  """Maps a canonical diffusers-style Krea 2 module path to the Flax param-tree
  path tuple, or None if the path names no Krea 2 module. Mirrors the rename
  logic of `load_and_convert_krea2_weights`."""
  parts = torch_path.split(".")
  # diffusers wraps the output projection in a container: to_out.0 -> to_out.
  if len(parts) >= 2 and parts[-1] == "0" and parts[-2] == "to_out":
    parts = parts[:-1]
  # kohya flattens the same container into to_out_0.
  if parts[-1] == "to_out_0":
    parts[-1] = "to_out"

  if parts[0] == "transformer_blocks":
    if len(parts) < 3 or not parts[1].isdigit():
      return None
    prefix = (f"blocks_{parts[1]}",)
    rest = parts[2:]
  elif parts[0] == "text_fusion":
    if len(parts) == 2 and parts[1] == "projector":
      return ("text_fusion", "projector")
    if len(parts) < 4 or parts[1] not in ("layerwise_blocks", "refiner_blocks") or not parts[2].isdigit():
      return None
    prefix = ("text_fusion", f"{parts[1]}_{parts[2]}")
    rest = parts[3:]
  elif len(parts) == 1 and parts[0] in ("img_in", "time_mod_proj"):
    return (parts[0],)
  elif len(parts) == 2 and parts[0] == "time_embed" and parts[1] in ("linear_1", "linear_2"):
    return tuple(parts)
  elif len(parts) == 2 and parts[0] == "txt_in" and parts[1] in ("linear_1", "linear_2", "norm"):
    return tuple(parts)
  elif len(parts) == 2 and parts[0] == "final_layer" and parts[1] in ("linear", "norm", "scale_shift_table"):
    return tuple(parts)
  else:
    return None

  if len(rest) == 1 and rest[0] in ("norm1", "norm2"):
    return (*prefix, rest[0])
  if len(rest) == 1 and rest[0] == "scale_shift_table" and prefix[0].startswith("blocks_"):
    return (*prefix, "scale_shift_table")
  if len(rest) == 2 and rest[0] == "attn" and rest[1] in _ATTN_LEAVES:
    return (*prefix, "attn", rest[1])
  if len(rest) == 2 and rest[0] == "ff" and rest[1] in _FF_RENAME:
    return (*prefix, "ff", _FF_RENAME[rest[1]])
  return None


def _leaf_kind(flax_path):
  last = flax_path[-1]
  if last == "scale_shift_table":
    return "table"
  if last in _NORM_LEAVES:
    return "norm"
  return "dense"


def _canonicalize_base(base: str):
  """Normalizes a LoRA key's module path (prefix stripped, dots restored) to the
  canonical diffusers-style Krea 2 path, or None if it can't be recognized."""
  for prefix in _STRIP_PREFIXES:
    if base.startswith(prefix):
      base = base[len(prefix) :]
      break

  if base.startswith(_KOHYA_UNET_PREFIX):
    base = base[len(_KOHYA_UNET_PREFIX) :]
    if base in _KOHYA_LITERALS:
      return _KOHYA_LITERALS[base]
    for pattern, rebuild in _KOHYA_UNDERSCORE_PATTERNS:
      m = pattern.match(base)
      if m:
        return rebuild(m)
    return None
  return base


def normalize_krea2_lora_state_dict(state_dict):
  """Parses a LoRA state dict into canonical per-module entries.

  Returns `(modules, skipped_te_keys, unmatched_keys)` where `modules` maps the
  canonical diffusers-style module path to a dict with any of the keys
  `down`, `up`, `alpha`, `diff`, `diff_b` (numpy values, alpha as float).
  Text-encoder keys are collected into `skipped_te_keys`; keys that name no
  Krea 2 module land in `unmatched_keys`.
  """
  modules = {}
  skipped_te_keys = []
  unmatched_keys = []

  for key, value in state_dict.items():
    if key.startswith(_TEXT_ENCODER_PREFIXES):
      skipped_te_keys.append(key)
      continue

    base, kind = None, None
    for pattern, k in _SUFFIX_PATTERNS:
      m = pattern.match(key)
      if m:
        base, kind = m.group("base"), k
        break
    if base is None:
      unmatched_keys.append(key)
      continue

    if "attn_qkv" in base or ".attn.qkv" in base:
      raise NotImplementedError(
          f"Krea 2 LoRA key '{key}' uses a fused qkv projection, which is not supported; "
          "export the LoRA with separate to_q/to_k/to_v projections."
      )

    canonical = _canonicalize_base(base)
    if canonical is None or krea2_torch_path_to_flax_path(canonical) is None:
      unmatched_keys.append(key)
      continue

    entry = modules.setdefault(canonical, {})
    if kind == "alpha":
      entry["alpha"] = float(np.asarray(_to_numpy(value)))
    else:
      entry[kind] = _to_numpy(value)

  return modules, skipped_te_keys, unmatched_keys


def parse_lora_metadata(sft_metadata=None, adapter_config=None):
  """Extracts default LoRA scaling settings from side-channel sources.

  PEFT keeps `lora_alpha`/`use_rslora` in `adapter_config.json` and modern
  diffusers embeds the same LoraConfig as JSON in the safetensors metadata
  (`lora_adapter_metadata`, keys prefixed per component). Kohya files carry
  `ss_network_alpha` in the safetensors metadata. Returns
  `{"lora_alpha": float | None, "use_rslora": bool, "alpha_pattern": dict}`.
  `adapter_config` (a parsed adapter_config.json dict) takes precedence over
  the safetensors metadata.
  """
  meta = {"lora_alpha": None, "use_rslora": False, "alpha_pattern": {}}

  def _take(cfg):
    if cfg.get("lora_alpha") is not None:
      meta["lora_alpha"] = float(cfg["lora_alpha"])
    if cfg.get("use_rslora") is not None:
      use_rslora = cfg["use_rslora"]
      meta["use_rslora"] = use_rslora if isinstance(use_rslora, bool) else str(use_rslora).lower() == "true"
    if isinstance(cfg.get("alpha_pattern"), dict):
      meta["alpha_pattern"] = {str(k): float(v) for k, v in cfg["alpha_pattern"].items()}

  sft_metadata = sft_metadata or {}
  raw = sft_metadata.get(DIFFUSERS_LORA_METADATA_KEY)
  if raw:
    try:
      packed = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
      max_logging.log(f"WARNING: could not parse safetensors '{DIFFUSERS_LORA_METADATA_KEY}' metadata; ignoring it.")
      packed = {}
    # Prefer transformer-prefixed keys; fall back to un-prefixed (flat) keys.
    transformer_cfg = {k[len("transformer.") :]: v for k, v in packed.items() if k.startswith("transformer.")}
    flat_cfg = {k: v for k, v in packed.items() if "." not in k}
    _take(transformer_cfg or flat_cfg)

  # Kohya stores the training-wide network alpha as a metadata string.
  if meta["lora_alpha"] is None and sft_metadata.get("ss_network_alpha") is not None:
    try:
      meta["lora_alpha"] = float(sft_metadata["ss_network_alpha"])
    except (TypeError, ValueError):
      pass

  if adapter_config:
    _take(adapter_config)
  return meta


def _resolve_network_alpha(torch_path, rank, explicit_alpha, lora_meta):
  """Per-module alpha: an explicit kohya `.alpha` tensor wins (plain alpha/rank
  semantics), then a PEFT `alpha_pattern` match, then the default `lora_alpha`.
  Metadata-derived alphas honor `use_rslora` (scaling alpha/sqrt(rank)) by
  returning `alpha * sqrt(rank)`, since `LoRALinearLayer` multiplies by
  `network_alpha / rank`. Returns None when no source defines an alpha (factor
  1.0, i.e. alpha == rank)."""
  if explicit_alpha is not None:
    return explicit_alpha
  if not lora_meta:
    return None
  alpha = None
  for pattern, pattern_alpha in lora_meta.get("alpha_pattern", {}).items():
    # Match PEFT's get_pattern_key semantics: patterns are regular
    # expressions matched against the complete module path, with an optional
    # dotted prefix. A substring check silently misses anchored/escaped
    # patterns and can over-match similarly named modules.
    if re.match(rf"(.*\.)?({pattern})$", torch_path):
      alpha = float(pattern_alpha)
      break
  if alpha is None:
    alpha = lora_meta.get("lora_alpha")
  if alpha is None:
    return None
  if lora_meta.get("use_rslora"):
    return alpha * math.sqrt(rank)
  return alpha


def convert_krea2_lora_to_flax(state_dict, adapter_name, weights_dtype=jnp.bfloat16, lora_meta=None):
  """Converts a Krea 2 LoRA state dict into Flax-ready tensors.

  Returns:
    flat_lora_params: {(*flax_module_path, "lora-{adapter}", "down"/"up", "kernel"): jnp array}
      matching the params `LoRALinearLayer` creates under interception.
    ranks_by_path: {flax_module_path: int} per-module LoRA rank.
    network_alphas_by_path: {flax_module_path: float | None}; resolved per module
      from the explicit `.alpha` tensor, then `lora_meta` (see
      `parse_lora_metadata` / `_resolve_network_alpha`); None means no source
      defined an alpha (kohya's implicit alpha == rank, i.e. factor 1.0).
    diff_updates: {flat_param_path: numpy delta} additive updates to base params
      (norm weights, scale_shift_tables, full-weight `.diff`, bias `.diff_b`),
      already in Flax layout (kernels transposed).
  """
  modules, skipped_te_keys, unmatched_keys = normalize_krea2_lora_state_dict(state_dict)
  if skipped_te_keys:
    max_logging.log(
        f"Ignoring {len(skipped_te_keys)} text-encoder LoRA key(s); "
        "Qwen3 text-encoder LoRA is not supported for Krea 2."
    )
  if unmatched_keys:
    max_logging.log(
        f"WARNING: {len(unmatched_keys)} LoRA key(s) did not match any Krea 2 module and were "
        f"skipped: {sorted(unmatched_keys)[:20]}"
    )

  lora_name = f"lora-{adapter_name}"
  flat_lora_params = {}
  ranks_by_path = {}
  network_alphas_by_path = {}
  diff_updates = {}

  for torch_path, entry in modules.items():
    flax_path = krea2_torch_path_to_flax_path(torch_path)
    kind = _leaf_kind(flax_path)

    down, up = entry.get("down"), entry.get("up")
    if (down is None) != (up is None):
      max_logging.log(f"WARNING: LoRA module '{torch_path}' has only one of down/up weights; skipping.")
    elif down is not None:
      if kind != "dense":
        max_logging.log(f"WARNING: LoRA down/up weights target non-linear module '{torch_path}'; skipping.")
      elif down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
        max_logging.log(
            f"WARNING: LoRA module '{torch_path}' has incompatible down/up shapes "
            f"{down.shape}/{up.shape}; expected 2-D tensors with a shared rank dimension; skipping."
        )
      else:
        rank = int(down.shape[0])
        # A rank wider than the input or output is unusual but valid. Krea 2's
        # text-fusion projector is a real example: it maps 12 -> 1 while a
        # full-coverage adapter can use rank 32.
        # PyTorch (out, in) -> Flax kernel (in, out): down (r, in) -> (in, r), up (out, r) -> (r, out).
        flat_lora_params[(*flax_path, lora_name, "down", "kernel")] = jnp.asarray(down.T, dtype=weights_dtype)
        flat_lora_params[(*flax_path, lora_name, "up", "kernel")] = jnp.asarray(up.T, dtype=weights_dtype)
        ranks_by_path[flax_path] = rank
        network_alphas_by_path[flax_path] = _resolve_network_alpha(torch_path, rank, entry.get("alpha"), lora_meta)

    diff = entry.get("diff")
    if diff is not None:
      if kind == "table":
        diff_updates[flax_path] = diff
      elif kind == "norm":
        diff_updates[(*flax_path, "weight")] = diff
      else:
        diff_updates[(*flax_path, "kernel")] = diff.T if diff.ndim == 2 else diff

    diff_b = entry.get("diff_b")
    if diff_b is not None:
      if kind == "dense":
        diff_updates[(*flax_path, "bias")] = diff_b
      else:
        max_logging.log(f"WARNING: bias diff targets non-linear module '{torch_path}'; skipping.")

  return flat_lora_params, ranks_by_path, network_alphas_by_path, diff_updates
