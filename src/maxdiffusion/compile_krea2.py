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

# Compile-only HBM estimator for Krea 2 on a target TPU topology. Every
# executable FlaxKrea2Pipeline.__call__ runs is lowered and compiled against
# abstract (sharded ShapeDtypeStruct) weights, so no checkpoint is loaded and
# no TPU is needed:
#
#   JAX_PLATFORMS=cpu python src/maxdiffusion/compile_krea2.py \
#     src/maxdiffusion/configs/base_krea2_turbo.yml compile_topology=v6e-1 height=2048 width=2048
#
# Named resolutions work like in generate_krea2.py, e.g. krea2_aspect_ratio=21:9 krea2_image_size=2k.

import json
import math
import os
import time

from absl import app
import flax
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.generate_krea2 import (
    build_krea2_transformer,
    build_qwen3_config,
    krea2_mesh_device_kind,
    resolve_generation_size,
    resolve_krea2_attention_kernel,
)
from maxdiffusion.kernels.krea2_attention import resolve_kernel_variant
from maxdiffusion.models.krea2.transformer_quant import describe_transformer_quantization, resolve_transformer_quantization

GIB = 1024**3
MIB = 1024**2

# name -> (topology_name, chips_per_host_bounds, usable HBM GiB per chip, peak bf16 TFLOP/s per chip)
TOPOLOGIES = {
    "v5e-1": ("v5e:1x1", (1, 1, 1), 15.75, 197),
    "v5e-4": ("v5e:2x2", (2, 2, 1), 15.75, 197),
    "v5e-8": ("v5e:2x4", (2, 2, 1), 15.75, 197),
    "v5e-16": ("v5e:4x4", (2, 2, 1), 15.75, 197),
    "v6e-1": ("v6e:1x1", (1, 1, 1), 31.25, 918),
    "v6e-4": ("v6e:2x2", (2, 2, 1), 31.25, 918),
    "v6e-8": ("v6e:2x4", (2, 2, 1), 31.25, 918),
    "v6e-16": ("v6e:4x4", (2, 2, 1), 31.25, 918),
}
# attention_flax.py's dispatcher only applies flash_min_seq_length (falling back
# to dot_product below it) for these kernels; others always run the kernel.
THRESHOLD_GATED_KERNELS = (
    "flash",
    "flash_custom",
    "tokamax_flash",
    "ulysses",
    "ulysses_custom",
    "ulysses_custom_fixed_m",
    "ulysses_ring",
)
_CHIP_SPECS = {"v5e": (15.75, 197), "v6e": (31.25, 918)}

# Qwen3-VL-4B text tower, used when text_encoder/config.json is unavailable.
DEFAULT_QWEN3_TEXT_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rope_theta": 5000000.0,
    "max_position_embeddings": 262144,
}

# Pipeline components in residency order -> key in the report's resident_weights dict.
COMPONENT_WEIGHT_KEYS = {"transformer": "transformer", "text_encoder": "qwen3", "vae": "vae"}
OFFLOADABLE_COMPONENTS = ("text_encoder", "transformer")


def hbm_verdict(peak, hbm_gib):
  """Returns (verdict, headroom GiB) of a per-chip peak against usable HBM; (None, None) if unknown."""
  if hbm_gib is None:
    return None, None
  usable = hbm_gib * GIB
  return ("OOM" if peak > usable else ("TIGHT" if peak > 0.9 * usable else "FITS")), (usable - peak) / GIB


def phase_estimates(phase_specs, records, weights, offload, hbm_gib):
  """Per-phase HBM peaks under per-phase weight residency.

  phase_specs: [(phase name, component the phase runs, executable names)]. The resident set of a phase is
  every component not offloaded plus the one the phase runs; its peak is those weights plus the largest
  executable subtotal (including dispatch run-ahead) among the phase's executables.
  """
  by_name = {r["name"]: r for r in records}
  phases = []
  for name, component, executables in phase_specs:
    resident = [c for c in COMPONENT_WEIGHT_KEYS if c not in offload or c == component]
    resident_weights = sum(weights[COMPONENT_WEIGHT_KEYS[c]] for c in resident)
    phase_records = [by_name[e] for e in executables]
    top = max(phase_records, key=lambda r: r["subtotal"] + r["dispatch_runahead"])
    activations = top["subtotal"] + top["dispatch_runahead"]
    peak = resident_weights + activations
    verdict, headroom = hbm_verdict(peak, hbm_gib)
    phases.append(
        {
            "phase": name,
            "component": component,
            "executables": list(executables),
            "resident_components": resident,
            "weights": resident_weights,
            "activations": activations,
            "activations_without_runahead": max(r["subtotal"] for r in phase_records),
            "peak_executable": top["name"],
            "peak": peak,
            "verdict": verdict,
            "headroom_gib": headroom,
        }
    )
  return phases


def resolve_topology(name):
  """Returns (topology_name, chips_per_host_bounds, hbm_gib, peak_tflops); unknown specs are None."""
  if name in TOPOLOGIES:
    return TOPOLOGIES[name]
  if ":" not in name:
    raise ValueError(f"Unknown compile_topology '{name}'. Use one of {sorted(TOPOLOGIES)} or a raw 'v6e:2x2' string.")
  bounds = (1, 1, 1) if name.split(":")[1] == "1x1" else (2, 2, 1)
  hbm_gib, peak_tflops = _CHIP_SPECS.get(name.split(":")[0], (None, None))
  return name, bounds, hbm_gib, peak_tflops


def build_meshes(config, devices):
  """Builds the transformer and VAE meshes exactly like generate_krea2.main, on the given devices."""
  try:
    num_slices = 1 + max(d.slice_index for d in devices)
  except Exception:
    num_slices = 1
  devices_per_slice = len(devices) // num_slices
  if config.batch_size == 1 and config.ici_tensor_parallelism == 1 and devices_per_slice > 1:
    max_logging.log(
        f"Auto-configuring Tensor Parallelism: ici_tensor_parallelism={devices_per_slice}, "
        f"ici_fsdp_parallelism=1 for batch_size=1 on {devices_per_slice} devices per slice "
        f"({num_slices} slice(s))."
    )
    pyconfig._config.keys["ici_tensor_parallelism"] = devices_per_slice
    pyconfig._config.keys["ici_fsdp_parallelism"] = 1

  devices_array = max_utils.create_device_mesh(config, devices=devices)
  mesh = Mesh(devices_array, config.mesh_axes)

  vae_spatial = getattr(config, "vae_spatial", -1)
  total_devices = math.prod(devices_array.shape)
  if vae_spatial == -1:
    vae_spatial = total_devices
  assert (
      total_devices % vae_spatial == 0
  ), f"total devices ({total_devices}) must be a multiple of vae_spatial ({vae_spatial})"
  vae_devices_array = devices_array.flatten().reshape(total_devices // vae_spatial, vae_spatial)
  vae_mesh = Mesh(vae_devices_array, ("redundant", "vae_spatial"))
  return mesh, vae_mesh


def resolve_model_configs(repo):
  """Returns (snapshot_dir, transformer_cfg, te_config); falls back to built-in defaults on any failure."""
  try:
    if os.path.isdir(repo):
      snapshot_dir = repo
    else:
      from huggingface_hub import hf_hub_download

      for component in ("transformer", "text_encoder", "vae"):
        path = hf_hub_download(repo, f"{component}/config.json")
      snapshot_dir = os.path.dirname(os.path.dirname(path))
    with open(os.path.join(snapshot_dir, "transformer", "config.json"), "r") as f:
      transformer_cfg = json.load(f)
    with open(os.path.join(snapshot_dir, "text_encoder", "config.json"), "r") as f:
      te_config = json.load(f)
    max_logging.log(f"Using model configs from {snapshot_dir}")
    return snapshot_dir, transformer_cfg, te_config
  except Exception as exc:  # noqa: BLE001 - any failure means "use defaults"
    max_logging.log(f"Warning: could not read model configs for '{repo}' ({exc}); using built-in Krea 2 defaults.")
    return None, {}, {"text_config": DEFAULT_QWEN3_TEXT_CONFIG}


def abstract_linen_params(init_fn, mesh, logical_axis_rules, dtype_fn, safe_shardings=False):
  """Evaluates a linen init abstractly and returns its params as sharded ShapeDtypeStructs.

  safe_shardings: replicate leaves whose sharding the mesh cannot divide (as generate_krea2 does for the
  quantized text encoder).
  """
  with mesh, nn_partitioning.axis_rules(logical_axis_rules):
    abstract_vars = jax.eval_shape(init_fn)
    logical_specs = nn.get_partition_spec(abstract_vars)
    shardings = nn.logical_to_mesh_sharding(logical_specs, mesh, logical_axis_rules)
  params = flax.core.unfreeze(
      jax.tree_util.tree_map(
          lambda x: x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x,
          abstract_vars["params"],
          is_leaf=lambda x: isinstance(x, flax_spmd.LogicallyPartitioned),
      )
  )
  shardings = flax.core.unfreeze(shardings["params"])
  if safe_shardings:
    from maxdiffusion.models.krea2.text_encoder_quant import safe_param_shardings

    shardings = safe_param_shardings(params, shardings, mesh)
  return jax.tree_util.tree_map_with_path(
      lambda path, x, sharding: jax.ShapeDtypeStruct(x.shape, dtype_fn(path, x), sharding=sharding),
      params,
      shardings,
  )


def abstract_vae(config, snapshot_dir, vae_mesh):
  """Builds the Qwen-Image VAE abstractly; every Param is replicated in weights_dtype like the loader."""
  from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan

  kwargs = {
      "mesh": vae_mesh,
      "dtype": config.activations_dtype,
      "weights_dtype": config.weights_dtype,
      "vae_decode_chunk": 1,
      "vae_encode_chunk": 4,
  }

  def create_model(rngs):
    if snapshot_dir is not None:
      return AutoencoderKLWan.from_config(snapshot_dir, subfolder="vae", rngs=rngs, **kwargs)
    return AutoencoderKLWan(rngs=rngs, **kwargs)

  rngs = nnx.Rngs(jax.random.key(config.seed if config.seed is not None else 0))
  vae = nnx.eval_shape(create_model, rngs=rngs)
  graphdef, state, rest = nnx.split(vae, nnx.Param, ...)
  replicated = NamedSharding(vae_mesh, P())
  state = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, config.weights_dtype, sharding=replicated), state)
  return vae, graphdef, state, rest


def tree_shard_bytes(tree):
  """Per-device bytes of a pytree of sharded ShapeDtypeStructs."""
  total = 0
  for leaf in jax.tree_util.tree_leaves(tree):
    shape = leaf.sharding.shard_shape(leaf.shape) if leaf.sharding is not None else leaf.shape
    total += math.prod(shape) * jnp.dtype(leaf.dtype).itemsize
  return total


def sds(shape, dtype, sharding):
  return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)


def clone(x):
  """A distinct buffer with the same shape, dtype and sharding (identity matters for add_retained)."""
  return sds(x.shape, x.dtype, x.sharding)


def xla_flops(compiled):
  """Per-device XLA flops from cost_analysis(), or None if unavailable."""
  try:
    cost = compiled.cost_analysis()
  except Exception:  # noqa: BLE001 - optional backend feature
    return None
  if isinstance(cost, (list, tuple)):
    cost = cost[0] if cost else None
  if not cost:
    return None
  return cost.get("flops")


def compile_executable(name, entry, param_args, activation_args, calls):
  """Lowers and compiles one cached_jit entry; returns (record, abstract outputs)."""
  max_logging.log(f"Compiling {name}...")
  t0 = time.perf_counter()
  lowered = entry.jitted.lower(*param_args, *activation_args)
  compiled = lowered.compile()
  compile_s = time.perf_counter() - t0
  outputs = jax.tree_util.tree_map(
      lambda info, sharding: sds(info.shape, info.dtype, sharding), lowered.out_info, compiled.output_shardings
  )
  mem = compiled.memory_analysis()
  activation_inputs = tree_shard_bytes(activation_args)
  # Donated inputs whose buffers the outputs reuse; counted in both
  # act_in and outputs otherwise.
  alias_size = getattr(mem, "alias_size_in_bytes", 0) or 0
  record = {
      "name": name,
      "calls": calls,
      "argument_size": mem.argument_size_in_bytes,
      "output_size": mem.output_size_in_bytes,
      "temp_size": mem.temp_size_in_bytes,
      "generated_code_size": mem.generated_code_size_in_bytes,
      "activation_inputs": activation_inputs,
      "alias_size": alias_size,
      # None until add_retained runs; it must run exactly once per record.
      "retained": None,
      "retained_buffers": None,
      "subtotal": mem.temp_size_in_bytes + activation_inputs + mem.output_size_in_bytes - alias_size,
      "dispatch_runahead": 0,
      "flops_per_device": xla_flops(compiled),
      "compile_seconds": compile_s,
  }
  max_logging.log(f" -> {name} compiled in {compile_s:.1f}s")
  return record, outputs


def add_retained(record, live_buffers, activation_args, outputs):
  """Adds the pipeline buffers live in __call__ during this executable that are not its own inputs/outputs."""
  if record["retained"] is not None:
    raise ValueError(f"add_retained called twice for {record['name']}")
  own = {id(leaf) for leaf in jax.tree_util.tree_leaves((activation_args, outputs))}
  retained = {name: buf for name, buf in live_buffers.items() if id(buf) not in own}
  record["retained"] = tree_shard_bytes(list(retained.values()))
  record["retained_buffers"] = sorted(retained)
  record["subtotal"] += record["retained"]


def fmt_bytes(num_bytes):
  if num_bytes >= GIB:
    return f"{num_bytes / GIB:.2f} GiB"
  return f"{num_bytes / MIB:.1f} MiB"


def print_report(report):
  """Prints the per-executable table and the per-chip HBM / time summary."""
  header = (
      f"{'executable':<22}{'calls':>7}{'args':>12}{'outputs':>12}{'temp':>12}{'code':>12}"
      f"{'act_in':>12}{'aliased':>12}{'retained':>12}{'subtotal':>12}{'GFLOP/dev':>12}{'compile':>10}"
  )
  max_logging.log("=" * len(header))
  max_logging.log(
      f"KREA 2 AOT MEMORY ESTIMATE: {report['topology']} ({report['num_devices']} chips), "
      f"{report['width']}x{report['height']}, batch {report['batch_size']}, "
      f"{report['num_inference_steps']} steps, CFG x{report['cfg_passes']}, "
      f"{'staged' if report['staged'] else 'monolithic'} transformer, mesh {report['mesh_shape']}"
  )
  max_logging.log("Per-device sizes from compiled.memory_analysis().")
  max_logging.log("=" * len(header))
  max_logging.log(header)
  for r in report["executables"]:
    flops = "n/a" if r["flops_per_device"] is None else f"{r['flops_per_device'] / 1e9:.1f}"
    max_logging.log(
        f"{r['name']:<22}{r['calls']:>7}{fmt_bytes(r['argument_size']):>12}{fmt_bytes(r['output_size']):>12}"
        f"{fmt_bytes(r['temp_size']):>12}{fmt_bytes(r['generated_code_size']):>12}"
        f"{fmt_bytes(r['activation_inputs']):>12}{fmt_bytes(r['alias_size']):>12}"
        f"{fmt_bytes(r['retained']):>12}{fmt_bytes(r['subtotal']):>12}{flops:>12}"
        f"{r['compile_seconds']:>9.1f}s"
    )
  max_logging.log("-" * len(header))
  max_logging.log(
      "subtotal = temp + act_in + outputs - aliased + retained (aliased = donated inputs reused as outputs; "
      "retained = pipeline buffers live in __call__ meanwhile)."
  )
  for r in report["executables"]:
    if r["retained_buffers"]:
      max_logging.log(f"  retained during {r['name']}: {', '.join(r['retained_buffers'])}")
  weights = report["resident_weights"]
  max_logging.log(report["text_encoder_residency"])
  max_logging.log(
      describe_transformer_quantization(report["transformer_quantization"], report["transformer_quant_targets"])
  )
  max_logging.log(
      f"Component weights per chip: transformer {fmt_bytes(weights['transformer'])} + "
      f"text_encoder (qwen3) {fmt_bytes(weights['qwen3'])} + vae {fmt_bytes(weights['vae'])} = "
      f"{fmt_bytes(weights['total'])}"
  )
  offload = report["offload_components"]
  max_logging.log(
      f"Offloaded components (host-resident, placed only for their phase): {', '.join(offload) if offload else 'none'}"
  )
  runahead = max(r["dispatch_runahead"] for r in report["executables"])
  if report["staged"] and report["staged_donate_hidden_states"]:
    max_logging.log(
        "Staged dispatch run-ahead: disabled by donation (krea2_staged_donate_hidden_states=True, "
        "each block's output aliases its hidden_states input)"
    )
  elif runahead:
    max_logging.log(
        f"Staged dispatch run-ahead: +{fmt_bytes(runahead)} (block outputs allocated ahead of execution); "
        f"peak without it would be {fmt_bytes(report['peak_estimate_without_runahead'])}"
    )
  max_logging.log("-" * len(header))
  phase_header = (
      f"{'phase':<12}{'resident components':<36}{'weights':>12}{'activations':>14}{'peak':>12}{'verdict':>10}"
      f"{'headroom':>12}"
  )
  max_logging.log(phase_header)
  for phase in report["phases"]:
    verdict = phase["verdict"] or "n/a"
    headroom = "n/a" if phase["headroom_gib"] is None else f"{phase['headroom_gib']:+.2f} GiB"
    max_logging.log(
        f"{phase['phase']:<12}{', '.join(phase['resident_components']):<36}{fmt_bytes(phase['weights']):>12}"
        f"{fmt_bytes(phase['activations']):>14}{fmt_bytes(phase['peak']):>12}{verdict:>10}{headroom:>12}"
    )
  max_logging.log(
      "activations = largest executable subtotal of the phase"
      f"{' (+ dispatch run-ahead)' if runahead else ''}; peak = resident weights + activations."
  )
  if report["swap_bytes_per_generation"]:
    max_logging.log(
        f"Per-generation host->HBM swap: {report['swap_bytes_per_generation'] / GIB:.2f} GiB "
        "(bandwidth must be measured on the VM)"
    )
  max_logging.log(
      f"Peak estimate per chip: {fmt_bytes(report['peak_estimate'])} "
      f"(phase {report['peak_phase']}, executable {report['peak_executable']})"
  )
  if report["hbm_usable_gib"] is None:
    max_logging.log("Verdict: n/a (unknown HBM capacity for this topology)")
  else:
    max_logging.log(
        f"Verdict: {report['verdict']} vs {report['hbm_usable_gib']:.2f} GiB usable HBM "
        f"(headroom {report['headroom_gib']:+.2f} GiB)"
    )
  if report["ideal_seconds_xla"] is None:
    max_logging.log("Ideal compute time: n/a (unknown peak TFLOP/s or missing flops)")
  elif report["attention"] == "dot_product":
    max_logging.log(
        f"Ideal compute time (XLA flops at peak): {report['ideal_seconds_xla']:.2f}s "
        "-- includes dot_product attention, which XLA counts"
    )
  elif report["attention"] in THRESHOLD_GATED_KERNELS and not report["attention_uses_kernel"]:
    max_logging.log(
        f"Ideal compute time (XLA flops at peak): {report['ideal_seconds_xla']:.2f}s "
        "-- includes attention, which XLA already counts: sequence length < flash_min_seq_length "
        f"({report['flash_min_seq_length']}) falls back from {report['attention']} to dot_product (no analytic addition)"
    )
  else:
    max_logging.log(
        f"Ideal compute time (XLA flops at peak): {report['ideal_seconds_xla']:.2f}s "
        f"-- excludes {report['attention']} attention kernels, which XLA does not count"
    )
    max_logging.log(
        f"Analytic attention flops: {report['attention_flops'] / 1e12:.1f} TFLOP; "
        f"ideal time including them: {report['ideal_seconds_with_attention']:.2f}s"
    )
  if report["ideal_seconds_xla"] is not None and report["transformer_quantization"]:
    max_logging.log(
        "Ideal compute time assumes the bf16 peak rate for every matmul, although the W8A8 int8 matmuls have a "
        "2x higher peak on v6e."
    )
  max_logging.log(
      "Caveats: Pallas VMEM scratch and the TPU runtime stack reservation (~0.4 GiB on v5e) are not included; "
      "expect ~5-10% runtime fragmentation on top of the peak; LoRA adapters are not included."
  )
  max_logging.log("=" * len(header))


def main(argv):
  jax.config.update("jax_use_shardy_partitioner", True)

  config_path = "src/maxdiffusion/configs/base_krea2.yml"
  custom_overrides = []
  if len(argv) > 1:
    if argv[1].endswith(".yml") or argv[1].endswith(".yaml"):
      config_path = argv[1]
      custom_overrides = argv[2:]
    else:
      custom_overrides = argv[1:]
  pyconfig.initialize(
      [None, config_path, "run_name=krea2_compile", "output_dir=output/", "skip_jax_distributed_system=True"]
      + custom_overrides
  )

  from maxdiffusion.models.krea2.util import KREA2_PROMPT_TEMPLATE_START_IDX
  from maxdiffusion.models.qwen3_flax import FlaxQwen3Model
  from maxdiffusion.models.krea2.text_encoder_quant import (
      describe_text_encoder_residency,
      quantize_text_encoder_model,
      resolve_text_encoder_quantization,
  )
  from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWanCache
  from maxdiffusion.pipelines.krea2.krea2_pipeline import (
      KREA2_PRELUDE_KEYS,
      KREA2_TEXT_CONTEXT_KEYS,
      FlaxKrea2Pipeline,
      is_classifier_free_guidance_enabled,
  )

  config = pyconfig.config
  topology = config.compile_topology
  if not topology:
    raise ValueError("Set compile_topology=<name>, e.g. compile_topology=v6e-4.")
  attention_kernel_choice = resolve_krea2_attention_kernel(config)
  # getattr defaults keep this script usable with configs that predate these keys.
  offload = tuple(getattr(config, "krea2_offload_components", None) or ())
  unknown = sorted(set(offload) - set(OFFLOADABLE_COMPONENTS))
  if unknown:
    raise ValueError(
        f"krea2_offload_components entries {unknown} are not supported; allowed: {list(OFFLOADABLE_COMPONENTS)}"
    )
  offload = tuple(c for c in OFFLOADABLE_COMPONENTS if c in offload)
  donate_hidden_states = bool(getattr(config, "krea2_staged_donate_hidden_states", True))
  topology_name, host_bounds, hbm_gib, peak_tflops = resolve_topology(topology)
  num_slices = max(int(config.compile_topology_num_slices), 1)
  devices = get_topology_desc(
      platform="tpu",
      topology_name=topology_name,
      chip_config_name="default",
      chips_per_host_bounds=host_bounds,
      num_slices=num_slices,
  ).devices
  max_logging.log(f"Target topology {topology} ({topology_name}): {len(devices)} device(s), {num_slices} slice(s)")
  mesh, vae_mesh = build_meshes(config, devices)
  vae_logical_axis_rules = getattr(config, "vae_logical_axis_rules", None)
  max_logging.log("LoRA is not modeled: compiling with lora_compile_spec=() and no interceptors.")

  snapshot_dir, transformer_cfg, te_config = resolve_model_configs(config.pretrained_model_name_or_path)
  qwen3_config = build_qwen3_config(te_config, config)
  qwen3_model = FlaxQwen3Model(qwen3_config)
  te_quantization, te_quant_tile_size, te_embed_on_host = resolve_text_encoder_quantization(config)
  if te_quantization == "int8":
    qwen3_model = quantize_text_encoder_model(qwen3_model, te_quant_tile_size)
  te_residency = describe_text_encoder_residency(te_quantization, te_quant_tile_size, te_embed_on_host)
  max_logging.log(te_residency)
  # int8 block kernels come straight out of the runtime model's abstract tree.
  transformer_quantization, transformer_quant_targets = resolve_transformer_quantization(config)
  transformer = build_krea2_transformer(transformer_cfg, config, mesh, quant_targets=transformer_quant_targets)
  max_logging.log(describe_transformer_quantization(transformer_quantization, transformer_quant_targets))

  batch = config.batch_size
  # Same choice as generate_krea2: the preset keys win over height/width.
  height, width, _ = resolve_generation_size(config)
  grid_h, grid_w = height // 16, width // 16
  seq_img = grid_h * grid_w
  seq_txt = config.max_sequence_length
  compaction_multiple = int(getattr(config, "krea2_text_compaction_multiple", 0) or 0)
  if compaction_multiple > 0:
    max_logging.log(
        f"krea2_text_compaction_multiple={compaction_multiple}: the text bucket is prompt-dependent, so the estimate "
        f"uses the worst case text length max_sequence_length={seq_txt}."
    )
  # flash_custom picks its kernel variant per chip; the topology devices report the target chip
  # (same helper as the kernel wrapper and generate_krea2's AOT meta).
  attention_device_kind = krea2_mesh_device_kind(transformer.mesh)
  attention_kernel_variant = (
      resolve_kernel_variant(attention_kernel_choice, attention_device_kind) if config.attention == "flash_custom" else None
  )
  max_logging.log(
      f"RoPE layout: {transformer.rope_layout}; attention kernel: {transformer.attention_kernel}"
      + (
          f" (krea2_attention_kernel {attention_kernel_choice} -> {attention_kernel_variant} on {attention_device_kind})"
          if attention_kernel_variant
          else ""
      )
  )
  seq_txt_full = seq_txt + KREA2_PROMPT_TEMPLATE_START_IDX
  in_channels = transformer.in_channels

  key = jax.random.PRNGKey(config.seed if config.seed is not None else 0)
  key, qwen_key = jax.random.split(key)

  def transformer_init_fn():
    return transformer.init(
        key,
        hidden_states=jnp.zeros((batch, seq_img, in_channels)),
        encoder_hidden_states=jnp.zeros((batch, seq_txt, transformer.num_text_layers, transformer.text_hidden_dim)),
        timestep=jnp.zeros((batch,)),
        img_ids=jnp.zeros((seq_img, 3)),
        txt_ids=jnp.zeros((seq_txt, 3)),
        encoder_attention_mask=jnp.ones((batch, seq_txt), dtype=jnp.bool_),
    )

  def qwen3_init_fn():
    mask = jnp.ones((batch, seq_txt_full), dtype=jnp.int32)
    if te_embed_on_host:
      embeds = jnp.zeros((batch, seq_txt_full, qwen3_config.hidden_size), dtype=qwen3_config.dtype)
      return qwen3_model.init(qwen_key, None, mask, inputs_embeds=embeds)
    return qwen3_model.init(qwen_key, jnp.zeros((batch, seq_txt_full), dtype=jnp.int32), mask)

  def quantized_leaf_dtype(path):
    """int8 qvalues; scales are stored in the text encoder compute dtype (generate_krea2); else None."""
    last = path[-1] if path else None
    name = getattr(last, "name", None)
    if name == "qvalue":
      return jnp.int8
    if name == "scale":
      return qwen3_config.dtype
    return None

  if config.weights_dtype == jnp.bfloat16:

    def qwen3_dtype(path, _):
      # Mirrors cast_dict_to_bfloat16_inplace(exclude_keywords=("norm",)) in generate_krea2.
      quantized = quantized_leaf_dtype(path)
      if quantized is not None:
        return quantized
      return jnp.float32 if "norm" in jax.tree_util.keystr(path) else jnp.bfloat16

  else:
    max_logging.log(
        f"Warning: weights_dtype={jnp.dtype(config.weights_dtype).name}, so the runtime keeps the Qwen3 checkpoint "
        "dtypes without casting; the estimate assumes a bf16 text_encoder checkpoint (every tensor bf16)."
    )

    def qwen3_dtype(path, _):
      quantized = quantized_leaf_dtype(path)
      return quantized if quantized is not None else jnp.bfloat16

  max_logging.log("Evaluating abstract parameter shapes and shardings...")
  t_params = abstract_linen_params(transformer_init_fn, mesh, config.logical_axis_rules, lambda _, x: x.dtype)
  q_params = abstract_linen_params(
      qwen3_init_fn, mesh, config.logical_axis_rules, qwen3_dtype, safe_shardings=bool(te_quantization)
  )
  vae, vae_graphdef, vae_state, vae_rest = abstract_vae(config, snapshot_dir, vae_mesh)

  pipeline = FlaxKrea2Pipeline(
      transformer=transformer,
      vae=vae,
      vae_cache=AutoencoderKLWanCache(vae),
      text_encoder=qwen3_model,
      tokenizer=None,
      scheduler=None,
      config=config,
      mesh=mesh,
      vae_mesh=vae_mesh,
      vae_logical_axis_rules=vae_logical_axis_rules,
      lora_compile_spec=(),
  )
  pipeline._setup_jit_functions()

  staged = bool(config.krea2_staged_transformer)
  steps = config.num_inference_steps
  cfg_passes = 2 if is_classifier_free_guidance_enabled(config.guidance_scale, config.do_classifier_free_guidance) else 1
  num_layers = transformer.num_layers
  replicated = NamedSharding(mesh, P())
  data = NamedSharding(mesh, P("data"))
  records = []
  # (record, activation_args, outputs) of each executable run inside the mesh, for add_retained.
  transformer_runs = []

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    text_ids = sds((batch, seq_txt_full), jnp.int32, replicated)
    if te_embed_on_host:
      # Host-gathered token embeddings replace the token ids.
      text_inputs = sds((batch, seq_txt_full, qwen3_config.hidden_size), qwen3_config.dtype, replicated)
    else:
      text_inputs = text_ids
    qwen3_args = (text_inputs, text_ids, text_ids)
    qwen3_record, prompt_embeds = compile_executable(
        "qwen3_forward", pipeline._jitted_qwen3_forward, (q_params,), qwen3_args, cfg_passes
    )
    records.append(qwen3_record)

    text_mask = sds((batch, seq_txt), jnp.bool_, data)
    text_context_args = (sds(prompt_embeds.shape, prompt_embeds.dtype, data), text_mask)
    text_context_record, text_hidden = compile_executable(
        "transformer_text_context",
        pipeline._jitted_transformer_text_context,
        ({k: t_params[k] for k in KREA2_TEXT_CONTEXT_KEYS},),
        text_context_args,
        cfg_passes,
    )
    records.append(text_context_record)

    step_inputs = (
        sds((batch, seq_img, in_channels), jnp.float32, data),
        sds(text_hidden.shape, text_hidden.dtype, data),
        text_mask,
        sds((batch, seq_img, 3), jnp.float32, data),
        sds((batch, seq_txt, 3), jnp.float32, data),
        sds((batch,), jnp.float32, replicated),
    )
    # txt_ids, img_ids and the scheduler sigmas/timesteps are created before encoding.
    # Under CFG the positive prompt embeds and mask are also held while the negative prompt is encoded.
    qwen3_live = {
        "img_ids": step_inputs[3],
        "txt_ids": step_inputs[4],
        "scheduler_sigmas": sds((steps,), jnp.float32, replicated),
        "scheduler_timesteps": sds((steps,), jnp.float32, replicated),
    }
    if cfg_passes == 2:
      qwen3_live["prompt_embeds"] = clone(prompt_embeds)
      qwen3_live["prompt_embeds_mask"] = sds((batch, seq_txt), jnp.bool_, replicated)
    add_retained(qwen3_record, qwen3_live, qwen3_args, prompt_embeds)
    if staged:
      prelude_params = {k: t_params[k] for k in KREA2_PRELUDE_KEYS}
      record, prelude_out = compile_executable(
          "transformer_prelude",
          pipeline._jitted_transformer_prelude,
          (prelude_params,),
          step_inputs,
          steps * cfg_passes,
      )
      records.append(record)
      transformer_runs.append((record, step_inputs, prelude_out))
      hidden_states, temb, temb_mod, rotary_emb, attention_mask = prelude_out
      block_args = (hidden_states, temb_mod, rotary_emb, attention_mask)
      record, hidden_states = compile_executable(
          "transformer_block",
          pipeline._jitted_transformer_block,
          (t_params["blocks_0"], {}),
          block_args,
          num_layers * steps * cfg_passes,
      )
      # Async dispatch runs the Python block loop ahead of the device, so every
      # block's output buffer is allocated before the earlier blocks finish: a
      # v5e-4 2048x2048 xprof showed num_layers + 1 hidden states live at peak.
      # Donating hidden_states makes each output alias its input, so the loop
      # allocates nothing ahead of execution.
      if not donate_hidden_states:
        record["dispatch_runahead"] = (num_layers - 1) * record["output_size"]
      records.append(record)
      transformer_runs.append((record, block_args, hidden_states))
      final_args = (hidden_states, temb, step_inputs[2])
      record, noise_pred = compile_executable(
          "transformer_final",
          pipeline._jitted_transformer_final,
          (t_params["final_layer"],),
          final_args,
          steps * cfg_passes,
      )
      records.append(record)
      transformer_runs.append((record, final_args, noise_pred))
    else:
      record, noise_pred = compile_executable(
          "transformer_step", pipeline._jitted_transformer_step, (t_params,), step_inputs, steps * cfg_passes
      )
      records.append(record)
      transformer_runs.append((record, step_inputs, noise_pred))

  # The raw prompt embeds are dropped once the text context is computed, so
  # only text_hidden (per prompt) stays live through the loop.
  text_context_live = {
      "latents": step_inputs[0],
      "prompt_embeds_mask": step_inputs[2],
      "img_ids": step_inputs[3],
      "txt_ids": step_inputs[4],
  }
  if cfg_passes == 2:
    # Positive context: the negative embeds are still live (the positive
    # embeds are this executable's own input).
    text_context_live["negative_prompt_embeds"] = clone(text_context_args[0])
    text_context_live["negative_prompt_embeds_mask"] = clone(text_mask)
  add_retained(text_context_record, text_context_live, text_context_args, text_hidden)
  # Pipeline-level buffers FlaxKrea2Pipeline.__call__ keeps live from the
  # denoise loop through the VAE decode. Under CFG the negative text context,
  # mask and txt_ids and the conditional noise_pred (held while the negative
  # pass runs) are distinct buffers, so the positive-pass arguments stand in for
  # the negative pass's.
  live_buffers = {
      "latents": step_inputs[0],
      "text_hidden": step_inputs[1],
      "prompt_embeds_mask": step_inputs[2],
      "img_ids": step_inputs[3],
      "txt_ids": step_inputs[4],
  }
  if cfg_passes == 2:
    live_buffers["negative_text_hidden"] = clone(step_inputs[1])
    live_buffers["negative_prompt_embeds_mask"] = clone(step_inputs[2])
    live_buffers["negative_txt_ids"] = clone(step_inputs[4])
    live_buffers["noise_pred"] = clone(noise_pred)
  # After the loop noise_pred (and neg_noise_pred under CFG) stay live through the VAE decode.
  vae_live = dict(live_buffers, noise_pred=clone(noise_pred))
  if cfg_passes == 2:
    vae_live["neg_noise_pred"] = clone(noise_pred)
  if staged:
    # temb is a prelude output consumed only by transformer_final, so it is live across every block.
    live_buffers["temb"] = temb
  for record, activation_args, outputs in transformer_runs:
    add_retained(record, live_buffers, activation_args, outputs)

  with vae_mesh, nn_partitioning.axis_rules(pipeline.vae_logical_axis_rules):
    latents_5d = sds((batch, 16, 1, height // 8, width // 8), config.activations_dtype, NamedSharding(vae_mesh, P()))
    record, images = compile_executable(
        "vae_decode", pipeline._jitted_vae_decode, (vae_graphdef, vae_state, vae_rest), (latents_5d,), 1
    )
    records.append(record)
    add_retained(record, vae_live, (latents_5d,), images)

  weights = {
      "transformer": tree_shard_bytes(t_params),
      "qwen3": tree_shard_bytes(q_params),
      "vae": tree_shard_bytes(vae_state),
  }
  weights["total"] = sum(weights.values())
  denoise_executables = ("transformer_text_context",) + (
      ("transformer_prelude", "transformer_block", "transformer_final") if staged else ("transformer_step",)
  )
  phase_specs = [
      ("A encode", "text_encoder", ("qwen3_forward",)),
      ("B denoise", "transformer", denoise_executables),
      ("C decode", "vae", ("vae_decode",)),
  ]
  phases = phase_estimates(phase_specs, records, weights, offload, hbm_gib)
  peak_phase = max(phases, key=lambda p: p["peak"])
  peak = peak_phase["peak"]
  peak_without_runahead = max(p["weights"] + p["activations_without_runahead"] for p in phases)
  verdict, headroom = hbm_verdict(peak, hbm_gib)
  # Both prompts (and the negative one under CFG) are encoded under one text_encoder placement, so each
  # offloaded component crosses host->HBM once per generation.
  swap_bytes = sum(weights[COMPONENT_WEIGHT_KEYS[c]] for c in offload)

  # cost_analysis() flops are per device (per SPMD partition), so each chip's
  # time is its own flops over its own peak.
  # XLA already counts dot_product (einsum) attention, including the fallback
  # attention_flax takes below flash_min_seq_length for THRESHOLD_GATED_KERNELS; Pallas/cuDNN kernels are
  # opaque to it, so their QK^T and PV matmuls are added analytically.
  ideal_xla, ideal_with_attention = None, None
  attention_flops = 0
  uses_kernel = config.attention != "dot_product" and (
      config.attention not in THRESHOLD_GATED_KERNELS or (seq_img + seq_txt) >= transformer.flash_min_seq_length
  )
  if uses_kernel:
    hidden_dim = transformer.num_attention_heads * transformer.attention_head_dim
    attention_flops = 4 * batch * (seq_img + seq_txt) ** 2 * hidden_dim * num_layers * steps * cfg_passes
  if peak_tflops is not None and all(r["flops_per_device"] is not None for r in records):
    peak_flops = peak_tflops * 1e12
    ideal_xla = sum(r["calls"] * r["flops_per_device"] for r in records) / peak_flops
    ideal_with_attention = ideal_xla + attention_flops / (peak_flops * len(devices))

  report = {
      "topology": topology,
      "topology_name": topology_name,
      "num_devices": len(devices),
      "mesh_shape": dict(mesh.shape),
      "vae_mesh_shape": dict(vae_mesh.shape),
      "height": height,
      "width": width,
      "batch_size": batch,
      "num_inference_steps": steps,
      "cfg_passes": cfg_passes,
      "staged": staged,
      "staged_donate_hidden_states": donate_hidden_states,
      "offload_components": list(offload),
      "text_encoder_quantization": te_quantization,
      "text_encoder_quant_tile_size": te_quant_tile_size if te_quantization else None,
      "text_embed_on_host": te_embed_on_host,
      "text_encoder_residency": te_residency,
      "transformer_quantization": transformer_quantization,
      "transformer_quant_targets": list(transformer_quant_targets),
      "attention": config.attention,
      "krea2_attention_kernel": attention_kernel_choice,
      "krea2_attention_kernel_variant": attention_kernel_variant,
      "attention_uses_kernel": uses_kernel,
      "flash_min_seq_length": transformer.flash_min_seq_length,
      "executables": records,
      "resident_weights": weights,
      "phases": phases,
      "peak_phase": peak_phase["phase"],
      "peak_executable": peak_phase["peak_executable"],
      "peak_estimate": peak,
      "peak_estimate_without_runahead": peak_without_runahead,
      "hbm_usable_gib": hbm_gib,
      "verdict": verdict,
      "headroom_gib": headroom,
      "swap_bytes_per_generation": swap_bytes,
      "peak_tflops": peak_tflops,
      "ideal_seconds_xla": ideal_xla,
      "attention_flops": attention_flops,
      "ideal_seconds_with_attention": ideal_with_attention,
  }
  print_report(report)

  os.makedirs(config.output_dir, exist_ok=True)
  suffix = "_staged" if staged else ""
  if staged and not donate_hidden_states:
    suffix += "_nodonate"
  if offload:
    suffix += "_offload-" + "-".join(offload)
  if te_quantization:
    suffix += f"_te-{te_quantization}"
  if te_embed_on_host:
    suffix += "_embed-host"
  if transformer_quantization:
    suffix += f"_tq-{transformer_quantization}"
  json_path = os.path.join(config.output_dir, f"compile_krea2_{topology}_{width}x{height}{suffix}.json")
  with open(json_path, "w") as f:
    json.dump(report, f, indent=2)
  max_logging.log(f"Wrote {json_path}")


if __name__ == "__main__":
  app.run(main)
