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

# Compares the bf16 and int8 weight-only Krea 2 text encoders on the hidden
# states Krea 2 actually consumes (the tapped layers, system prefix dropped,
# valid tokens only). Runs on CPU with float32 activations (XLA:CPU cannot
# lower some bf16 vector ops); the weights are the real checkpoint.
#
#   JAX_PLATFORMS=cpu PYTHONPATH=src python -m maxdiffusion.tools.krea2_text_encoder_quant_check --num-prompts 8
#
# Variants: "f32scale" keeps the float32 per-tile scales; "bf16scale" rounds
# them to bfloat16 (what generate_krea2 stores for bf16 runs) and computes in
# float32, so it covers the scale rounding but not the bf16 matmul rounding.

import argparse
import gc
import json
import os
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from maxdiffusion.generate_krea2 import build_qwen3_config
from maxdiffusion.models.krea2.text_encoder_quant import (
    quantize_text_encoder_model,
    quantize_text_encoder_params,
    tree_nbytes,
)
from maxdiffusion.models.krea2.util import (
    KREA2_PROMPT_TEMPLATE_START_IDX,
    KREA2_TEXT_ENCODER_SELECT_LAYERS,
    load_krea2_tokenizer,
)
from maxdiffusion.models.qwen3_flax import (
    FlaxQwen3Model,
    load_and_convert_qwen3_weights,
    load_qwen3_embedding_table,
)
from maxdiffusion.pipelines.krea2.krea2_pipeline import krea2_text_encoder_hidden_states, tokenize_krea2_prompts

GIB = 1024**3
KEY_PREFIX = "model.language_model."

PROMPTS = (
    "a fox in the snow",
    "portrait photo of an old fisherman, golden hour",
    "A cozy reading nook with a green velvet armchair, a stack of worn books, a steaming mug of tea on a small wooden "
    "side table, and soft afternoon light filtering through lace curtains.",
    "Product shot of a matte black mechanical wristwatch on a slab of wet slate, water droplets, dramatic rim "
    "lighting, shallow depth of field, 85mm lens.",
    'A hand-painted wooden shop sign that reads "BAKERY & COFFEE - EST. 1952" hanging above a blue door.',
    "isometric pixel art of a tiny floating island with a lighthouse, waterfalls pouring off the edges, clouds",
    "Ein Aquarell von Kirschblüten an einem Fluss in Kyoto, mit einer kleinen Holzbrücke und Laternen im Abendlicht.",
    # ~400 tokens.
    " ".join(
        [
            "An expansive, highly detailed matte painting of a sprawling solarpunk city built into the terraced slopes"
            " of a mountain valley at dawn. Tiered gardens overflow with ferns, citrus trees and flowering vines that"
            " spill over white limestone balconies, while slender wind turbines shaped like seed pods turn slowly"
            " along the ridgeline. A wide river winds through the center of the valley, crossed by arched bridges of"
            " timber and glass, and small electric ferries with striped canvas awnings carry commuters between"
            " floating markets where vendors sell bright produce, woven baskets and hand-thrown ceramics.",
            "In the foreground a young cartographer in a weathered ochre coat sits on a stone parapet with a leather"
            " satchel, sketching the skyline in a notebook; a curious red panda perches beside her, sniffing a paper"
            " bag of pastries. Warm low-angle sunlight catches the mist rising off the water and throws long blue"
            " shadows across the cobblestones, and the reflections of hanging lanterns shimmer in shallow puddles"
            " left by an early rain.",
            "Further back, a monorail glides between towers clad in photovoltaic tiles that shimmer like dragonfly"
            " wings, rooftop greenhouses glow with violet grow lights, and flocks of swifts circle a clock tower whose"
            " face is a mosaic of recycled glass. The atmosphere is optimistic and serene, with soft volumetric light,"
            " crisp architectural detail, subtle film grain, a palette of teal, amber and cream, cinematic"
            " composition, ultra wide angle, 8k resolution, in the style of a classic illustrated travel poster"
            " crossed with contemporary concept art.",
        ]
    ),
)


def to_device_inplace(tree, device):
  """Moves every leaf of a nested dict to `device`, releasing each host leaf as it goes."""
  for key in list(tree):
    value = tree[key]
    if isinstance(value, dict):
      to_device_inplace(value, device)
    else:
      tree[key] = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), value)
      del value
  return tree


def metrics(q, ref, valid):
  """Per tapped layer + overall metrics over valid tokens. q/ref: (S, L, H), valid: (S,) bool."""
  q = q[valid].astype(np.float64)
  ref = ref[valid].astype(np.float64)
  diff = q - ref
  rows = []
  for layer in range(ref.shape[1]):
    rows.append(_stats(diff[:, layer], q[:, layer], ref[:, layer]))
  rows.append(_stats(diff.reshape(-1, ref.shape[-1]), q.reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])))
  return np.stack(rows)


def pool(stats_list):
  """Pools per-prompt stats: sums for the L2/cosine accumulators, max for the max-abs columns."""
  stacked = np.stack(stats_list)
  pooled = stacked.sum(axis=0)
  pooled[..., 4:] = stacked[..., 4:].max(axis=0)
  return pooled


def _stats(diff, q, ref):
  """Sums so prompts can be pooled: (sq_err, sq_ref, cos_sum, n_tokens, max_abs_err, max_abs_ref)."""
  cos = np.sum(q * ref, axis=-1) / np.maximum(np.linalg.norm(q, axis=-1) * np.linalg.norm(ref, axis=-1), 1e-30)
  return np.array(
      [np.sum(diff**2), np.sum(ref**2), np.sum(cos), cos.shape[0], np.max(np.abs(diff)), np.max(np.abs(ref))],
      dtype=np.float64,
  )


def run_model(name, model, params, batches, embed_table=None):
  fwd = jax.jit(lambda p, x, m, pos: krea2_text_encoder_hidden_states(model, p, x, m, pos))
  outputs, times = [], []
  for idx, (ids, mask, pos) in enumerate(batches):
    x = jnp.asarray(embed_table[ids]) if embed_table is not None else jnp.asarray(ids)
    t0 = time.perf_counter()
    out = np.asarray(fwd(params, x, jnp.asarray(mask), jnp.asarray(pos)))
    times.append(time.perf_counter() - t0)
    outputs.append(out[0])
    print(f"  [{name}] prompt {idx}: {times[-1]:.1f}s", flush=True)
  return outputs, times


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", default="krea/Krea-2-Turbo")
  parser.add_argument("--num-prompts", type=int, default=len(PROMPTS))
  parser.add_argument("--max-sequence-length", type=int, default=512)
  parser.add_argument("--tile-size", type=int, default=128)
  parser.add_argument("--variants", default="f32scale,bf16scale", help="comma list of f32scale, bf16scale")
  parser.add_argument("--json-out", default="", help="optional path for the metrics as JSON")
  args = parser.parse_args()
  variants = [v for v in args.variants.split(",") if v]
  unknown = set(variants) - {"f32scale", "bf16scale"}
  if unknown:
    raise ValueError(f"unknown variants {sorted(unknown)}")

  from huggingface_hub import snapshot_download

  if os.path.isdir(args.repo):
    snapshot_dir = args.repo
  else:
    snapshot_dir = snapshot_download(repo_id=args.repo, allow_patterns=["text_encoder/*", "tokenizer/*"])
  te_path = os.path.join(snapshot_dir, "text_encoder")
  with open(os.path.join(te_path, "config.json"), "r") as f:
    te_config = json.load(f)
  config = build_qwen3_config(te_config, types.SimpleNamespace(weights_dtype=jnp.float32))
  tokenizer = load_krea2_tokenizer(os.path.join(snapshot_dir, "tokenizer"), snapshot_dir)
  cpu = jax.local_devices(backend="cpu")[0]

  prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]
  batches = [tokenize_krea2_prompts(tokenizer, [p], args.max_sequence_length) for p in prompts]
  valid = [mask[0, KREA2_PROMPT_TEMPLATE_START_IDX:].astype(bool) for _, mask, _ in batches]
  for i, (p, v) in enumerate(zip(prompts, valid)):
    print(f"prompt {i}: {int(v.sum())} valid tokens (prompt + suffix) | {p[:60]!r}")

  seq = args.max_sequence_length + KREA2_PROMPT_TEMPLATE_START_IDX
  ids_dummy = jnp.zeros((1, seq), jnp.int32)
  mask_dummy = jnp.ones((1, seq), jnp.int32)

  # 1. bf16 reference (weights as loaded, float32 activations).
  model = FlaxQwen3Model(config)
  template = nn.unbox(jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), ids_dummy, mask_dummy))["params"])
  t0 = time.perf_counter()
  params = load_and_convert_qwen3_weights(te_path, template, config, key_prefix=KEY_PREFIX)
  print(f"loaded reference weights in {time.perf_counter() - t0:.1f}s: {tree_nbytes(params) / GIB:.2f} GiB")
  params = to_device_inplace(params, cpu)
  ref_out, ref_times = run_model("bf16", model, params, batches)

  # 2. int8 weight-only, embedding lookup on the host.
  table = load_qwen3_embedding_table(te_path, key_prefix=KEY_PREFIX)
  if not np.array_equal(table.view(np.uint16), np.asarray(params["embed_tokens"]["embedding"]).view(np.uint16)):
    raise AssertionError("host embedding table differs from the in-model embedding")
  del params["embed_tokens"]
  qmodel = quantize_text_encoder_model(model, args.tile_size)
  embeds_dummy = jnp.zeros((1, seq, config.hidden_size), jnp.float32)
  abstract_q = jax.eval_shape(lambda: qmodel.init(jax.random.PRNGKey(0), None, mask_dummy, inputs_embeds=embeds_dummy))[
      "params"
  ]
  t0 = time.perf_counter()
  bf16_bytes = tree_nbytes(params)
  qparams = quantize_text_encoder_params(params, abstract_q, device=cpu)
  quant_s = time.perf_counter() - t0
  print(f"quantized in {quant_s:.1f}s: {bf16_bytes / GIB:.2f} GiB -> {tree_nbytes(qparams) / GIB:.2f} GiB (f32 scales)")
  del params
  gc.collect()

  results = {}
  for variant in variants:
    vparams = qparams
    if variant == "bf16scale":
      from qwix._src.providers.ptq import WithAux

      def round_scale(leaf):
        if not isinstance(leaf, WithAux):
          return leaf
        scale = np.asarray(leaf.array.scale).astype(jnp.bfloat16).astype(np.float32)
        return leaf.replace(array=leaf.array.replace(scale=scale))

      vparams = jax.tree_util.tree_map(round_scale, qparams, is_leaf=lambda x: isinstance(x, WithAux))
    vparams = jax.device_put(vparams, cpu)
    q_out, q_times = run_model(f"int8-{variant}", qmodel, vparams, batches, embed_table=table)
    del vparams
    prompt_stats = [metrics(q, r, v) for q, r, v in zip(q_out, ref_out, valid)]
    results[variant] = (pool(prompt_stats), [st[-1] for st in prompt_stats], q_times)

  # Report.
  def fmt_row(label, st):
    sq_err, sq_ref, cos_sum, n, max_err, max_ref = st
    return f"{label:<10}{np.sqrt(sq_err / sq_ref):>12.3e}{cos_sum / n:>14.6f}{max_err:>14.4g}{max_ref:>14.4g}"

  header = f"{'layer':<10}{'rel L2':>12}{'mean cos':>14}{'max |err|':>14}{'max |ref|':>14}"
  for variant, (stats, per_prompt, q_times) in results.items():
    print()
    print(f"=== int8 weight-only (tile {args.tile_size}, {variant}), host embedding vs bf16 reference ===")
    print(
        f"{args.num_prompts} prompts, {int(sum(v.sum() for v in valid))} valid tokens, tapped layers "
        f"{list(KREA2_TEXT_ENCODER_SELECT_LAYERS)} (all_hidden_states index)"
    )
    print(header)
    for layer_idx, st in zip(KREA2_TEXT_ENCODER_SELECT_LAYERS, stats[:-1]):
      print(fmt_row(f"h[{layer_idx}]", st))
    print(fmt_row("overall", stats[-1]))
    print("per prompt (overall):")
    for i, st in enumerate(per_prompt):
      print(fmt_row(f"prompt {i}", st) + f"   ({int(valid[i].sum())} tokens)")
    print(
        f"time: bf16 first {ref_times[0]:.1f}s (incl. compile), steady {np.mean(ref_times[1:] or ref_times):.1f}s/prompt; "
        f"int8 first {q_times[0]:.1f}s, steady {np.mean(q_times[1:] or q_times):.1f}s/prompt; quantize {quant_s:.1f}s"
    )

  if args.json_out:
    payload = {
        variant: {
            "layers": {
                str(layer_idx): {
                    "rel_l2": float(np.sqrt(st[0] / st[1])),
                    "mean_cos": float(st[2] / st[3]),
                    "max_abs_err": float(st[4]),
                }
                for layer_idx, st in zip(list(KREA2_TEXT_ENCODER_SELECT_LAYERS) + ["overall"], stats)
            },
            "int8_seconds": q_times,
        }
        for variant, (stats, _, q_times) in results.items()
    }
    payload["bf16_seconds"] = ref_times
    with open(args.json_out, "w") as f:
      json.dump(payload, f, indent=2)


if __name__ == "__main__":
  main()
