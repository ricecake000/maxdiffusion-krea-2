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

# CPU tests for Krea 2 HBM residency options: donated staged residual stream
# and per-phase placement of offloaded component weights; text compaction and
# the precompile -> lazy-load round trip of the AOT cache.

import glob
import os
import tempfile
import types
import unittest
from unittest import mock

import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion import aot_cache, generate_krea2
from maxdiffusion.models.krea2.resolution_presets import Krea2Resolution
from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
from maxdiffusion.models.krea2.util import prepare_krea2_image_ids, prepare_krea2_text_ids
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.pipelines.krea2 import krea2_pipeline
from maxdiffusion.pipelines.krea2.krea2_pipeline import (
    FlaxKrea2Pipeline,
    free_params,
    params_device_bytes,
    place_params,
)

_B, _GRID_H, _GRID_W, _S_TXT = 1, 2, 2, 3


def _unbox(params):
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x,
      params,
      is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
  )


def _tiny_transformer(in_channels=16, rope_layout="interleaved", attention_kernel="dot_product"):
  return Krea2Transformer2DModel(
      rope_layout=rope_layout,
      attention_kernel=attention_kernel,
      in_channels=in_channels,
      num_layers=2,
      attention_head_dim=8,
      num_attention_heads=4,
      num_key_value_heads=2,
      intermediate_size=64,
      timestep_embed_dim=16,
      text_hidden_dim=24,
      num_text_layers=3,
      text_num_attention_heads=4,
      text_num_key_value_heads=4,
      text_intermediate_size=48,
      num_layerwise_text_blocks=2,
      num_refiner_text_blocks=2,
      axes_dims_rope=(4, 2, 2),
  )


def _config(**overrides):
  values = {
      "max_sequence_length": _S_TXT,
      "logical_axis_rules": (),
      "krea2_staged_transformer": True,
      "krea2_staged_donate_hidden_states": True,
      "activations_dtype": jnp.float32,
      "weights_dtype": jnp.float32,
      "seed": 0,
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


def _mesh():
  return Mesh(np.array(jax.devices()[:1]), ("data",))


def _pipeline(transformer, **config_overrides):
  return FlaxKrea2Pipeline(
      transformer=transformer,
      vae=None,
      vae_cache=None,
      text_encoder=None,
      tokenizer=None,
      scheduler=None,
      config=_config(**config_overrides),
      mesh=_mesh(),
  )


def _inputs():
  s_img = _GRID_H * _GRID_W
  latents = jnp.array(np.random.RandomState(0).randn(_B, s_img, 16), dtype=jnp.float32)
  prompt_embeds = jnp.array(np.random.RandomState(1).randn(_B, _S_TXT, 3, 24), dtype=jnp.float32)
  text_mask = jnp.array([[True, True, False]])
  img_ids = prepare_krea2_image_ids(_B, _GRID_H, _GRID_W)
  txt_ids = prepare_krea2_text_ids(_B, _S_TXT)
  t_vec = jnp.full((_B,), 0.5, dtype=jnp.float32)
  return latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec


class Krea2StagedDonationTest(unittest.TestCase):

  def setUp(self):
    self.transformer = _tiny_transformer()
    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = _inputs()
    self.inputs = (latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec)
    self.params = _unbox(
        self.transformer.init(jax.random.PRNGKey(0), latents, prompt_embeds, t_vec, img_ids, txt_ids, text_mask)["params"]
    )

  def _step_inputs(self, pipeline):
    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = self.inputs
    text_params = {key: self.params[key] for key in krea2_pipeline.KREA2_TEXT_CONTEXT_KEYS}
    text_hidden = pipeline._jitted_transformer_text_context(text_params, prompt_embeds, text_mask)
    return latents, text_hidden, text_mask, img_ids, txt_ids, t_vec

  def _step(self, donate):
    pipeline = _pipeline(self.transformer, krea2_staged_donate_hidden_states=donate)
    pipeline._setup_jit_functions()
    return pipeline, np.asarray(pipeline._jitted_transformer_step(self.params, *self._step_inputs(pipeline)))

  def test_donated_step_matches_undonated_and_monolithic(self):
    _, donated = self._step(donate=True)
    _, undonated = self._step(donate=False)
    np.testing.assert_allclose(donated, undonated, rtol=1e-6, atol=1e-6)

    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = self.inputs
    expected = self.transformer.apply(
        {"params": self.params}, latents, prompt_embeds, t_vec, img_ids, txt_ids, text_mask
    ).sample
    np.testing.assert_allclose(donated, np.asarray(expected), rtol=1e-5, atol=1e-5)

    # Only internal residual buffers are donated; caller inputs survive.
    for x in self.inputs:
      self.assertFalse(x.is_deleted())
    for leaf in jax.tree_util.tree_leaves(self.params):
      self.assertFalse(leaf.is_deleted())

  def _prelude_outputs(self, pipeline):
    prelude_keys = ("img_in", "time_embed", "time_mod_proj")
    return pipeline._jitted_transformer_prelude(
        {key: self.params[key] for key in prelude_keys}, *self._step_inputs(pipeline)
    )

  def test_block_donates_hidden_states(self):
    pipeline, _ = self._step(donate=True)
    hidden_states, _, temb_mod, rotary_emb, attention_mask = self._prelude_outputs(pipeline)
    out = pipeline._jitted_transformer_block(
        self.params["blocks_0"], {}, hidden_states, temb_mod, rotary_emb, attention_mask
    )
    jax.block_until_ready(out)
    self.assertTrue(hidden_states.is_deleted())
    self.assertFalse(temb_mod.is_deleted())
    self.assertEqual(out.shape, hidden_states.shape)

  def test_staged_rotate_half_step_matches_monolithic(self):
    transformer = _tiny_transformer(rope_layout="rotate_half")
    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = self.inputs
    expected = transformer.apply(
        {"params": self.params}, latents, prompt_embeds, t_vec, img_ids, txt_ids, text_mask
    ).sample
    for staged in (True, False):
      pipeline = _pipeline(transformer, krea2_staged_transformer=staged)
      pipeline._setup_jit_functions()
      actual = pipeline._jitted_transformer_step(self.params, *self._step_inputs(pipeline))
      np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_block_keeps_hidden_states_without_donation(self):
    pipeline, _ = self._step(donate=False)
    hidden_states, _, temb_mod, rotary_emb, attention_mask = self._prelude_outputs(pipeline)
    out = pipeline._jitted_transformer_block(
        self.params["blocks_0"], {}, hidden_states, temb_mod, rotary_emb, attention_mask
    )
    jax.block_until_ready(out)
    self.assertFalse(hidden_states.is_deleted())


class Krea2ParamResidencyTest(unittest.TestCase):

  def _host_tree(self):
    return {"a": {"kernel": np.arange(12, dtype=np.float32).reshape(3, 4)}, "b": np.ones((5,), np.float32)}

  def _shardings(self, tree):
    sharding = NamedSharding(_mesh(), P())
    return jax.tree_util.tree_map(lambda _: sharding, tree)

  def test_place_and_free_numpy_host_tree(self):
    host = self._host_tree()
    placed = place_params(host, self._shardings(host))
    for leaf in jax.tree_util.tree_leaves(placed):
      self.assertIsInstance(leaf, jax.Array)
      self.assertFalse(leaf.is_deleted())
    np.testing.assert_array_equal(np.asarray(placed["a"]["kernel"]), host["a"]["kernel"])
    self.assertEqual(params_device_bytes(placed), (12 + 5) * 4)

    self.assertIsNone(free_params(placed, host))
    for leaf in jax.tree_util.tree_leaves(placed):
      self.assertTrue(leaf.is_deleted())
    # Host tree is untouched and can be placed again for the next generation.
    np.testing.assert_array_equal(host["b"], np.ones((5,), np.float32))
    again = place_params(host, self._shardings(host))
    np.testing.assert_array_equal(np.asarray(again["b"]), host["b"])

  def test_free_keeps_host_buffers_aliased_on_same_device(self):
    # On a CPU-only run the "host" tree already lives on the target device and
    # placement may alias it; freeing must not delete the host copy.
    host = jax.tree_util.tree_map(jnp.asarray, self._host_tree())
    placed = place_params(host, self._shardings(host))
    free_params(placed, host)
    for leaf in jax.tree_util.tree_leaves(host):
      self.assertFalse(leaf.is_deleted())
    np.testing.assert_array_equal(np.asarray(host["b"]), np.ones((5,), np.float32))

  def test_swap_in_records_trace(self):
    host = self._host_tree()
    pipeline = FlaxKrea2Pipeline(
        transformer=_tiny_transformer(),
        vae=None,
        vae_cache=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
        config=_config(),
        mesh=_mesh(),
        offload_components=("text_encoder",),
        param_shardings={"text_encoder": self._shardings(host)},
    )
    trace = {}
    placed = pipeline._swap_in("text_encoder", host, trace)
    self.assertGreaterEqual(trace["text_encoder_swap_in"], 0.0)
    self.assertEqual(trace["text_encoder_swap_bytes"], (12 + 5) * 4)
    free_params(placed, host)

  def test_offload_component_validation(self):
    kwargs = {
        "transformer": _tiny_transformer(),
        "vae": None,
        "vae_cache": None,
        "text_encoder": None,
        "tokenizer": None,
        "scheduler": None,
        "config": _config(),
        "mesh": _mesh(),
    }
    with self.assertRaisesRegex(ValueError, "vae"):
      FlaxKrea2Pipeline(**kwargs, offload_components=("vae",), param_shardings={"vae": {}})
    with self.assertRaisesRegex(ValueError, "transformer"):
      FlaxKrea2Pipeline(**kwargs, offload_components=("transformer",), param_shardings={"text_encoder": {}})
    pipeline = FlaxKrea2Pipeline(**kwargs)
    self.assertEqual(pipeline.offload_components, frozenset())


class Krea2QuantizedTextEncoderResidencyTest(unittest.TestCase):
  """int8 text encoder + offload: place/free a qwix-quantized host tree."""

  _RULES = (("vocab", None), ("embed", None), ("mlp", "data"), ("heads", "data"), ("kv", None), ("norm", None))

  def test_place_forward_free_quantized_tree(self):
    from flax import linen as nn
    from flax.linen import partitioning as nn_partitioning
    from qwix._src.core.qarray import QArray
    from qwix._src.providers.ptq import WithAux

    from maxdiffusion.models.krea2.text_encoder_quant import (
        quantize_text_encoder_model,
        quantize_text_encoder_params,
        safe_param_shardings,
    )
    from maxdiffusion.models.qwen3_flax import FlaxQwen3Model
    from maxdiffusion.tests.krea2_text_encoder_quant_test import _TILE
    from maxdiffusion.tests.krea2_text_encoder_quant_test import _config as _qwen3_config

    model = FlaxQwen3Model(_qwen3_config())
    qmodel = quantize_text_encoder_model(model, _TILE)
    rng = np.random.RandomState(0)
    ids = jnp.asarray(rng.randint(0, 100, size=(1, 8)), dtype=jnp.int32)
    mask = jnp.ones_like(ids)
    position_ids = jnp.cumsum(mask, axis=-1) - 1
    host = jax.tree_util.tree_map(np.asarray, nn.unbox(model.init(jax.random.PRNGKey(0), ids, mask)["params"]))

    # Same derivation as generate_krea2: boxed abstract tree -> logical specs -> mesh shardings.
    mesh = _mesh()
    with mesh, nn_partitioning.axis_rules(self._RULES):
      abstract = jax.eval_shape(lambda: qmodel.init(jax.random.PRNGKey(0), ids, mask))["params"]
      shardings = nn.logical_to_mesh_sharding(nn.get_partition_spec(abstract), mesh, self._RULES)
    shardings = safe_param_shardings(nn.unbox(abstract), shardings, mesh)
    host_q = quantize_text_encoder_params(host, abstract)
    kernel = host_q["layers_0"]["mlp"]["down_proj"]["kernel"]
    self.assertIsInstance(kernel, WithAux)
    self.assertIsInstance(kernel.array, QArray)
    self.assertIsInstance(kernel.array.qvalue, np.ndarray)

    expected_last, _ = qmodel.apply({"params": host_q}, ids, mask, position_ids=position_ids)
    device = jax.devices()[0]
    fwd = jax.jit(lambda p: qmodel.apply({"params": p}, ids, mask, position_ids=position_ids)[0])
    for _ in range(2):  # a second cycle mirrors the next generation's `_swap_in`
      placed = place_params(host_q, shardings)
      self.assertEqual(jax.tree_util.tree_structure(placed), jax.tree_util.tree_structure(host_q))
      placed_kernel = placed["layers_0"]["mlp"]["down_proj"]["kernel"]
      self.assertIsInstance(placed_kernel, WithAux)
      self.assertIsInstance(placed_kernel.array, QArray)
      self.assertEqual(placed_kernel.array.qvalue.dtype, jnp.int8)
      leaves = jax.tree_util.tree_leaves(placed)
      self.assertEqual(len(leaves), len(jax.tree_util.tree_leaves(host_q)))
      for leaf in leaves:
        self.assertIsInstance(leaf, jax.Array)
        self.assertEqual(leaf.devices(), {device})
        self.assertFalse(leaf.is_deleted())
      self.assertEqual(params_device_bytes(placed), sum(x.nbytes for x in jax.tree_util.tree_leaves(host_q)))

      last = np.asarray(fwd(placed))
      self.assertTrue(np.all(np.isfinite(last)))
      np.testing.assert_allclose(last, np.asarray(expected_last), rtol=1e-5, atol=1e-5)

      self.assertIsNone(free_params(placed, host_q))
      for leaf in leaves:
        self.assertTrue(leaf.is_deleted())
      # The host tree survives for the next placement.
      self.assertIsInstance(host_q["layers_0"]["mlp"]["down_proj"]["kernel"].array.qvalue, np.ndarray)


class Krea2OffloadCallTest(unittest.TestCase):
  """Runs `__call__` end to end on CPU with stub text encoder and VAE."""

  def setUp(self):
    # 32x32 pixels -> 2x2 grid of packed 64-channel latents (16 VAE channels).
    self.transformer = _tiny_transformer(in_channels=64)
    self.latents = np.random.RandomState(0).randn(_B, _GRID_H * _GRID_W, 64).astype(np.float32)
    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = _inputs()
    del latents
    params = self.transformer.init(
        jax.random.PRNGKey(0), jnp.asarray(self.latents), prompt_embeds, t_vec, img_ids, txt_ids, text_mask
    )["params"]
    # Host trees as generate_krea2 keeps them for offloaded components.
    self.host_params = jax.tree_util.tree_map(np.asarray, _unbox(params))
    self.host_qwen3 = {"embed": np.ones((4, 24), np.float32)}
    self.prompt_embeds = prompt_embeds
    self.text_mask = text_mask

  def _run(self, offload_components, text_mask=None, transformer=None, call_kwargs=None, **config_overrides):
    sharding = NamedSharding(_mesh(), P())
    shardings = {
        "transformer": jax.tree_util.tree_map(lambda _: sharding, self.host_params),
        "text_encoder": jax.tree_util.tree_map(lambda _: sharding, self.host_qwen3),
    }
    pipeline = FlaxKrea2Pipeline(
        transformer=transformer or self.transformer,
        vae=types.SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
        vae_cache=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=FlaxFlowMatchScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            time_shift_type="exponential",
        ),
        config=_config(is_distilled=True, **config_overrides),
        mesh=_mesh(),
        offload_components=offload_components,
        param_shardings=shardings,
    )
    pipeline._setup_jit_functions()
    seen = {}

    mask = self.text_mask if text_mask is None else text_mask

    def fake_encode_prompt(prompts, qwen3_params):
      seen.setdefault("qwen3_params", []).append(qwen3_params)
      return self.prompt_embeds, mask

    def fake_vae_decode(graphdef, state, rest_of_state, latents_5d):
      seen["latents_5d"] = np.asarray(latents_5d)
      return jnp.zeros((_B, 1, 32, 32, 3))

    pipeline.encode_prompt = fake_encode_prompt
    pipeline._jitted_vae_decode = fake_vae_decode
    if offload_components:
      params, qwen3_params = self.host_params, self.host_qwen3
    else:
      params = place_params(self.host_params, shardings["transformer"])
      qwen3_params = place_params(self.host_qwen3, shardings["text_encoder"])
    freed = seen.setdefault("freed", [])

    def spy_free_params(tree, host_tree=None):
      freed.append(tree)
      return free_params(tree, host_tree)

    with (
        mock.patch.object(krea2_pipeline.nnx, "split", return_value=(None, None, None)),
        mock.patch.object(krea2_pipeline, "free_params", side_effect=spy_free_params),
    ):
      _, trace = pipeline(
          "a fox",
          params,
          qwen3_params,
          height=32,
          width=32,
          num_inference_steps=2,
          guidance_scale=1.5,
          do_classifier_free_guidance=True,
          latents=self.latents,
          save_outputs=False,
          **(call_kwargs or {}),
      )
    return trace, seen

  def test_offloaded_call_matches_resident_and_frees_device_params(self):
    resident_trace, resident_seen = self._run(())
    offload_trace, offload_seen = self._run(("text_encoder", "transformer"))

    np.testing.assert_allclose(offload_seen["latents_5d"], resident_seen["latents_5d"], rtol=1e-6, atol=1e-6)
    for key in ("text_encoder_swap_in", "text_encoder_swap_bytes", "transformer_swap_in", "transformer_swap_bytes"):
      self.assertNotIn(key, resident_trace)
      self.assertIn(key, offload_trace)
    self.assertEqual(offload_trace["text_encoder_swap_bytes"], 4 * 24 * 4)
    self.assertEqual(offload_trace["transformer_swap_bytes"], params_device_bytes(self.host_params))

    # CFG encodes both prompts from one placement, which is freed after Phase A.
    placed_qwen3 = offload_seen["qwen3_params"]
    self.assertEqual(len(placed_qwen3), 2)
    self.assertIs(placed_qwen3[0], placed_qwen3[1])
    self.assertTrue(placed_qwen3[0]["embed"].is_deleted())
    self.assertEqual(resident_seen["freed"], [])
    self.assertEqual(len(offload_seen["freed"]), 2)
    self.assertIs(offload_seen["freed"][0], placed_qwen3[0])
    for leaf in jax.tree_util.tree_leaves(offload_seen["freed"][1]):
      self.assertTrue(leaf.is_deleted())

  def test_text_compaction_matches_uncompacted_call(self):
    # Mid-sequence padding [valid | PAD | valid] like the Krea 2 template.
    mid_padded = jnp.array([[True, False, True]])
    _, plain_seen = self._run((), text_mask=mid_padded)
    _, compact_seen = self._run(
        ("text_encoder",), text_mask=mid_padded, krea2_text_compaction_multiple=2, krea2_staged_transformer=False
    )
    np.testing.assert_allclose(compact_seen["latents_5d"], plain_seen["latents_5d"], rtol=1e-5, atol=1e-5)

  def test_compact_text_embeddings_min_tokens(self):
    # Mid-sequence padding; the batch's largest valid count is 3 (row 1).
    mask = np.array(
        [
            [True, False, False, True, False, False, False],
            [False, True, True, False, True, False, False],
        ]
    )
    embeds = jnp.arange(2 * 7, dtype=jnp.float32).reshape(2, 7, 1, 1)
    compact = krea2_pipeline.compact_text_embeddings

    def bucket(**kwargs):
      return compact(embeds, mask, 2, **kwargs)[1].shape[1]

    self.assertEqual(bucket(), 4)  # round_up(3, 2)
    self.assertEqual(bucket(min_tokens=0), 4)
    self.assertEqual(bucket(min_tokens=1), 4)  # below the valid count: no effect
    self.assertEqual(bucket(min_tokens=5), 6)  # a non-multiple minimum is rounded up
    self.assertEqual(bucket(min_tokens=6), 6)
    self.assertEqual(bucket(min_tokens=100), 7)  # clipped to the sequence length

    out_embeds, out_mask = compact(embeds, mask, 2, min_tokens=5)
    self.assertEqual(out_embeds.shape, (2, 6, 1, 1))
    # Prefix mask, valid tokens first in their original order, then padding.
    np.testing.assert_array_equal(
        np.asarray(out_mask), [[True, True, False, False, False, False], [True, True, True, False, False, False]]
    )
    np.testing.assert_array_equal(np.asarray(out_embeds[0, :2, 0, 0]), [0.0, 3.0])
    np.testing.assert_array_equal(np.asarray(out_embeds[1, :3, 0, 0]), [8.0, 9.0, 11.0])

    with self.assertRaisesRegex(ValueError, "min_tokens"):
      compact(embeds, mask, 2, min_tokens=-1)

  def test_forced_text_bucket_matches_compacted_call(self):
    # A forced larger bucket only appends masked padding tokens: same image.
    mid_padded = jnp.array([[True, False, True]])
    buckets = []
    real_compact = krea2_pipeline.compact_text_embeddings

    def spy_compact(embeds, mask, multiple, **kwargs):
      out = real_compact(embeds, mask, multiple, **kwargs)
      buckets.append((kwargs.get("min_tokens", 0), out[1].shape[1]))
      return out

    config = {"krea2_text_compaction_multiple": 2, "krea2_staged_transformer": False}
    with mock.patch.object(krea2_pipeline, "compact_text_embeddings", side_effect=spy_compact):
      _, compact_seen = self._run((), text_mask=mid_padded, **config)
      _, forced_seen = self._run((), text_mask=mid_padded, call_kwargs={"min_text_tokens": 3}, **config)
    # Prompt + negative prompt (CFG); round_up(3, 2) = 4 is clipped to the text length 3.
    self.assertEqual(buckets, [(0, 2), (0, 2), (3, 3), (3, 3)])
    np.testing.assert_allclose(forced_seen["latents_5d"], compact_seen["latents_5d"], rtol=1e-5, atol=1e-5)

  def test_flash_custom_always_compacts_to_a_prefix_mask(self):
    # flash_custom needs a prefix key mask, so compaction is forced (with the
    # full length as the bucket) even when krea2_text_compaction_multiple=0.
    mid_padded = jnp.array([[True, False, True]])
    transformer = _tiny_transformer(in_channels=64, attention_kernel="flash_custom")
    calls = []
    real_compact = krea2_pipeline.compact_text_embeddings

    def spy_compact(embeds, mask, multiple):
      out = real_compact(embeds, mask, multiple)
      calls.append((multiple, np.asarray(out[1])))
      return out

    with mock.patch.object(krea2_pipeline, "compact_text_embeddings", side_effect=spy_compact):
      _, custom_seen = self._run((), text_mask=mid_padded, transformer=transformer, krea2_text_compaction_multiple=0)
    self.assertEqual([multiple for multiple, _ in calls], [_S_TXT, _S_TXT])  # positive + negative (CFG)
    for _, mask in calls:
      np.testing.assert_array_equal(mask, [[True, True, False]])
    # Tiny sequences fall back to masked dot-product attention: same result as dot_product.
    _, plain_seen = self._run((), text_mask=mid_padded)
    np.testing.assert_allclose(custom_seen["latents_5d"], plain_seen["latents_5d"], rtol=1e-5, atol=1e-5)
    # Host trees are left intact for the next generation.
    self.assertIsInstance(self.host_qwen3["embed"], np.ndarray)
    self.assertTrue(all(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(self.host_params)))


@aot_cache.cached_jit
def _tiny_vae_decode(graphdef, state, rest_of_state, latents_5d):
  """Stand-in for vae_decode_pass: a cached executable whose output depends on the latents."""
  del graphdef, state, rest_of_state
  return jnp.tanh(latents_5d) * 2.0


class Krea2PrecompileRoundTripTest(unittest.TestCase):
  """What generate_krea2.run_precompile saves is what a later lazy-loading run loads and runs.

  The real pipeline with the tiny transformer; the text encoder and the VAE
  are stubbed like in Krea2OffloadCallTest, the VAE through a cached entry.
  """

  _S_TXT = 6
  _META = {"test": "krea2_precompile_round_trip"}
  # 1 valid token: natural bucket 2 (multiple 2), so precompile has to force 4.
  _SHORT_MASK = [[True, False, False, False, False, False]]
  # 3 valid tokens (mid-sequence padding): natural bucket 4.
  _RUN_MASK = [[True, True, False, True, False, False]]

  @staticmethod
  def _reset_aot_state():
    """Puts the process-global AOT cache back to its never-installed state.

    Other tests expect it disabled and every entry empty, whatever order they
    run in. `_STATE.generation` is left alone: it only ever has to change.
    """
    aot_cache.wait_for_loads()
    state = aot_cache._STATE
    state.enabled = False
    state.lazy_load = False
    state.warmup_only = False
    state.cache_dir = ""
    state.fingerprint = ""
    state.mesh = None
    for entry in aot_cache._REGISTRY:
      with entry._lock:
        entry._compiled.clear()
        entry._out_specs.clear()
        entry._pending.clear()
        entry._adapters.clear()
        entry._on_disk.clear()
        entry._lazy_tried.clear()

  def setUp(self):
    # Registered first, so the reset also runs when a later setUp step fails.
    self.addCleanup(self._reset_aot_state)
    self._reset_aot_state()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.cache_dir = tmp.name
    # nnx.split of the stub VAE: nothing to split.
    patcher = mock.patch.object(krea2_pipeline.nnx, "split", return_value=(None, None, None))
    patcher.start()
    self.addCleanup(patcher.stop)

    self.transformer = _tiny_transformer(in_channels=64)
    self.latents = np.random.RandomState(0).randn(_B, _GRID_H * _GRID_W, 64).astype(np.float32)
    self.prompt_embeds = jnp.array(np.random.RandomState(1).randn(_B, self._S_TXT, 3, 24), dtype=jnp.float32)
    _, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = _inputs()
    params = self.transformer.init(
        jax.random.PRNGKey(0), jnp.asarray(self.latents), prompt_embeds, t_vec, img_ids, txt_ids, text_mask
    )["params"]
    host_params = jax.tree_util.tree_map(np.asarray, _unbox(params))
    sharding = NamedSharding(_mesh(), P())
    self.params = place_params(host_params, jax.tree_util.tree_map(lambda _: sharding, host_params))
    self.qwen3_params = {"embed": jnp.ones((4, 24), jnp.float32)}
    # 32x32 pixels: a 2x2 grid of packed latents, like self.latents.
    self.resolution = Krea2Resolution(image_size="1k", aspect_ratio="1:1", height=32, width=32)

  def _pipeline(self, text_mask):
    pipeline = FlaxKrea2Pipeline(
        transformer=self.transformer,
        vae=types.SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
        vae_cache=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=FlaxFlowMatchScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            time_shift_type="exponential",
        ),
        config=_config(
            is_distilled=True,
            max_sequence_length=self._S_TXT,
            krea2_text_compaction_multiple=2,
            krea2_staged_transformer=False,
        ),
        mesh=_mesh(),
    )
    pipeline._setup_jit_functions()
    seen = {}
    mask = jnp.array(text_mask)

    def fake_encode_prompt(prompts, qwen3_params):
      return self.prompt_embeds, mask

    def vae_decode(graphdef, state, rest_of_state, latents_5d):
      images = _tiny_vae_decode(graphdef, state, rest_of_state, latents_5d)
      seen["latents_5d"] = np.asarray(latents_5d)
      seen["images"] = np.asarray(images)
      return images

    pipeline.encode_prompt = fake_encode_prompt
    pipeline._jitted_vae_decode = vae_decode
    return pipeline, seen

  def _call_kwargs(self):
    return {
        "params": self.params,
        "qwen3_params": self.qwen3_params,
        "num_inference_steps": 2,
        "guidance_scale": 1.5,
        "do_classifier_free_guidance": True,
        "latents": self.latents,
    }

  def _run(self, pipeline):
    _, trace = pipeline(
        "a fox",
        height=self.resolution.height,
        width=self.resolution.width,
        save_outputs=False,
        **self._call_kwargs(),
    )
    return trace

  def _entries(self, pipeline):
    return {
        "transformer_step": pipeline._jitted_transformer_step,
        "text_context": pipeline._jitted_transformer_text_context,
        "vae_decode": _tiny_vae_decode,
    }

  def test_precompiled_executables_are_loaded_by_a_later_run(self):
    # (1) Reference: AOT cache disabled.
    reference_pipeline, reference_seen = self._pipeline(self._RUN_MASK)
    reference_trace = self._run(reference_pipeline)
    self.assertEqual((reference_trace["text_tokens"], reference_trace["negative_text_tokens"]), (4, 4))

    # (2) Precompile one resolution with a bucket larger than the prompt's natural one.
    precompile_pipeline, _ = self._pipeline(self._SHORT_MASK)
    aot_cache.install(self.cache_dir, self._META, _mesh())
    aot_cache.wait_for_loads()
    plan = [(self.resolution, 4)]
    records = generate_krea2.run_precompile(precompile_pipeline, plan, self._call_kwargs(), ["a fox"])
    self.assertEqual(len(records), 1)
    self.assertEqual((records[0]["text_tokens"], records[0]["negative_text_tokens"]), (4, 4))
    self.assertEqual(records[0]["requested_text_tokens"], 4)
    self.assertGreaterEqual(records[0]["saved"], 3)
    written = {
        name: glob.glob(os.path.join(self.cache_dir, f"{entry.name}-*.aotx"))
        for name, entry in self._entries(precompile_pipeline).items()
    }
    for name, paths in written.items():
      self.assertTrue(paths, f"no executable written for {name}")

    # (3) A new process: fresh pipeline entries, lazy install, nothing loaded yet.
    run_pipeline, run_seen = self._pipeline(self._RUN_MASK)
    aot_cache.install(self.cache_dir, self._META, _mesh(), lazy_load=True)
    entries = self._entries(run_pipeline)
    for name, entry in entries.items():
      self.assertEqual(entry._compiled, {}, name)

    # Any jit fallback or compile would mean a cache miss.
    def no_compile(*args, **kwargs):
      raise AssertionError("cache miss: the run tried to jit or compile")

    with (
        mock.patch.object(aot_cache._AotEntry, "_adapter_for", side_effect=no_compile),
        mock.patch.object(aot_cache._AotEntry, "_compile_and_record", side_effect=no_compile),
    ):
      run_trace = self._run(run_pipeline)
    self.assertEqual((run_trace["text_tokens"], run_trace["negative_text_tokens"]), (4, 4))

    # (4) Every executable came from disk, and the result matches the reference.
    for name, entry in entries.items():
      self.assertTrue(entry._compiled, name)
      self.assertEqual(set(entry._compiled), entry._on_disk, name)
      self.assertEqual(entry._pending, {}, name)
      self.assertEqual(len(entry._on_disk), len(written[name]), name)
    np.testing.assert_allclose(run_seen["latents_5d"], reference_seen["latents_5d"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run_seen["images"], reference_seen["images"], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
