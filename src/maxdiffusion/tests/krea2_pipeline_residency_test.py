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
# and per-phase placement of offloaded component weights.

import types
import unittest
from unittest import mock

import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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


def _tiny_transformer(in_channels=16):
  return Krea2Transformer2DModel(
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
      attention_kernel="dot_product",
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

  def _step(self, donate):
    pipeline = _pipeline(self.transformer, krea2_staged_donate_hidden_states=donate)
    pipeline._setup_jit_functions()
    return pipeline, np.asarray(pipeline._jitted_transformer_step(self.params, *self.inputs))

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
    prelude_keys = ("img_in", "time_embed", "time_mod_proj", "text_fusion", "txt_in")
    return pipeline._jitted_transformer_prelude({key: self.params[key] for key in prelude_keys}, *self.inputs)

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

  def _run(self, offload_components):
    sharding = NamedSharding(_mesh(), P())
    shardings = {
        "transformer": jax.tree_util.tree_map(lambda _: sharding, self.host_params),
        "text_encoder": jax.tree_util.tree_map(lambda _: sharding, self.host_qwen3),
    }
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
        config=_config(is_distilled=True),
        mesh=_mesh(),
        offload_components=offload_components,
        param_shardings=shardings,
    )
    pipeline._setup_jit_functions()
    seen = {}

    def fake_encode_prompt(prompts, qwen3_params):
      seen.setdefault("qwen3_params", []).append(qwen3_params)
      return self.prompt_embeds, self.text_mask

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
    # Host trees are left intact for the next generation.
    self.assertIsInstance(self.host_qwen3["embed"], np.ndarray)
    self.assertTrue(all(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(self.host_params)))


if __name__ == "__main__":
  unittest.main()
