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

# CPU tests for Krea 2 rotate-half RoPE: the load-time q/k weight permutation
# makes the rotate_half model reproduce the interleaved model exactly.

import unittest

import flax
import flax.linen as nn
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.loaders.krea2_lora_pipeline import Krea2LoraLoaderMixin, insert_lora_params
from maxdiffusion.models.attention_flax import apply_rope
from maxdiffusion.models.embeddings_flax import FluxPosEmbed
from maxdiffusion.models.krea2.lora_util import convert_krea2_lora_to_flax
from maxdiffusion.models.krea2.transformer_krea2_flax import (
    Krea2Attention,
    Krea2Transformer2DModel,
    apply_rope_rotate_half,
    krea2_rotary_tables,
)
from maxdiffusion.models.krea2.util import (
    permute_rope_weights_to_rotate_half,
    prepare_krea2_image_ids,
    prepare_krea2_text_ids,
    rotate_half_permutation,
)

_HEAD_DIM, _HEADS, _KV_HEADS = 8, 4, 2
_AXES = (4, 2, 2)


def _unbox(params):
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x,
      params,
      is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
  )


def _tiny_model(**overrides):
  kwargs = dict(
      in_channels=16,
      num_layers=2,
      attention_head_dim=_HEAD_DIM,
      num_attention_heads=_HEADS,
      num_key_value_heads=_KV_HEADS,
      intermediate_size=64,
      timestep_embed_dim=16,
      text_hidden_dim=24,
      num_text_layers=3,
      text_num_attention_heads=4,
      text_num_key_value_heads=4,
      text_intermediate_size=48,
      num_layerwise_text_blocks=2,
      num_refiner_text_blocks=2,
      axes_dims_rope=_AXES,
  )
  kwargs.update(overrides)
  return Krea2Transformer2DModel(**kwargs)


def _inputs(batch_size=1, grid_h=2, grid_w=3, s_txt=5):
  rng = np.random.RandomState(0)
  hs = jnp.asarray(rng.randn(batch_size, grid_h * grid_w, 16), dtype=jnp.float32)
  ehs = jnp.asarray(rng.randn(batch_size, s_txt, 3, 24), dtype=jnp.float32)
  t = jnp.full((batch_size,), 0.4)
  img_ids = prepare_krea2_image_ids(batch_size, grid_h, grid_w)
  txt_ids = prepare_krea2_text_ids(batch_size, s_txt)
  # Mid-sequence padding like the Krea 2 template.
  mask = jnp.asarray([[True, True, False, False, True]] * batch_size)
  return hs, ehs, t, img_ids, txt_ids, mask


def _randomize(params, seed=0):
  """Random non-zero values everywhere (the init leaves norms at zero, which
  would hide a missing norm_q/norm_k permutation)."""
  leaves, treedef = jax.tree_util.tree_flatten(params)
  rng = np.random.RandomState(seed)
  new_leaves = [jnp.asarray(0.3 * rng.randn(*leaf.shape), dtype=leaf.dtype) for leaf in leaves]
  return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _init_params(model, inputs):
  return flax.core.unfreeze(_randomize(_unbox(model.init(jax.random.PRNGKey(0), *inputs)["params"])))


class RotaryTableTest(unittest.TestCase):

  def test_interleaved_tables_match_flux_pos_embed(self):
    ids = jnp.concatenate([prepare_krea2_image_ids(1, 3, 5)[0], prepare_krea2_text_ids(1, 4)[0]], axis=0)
    for axes, theta in ((_AXES, 1000.0), ((32, 48, 48), 1000.0)):
      cos, sin = krea2_rotary_tables(ids, axes, theta, "interleaved")
      ref_cos, ref_sin = FluxPosEmbed(theta=theta, axes_dim=axes, return_tuple=True).apply({}, ids)
      np.testing.assert_array_equal(np.asarray(cos), np.asarray(ref_cos))
      np.testing.assert_array_equal(np.asarray(sin), np.asarray(ref_sin))

  def test_rotate_half_table_width_is_half_head_dim(self):
    ids = prepare_krea2_image_ids(1, 3, 5)[0]
    cos, sin = krea2_rotary_tables(ids, (32, 48, 48), 1000.0, "rotate_half")
    self.assertEqual(cos.shape, (15, 64))
    self.assertEqual(sin.shape, (15, 64))
    self.assertEqual(cos.dtype, jnp.float32)
    # Each rotate-half entry is the shared angle of one interleaved pair.
    full_cos, full_sin = krea2_rotary_tables(ids, (32, 48, 48), 1000.0, "interleaved")
    np.testing.assert_array_equal(np.asarray(cos), np.asarray(full_cos)[:, 0::2])
    np.testing.assert_array_equal(np.asarray(sin), np.asarray(full_sin)[:, 1::2])

  def test_invalid_layout_is_rejected(self):
    with self.assertRaises(ValueError):
      krea2_rotary_tables(jnp.zeros((2, 3)), _AXES, 1000.0, "bogus")
    attn = Krea2Attention(dim=16, num_heads=2, num_kv_heads=2, head_dim=8, rope_layout="bogus")
    with self.assertRaises(ValueError):
      attn.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 16)))

  def test_rotate_half_of_permuted_vector_is_permuted_interleaved_rope(self):
    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.randn(2, 3, 7, _HEAD_DIM), dtype=jnp.float32)
    ids = prepare_krea2_image_ids(1, 1, 7)[0]
    full = krea2_rotary_tables(ids, _AXES, 1000.0, "interleaved")
    half = krea2_rotary_tables(ids, _AXES, 1000.0, "rotate_half")
    perm = rotate_half_permutation(_HEAD_DIM)
    expected, _ = apply_rope(x, x, full)
    actual = apply_rope_rotate_half(x[..., perm], *half)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected)[..., perm], rtol=1e-6, atol=1e-6)
    # Output keeps the input dtype.
    self.assertEqual(apply_rope_rotate_half(x.astype(jnp.bfloat16), *half).dtype, jnp.bfloat16)


class PermutationTest(unittest.TestCase):

  def test_permutation_index_map(self):
    np.testing.assert_array_equal(rotate_half_permutation(8), [0, 2, 4, 6, 1, 3, 5, 7])

  def test_permutes_only_block_q_k_and_their_norms(self):
    model = _tiny_model()
    inputs = _inputs()
    params = jax.tree_util.tree_map(np.asarray, _init_params(model, inputs))
    permuted = permute_rope_weights_to_rotate_half(params, _HEADS, _KV_HEADS, _HEAD_DIM)

    perm = rotate_half_permutation(_HEAD_DIM)
    q_index = np.concatenate([h * _HEAD_DIM + perm for h in range(_HEADS)])
    k_index = np.concatenate([h * _HEAD_DIM + perm for h in range(_KV_HEADS)])
    for i in range(model.num_layers):
      old, new = params[f"blocks_{i}"]["attn"], permuted[f"blocks_{i}"]["attn"]
      np.testing.assert_array_equal(new["to_q"]["kernel"], old["to_q"]["kernel"][:, q_index])
      np.testing.assert_array_equal(new["to_k"]["kernel"], old["to_k"]["kernel"][:, k_index])
      # new[i] = old[2i], new[i + D/2] = old[2i + 1] within the first head.
      np.testing.assert_array_equal(new["to_q"]["kernel"][:, 1], old["to_q"]["kernel"][:, 2])
      np.testing.assert_array_equal(new["to_q"]["kernel"][:, _HEAD_DIM // 2], old["to_q"]["kernel"][:, 1])
      np.testing.assert_array_equal(new["norm_q"]["weight"], old["norm_q"]["weight"][perm])
      np.testing.assert_array_equal(new["norm_k"]["weight"], old["norm_k"]["weight"][perm])
      for name in ("to_v", "to_gate", "to_out"):
        self.assertIs(new[name], old[name])
    # No RoPE in text fusion: untouched (shared, not copied).
    self.assertIs(permuted["text_fusion"], params["text_fusion"])
    for key in ("img_in", "txt_in", "final_layer", "time_embed", "time_mod_proj"):
      self.assertIs(permuted[key], params[key])
    # The input tree is not modified.
    self.assertFalse(np.array_equal(params["blocks_0"]["attn"]["to_q"]["kernel"], permuted["blocks_0"]["attn"]["to_q"]["kernel"]))

  def test_works_on_jax_arrays(self):
    model = _tiny_model()
    params = _init_params(model, _inputs())
    permuted = permute_rope_weights_to_rotate_half(params, _HEADS, _KV_HEADS, _HEAD_DIM)
    self.assertIsInstance(permuted["blocks_0"]["attn"]["to_q"]["kernel"], jax.Array)


class RotateHalfModelEquivalenceTest(unittest.TestCase):

  def test_full_forward_matches_interleaved(self):
    inputs = _inputs(batch_size=2)
    interleaved = _tiny_model()
    rotate_half = _tiny_model(rope_layout="rotate_half")
    params = _init_params(interleaved, inputs)
    permuted = permute_rope_weights_to_rotate_half(params, _HEADS, _KV_HEADS, _HEAD_DIM)

    expected = interleaved.apply({"params": params}, *inputs).sample
    actual = rotate_half.apply({"params": permuted}, *inputs).sample
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    # Sanity: without the permutation the rotate-half model differs.
    wrong = rotate_half.apply({"params": params}, *inputs).sample
    self.assertTrue(np.any(np.abs(np.asarray(wrong) - np.asarray(expected)) > 1e-4))

  def test_full_forward_with_lora_on_q_and_k_matches_interleaved(self):
    inputs = _inputs()
    rng = np.random.RandomState(5)

    def rand(*shape):
      return rng.randn(*shape).astype(np.float32)

    state_dict = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": rand(2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": rand(32, 2),
        "lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight": rand(3, 32),
        "lora_unet_transformer_blocks_0_attn_to_k.lora_up.weight": rand(16, 3),
        "lora_unet_transformer_blocks_1_attn_to_k.lora_down.weight": rand(2, 32),
        "lora_unet_transformer_blocks_1_attn_to_k.lora_up.weight": rand(16, 2),
        "lora_unet_transformer_blocks_1_attn_to_v.lora_down.weight": rand(2, 32),
        "lora_unet_transformer_blocks_1_attn_to_v.lora_up.weight": rand(16, 2),
    }
    flat_lora, ranks, alphas, _ = convert_krea2_lora_to_flax(state_dict, "x", weights_dtype=jnp.float32)
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, "x", scale=0.8)

    interleaved = _tiny_model()
    rotate_half = _tiny_model(rope_layout="rotate_half")
    with nn.intercept_methods(interceptor):
      params = _init_params(interleaved, inputs)
    params = insert_lora_params(params, flat_lora)
    permuted = permute_rope_weights_to_rotate_half(params, _HEADS, _KV_HEADS, _HEAD_DIM)

    q_lora_old = params["blocks_0"]["attn"]["to_q"]["lora-x"]
    q_lora_new = permuted["blocks_0"]["attn"]["to_q"]["lora-x"]
    self.assertIs(q_lora_new["down"], q_lora_old["down"])
    self.assertFalse(np.array_equal(np.asarray(q_lora_new["up"]["kernel"]), np.asarray(q_lora_old["up"]["kernel"])))
    self.assertIs(permuted["blocks_1"]["attn"]["to_v"], params["blocks_1"]["attn"]["to_v"])

    with nn.intercept_methods(interceptor):
      expected = interleaved.apply({"params": params}, *inputs).sample
      actual = rotate_half.apply({"params": permuted}, *inputs).sample
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

    # The LoRA contributes: dropping the q/k up permutation breaks equivalence.
    unpermuted_lora = flax.core.unfreeze(flax.core.freeze(permuted))
    unpermuted_lora["blocks_0"]["attn"]["to_q"]["lora-x"] = params["blocks_0"]["attn"]["to_q"]["lora-x"]
    with nn.intercept_methods(interceptor):
      wrong = rotate_half.apply({"params": unpermuted_lora}, *inputs).sample
    self.assertTrue(np.any(np.abs(np.asarray(wrong) - np.asarray(expected)) > 1e-4))


if __name__ == "__main__":
  unittest.main()
