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

# CPU-runnable unit tests for the Krea 2 (K2) Flax transformer and utilities.

import math
import unittest

import flax
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.krea2.transformer_krea2_flax import (
    Krea2Attention,
    Krea2RMSNorm,
    Krea2TextFusion,
    Krea2TimestepEmbedding,
    Krea2Transformer2DModel,
)
from maxdiffusion.models.krea2.util import (
    calculate_krea2_shift,
    load_and_convert_krea2_weights,
    prepare_krea2_image_ids,
    prepare_krea2_text_ids,
    round_up_to_multiple,
)


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
  kwargs.update(overrides)
  return Krea2Transformer2DModel(**kwargs)


class Krea2RMSNormTest(unittest.TestCase):

  def test_zero_weight_is_plain_rmsnorm(self):
    """The checkpoint stores zero-centered scales: weight=0 => multiplier of 1."""
    norm = Krea2RMSNorm(dim=8, eps=1e-5)
    x = jnp.array(np.random.RandomState(0).randn(2, 3, 8), dtype=jnp.float32)
    params = norm.init(jax.random.PRNGKey(0), x)["params"]
    self.assertTrue(np.allclose(np.asarray(params["weight"]), 0.0))
    self.assertEqual(params["weight"].dtype, jnp.float32)

    out = norm.apply({"params": params}, x)
    expected = x / np.sqrt(np.mean(np.square(np.asarray(x)), axis=-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

  def test_weight_offsets_scale_by_one(self):
    norm = Krea2RMSNorm(dim=4, eps=1e-5)
    x = jnp.ones((1, 4))
    params = {"weight": jnp.full((4,), 0.5, dtype=jnp.float32)}
    out = norm.apply({"params": params}, x)
    # rms of ones is 1 (up to eps), so output ~= 1 + weight
    np.testing.assert_allclose(np.asarray(out), 1.5, rtol=1e-4)


class Krea2TimestepEmbeddingTest(unittest.TestCase):

  def test_cos_first_sinusoid(self):
    embed = Krea2TimestepEmbedding(embed_dim=8, hidden_size=8)
    t = jnp.array([0.5])
    params = _unbox(embed.init(jax.random.PRNGKey(0), t)["params"])

    half = 4
    freqs = np.exp(-math.log(1e4) * np.arange(half) / half)
    args = (0.5 * 1e3) * freqs
    expected_emb = np.concatenate([np.cos(args), np.sin(args)])[None, None, :]

    # Identity-like check: probe through linear layers set to identity.
    params = flax.core.unfreeze(params)
    params["linear_1"]["kernel"] = jnp.eye(8)
    params["linear_1"]["bias"] = jnp.zeros((8,))
    params["linear_2"]["kernel"] = jnp.eye(8)
    params["linear_2"]["bias"] = jnp.zeros((8,))
    out = embed.apply({"params": params}, t)
    self.assertEqual(out.shape, (1, 1, 8))
    expected = jax.nn.gelu(jnp.array(expected_emb), approximate=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-4, atol=1e-5)


class Krea2AttentionTest(unittest.TestCase):

  def test_gqa_shapes_and_gate(self):
    attn = Krea2Attention(dim=16, num_heads=4, num_kv_heads=2, head_dim=4, use_rope=False)
    x = jnp.array(np.random.RandomState(0).randn(2, 5, 16), dtype=jnp.float32)
    params = _unbox(attn.init(jax.random.PRNGKey(0), x)["params"])

    # GQA: kv projections have half the output width of q.
    self.assertEqual(params["to_q"]["kernel"].shape, (16, 16))
    self.assertEqual(params["to_k"]["kernel"].shape, (16, 8))
    self.assertEqual(params["to_v"]["kernel"].shape, (16, 8))
    self.assertEqual(params["to_gate"]["kernel"].shape, (16, 16))

    out = attn.apply({"params": params}, x)
    self.assertEqual(out.shape, (2, 5, 16))

    # The sigmoid gate must modulate the attention output: zeroing the gate
    # projection forces sigmoid(0)=0.5 gating, which changes the output.
    params_no_gate = flax.core.unfreeze(flax.core.freeze(params))
    params_no_gate["to_gate"]["kernel"] = jnp.zeros_like(params_no_gate["to_gate"]["kernel"])
    out_no_gate = attn.apply({"params": params_no_gate}, x)
    self.assertTrue(np.any(np.abs(np.asarray(out) - np.asarray(out_no_gate)) > 1e-6))

  def test_key_padding_mask_excludes_padded_keys(self):
    attn = Krea2Attention(dim=8, num_heads=2, num_kv_heads=2, head_dim=4, use_rope=False)
    rng = np.random.RandomState(0)
    x = jnp.array(rng.randn(1, 6, 8), dtype=jnp.float32)
    params = _unbox(attn.init(jax.random.PRNGKey(0), x)["params"])

    mask = jnp.array([[1, 1, 1, 1, 0, 0]], dtype=jnp.bool_)
    out1 = attn.apply({"params": params}, x, mask)
    # Perturb the padded (masked-out) tokens; valid-token outputs must not change.
    x2 = np.asarray(x).copy()
    x2[:, 4:] += 10.0
    out2 = attn.apply({"params": params}, jnp.array(x2), mask)
    np.testing.assert_allclose(np.asarray(out1[:, :4]), np.asarray(out2[:, :4]), rtol=1e-5, atol=1e-5)


class Krea2TextFusionTest(unittest.TestCase):

  def test_output_shape_and_mask_invariance(self):
    fusion = Krea2TextFusion(
        num_text_layers=3,
        dim=16,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=32,
        num_layerwise_blocks=1,
        num_refiner_blocks=1,
    )
    rng = np.random.RandomState(0)
    x = jnp.array(rng.randn(2, 6, 3, 16), dtype=jnp.float32)
    mask = jnp.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=jnp.bool_)
    params = _unbox(fusion.init(jax.random.PRNGKey(0), x, mask)["params"])

    out1 = fusion.apply({"params": params}, x, mask)
    self.assertEqual(out1.shape, (2, 6, 16))

    # Valid token outputs must be invariant to the content of padded tokens.
    x2 = np.asarray(x).copy()
    x2[0, 4:] += 5.0
    out2 = fusion.apply({"params": params}, jnp.array(x2), mask)
    np.testing.assert_allclose(np.asarray(out1[0, :4]), np.asarray(out2[0, :4]), rtol=1e-5, atol=1e-5)


class Krea2TransformerModelTest(unittest.TestCase):

  def test_forward_shape(self):
    model = _tiny_model()
    B, S_img, S_txt = 2, 12, 7
    hs = jnp.array(np.random.RandomState(0).randn(B, S_img, 16), dtype=jnp.float32)
    ehs = jnp.array(np.random.RandomState(1).randn(B, S_txt, 3, 24), dtype=jnp.float32)
    t = jnp.full((B,), 0.5)
    img_ids = prepare_krea2_image_ids(B, 3, 4)
    txt_ids = prepare_krea2_text_ids(B, S_txt)
    mask = jnp.ones((B, S_txt), dtype=jnp.bool_)

    params = model.init(jax.random.PRNGKey(0), hs, ehs, t, img_ids, txt_ids, mask)["params"]
    out = model.apply({"params": params}, hs, ehs, t, img_ids, txt_ids, mask)
    self.assertEqual(out.sample.shape, (B, S_img, 16))

  def test_output_depends_on_timestep(self):
    model = _tiny_model()
    B, S_img, S_txt = 1, 4, 3
    hs = jnp.ones((B, S_img, 16))
    ehs = jnp.ones((B, S_txt, 3, 24))
    img_ids = prepare_krea2_image_ids(B, 2, 2)
    txt_ids = prepare_krea2_text_ids(B, S_txt)
    mask = jnp.ones((B, S_txt), dtype=jnp.bool_)
    params = model.init(jax.random.PRNGKey(0), hs, ehs, jnp.full((B,), 0.5), img_ids, txt_ids, mask)["params"]
    out1 = model.apply({"params": params}, hs, ehs, jnp.full((B,), 1.0), img_ids, txt_ids, mask).sample
    out2 = model.apply({"params": params}, hs, ehs, jnp.full((B,), 0.1), img_ids, txt_ids, mask).sample
    self.assertTrue(np.any(np.abs(np.asarray(out1) - np.asarray(out2)) > 1e-6))


class Krea2WeightConversionTest(unittest.TestCase):

  def test_converter_maps_all_keys(self):
    """Round-trip: synthesize a diffusers-style state dict for a tiny config and
    verify every Flax parameter is populated with the correctly transposed value."""
    import tempfile
    from safetensors.numpy import save_file

    model = _tiny_model()
    B, S_img, S_txt = 1, 4, 3
    hs = jnp.ones((B, S_img, 16))
    ehs = jnp.ones((B, S_txt, 3, 24))
    img_ids = prepare_krea2_image_ids(B, 2, 2)
    txt_ids = prepare_krea2_text_ids(B, S_txt)
    mask = jnp.ones((B, S_txt), dtype=jnp.bool_)
    params = _unbox(model.init(jax.random.PRNGKey(0), hs, ehs, jnp.full((B,), 0.5), img_ids, txt_ids, mask)["params"])
    params = flax.core.unfreeze(params)

    # Build the diffusers-format state dict from the flax tree.
    rng = np.random.RandomState(0)
    pt_state = {}

    def fake(shape):
      return rng.randn(*shape).astype(np.float32)

    def add_linear(pt_key, flax_leaf, bias_key=None, bias_leaf=None):
      in_dim, out_dim = flax_leaf.shape
      pt_state[pt_key] = fake((out_dim, in_dim))
      if bias_key is not None:
        pt_state[bias_key] = fake(bias_leaf.shape)

    def add_attention(pt_prefix, flax_attn):
      add_linear(pt_prefix + "to_q.weight", flax_attn["to_q"]["kernel"])
      add_linear(pt_prefix + "to_k.weight", flax_attn["to_k"]["kernel"])
      add_linear(pt_prefix + "to_v.weight", flax_attn["to_v"]["kernel"])
      add_linear(pt_prefix + "to_gate.weight", flax_attn["to_gate"]["kernel"])
      add_linear(pt_prefix + "to_out.0.weight", flax_attn["to_out"]["kernel"])
      pt_state[pt_prefix + "norm_q.weight"] = fake(flax_attn["norm_q"]["weight"].shape)
      pt_state[pt_prefix + "norm_k.weight"] = fake(flax_attn["norm_k"]["weight"].shape)

    def add_swiglu(pt_prefix, flax_ff):
      add_linear(pt_prefix + "gate.weight", flax_ff["gate_proj"]["kernel"])
      add_linear(pt_prefix + "up.weight", flax_ff["up_proj"]["kernel"])
      add_linear(pt_prefix + "down.weight", flax_ff["down_proj"]["kernel"])

    def add_fusion_block(pt_prefix, flax_block):
      pt_state[pt_prefix + "norm1.weight"] = fake(flax_block["norm1"]["weight"].shape)
      pt_state[pt_prefix + "norm2.weight"] = fake(flax_block["norm2"]["weight"].shape)
      add_attention(pt_prefix + "attn.", flax_block["attn"])
      add_swiglu(pt_prefix + "ff.", flax_block["ff"])

    add_linear("img_in.weight", params["img_in"]["kernel"], "img_in.bias", params["img_in"]["bias"])
    for name in ("linear_1", "linear_2"):
      add_linear(
          f"time_embed.{name}.weight",
          params["time_embed"][name]["kernel"],
          f"time_embed.{name}.bias",
          params["time_embed"][name]["bias"],
      )
    add_linear(
        "time_mod_proj.weight", params["time_mod_proj"]["kernel"], "time_mod_proj.bias", params["time_mod_proj"]["bias"]
    )
    for i in range(2):
      add_fusion_block(f"text_fusion.layerwise_blocks.{i}.", params["text_fusion"][f"layerwise_blocks_{i}"])
      add_fusion_block(f"text_fusion.refiner_blocks.{i}.", params["text_fusion"][f"refiner_blocks_{i}"])
    add_linear("text_fusion.projector.weight", params["text_fusion"]["projector"]["kernel"])
    pt_state["txt_in.norm.weight"] = fake(params["txt_in"]["norm"]["weight"].shape)
    for name in ("linear_1", "linear_2"):
      add_linear(
          f"txt_in.{name}.weight",
          params["txt_in"][name]["kernel"],
          f"txt_in.{name}.bias",
          params["txt_in"][name]["bias"],
      )
    for i in range(2):
      prefix = f"transformer_blocks.{i}."
      block = params[f"blocks_{i}"]
      pt_state[prefix + "scale_shift_table"] = fake(block["scale_shift_table"].shape)
      pt_state[prefix + "norm1.weight"] = fake(block["norm1"]["weight"].shape)
      pt_state[prefix + "norm2.weight"] = fake(block["norm2"]["weight"].shape)
      add_attention(prefix + "attn.", block["attn"])
      add_swiglu(prefix + "ff.", block["ff"])
    pt_state["final_layer.scale_shift_table"] = fake(params["final_layer"]["scale_shift_table"].shape)
    pt_state["final_layer.norm.weight"] = fake(params["final_layer"]["norm"]["weight"].shape)
    add_linear(
        "final_layer.linear.weight",
        params["final_layer"]["linear"]["kernel"],
        "final_layer.linear.bias",
        params["final_layer"]["linear"]["bias"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      import os

      ckpt = os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")
      save_file(dict(pt_state), ckpt)
      converted = load_and_convert_krea2_weights(tmpdir, params, num_layers=2)

    # Spot-check transposition and verbatim loads.
    np.testing.assert_allclose(np.asarray(converted["img_in"]["kernel"]), pt_state["img_in.weight"].T)
    np.testing.assert_allclose(
        np.asarray(converted["blocks_1"]["attn"]["to_k"]["kernel"]), pt_state["transformer_blocks.1.attn.to_k.weight"].T
    )
    np.testing.assert_allclose(
        np.asarray(converted["blocks_0"]["scale_shift_table"]), pt_state["transformer_blocks.0.scale_shift_table"]
    )
    np.testing.assert_allclose(
        np.asarray(converted["text_fusion"]["projector"]["kernel"]), pt_state["text_fusion.projector.weight"].T
    )
    np.testing.assert_allclose(np.asarray(converted["final_layer"]["norm"]["weight"]), pt_state["final_layer.norm.weight"])
    # Norms and tables stay float32.
    self.assertEqual(converted["blocks_0"]["norm1"]["weight"].dtype, jnp.float32)
    self.assertEqual(converted["final_layer"]["scale_shift_table"].dtype, jnp.float32)

    # Every leaf must be a concrete array (no leftover ShapeDtypeStructs).
    for leaf in jax.tree_util.tree_leaves(converted):
      self.assertFalse(isinstance(leaf, jax.ShapeDtypeStruct))


class Krea2ShiftTest(unittest.TestCase):

  def test_endpoints(self):
    self.assertAlmostEqual(calculate_krea2_shift(256), 0.5, places=6)
    self.assertAlmostEqual(calculate_krea2_shift(6400), 1.15, places=6)
    # 1024x1024 -> (1024/16)^2 = 4096 tokens
    mu = calculate_krea2_shift(4096)
    self.assertTrue(0.5 < mu < 1.15)

  def test_round_up_to_multiple(self):
    self.assertEqual(round_up_to_multiple(1024, 16), 1024)
    self.assertEqual(round_up_to_multiple(1025, 16), 1040)
    self.assertEqual(round_up_to_multiple(1039, 16), 1040)
    self.assertEqual(round_up_to_multiple(16, 16), 16)
    self.assertEqual(round_up_to_multiple(1, 16), 16)

  def test_position_id_helpers(self):
    txt_ids = prepare_krea2_text_ids(2, 5)
    self.assertEqual(txt_ids.shape, (2, 5, 3))
    self.assertTrue(np.all(np.asarray(txt_ids) == 0))

    img_ids = prepare_krea2_image_ids(1, 2, 3)
    self.assertEqual(img_ids.shape, (1, 6, 3))
    np.testing.assert_array_equal(np.asarray(img_ids[0, :, 0]), 0)
    np.testing.assert_array_equal(np.asarray(img_ids[0, :, 1]), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(np.asarray(img_ids[0, :, 2]), [0, 1, 2, 0, 1, 2])


if __name__ == "__main__":
  unittest.main()
