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
from unittest import mock

import flax
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.kernels.krea2_qk_prep import qk_prep_reference
from maxdiffusion.models.attention_flax import AttentionOp
from maxdiffusion.models.krea2.transformer_krea2_flax import (
    Krea2Attention,
    Krea2RMSNorm,
    Krea2TextFusion,
    Krea2TimestepEmbedding,
    Krea2Transformer2DModel,
    Krea2TransformerBlock,
    krea2_rotary_tables,
)
from maxdiffusion.models.krea2.util import (
    calculate_krea2_shift,
    load_and_convert_krea2_weights,
    prepare_krea2_image_ids,
    prepare_krea2_text_ids,
    round_up_to_multiple,
)
from maxdiffusion.pipelines.krea2.krea2_pipeline import compact_text_embeddings, is_classifier_free_guidance_enabled


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

  def test_mask_padding_tokens_reaches_shared_attention_op(self):
    seen_values = []

    def fake_apply_attention(module, query, key, value, attention_mask=None):
      del key, value, attention_mask
      seen_values.append(module.mask_padding_tokens)
      return query

    attn = Krea2Attention(
        dim=8,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        use_rope=False,
        attention_kernel="flash",
        mask_padding_tokens=False,
    )
    x = jnp.ones((1, 4, 8), dtype=jnp.float32)
    with mock.patch.object(AttentionOp, "apply_attention", fake_apply_attention):
      params = attn.init(jax.random.PRNGKey(0), x)["params"]
      attn.apply({"params": params}, x)

    self.assertTrue(seen_values)
    self.assertTrue(all(value is False for value in seen_values))

  def test_flash_custom_receives_4d_unexpanded_gqa_after_rope(self):
    """flash_custom contract: (B, H, L, D) q and (B, H_kv, L, D) k/v, unscaled,
    with the int32 key mask; returns (B, L, H*D). Interleaved RoPE: q/k after
    norm and RoPE. rotate_half: RAW q/k plus the "krea2_qk_prep" context (norm
    weights, eps, half tables) the wrapper's fused prep kernel applies."""
    seen = {}
    head_dim, heads, kv_heads = 8, 4, 2

    def fake_apply_attention(module, query, key, value, attention_mask=None, extra_context=None):
      seen.update(query=query.shape, key=key.shape, value=value.shape, mask=attention_mask, extra=extra_context)
      if extra_context:
        prep = extra_context["krea2_qk_prep"]

        def prepare(x, weight):
          return qk_prep_reference(jnp.swapaxes(x, 1, 2), weight, prep["eps"], prep["cos"], prep["sin"])

        query = prepare(query, prep["q_norm_weight"])
        key = prepare(key, prep["k_norm_weight"])
      # Reference attention in the kernel's own terms: expand GQA, scale, softmax.
      repeats = query.shape[1] // key.shape[1]
      k = jnp.repeat(key, repeats, axis=1).astype(jnp.float32)
      v = jnp.repeat(value, repeats, axis=1).astype(jnp.float32)
      scores = jnp.einsum("bhqd,bhkd->bhqk", query.astype(jnp.float32), k) * module.scale
      if attention_mask is not None:
        scores = jnp.where(attention_mask[:, None, None, :] > 0, scores, -1e9)
      out = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)
      b, h, l, d = out.shape
      return jnp.transpose(out, (0, 2, 1, 3)).reshape(b, l, h * d)

    rng = np.random.RandomState(0)
    x = jnp.array(rng.randn(2, 6, 32), dtype=jnp.float32)
    ids = prepare_krea2_image_ids(1, 2, 3)[0]
    mask = jnp.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=jnp.bool_)
    for layout in ("interleaved", "rotate_half"):
      rotary = krea2_rotary_tables(ids, (4, 2, 2), 1000.0, layout)
      reference = Krea2Attention(
          dim=32, num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, rope_layout=layout
      )
      custom = Krea2Attention(
          dim=32,
          num_heads=heads,
          num_kv_heads=kv_heads,
          head_dim=head_dim,
          rope_layout=layout,
          attention_kernel="flash_custom",
          flash_min_seq_length=0,
      )
      params = _unbox(reference.init(jax.random.PRNGKey(0), x, mask, rotary)["params"])
      expected = reference.apply({"params": params}, x, mask, rotary)
      seen.clear()
      with mock.patch.object(AttentionOp, "apply_attention", fake_apply_attention):
        actual = custom.apply({"params": params}, x, mask, rotary)
      self.assertEqual(seen["query"], (2, heads, 6, head_dim))
      self.assertEqual(seen["key"], (2, kv_heads, 6, head_dim))
      self.assertEqual(seen["value"], (2, kv_heads, 6, head_dim))
      self.assertEqual(seen["mask"].dtype, jnp.int32)
      np.testing.assert_array_equal(np.asarray(seen["mask"]), np.asarray(mask).astype(np.int32))
      if layout == "rotate_half":
        prep = seen["extra"]["krea2_qk_prep"]
        self.assertEqual(sorted(prep), ["cos", "eps", "k_norm_weight", "q_norm_weight", "sin"])
        np.testing.assert_array_equal(np.asarray(prep["q_norm_weight"]), np.asarray(params["norm_q"]["weight"]))
        np.testing.assert_array_equal(np.asarray(prep["k_norm_weight"]), np.asarray(params["norm_k"]["weight"]))
        self.assertEqual(prep["cos"].shape, (6, head_dim // 2))
      else:
        self.assertIsNone(seen["extra"])
      np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


class Krea2GuidanceTest(unittest.TestCase):

  def test_explicit_switch_controls_guidance(self):
    self.assertTrue(is_classifier_free_guidance_enabled(3.5, True))
    self.assertFalse(is_classifier_free_guidance_enabled(3.5, False))
    self.assertFalse(is_classifier_free_guidance_enabled(0.0, True))

  def test_direct_pipeline_calls_keep_scale_based_default(self):
    self.assertTrue(is_classifier_free_guidance_enabled(3.5))
    self.assertFalse(is_classifier_free_guidance_enabled(0.0))


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

  def test_staged_components_match_monolithic_forward(self):
    model = _tiny_model()
    B, S_img, S_txt = 1, 4, 3
    hs = jnp.array(np.random.RandomState(2).randn(B, S_img, 16), dtype=jnp.float32)
    ehs = jnp.array(np.random.RandomState(3).randn(B, S_txt, 3, 24), dtype=jnp.float32)
    timestep = jnp.full((B,), 0.5)
    img_ids = prepare_krea2_image_ids(B, 2, 2)
    txt_ids = prepare_krea2_text_ids(B, S_txt)
    mask = jnp.ones((B, S_txt), dtype=jnp.bool_)
    params = model.init(jax.random.PRNGKey(0), hs, ehs, timestep, img_ids, txt_ids, mask)["params"]

    expected = model.apply({"params": params}, hs, ehs, timestep, img_ids, txt_ids, mask).sample
    text_keys = ("text_fusion", "txt_in")
    text_hidden = model.apply(
        {"params": {key: params[key] for key in text_keys}},
        ehs,
        mask,
        method=model.encode_text_context,
    )
    prelude_keys = ("img_in", "time_embed", "time_mod_proj")
    prelude_params = {key: params[key] for key in prelude_keys}
    hidden, temb, temb_mod, rotary_emb, attention_mask = model.apply(
        {"params": prelude_params},
        hs,
        text_hidden,
        timestep,
        img_ids,
        txt_ids,
        mask,
        method=model.prepare_inputs,
    )

    block = Krea2TransformerBlock(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_kv_heads=2,
    )
    for block_idx in range(model.num_layers):
      hidden = block.apply(
          {"params": params[f"blocks_{block_idx}"]},
          hidden,
          temb_mod=temb_mod,
          image_rotary_emb=rotary_emb,
          attention_mask=attention_mask,
      )
    actual = model.apply(
        {"params": {"final_layer": params["final_layer"]}},
        hidden,
        temb,
        S_img,
        method=model.finalize_output,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


class Krea2FlashCustomFallbackTest(unittest.TestCase):

  def test_short_sequences_use_masked_dot_product(self):
    """Below flash_min_seq_length, flash_custom must not reach the shared
    dispatcher (whose dot_product fallback ignores the mask and 4-D GQA)."""
    B, S_img, S_txt = 2, 6, 5
    rng = np.random.RandomState(6)
    hs = jnp.array(rng.randn(B, S_img, 16), dtype=jnp.float32)
    ehs = jnp.array(rng.randn(B, S_txt, 3, 24), dtype=jnp.float32)
    t = jnp.full((B,), 0.5)
    img_ids = prepare_krea2_image_ids(B, 2, 3)
    txt_ids = prepare_krea2_text_ids(B, S_txt)
    mask = jnp.array([[True, True, True, False, False], [True, False, False, False, False]])
    reference = _tiny_model(attention_kernel="dot_product")
    custom = _tiny_model(attention_kernel="flash_custom", flash_min_seq_length=512)
    params = reference.init(jax.random.PRNGKey(0), hs, ehs, t, img_ids, txt_ids, mask)["params"]
    expected = reference.apply({"params": params}, hs, ehs, t, img_ids, txt_ids, mask).sample

    def fail(*args, **kwargs):
      raise AssertionError("flash_custom attention op called below flash_min_seq_length")

    with mock.patch.object(AttentionOp, "apply_attention", fail):
      actual = custom.apply({"params": params}, hs, ehs, t, img_ids, txt_ids, mask).sample
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


class Krea2SequenceLayoutTest(unittest.TestCase):

  def _setup(self, mask_row=(True, True, False, False, True, False)):
    model = _tiny_model()
    B, S_txt = 2, len(mask_row)
    rng = np.random.RandomState(4)
    hs = jnp.array(rng.randn(B, 6, 16), dtype=jnp.float32)
    ehs = jnp.array(rng.randn(B, S_txt, 3, 24), dtype=jnp.float32)
    timestep = jnp.full((B,), 0.3)
    img_ids = prepare_krea2_image_ids(B, 2, 3)
    mask = jnp.array([mask_row, (True,) * 3 + (False,) * (S_txt - 3)])
    params = model.init(jax.random.PRNGKey(0), hs, ehs, timestep, img_ids, prepare_krea2_text_ids(B, S_txt), mask)[
        "params"
    ]
    params = _unbox(params)
    # Non-zero modulation tables / norms so every path matters.
    leaves, treedef = jax.tree_util.tree_flatten(params)
    params = jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(0.3 * rng.randn(*x.shape), dtype=x.dtype) for x in leaves]
    )
    return model, params, hs, ehs, timestep, img_ids, mask

  def test_image_text_order_matches_reference_text_image_order(self):
    """[image | text] must equal the reference [text, image] sequence order."""
    model, params, hs, ehs, timestep, img_ids, mask = self._setup()
    S_img, S_txt = hs.shape[1], ehs.shape[1]
    txt_ids = prepare_krea2_text_ids(hs.shape[0], S_txt)
    actual = model.apply({"params": params}, hs, ehs, timestep, img_ids, txt_ids, mask).sample

    text_hidden = model.apply({"params": params}, ehs, mask, method=model.encode_text_context)
    hidden, temb, temb_mod, (cos, sin), attn_mask = model.apply(
        {"params": params}, hs, text_hidden, timestep, img_ids, txt_ids, mask, method=model.prepare_inputs
    )
    np.testing.assert_array_equal(np.asarray(attn_mask[:, :S_img]), True)
    np.testing.assert_array_equal(np.asarray(attn_mask[:, S_img:]), np.asarray(mask))
    # Rebuild the reference [text, image] order and drop the leading text tokens.
    roll = lambda x, axis: jnp.concatenate(  # noqa: E731
        [jax.lax.slice_in_dim(x, S_img, S_img + S_txt, axis=axis), jax.lax.slice_in_dim(x, 0, S_img, axis=axis)],
        axis=axis,
    )
    hidden, rotary, attn_mask = roll(hidden, 1), (roll(cos, 0), roll(sin, 0)), roll(attn_mask, 1)
    block = Krea2TransformerBlock(hidden_size=32, intermediate_size=64, num_heads=4, num_kv_heads=2)
    for block_idx in range(model.num_layers):
      hidden = block.apply(
          {"params": params[f"blocks_{block_idx}"]},
          hidden,
          temb_mod=temb_mod,
          image_rotary_emb=rotary,
          attention_mask=attn_mask,
      )
    expected = model.apply(
        {"params": params}, hidden[:, S_txt:], temb, S_img, method=model.finalize_output
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_compact_text_embeddings(self):
    rng = np.random.RandomState(0)
    embeds = jnp.array(rng.randn(2, 8, 2, 3), dtype=jnp.float32)
    mask = jnp.array(
        [[1, 1, 0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 1]], dtype=jnp.bool_
    )
    out, out_mask = compact_text_embeddings(embeds, mask, 2)
    # max valid count 4 -> bucket 4; valid tokens first, stable order.
    self.assertEqual(out.shape, (2, 4, 2, 3))
    np.testing.assert_array_equal(np.asarray(out_mask), [[1, 1, 1, 1], [1, 1, 0, 0]])
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(embeds)[0, [0, 1, 5, 6]])
    np.testing.assert_array_equal(np.asarray(out[1, :2]), np.asarray(embeds)[1, [0, 7]])
    # Bucket rounds up and is clipped to the input length.
    self.assertEqual(compact_text_embeddings(embeds, mask, 3)[0].shape[1], 6)
    self.assertEqual(compact_text_embeddings(embeds, mask, 128)[0].shape[1], 8)
    with self.assertRaises(ValueError):
      compact_text_embeddings(embeds, mask, 0)

  def test_compacted_text_gives_identical_output(self):
    model, params, hs, ehs, timestep, img_ids, mask = self._setup()
    B = hs.shape[0]
    expected = model.apply(
        {"params": params}, hs, ehs, timestep, img_ids, prepare_krea2_text_ids(B, ehs.shape[1]), mask
    ).sample
    compact_ehs, compact_mask = compact_text_embeddings(ehs, mask, 2)
    self.assertEqual(compact_ehs.shape[1], 4)
    actual = model.apply(
        {"params": params},
        hs,
        compact_ehs,
        timestep,
        img_ids,
        prepare_krea2_text_ids(B, compact_ehs.shape[1]),
        compact_mask,
    ).sample
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


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
