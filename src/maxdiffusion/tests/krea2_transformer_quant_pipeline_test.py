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

# CPU tests for the Krea 2 transformer W8A8 path through the pipeline and the
# generate_krea2 load flow: staged vs monolithic steps, placement of the
# quantized host tree, LoRA on quantized projections and the model builder
# (float32 activations throughout).

import re
import types
import unittest
from contextlib import ExitStack

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
from jax.sharding import PartitionSpec as P

from maxdiffusion.generate_krea2 import (
    build_krea2_transformer,
    flash_custom_block_selection_aot_meta,
    transformer_quantization_aot_meta,
)
from maxdiffusion.kernels.krea2_attention import KREA2_BLOCK_SELECTION_REVISION
from maxdiffusion.loaders.krea2_lora_pipeline import Krea2LoraLoaderMixin, insert_lora_params, make_lora_compile_spec
from maxdiffusion.models.krea2.lora_util import convert_krea2_lora_to_flax
from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
from maxdiffusion.models.krea2.transformer_quant import (
    KREA2_DEFAULT_QUANT_TARGETS,
    KREA2_TRANSFORMER_QUANT_REVISION,
    check_transformer_param_tree,
    quantize_transformer_params,
)
from maxdiffusion.pipelines.krea2 import krea2_pipeline
from maxdiffusion.pipelines.krea2.krea2_pipeline import (
    FlaxKrea2Pipeline,
    free_params,
    params_device_bytes,
    place_params,
)
from maxdiffusion.tests.krea2_pipeline_residency_test import _config, _inputs, _mesh, _unbox

_TARGETS = KREA2_DEFAULT_QUANT_TARGETS
_TINY_CFG = {
    "in_channels": 16,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 4,  # hidden_size = 32
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "timestep_embed_dim": 16,
    "text_hidden_dim": 24,
    "num_text_layers": 3,
    "text_num_attention_heads": 4,
    "text_num_key_value_heads": 4,
    "text_intermediate_size": 48,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": (4, 2, 2),
}
_I8_DOT = re.compile(r"dot_general.*\(tensor<[0-9x]*xi8>, tensor<[0-9x]*xi8>\) -> tensor<[0-9x]*xi32>")


def _tiny_transformer(quant_targets=_TARGETS, **overrides):
  return Krea2Transformer2DModel(**{**_TINY_CFG, **overrides}, quant_targets=quant_targets)


def _model_args():
  """Positional `Krea2Transformer2DModel.__call__` arguments for `_inputs()`."""
  latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = _inputs()
  return latents, prompt_embeds, t_vec, img_ids, txt_ids, text_mask


def _intercepted(interceptors):
  stack = ExitStack()
  for interceptor in interceptors:
    stack.enter_context(nn.intercept_methods(interceptor))
  return stack


def _abstract_params(model, interceptors=()):
  with _intercepted(interceptors):
    return jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), *_model_args()))["params"]


def _random_float_params(interceptors=(), seed=0):
  """Non-trivial float params of the unquantized tiny model as a numpy tree.

  Like generate_krea2, the float structure is evaluated under the LoRA
  interceptors, so it includes the (random) `lora-*` leaves.
  """
  params = _unbox(_abstract_params(_tiny_transformer(()), interceptors))
  leaves, treedef = jax.tree_util.tree_flatten(params)
  rng = np.random.RandomState(seed)
  new_leaves = [(0.3 * rng.randn(*leaf.shape)).astype(leaf.dtype) for leaf in leaves]
  return flax.core.unfreeze(jax.tree_util.tree_unflatten(treedef, new_leaves))


def _pipeline(transformer, lora_compile_spec=(), **config_overrides):
  return FlaxKrea2Pipeline(
      transformer=transformer,
      vae=None,
      vae_cache=None,
      text_encoder=None,
      tokenizer=None,
      scheduler=None,
      config=_config(**config_overrides),
      mesh=_mesh(),
      lora_compile_spec=lora_compile_spec,
  )


def _step_inputs(pipeline, params):
  latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = _inputs()
  text_params = {key: params[key] for key in krea2_pipeline.KREA2_TEXT_CONTEXT_KEYS}
  text_hidden = pipeline._jitted_transformer_text_context(text_params, prompt_embeds, text_mask)
  return latents, text_hidden, text_mask, img_ids, txt_ids, t_vec


class QuantizedPipelineStepTest(unittest.TestCase):

  def setUp(self):
    self.transformer = _tiny_transformer()
    self.params = jax.tree_util.tree_map(
        jnp.asarray, quantize_transformer_params(_random_float_params(), _TARGETS, np.float32)
    )

  def test_staged_and_monolithic_steps_match_direct_apply(self):
    expected = np.asarray(self.transformer.apply({"params": self.params}, *_model_args()).sample)
    for staged, donate in ((True, True), (True, False), (False, True)):
      with self.subTest(staged=staged, donate=donate):
        pipeline = _pipeline(
            self.transformer, krea2_staged_transformer=staged, krea2_staged_donate_hidden_states=donate
        )
        pipeline._setup_jit_functions()
        actual = pipeline._jitted_transformer_step(self.params, *_step_inputs(pipeline, self.params))
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
    # Donation only touches internal residual buffers; the params survive.
    for leaf in jax.tree_util.tree_leaves(self.params):
      self.assertFalse(leaf.is_deleted())

  def test_staged_block_runs_int8_matmuls(self):
    pipeline = _pipeline(self.transformer, krea2_staged_donate_hidden_states=False)
    pipeline._setup_jit_functions()
    prelude_params = {key: self.params[key] for key in krea2_pipeline.KREA2_PRELUDE_KEYS}
    hidden_states, _, temb_mod, rotary_emb, attention_mask = pipeline._jitted_transformer_prelude(
        prelude_params, *_step_inputs(pipeline, self.params)
    )
    hlo = pipeline._jitted_transformer_block.jitted.lower(
        self.params["blocks_0"], {}, hidden_states, temb_mod, rotary_emb, attention_mask
    ).as_text()
    self.assertEqual(len(_I8_DOT.findall(hlo)), len(_TARGETS))


class QuantizedParamResidencyTest(unittest.TestCase):
  """generate_krea2's placement of a quantized host tree, and offload's place/free cycle."""

  _RULES = (("embed", None), ("mlp", "data"), ("heads", "data"))

  def test_place_forward_free_quantized_host_tree(self):
    transformer = _tiny_transformer()
    # Same derivation as generate_krea2: runtime (quantized) abstract tree -> logical specs -> mesh shardings.
    mesh = _mesh()
    with mesh, nn_partitioning.axis_rules(self._RULES):
      abstract = _abstract_params(transformer)
      shardings = flax.core.freeze(nn.logical_to_mesh_sharding(nn.get_partition_spec(abstract), mesh, self._RULES))
    host_q = quantize_transformer_params(_random_float_params(), _TARGETS, np.float32)
    check_transformer_param_tree(host_q, abstract)
    host = flax.core.freeze(host_q)
    self.assertEqual(jax.tree_util.tree_structure(host), jax.tree_util.tree_structure(shardings))
    self.assertEqual(shardings["blocks_0"]["attn"]["to_q"]["kernel"].spec, P(None, "data"))
    self.assertEqual(shardings["blocks_0"]["attn"]["to_q"]["kernel_scale"].spec, P())
    host_kernel = host["blocks_0"]["ff"]["down_proj"]["kernel"]
    self.assertIsInstance(host_kernel, np.ndarray)
    self.assertEqual(host_kernel.dtype, np.int8)

    expected = np.asarray(transformer.apply({"params": host}, *_model_args()).sample)
    fwd = jax.jit(lambda p: transformer.apply({"params": p}, *_model_args()).sample)
    host_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(host))
    for _ in range(2):  # a second cycle mirrors the next generation's `_swap_in`
      placed = place_params(host, shardings)
      leaves = jax.tree_util.tree_leaves(placed)
      self.assertEqual(len(leaves), len(jax.tree_util.tree_leaves(host)))
      for leaf in leaves:
        self.assertIsInstance(leaf, jax.Array)
        self.assertFalse(leaf.is_deleted())
      self.assertEqual(placed["blocks_0"]["ff"]["down_proj"]["kernel"].dtype, jnp.int8)
      self.assertEqual(params_device_bytes(placed), host_bytes)
      np.testing.assert_allclose(np.asarray(fwd(placed)), expected, rtol=1e-5, atol=1e-5)

      self.assertIsNone(free_params(placed, host))
      for leaf in leaves:
        self.assertTrue(leaf.is_deleted())
      # The host tree survives for the next placement.
      self.assertIs(host["blocks_0"]["ff"]["down_proj"]["kernel"], host_kernel)


class QuantizedLoraTest(unittest.TestCase):
  """LoRA on quantized block projections, loaded in generate_krea2's order."""

  def _adapter(self, name, scale, state_dict):
    flat_lora, ranks, alphas, diffs = convert_krea2_lora_to_flax(state_dict, name, weights_dtype=jnp.float32)
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, name, scale)
    return flat_lora, interceptor, make_lora_compile_spec(name, scale, ranks, alphas, diffs)

  def _load(self, adapters):
    """Float init under the interceptors -> insert LoRA -> quantize -> check vs the runtime tree."""
    interceptors = [interceptor for _, interceptor, _ in adapters]
    params = _random_float_params(interceptors)
    for flat_lora, _, _ in adapters:
      params = insert_lora_params(params, flat_lora)
    params = quantize_transformer_params(params, _TARGETS, np.float32)
    abstract = _abstract_params(_tiny_transformer(), interceptors)
    check_transformer_param_tree(params, abstract)
    return params, abstract

  def _style_state(self, rng):
    return {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": rng.randn(2, 32).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": rng.randn(32, 2).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(2.0),
    }

  def test_eval_shape_tree_has_lora_under_quantized_projection(self):
    params, abstract = self._load([self._adapter("style", 1.0, self._style_state(np.random.RandomState(3)))])
    flat_abstract = flatten_dict(flax.core.unfreeze(_unbox(abstract)))
    prefix = ("blocks_0", "attn", "to_q")
    self.assertEqual(flat_abstract[prefix + ("kernel",)].dtype, jnp.int8)
    self.assertIn(prefix + ("kernel_scale",), flat_abstract)
    self.assertEqual(flat_abstract[prefix + ("lora-style", "down", "kernel")].shape, (32, 2))
    self.assertEqual(flat_abstract[prefix + ("lora-style", "up", "kernel")].shape, (2, 32))
    proj = params["blocks_0"]["attn"]["to_q"]
    self.assertEqual(set(proj), {"kernel", "kernel_scale", "lora-style"})
    self.assertEqual(proj["kernel"].dtype, np.int8)

  def test_lora_changes_quantized_output_and_zero_scale_is_inert(self):
    state = self._style_state(np.random.RandomState(3))
    adapter = self._adapter("style", 1.0, state)
    params, _ = self._load([adapter])
    model = _tiny_transformer()

    base_out = np.asarray(model.apply({"params": params}, *_model_args()).sample)
    with _intercepted([adapter[1]]):
      lora_out = np.asarray(model.apply({"params": params}, *_model_args()).sample)
    self.assertTrue(np.any(np.abs(lora_out - base_out) > 1e-6))

    _, zero_interceptor, _ = self._adapter("style", 0.0, state)
    with _intercepted([zero_interceptor]):
      zero_out = np.asarray(model.apply({"params": params}, *_model_args()).sample)
    np.testing.assert_allclose(zero_out, base_out, rtol=1e-6, atol=1e-6)

  def test_staged_explicit_lora_matches_interceptor(self):
    rng = np.random.RandomState(7)
    # LoRA on quantized (to_q, to_out, down_proj) and float (to_k) projections.
    style_state = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": rng.randn(2, 32).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": rng.randn(32, 2).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
        "lora_unet_transformer_blocks_1_ff_down.lora_down.weight": rng.randn(3, 64).astype(np.float32),
        "lora_unet_transformer_blocks_1_ff_down.lora_up.weight": rng.randn(32, 3).astype(np.float32),
    }
    detail_state = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": rng.randn(1, 32).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": rng.randn(32, 1).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight": rng.randn(2, 32).astype(np.float32),
        "lora_unet_transformer_blocks_0_attn_to_k.lora_up.weight": rng.randn(16, 2).astype(np.float32),
        "lora_unet_transformer_blocks_1_attn_to_out.lora_down.weight": rng.randn(2, 32).astype(np.float32),
        "lora_unet_transformer_blocks_1_attn_to_out.lora_up.weight": rng.randn(32, 2).astype(np.float32),
        "lora_unet_transformer_blocks_1_attn_to_out.alpha": np.float32(2.0),
    }
    adapters = [self._adapter("style", 0.7, style_state), self._adapter("detail", 0.25, detail_state)]
    params, _ = self._load(adapters)
    self.assertEqual(params["blocks_0"]["attn"]["to_k"]["kernel"].dtype, np.float32)
    self.assertIn("lora-detail", params["blocks_0"]["attn"]["to_k"])
    params = jax.tree_util.tree_map(jnp.asarray, params)
    interceptors = [interceptor for _, interceptor, _ in adapters]
    lora_compile_spec = tuple(spec for _, _, spec in adapters)
    model = _tiny_transformer()

    with _intercepted(interceptors):
      expected = np.asarray(model.apply({"params": params}, *_model_args()).sample)
    base = np.asarray(model.apply({"params": params}, *_model_args()).sample)
    self.assertTrue(np.any(np.abs(expected - base) > 1e-6))

    # generate_krea2 runs every pipeline call under the interceptors; the staged
    # block (module paths without `blocks_i`) gets its LoRA explicitly instead.
    for staged in (True, False):
      with self.subTest(staged=staged):
        pipeline = _pipeline(model, lora_compile_spec=lora_compile_spec, krea2_staged_transformer=staged)
        pipeline._setup_jit_functions()
        with _intercepted(interceptors):
          actual = pipeline._jitted_transformer_step(params, *_step_inputs(pipeline, params))
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def _build_config(**overrides):
  values = {
      "attention": "dot_product",
      "flash_block_sizes": {},
      "mask_padding_tokens": True,
      "krea2_rope_layout": "interleaved",
      "activations_dtype": jnp.float32,
      "weights_dtype": jnp.float32,
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


class BuildTransformerTest(unittest.TestCase):

  def test_quant_targets_follow_the_config(self):
    yml_targets = ["to_q", "to_gate", "to_out", "gate_proj", "up_proj", "down_proj"]
    cases = (
        ({}, ()),  # configs that predate the keys
        ({"krea2_transformer_quantization": "", "krea2_transformer_quant_targets": yml_targets}, ()),
        ({"krea2_transformer_quantization": "w8a8", "krea2_transformer_quant_targets": yml_targets}, _TARGETS),
        ({"krea2_transformer_quantization": "w8a8"}, _TARGETS),
        (
            {"krea2_transformer_quantization": "w8a8", "krea2_transformer_quant_targets": "up_proj,to_k"},
            ("to_k", "up_proj"),
        ),
    )
    for overrides, expected in cases:
      with self.subTest(overrides=overrides):
        transformer = build_krea2_transformer(_TINY_CFG, _build_config(**overrides), None)
        self.assertEqual(transformer.quant_targets, expected)
    with self.assertRaisesRegex(ValueError, "int4"):
      build_krea2_transformer(_TINY_CFG, _build_config(krea2_transformer_quantization="int4"), None)

  def test_quant_targets_override(self):
    config = _build_config(krea2_transformer_quantization="w8a8")
    # generate_krea2's float load model.
    load_model = build_krea2_transformer(_TINY_CFG, config, None, quant_targets=())
    self.assertEqual(load_model.quant_targets, ())
    self.assertEqual(
        build_krea2_transformer(_TINY_CFG, config, None, quant_targets=["up_proj", "to_q"]).quant_targets,
        ("to_q", "up_proj"),
    )

    runtime_model = build_krea2_transformer(_TINY_CFG, config, None)
    runtime = flatten_dict(flax.core.unfreeze(_unbox(_abstract_params(runtime_model))))
    load = flatten_dict(flax.core.unfreeze(_unbox(_abstract_params(load_model))))
    self.assertEqual(runtime[("blocks_1", "ff", "up_proj", "kernel")].dtype, jnp.int8)
    self.assertEqual(load[("blocks_1", "ff", "up_proj", "kernel")].dtype, jnp.float32)
    self.assertEqual(set(runtime) - set(load), {path for path in runtime if path[-1] == "kernel_scale"})


class TransformerQuantizationAotMetaTest(unittest.TestCase):

  def test_off_adds_no_key(self):
    # Unquantized setups keep the AOT cache fingerprint they had before the key existed.
    self.assertEqual(transformer_quantization_aot_meta("", ()), {})
    self.assertEqual(transformer_quantization_aot_meta("", ("to_q",)), {})

  def test_on_records_mode_targets_and_revision(self):
    revision = f"r{KREA2_TRANSFORMER_QUANT_REVISION}"
    self.assertEqual(
        transformer_quantization_aot_meta("w8a8", _TARGETS),
        {"krea2_transformer_quantization": f"w8a8:to_q,to_gate,to_out,gate_proj,up_proj,down_proj:{revision}"},
    )
    self.assertEqual(
        transformer_quantization_aot_meta("w8a8", ("to_k", "up_proj")),
        {"krea2_transformer_quantization": f"w8a8:to_k,up_proj:{revision}"},
    )


class FlashCustomBlockSelectionAotMetaTest(unittest.TestCase):

  def test_other_kernels_add_no_key(self):
    # Other attention kernels keep the AOT cache fingerprint they had before the key existed.
    for attention in ("flash", "dot_product", "cudnn_flash_te", ""):
      self.assertEqual(flash_custom_block_selection_aot_meta(attention), {})

  def test_flash_custom_records_revision(self):
    self.assertEqual(
        flash_custom_block_selection_aot_meta("flash_custom"),
        {"krea2_block_selection": f"r{KREA2_BLOCK_SELECTION_REVISION}"},
    )


if __name__ == "__main__":
  unittest.main()
