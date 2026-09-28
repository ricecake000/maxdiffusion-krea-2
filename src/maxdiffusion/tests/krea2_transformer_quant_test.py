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

# CPU tests for the Krea 2 transformer int8 W8A8 path: config resolution, the
# activation/kernel quantizers, `Krea2QuantDense`, host-side param
# quantization and the quantized tiny model (float32 activations throughout;
# bfloat16 appears only in numpy code or in eval_shape).

import re
import types
import unittest
from unittest import mock

import flax
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import pyconfig
from maxdiffusion.models.attention_flax import AttentionOp
from maxdiffusion.models.krea2 import transformer_krea2_flax
from maxdiffusion.models.krea2.transformer_krea2_flax import (
    Krea2Attention,
    Krea2Transformer2DModel,
    Krea2TransformerBlock,
    krea2_rotary_tables,
)
from maxdiffusion.models.krea2.transformer_quant import (
    KREA2_DEFAULT_QUANT_TARGETS,
    KREA2_QUANT_TARGETS,
    Krea2QuantDense,
    check_transformer_param_tree,
    describe_transformer_quantization,
    normalize_quant_targets,
    quantize_activation,
    quantize_kernel,
    quantize_transformer_params,
    resolve_transformer_quantization,
)
from maxdiffusion.models.krea2.util import (
    permute_rope_weights_to_rotate_half,
    prepare_krea2_image_ids,
    prepare_krea2_text_ids,
)

_HEAD_DIM, _HEADS, _KV_HEADS = 8, 4, 2
_HIDDEN = _HEAD_DIM * _HEADS
_ATTN_TARGETS = ("to_q", "to_k", "to_v", "to_gate", "to_out")
_FF_TARGETS = ("gate_proj", "up_proj", "down_proj")


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
      axes_dims_rope=(4, 2, 2),
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
  mask = jnp.asarray([[True, True, False, False, True]] * batch_size)
  return hs, ehs, t, img_ids, txt_ids, mask


def _float_params(inputs, seed=0):
  """Randomized (non-trivial) float params of the unquantized tiny model, as a numpy tree."""
  params = _unbox(_tiny_model().init(jax.random.PRNGKey(0), *inputs)["params"])
  leaves, treedef = jax.tree_util.tree_flatten(params)
  rng = np.random.RandomState(seed)
  new_leaves = [(0.3 * rng.randn(*leaf.shape)).astype(leaf.dtype) for leaf in leaves]
  return flax.core.unfreeze(jax.tree_util.tree_unflatten(treedef, new_leaves))


def _flat(tree):
  return flax.traverse_util.flatten_dict(flax.core.unfreeze(tree))


def _rel_l2(actual, expected):
  actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
  return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


class ConfigResolutionTest(unittest.TestCase):

  def test_normalize_quant_targets(self):
    self.assertEqual(normalize_quant_targets(None), ())
    self.assertEqual(normalize_quant_targets([]), ())
    self.assertEqual(normalize_quant_targets(""), ())
    self.assertEqual(normalize_quant_targets("''"), ())
    self.assertEqual(normalize_quant_targets("[]"), ())
    # Canonical order, de-duplicated.
    self.assertEqual(normalize_quant_targets(["down_proj", "to_q", "to_q"]), ("to_q", "down_proj"))
    self.assertEqual(normalize_quant_targets(("gate_proj", "to_out")), ("to_out", "gate_proj"))
    # Command-line / yaml string forms.
    self.assertEqual(normalize_quant_targets("to_q,gate_proj"), ("to_q", "gate_proj"))
    self.assertEqual(normalize_quant_targets(" gate_proj , to_q "), ("to_q", "gate_proj"))
    self.assertEqual(normalize_quant_targets("['to_gate', 'to_q']"), ("to_q", "to_gate"))
    self.assertEqual(normalize_quant_targets(list(reversed(KREA2_QUANT_TARGETS))), KREA2_QUANT_TARGETS)
    with self.assertRaisesRegex(ValueError, "to_qq"):
      normalize_quant_targets(["to_q", "to_qq"])
    with self.assertRaisesRegex(ValueError, "img_in"):
      normalize_quant_targets("img_in")

  def test_resolve_transformer_quantization(self):
    cfg = types.SimpleNamespace
    self.assertEqual(resolve_transformer_quantization(cfg()), ("", ()))
    self.assertEqual(resolve_transformer_quantization(cfg(krea2_transformer_quantization=None)), ("", ()))
    for off in ("", "''", '""', "none", "bf16", "bfloat16", " None "):
      self.assertEqual(
          resolve_transformer_quantization(cfg(krea2_transformer_quantization=off, krea2_transformer_quant_targets="to_q")),
          ("", ()),
      )
    # Default targets when the key is missing or None.
    self.assertEqual(
        resolve_transformer_quantization(cfg(krea2_transformer_quantization="w8a8")), ("w8a8", KREA2_DEFAULT_QUANT_TARGETS)
    )
    self.assertEqual(
        resolve_transformer_quantization(cfg(krea2_transformer_quantization="'W8A8'", krea2_transformer_quant_targets=None)),
        ("w8a8", KREA2_DEFAULT_QUANT_TARGETS),
    )
    self.assertEqual(
        resolve_transformer_quantization(
            cfg(krea2_transformer_quantization="w8a8", krea2_transformer_quant_targets="up_proj,to_k,up_proj")
        ),
        ("w8a8", ("to_k", "up_proj")),
    )
    with self.assertRaisesRegex(ValueError, "int4"):
      resolve_transformer_quantization(cfg(krea2_transformer_quantization="int4"))
    for empty in ("", "''", [], "[]"):
      with self.assertRaises(ValueError):
        resolve_transformer_quantization(cfg(krea2_transformer_quantization="w8a8", krea2_transformer_quant_targets=empty))
    with self.assertRaisesRegex(ValueError, "bogus"):
      resolve_transformer_quantization(cfg(krea2_transformer_quantization="w8a8", krea2_transformer_quant_targets="bogus"))

  def test_documented_command_line_form(self):
    # pyconfig parses a command-line override of a list-valued yml key with this function.
    raw = pyconfig.string_to_list("['gate_proj','up_proj','down_proj']")
    self.assertEqual(
        resolve_transformer_quantization(
            types.SimpleNamespace(krea2_transformer_quantization="w8a8", krea2_transformer_quant_targets=raw)
        ),
        ("w8a8", ("gate_proj", "up_proj", "down_proj")),
    )
    raw = pyconfig.string_to_list("['to_q','gate_proj']")
    self.assertEqual(
        resolve_transformer_quantization(
            types.SimpleNamespace(krea2_transformer_quantization="w8a8", krea2_transformer_quant_targets=raw)
        ),
        ("w8a8", ("to_q", "gate_proj")),
    )
    # The bare comma-separated form does not survive pyconfig's parser.
    with self.assertRaises(ValueError):
      pyconfig.string_to_list("to_q,gate_proj")

  def test_describe(self):
    self.assertEqual(describe_transformer_quantization("", ()), "transformer: unquantized")
    self.assertEqual(
        describe_transformer_quantization("w8a8", ("to_q", "down_proj")),
        "transformer: int8 W8A8 matmuls (to_q, down_proj); other projections unquantized",
    )


class QuantizeActivationTest(unittest.TestCase):

  def test_per_token_quantization(self):
    rng = np.random.RandomState(0)
    x = rng.randn(2, 5, 16).astype(np.float32) * rng.uniform(0.01, 10.0, size=(2, 5, 1)).astype(np.float32)
    x[1, 2] = 0.0
    x_q, scale = quantize_activation(jnp.asarray(x), jnp.float32)
    x_q, scale = np.asarray(x_q), np.asarray(scale)
    self.assertEqual(x_q.dtype, np.int8)
    self.assertEqual(x_q.shape, x.shape)
    self.assertEqual(scale.shape, (2, 5, 1))
    self.assertEqual(scale.dtype, np.float32)
    self.assertTrue(np.all(np.abs(x_q.astype(np.int32)) <= 127))
    err = np.abs(x - x_q.astype(np.float32) * scale)
    self.assertTrue(np.all(err <= scale / 2 * (1 + 1e-5) + 1e-12))
    # The token absmax maps to +-127.
    self.assertTrue(np.all(np.abs(x_q[0].astype(np.int32)).max(axis=-1) == 127))
    # All-zero token: scale 1, zeros.
    self.assertEqual(scale[1, 2, 0], 1.0)
    np.testing.assert_array_equal(x_q[1, 2], 0)

  def test_tokens_are_independent(self):
    rng = np.random.RandomState(1)
    x = rng.randn(1, 4, 8).astype(np.float32)
    x_q, scale = quantize_activation(jnp.asarray(x), jnp.float32)
    x2 = x.copy()
    x2[0, 1] *= 100.0
    x2_q, scale2 = quantize_activation(jnp.asarray(x2), jnp.float32)
    for token in (0, 2, 3):
      np.testing.assert_array_equal(np.asarray(x2_q)[0, token], np.asarray(x_q)[0, token])
      self.assertEqual(np.asarray(scale2)[0, token, 0], np.asarray(scale)[0, token, 0])
    np.testing.assert_array_equal(np.asarray(x2_q)[0, 1], np.asarray(x_q)[0, 1])
    np.testing.assert_allclose(np.asarray(scale2)[0, 1, 0], 100.0 * np.asarray(scale)[0, 1, 0], rtol=1e-6)


class QuantizeKernelTest(unittest.TestCase):

  def test_per_column_scale_and_error(self):
    rng = np.random.RandomState(0)
    w = rng.randn(12, 6).astype(np.float32) * np.array([1e-3, 0.1, 1.0, 10.0, 3.0, 1.0], np.float32)
    w[:, 5] = 0.0
    w_before = w.copy()
    kernel, scale = quantize_kernel(w, np.float32)
    self.assertEqual(kernel.dtype, np.int8)
    self.assertEqual(kernel.shape, (12, 6))
    self.assertEqual(scale.shape, (6,))
    self.assertEqual(scale.dtype, np.float32)
    np.testing.assert_allclose(scale[:5], np.abs(w[:, :5]).max(axis=0) / 127, rtol=1e-6)
    self.assertTrue(np.all(np.abs(kernel[:, :5].astype(np.int32)).max(axis=0) == 127))
    err = np.abs(w - kernel.astype(np.float32) * scale)
    self.assertTrue(np.all(err <= scale / 2 * (1 + 1e-5)))
    # Zero column: scale 1, zeros.
    self.assertEqual(scale[5], 1.0)
    np.testing.assert_array_equal(kernel[:, 5], 0)
    # The input is not modified.
    np.testing.assert_array_equal(w, w_before)

  def test_bfloat16_input_and_scale(self):
    rng = np.random.RandomState(1)
    w = (rng.randn(16, 4) * 0.02).astype(jnp.bfloat16)
    kernel, scale = quantize_kernel(w, jnp.bfloat16)
    self.assertEqual(kernel.dtype, np.int8)
    self.assertEqual(scale.dtype, jnp.bfloat16)
    self.assertEqual(scale.shape, (4,))
    s32 = scale.astype(np.float32)
    err = np.abs(w.astype(np.float32) - kernel.astype(np.float32) * s32)
    self.assertTrue(np.all(err <= s32 / 2 * (1 + 1e-5)))
    # Accepts jax arrays too (converted on the host).
    kernel_j, scale_j = quantize_kernel(jnp.asarray(w.astype(np.float32)), np.float32)
    self.assertIsInstance(kernel_j, np.ndarray)
    self.assertEqual(scale_j.dtype, np.float32)

  def test_invalid_inputs(self):
    with self.assertRaises(ValueError):
      quantize_kernel(np.ones((4,), np.float32), np.float32)
    with self.assertRaises(ValueError):
      quantize_kernel(np.ones((2, 3, 4), np.float32), np.float32)
    with self.assertRaises(ValueError):
      quantize_kernel(np.ones((4, 4), np.int8), np.float32)
    for bad in (np.inf, np.nan):
      w = np.ones((4, 3), np.float32)
      w[2, 1] = bad
      with self.assertRaises(ValueError):
        quantize_kernel(w, np.float32)


class Krea2QuantDenseTest(unittest.TestCase):

  def _params(self, w, scale_dtype=np.float32):
    kernel, scale = quantize_kernel(w, scale_dtype)
    return {"params": {"kernel": jnp.asarray(kernel), "kernel_scale": jnp.asarray(scale)}}

  def test_matches_float_reference(self):
    rng = np.random.RandomState(0)
    x = rng.randn(2, 7, 32).astype(np.float32)
    w = (rng.randn(32, 24) * 0.2).astype(np.float32)
    dense = Krea2QuantDense(24, kernel_axes=("embed", "mlp"))
    variables = self._params(w)
    out = dense.apply(variables, jnp.asarray(x))
    self.assertEqual(out.shape, (2, 7, 24))
    self.assertEqual(out.dtype, jnp.float32)
    self.assertLess(_rel_l2(out, x @ w), 0.02)
    # Shared pre-quantized inputs give the same result as self-quantization.
    shared = dense.apply(variables, jnp.asarray(x), quantize_activation(jnp.asarray(x), jnp.float32))
    np.testing.assert_array_equal(np.asarray(shared), np.asarray(out))

  def test_exact_for_representable_values(self):
    rng = np.random.RandomState(1)
    x_int = rng.randint(-127, 128, size=(3, 5, 16))
    x_int[..., 3] = 127  # every token reaches the absmax, so its scale is exactly 2**-3
    w_int = rng.randint(-127, 128, size=(16, 6))
    w_int[7] = -127  # every column reaches the absmax
    w_scale = 2.0 ** -np.arange(2, 8, dtype=np.float32)
    x = (x_int * 2.0**-3).astype(np.float32)
    w = (w_int * w_scale).astype(np.float32)
    dense = Krea2QuantDense(6, kernel_axes=("embed", "mlp"))
    variables = self._params(w)
    np.testing.assert_array_equal(np.asarray(variables["params"]["kernel"]), w_int)
    np.testing.assert_array_equal(np.asarray(variables["params"]["kernel_scale"]), w_scale)
    out = dense.apply(variables, jnp.asarray(x))
    expected = ((x_int @ w_int) * 2.0**-3 * w_scale).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(out), expected)

  def test_unflatten_output_equals_flat_output(self):
    rng = np.random.RandomState(2)
    x = jnp.asarray(rng.randn(2, 5, 64).astype(np.float32))
    w = (rng.randn(64, 32) * 0.2).astype(np.float32)
    variables = self._params(w)
    flat = Krea2QuantDense(32, kernel_axes=("embed", "mlp"))
    quantized = quantize_activation(x, jnp.float32)
    expected = np.asarray(flat.apply(variables, x))
    np.testing.assert_array_equal(np.asarray(flat.apply(variables, x, quantized)), expected)
    for unflatten in ((4, 8), (2, 2, 8)):
      with self.subTest(unflatten=unflatten):
        dense = Krea2QuantDense(32, kernel_axes=("embed", "mlp"), unflatten=unflatten)
        for args in ((x,), (x, quantized)):
          out = dense.apply(variables, *args)
          self.assertEqual(out.shape, (2, 5, 32))
          self.assertEqual(out.dtype, jnp.float32)
          np.testing.assert_array_equal(np.asarray(out), expected)

  def test_empty_or_trivial_unflatten_equals_flat_output(self):
    rng = np.random.RandomState(3)
    for features, inputs_shape, unflatten in ((1, (2, 3), ()), (1, (2, 3), (1,)), (32, (2, 5, 64), ())):
      with self.subTest(features=features, unflatten=unflatten):
        x = jnp.asarray(rng.randn(*inputs_shape).astype(np.float32))
        variables = self._params((rng.randn(inputs_shape[-1], features) * 0.2).astype(np.float32))
        expected = np.asarray(Krea2QuantDense(features, kernel_axes=("embed", "mlp")).apply(variables, x))
        out = np.asarray(Krea2QuantDense(features, kernel_axes=("embed", "mlp"), unflatten=unflatten).apply(variables, x))
        self.assertEqual(out.shape, inputs_shape[:-1] + (features,))
        np.testing.assert_array_equal(out, expected)

  def test_unflatten_must_match_features(self):
    dense = Krea2QuantDense(32, kernel_axes=("embed", "mlp"), unflatten=(4, 7))
    variables = self._params(np.ones((16, 32), np.float32))
    with self.assertRaisesRegex(ValueError, r"unflatten=\(4, 7\).*features=32"):
      dense.apply(variables, jnp.ones((1, 2, 16), jnp.float32))

  def test_float16_rescale_does_not_overflow(self):
    # int32 accumulator 1024 * 127 * 127 = 16516096 exceeds the float16 max (65504),
    # so casting it to float16 before the rescale would give inf.
    self.assertTrue(np.isinf(np.float16(1024 * 127 * 127)))
    for unflatten in (None, (2, 2)):
      with self.subTest(unflatten=unflatten):
        dense = Krea2QuantDense(4, kernel_axes=("embed", "mlp"), dtype=jnp.float16, unflatten=unflatten)
        kernel_scale = np.full((4,), 2.0**-9, np.float32)
        variables = {"params": {"kernel": jnp.full((1024, 4), 127, jnp.int8), "kernel_scale": jnp.asarray(kernel_scale)}}
        x = jnp.ones((1, 2, 1024), jnp.float16)
        out = np.asarray(dense.apply(variables, x))
        self.assertEqual(out.dtype, np.float16)
        self.assertEqual(out.shape, (1, 2, 4))
        self.assertTrue(np.all(np.isfinite(out)))
        # x_scale is 1/127 rounded to float16; the true product is 1024 * 127 * 2**-9 = 254.
        x_scale = float(np.float16(1.0 / 127))
        expected = 1024 * 127 * 127 * x_scale * 2.0**-9
        np.testing.assert_allclose(out.astype(np.float64), expected, rtol=2**-10)
        np.testing.assert_allclose(out.astype(np.float64), 254.0, rtol=2e-3)

  def test_rescale_dtype_in_lowered_hlo(self):
    # Lowering only (no bf16/f16 execution); explicit quantized_inputs keep the activation
    # quantizer's f32 math out of the text.
    def lowered(dtype):
      dense = Krea2QuantDense(24, kernel_axes=("embed", "mlp"), dtype=dtype, param_dtype=dtype)
      params = {"kernel": jax.ShapeDtypeStruct((64, 24), jnp.int8), "kernel_scale": jax.ShapeDtypeStruct((24,), dtype)}
      x = jax.ShapeDtypeStruct((1, 3, 64), dtype)
      quantized = (jax.ShapeDtypeStruct((1, 3, 64), jnp.int8), jax.ShapeDtypeStruct((1, 3, 1), dtype))
      return jax.jit(lambda p, x, q: dense.apply({"params": p}, x, q)).lower(params, x, quantized).as_text()

    multiply = re.compile(r"stablehlo\.multiply .*: tensor<1x3x24x(\w+)>")
    # bfloat16 holds the int32 accumulator: the benchmarked bf16 rescale, no f32 matmul-output tensor.
    bf16 = lowered(jnp.bfloat16)
    self.assertEqual(multiply.findall(bf16), ["bf16", "bf16"])
    self.assertIn("-> tensor<1x3x24xbf16>", bf16)
    self.assertNotIn("tensor<1x3x24xf32>", bf16)
    # float16 cannot (max 65504): the rescale runs in f32, then one convert to f16.
    f16 = lowered(jnp.float16)
    self.assertEqual(multiply.findall(f16), ["f32", "f32"])
    self.assertEqual(len(re.findall(r"-> tensor<1x3x24xf16>", f16)), 1)

  def test_unflatten_rescales_in_head_layout_in_lowered_hlo(self):
    # Lowering only (no bf16 execution): the int32 accumulator is reshaped to the head layout
    # before the rescale, so a caller's reshape to heads can fold into the matmul.
    dtype = jnp.bfloat16
    dense = Krea2QuantDense(32, kernel_axes=("embed", "mlp"), dtype=dtype, param_dtype=dtype, unflatten=(4, 8))
    params = {"kernel": jax.ShapeDtypeStruct((64, 32), jnp.int8), "kernel_scale": jax.ShapeDtypeStruct((32,), dtype)}
    x = jax.ShapeDtypeStruct((1, 3, 64), dtype)
    quantized = (jax.ShapeDtypeStruct((1, 3, 64), jnp.int8), jax.ShapeDtypeStruct((1, 3, 1), dtype))
    hlo = jax.jit(lambda p, x, q: dense.apply({"params": p}, x, q)).lower(params, x, quantized).as_text()
    lines = hlo.splitlines()

    def first_line(pattern):
      return next(i for i, line in enumerate(lines) if re.search(pattern, line))

    dot = re.search(r"(%\w+) = stablehlo\.dot_general .* -> tensor<1x3x32xi32>", hlo)
    self.assertIsNotNone(dot, hlo)
    reshape = first_line(re.escape(f"stablehlo.reshape {dot.group(1)} : (tensor<1x3x32xi32>) -> tensor<1x3x4x8xi32>"))
    self.assertLess(reshape, first_line(r"stablehlo\.multiply"))
    self.assertEqual(re.findall(r"stablehlo\.multiply .*: tensor<([0-9x]+\w+)>", hlo), ["1x3x4x8xbf16"] * 2)
    self.assertRegex(hlo, r"return %\w+ : tensor<1x3x32xbf16>")
    self.assertNotIn("tensor<1x3x32xf32>", hlo)

  def test_init_and_int8_dot_general(self):
    dense = Krea2QuantDense(24, kernel_axes=("embed", "mlp"), param_dtype=jnp.float32)
    x = jnp.ones((1, 4, 32), jnp.float32)
    params = dense.init(jax.random.PRNGKey(0), x)["params"]
    self.assertIsInstance(params["kernel"], flax_spmd.LogicallyPartitioned)
    self.assertEqual(params["kernel"].names, ("embed", "mlp"))
    self.assertEqual(params["kernel"].unbox().dtype, jnp.int8)
    self.assertEqual(params["kernel"].unbox().shape, (32, 24))
    self.assertNotIsInstance(params["kernel_scale"], flax_spmd.LogicallyPartitioned)
    self.assertEqual(params["kernel_scale"].shape, (24,))
    hlo = jax.jit(lambda p, x: dense.apply({"params": p}, x)).lower(_unbox(params), x).as_text()
    self.assertRegex(hlo, r"dot_general.*\(tensor<[0-9x]*xi8>, tensor<[0-9x]*xi8>\) -> tensor<[0-9x]*xi32>")


class QuantizedModelTreeTest(unittest.TestCase):

  def _abstract(self, model, inputs):
    return _unbox(jax.eval_shape(model.init, jax.random.PRNGKey(0), *inputs)["params"])

  def test_param_tree(self):
    inputs = _inputs()
    for weights_dtype in (jnp.float32, jnp.bfloat16):
      float_tree = _flat(self._abstract(_tiny_model(weights_dtype=weights_dtype), inputs))
      quant_tree = _flat(
          self._abstract(_tiny_model(weights_dtype=weights_dtype, quant_targets=KREA2_DEFAULT_QUANT_TARGETS), inputs)
      )
      for i in range(2):
        for group, names in (("attn", _ATTN_TARGETS), ("ff", _FF_TARGETS)):
          for name in names:
            prefix = (f"blocks_{i}", group, name)
            kernel = quant_tree[prefix + ("kernel",)]
            self.assertEqual(kernel.shape, float_tree[prefix + ("kernel",)].shape)
            if name in KREA2_DEFAULT_QUANT_TARGETS:
              self.assertEqual(kernel.dtype, jnp.int8)
              self.assertEqual(quant_tree[prefix + ("kernel_scale",)].dtype, weights_dtype)
              self.assertEqual(quant_tree[prefix + ("kernel_scale",)].shape, (kernel.shape[1],))
            else:
              self.assertEqual(kernel.dtype, weights_dtype)
              self.assertNotIn(prefix + ("kernel_scale",), quant_tree)
      # Nothing outside blocks_* is quantized.
      for path, leaf in quant_tree.items():
        if not path[0].startswith("blocks_"):
          self.assertNotEqual(path[-1], "kernel_scale", path)
          self.assertEqual(leaf.dtype, float_tree[path].dtype, path)
      scale_paths = {path for path in quant_tree if path[-1] == "kernel_scale"}
      self.assertEqual(len(scale_paths), 2 * len(KREA2_DEFAULT_QUANT_TARGETS))
      self.assertEqual(set(quant_tree) - scale_paths, set(float_tree))

  def test_empty_targets_tree_is_unchanged(self):
    inputs = _inputs()
    model = _tiny_model(quant_targets=())
    empty = _flat(self._abstract(model, inputs))
    # Hard-coded float block layout (independent of the quantization code path).
    block = {"/".join(path[1:]): leaf for path, leaf in empty.items() if path[0] == "blocks_0"}
    self.assertEqual(
        set(block),
        {
            "attn/to_q/kernel",
            "attn/to_k/kernel",
            "attn/to_v/kernel",
            "attn/to_gate/kernel",
            "attn/to_out/kernel",
            "attn/norm_q/weight",
            "attn/norm_k/weight",
            "ff/gate_proj/kernel",
            "ff/up_proj/kernel",
            "ff/down_proj/kernel",
            "norm1/weight",
            "norm2/weight",
            "scale_shift_table",
        },
    )
    for path, leaf in block.items():
      if path.endswith("/kernel"):
        self.assertEqual(leaf.dtype, jnp.dtype(model.weights_dtype), path)
        self.assertTrue(jnp.issubdtype(leaf.dtype, jnp.floating), path)
    self.assertEqual(block["attn/to_q/kernel"].shape, (_HIDDEN, _HEADS * _HEAD_DIM))
    self.assertEqual(block["attn/to_k/kernel"].shape, (_HIDDEN, _KV_HEADS * _HEAD_DIM))
    self.assertEqual(block["ff/gate_proj/kernel"].shape, (_HIDDEN, 64))
    self.assertFalse([path for path in empty if path[-1] == "kernel_scale"])
    # The default (no quant_targets argument) builds the same tree.
    default = _flat(self._abstract(_tiny_model(), inputs))
    self.assertEqual(set(default), set(empty))
    for path, leaf in default.items():
      self.assertEqual((leaf.shape, leaf.dtype), (empty[path].shape, empty[path].dtype))

  def test_qkv_projections_rescale_in_head_layout(self):
    inputs = _inputs()
    model = _tiny_model(quant_targets=KREA2_QUANT_TARGETS)
    bound = model.bind({"params": model.init(jax.random.PRNGKey(0), *inputs)["params"]})
    layouts = {
        "to_q": (_HEADS, _HEAD_DIM),
        "to_k": (_KV_HEADS, _HEAD_DIM),
        "to_v": (_KV_HEADS, _HEAD_DIM),
        "to_gate": None,
        "to_out": None,
    }
    for i in range(model.num_layers):
      for group, names in (("attn", _ATTN_TARGETS), ("ff", _FF_TARGETS)):
        for name in names:
          proj = getattr(getattr(bound.blocks[i], group), name)
          self.assertIsInstance(proj, Krea2QuantDense)
          self.assertEqual(proj.unflatten, layouts.get(name), (i, name))

  def test_head_layout_keeps_param_tree_and_output(self):
    inputs = _inputs()
    model = _tiny_model(quant_targets=KREA2_QUANT_TARGETS)
    real = transformer_krea2_flax._projection

    def flat_projection(*args, unflatten=None, **kwargs):
      del unflatten
      return real(*args, **kwargs)

    tree = _flat(self._abstract(model, inputs))
    params = quantize_transformer_params(_float_params(inputs), KREA2_QUANT_TARGETS, np.float32)
    actual = model.apply({"params": params}, *inputs).sample
    with mock.patch.object(transformer_krea2_flax, "_projection", flat_projection):
      flat_tree = _flat(self._abstract(model, inputs))
      expected = model.apply({"params": params}, *inputs).sample
    self.assertEqual(set(tree), set(flat_tree))
    for path, leaf in tree.items():
      self.assertEqual((leaf.shape, leaf.dtype), (flat_tree[path].shape, flat_tree[path].dtype), path)
    for name, features in (("to_q", _HEADS * _HEAD_DIM), ("to_k", _KV_HEADS * _HEAD_DIM), ("to_v", _KV_HEADS * _HEAD_DIM)):
      prefix = ("blocks_0", "attn", name)
      self.assertEqual((tree[prefix + ("kernel",)].shape, tree[prefix + ("kernel",)].dtype), ((_HIDDEN, features), jnp.int8))
      self.assertEqual(tree[prefix + ("kernel_scale",)].shape, (features,))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

  def test_unknown_target_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "projector"):
      jax.eval_shape(_tiny_model(quant_targets=("to_q", "projector")).init, jax.random.PRNGKey(0), *_inputs())


class QuantizedForwardTest(unittest.TestCase):

  def test_quantized_output_is_close_to_float(self):
    inputs = _inputs(batch_size=2)
    params = _float_params(inputs)
    expected = _tiny_model().apply({"params": params}, *inputs).sample
    quant_params = quantize_transformer_params(params, KREA2_DEFAULT_QUANT_TARGETS, np.float32)
    actual = _tiny_model(quant_targets=KREA2_DEFAULT_QUANT_TARGETS).apply({"params": quant_params}, *inputs).sample
    error = _rel_l2(actual, expected)
    print(f"tiny model W8A8 vs float relative L2 error: {error:.5f}")
    self.assertGreater(error, 0.0)
    # Measured 0.027 for this tiny model and seed; the bound is ~3x that.
    self.assertLess(error, 0.08)

  def test_activation_quantized_once_per_distinct_input(self):
    inputs = _inputs()
    real = transformer_krea2_flax.quantize_activation
    for targets, per_block in ((KREA2_DEFAULT_QUANT_TARGETS, 4), (("to_k",), 1), ((), 0), (KREA2_QUANT_TARGETS, 4)):
      model = _tiny_model(quant_targets=targets)
      with mock.patch.object(transformer_krea2_flax, "quantize_activation", wraps=real) as counted:
        jax.eval_shape(model.init, jax.random.PRNGKey(0), *inputs)
      self.assertEqual(counted.call_count, per_block * model.num_layers, targets)

  def test_staged_blocks_match_monolithic_forward(self):
    targets = KREA2_DEFAULT_QUANT_TARGETS
    model = _tiny_model(quant_targets=targets)
    hs, ehs, timestep, img_ids, txt_ids, mask = _inputs()
    params = quantize_transformer_params(_float_params(_inputs()), targets, np.float32)
    expected = model.apply({"params": params}, hs, ehs, timestep, img_ids, txt_ids, mask).sample

    text_hidden = model.apply({"params": params}, ehs, mask, method=model.encode_text_context)
    hidden, temb, temb_mod, rotary_emb, attention_mask = model.apply(
        {"params": params}, hs, text_hidden, timestep, img_ids, txt_ids, mask, method=model.prepare_inputs
    )
    block = Krea2TransformerBlock(
        hidden_size=_HIDDEN, intermediate_size=64, num_heads=_HEADS, num_kv_heads=_KV_HEADS, quant_targets=targets
    )
    for block_idx in range(model.num_layers):
      hidden = block.apply(
          {"params": params[f"blocks_{block_idx}"]},
          hidden,
          temb_mod=temb_mod,
          image_rotary_emb=rotary_emb,
          attention_mask=attention_mask,
      )
    actual = model.apply({"params": params}, hidden, temb, hs.shape[1], method=model.finalize_output)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_rotate_half_matches_interleaved(self):
    targets = KREA2_QUANT_TARGETS  # includes to_k
    inputs = _inputs(batch_size=2)
    params = _float_params(inputs)
    permuted = permute_rope_weights_to_rotate_half(params, _HEADS, _KV_HEADS, _HEAD_DIM)
    interleaved_params = quantize_transformer_params(params, targets, np.float32)
    rotate_half_params = quantize_transformer_params(permuted, targets, np.float32)
    expected = _tiny_model(quant_targets=targets).apply({"params": interleaved_params}, *inputs).sample
    actual = _tiny_model(quant_targets=targets, rope_layout="rotate_half").apply(
        {"params": rotate_half_params}, *inputs
    ).sample
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    # Per-column scales permute with the columns.
    q_scale = interleaved_params["blocks_0"]["attn"]["to_q"]["kernel_scale"]
    q_scale_permuted = rotate_half_params["blocks_0"]["attn"]["to_q"]["kernel_scale"]
    self.assertFalse(np.array_equal(q_scale, q_scale_permuted))
    np.testing.assert_array_equal(np.sort(q_scale), np.sort(q_scale_permuted))

  def test_flash_custom_quantizes_to_out(self):
    def fake_apply_attention(module, query, key, value, attention_mask=None):
      repeats = query.shape[1] // key.shape[1]
      k = jnp.repeat(key, repeats, axis=1).astype(jnp.float32)
      v = jnp.repeat(value, repeats, axis=1).astype(jnp.float32)
      scores = jnp.einsum("bhqd,bhkd->bhqk", query.astype(jnp.float32), k) * module.scale
      if attention_mask is not None:
        scores = jnp.where(attention_mask[:, None, None, :] > 0, scores, -1e9)
      out = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)
      b, h, l, d = out.shape
      return jnp.transpose(out, (0, 2, 1, 3)).reshape(b, l, h * d)

    targets = ("to_q", "to_gate", "to_out")
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(2, 6, 32), dtype=jnp.float32)
    ids = prepare_krea2_image_ids(1, 2, 3)[0]
    mask = jnp.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=jnp.bool_)
    rotary = krea2_rotary_tables(ids, (4, 2, 2), 1000.0)
    attn_kwargs = dict(dim=32, num_heads=_HEADS, num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, quant_targets=targets)
    reference = Krea2Attention(**attn_kwargs)
    custom = Krea2Attention(attention_kernel="flash_custom", flash_min_seq_length=0, **attn_kwargs)
    float_params = _unbox(Krea2Attention(**{**attn_kwargs, "quant_targets": ()}).init(jax.random.PRNGKey(0), x)["params"])
    params = quantize_transformer_params({"blocks_0": {"attn": float_params}}, targets, np.float32)["blocks_0"]["attn"]
    self.assertEqual(params["to_out"]["kernel"].dtype, np.int8)

    expected = reference.apply({"params": params}, x, mask, rotary)
    real = transformer_krea2_flax.quantize_activation
    with mock.patch.object(AttentionOp, "apply_attention", fake_apply_attention), mock.patch.object(
        transformer_krea2_flax, "quantize_activation", wraps=real
    ) as counted:
      actual = custom.apply({"params": params}, x, mask, rotary)
    # hidden_states (to_q/to_gate) and the gated attention output (to_out).
    self.assertEqual(counted.call_count, 2)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    # to_out really consumes the int8 kernel on this branch: a different scale changes the output.
    rescaled = flax.core.unfreeze(params)
    rescaled["to_out"] = {**params["to_out"], "kernel_scale": params["to_out"]["kernel_scale"] * 2}
    with mock.patch.object(AttentionOp, "apply_attention", fake_apply_attention):
      doubled = custom.apply({"params": rescaled}, x, mask, rotary)
    np.testing.assert_allclose(np.asarray(doubled), 2 * np.asarray(actual), rtol=1e-5, atol=1e-6)


class QuantizeTransformerParamsTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.inputs = _inputs()
    cls.params = _float_params(cls.inputs)

  def _abstract(self, targets):
    model = _tiny_model(quant_targets=targets)
    return jax.eval_shape(model.init, jax.random.PRNGKey(0), *self.inputs)["params"]

  def test_result_matches_model_tree_and_shares_leaves(self):
    params = self.params
    before = jax.tree_util.tree_map(np.copy, params)
    quantized = quantize_transformer_params(flax.core.freeze(params), KREA2_DEFAULT_QUANT_TARGETS, np.float32)
    self.assertIsInstance(quantized, dict)
    self.assertIsInstance(quantized["blocks_0"]["attn"], dict)
    # Boxed FrozenDict abstract tree from eval_shape.
    check_transformer_param_tree(quantized, self._abstract(KREA2_DEFAULT_QUANT_TARGETS))
    # Input untouched.
    self.assertNotIn("kernel_scale", params["blocks_0"]["attn"]["to_q"])
    jax.tree_util.tree_map(np.testing.assert_array_equal, params, before)
    # Unmodified leaves are shared.
    self.assertIs(quantized["blocks_0"]["attn"]["to_k"]["kernel"], params["blocks_0"]["attn"]["to_k"]["kernel"])
    self.assertIs(quantized["blocks_1"]["norm1"]["weight"], params["blocks_1"]["norm1"]["weight"])
    self.assertIs(
        quantized["text_fusion"]["refiner_blocks_0"]["ff"]["up_proj"]["kernel"],
        params["text_fusion"]["refiner_blocks_0"]["ff"]["up_proj"]["kernel"],
    )
    self.assertEqual(quantized["text_fusion"]["refiner_blocks_0"]["ff"]["up_proj"]["kernel"].dtype, np.float32)
    self.assertEqual(quantized["blocks_1"]["ff"]["down_proj"]["kernel"].dtype, np.int8)

  def test_lora_subtree_is_preserved(self):
    params = flax.core.unfreeze(flax.core.freeze(self.params))
    lora = {"down": {"kernel": np.ones((_HIDDEN, 2), np.float32)}, "up": {"kernel": np.ones((2, _HIDDEN), np.float32)}}
    params["blocks_0"]["attn"]["to_q"]["lora-x"] = lora
    quantized = quantize_transformer_params(params, ["to_q"], np.float32)
    proj = quantized["blocks_0"]["attn"]["to_q"]
    self.assertEqual(set(proj), {"kernel", "kernel_scale", "lora-x"})
    self.assertIs(proj["lora-x"]["down"]["kernel"], lora["down"]["kernel"])
    self.assertIs(proj["lora-x"]["up"]["kernel"], lora["up"]["kernel"])
    self.assertEqual(proj["kernel"].dtype, np.int8)

  def test_custom_targets(self):
    for targets in (_FF_TARGETS, KREA2_QUANT_TARGETS, "gate_proj,to_v"):
      quantized = quantize_transformer_params(self.params, targets, np.float32)
      check_transformer_param_tree(quantized, self._abstract(normalize_quant_targets(targets)))
    ff_only = quantize_transformer_params(self.params, _FF_TARGETS, np.float32)
    self.assertEqual(ff_only["blocks_0"]["attn"]["to_q"]["kernel"].dtype, np.float32)
    self.assertEqual(ff_only["blocks_0"]["ff"]["up_proj"]["kernel"].dtype, np.int8)
    # Empty targets: unchanged tree (as plain dicts).
    unchanged = quantize_transformer_params(flax.core.freeze(self.params), (), np.float32)
    self.assertIsInstance(unchanged, dict)
    self.assertIs(unchanged["blocks_0"]["attn"]["to_q"]["kernel"], self.params["blocks_0"]["attn"]["to_q"]["kernel"])

  def test_bfloat16_scales(self):
    quantized = quantize_transformer_params(self.params, ("to_q",), jnp.bfloat16)
    self.assertEqual(quantized["blocks_0"]["attn"]["to_q"]["kernel_scale"].dtype, jnp.bfloat16)

  def test_errors(self):
    missing = flax.core.unfreeze(flax.core.freeze(self.params))
    del missing["blocks_1"]["ff"]["up_proj"]
    with self.assertRaisesRegex(ValueError, "blocks_1/ff/up_proj"):
      quantize_transformer_params(missing, KREA2_DEFAULT_QUANT_TARGETS, np.float32)
    no_ff = flax.core.unfreeze(flax.core.freeze(self.params))
    del no_ff["blocks_0"]["ff"]
    with self.assertRaisesRegex(ValueError, "blocks_0/ff/gate_proj"):
      quantize_transformer_params(no_ff, ("gate_proj",), np.float32)
    quantized = quantize_transformer_params(self.params, ("to_q",), np.float32)
    with self.assertRaisesRegex(ValueError, "blocks_0/attn/to_q"):
      quantize_transformer_params(quantized, ("to_q",), np.float32)
    with self.assertRaisesRegex(ValueError, "blocks_"):
      quantize_transformer_params({"text_fusion": self.params["text_fusion"]}, ("to_q",), np.float32)
    with self.assertRaisesRegex(ValueError, "bogus"):
      quantize_transformer_params(self.params, ("bogus",), np.float32)

  def test_check_param_tree_errors(self):
    abstract = self._abstract(KREA2_DEFAULT_QUANT_TARGETS)
    good = quantize_transformer_params(self.params, KREA2_DEFAULT_QUANT_TARGETS, np.float32)
    check_transformer_param_tree(good, abstract)
    check_transformer_param_tree(good, _unbox(abstract))

    def mutated(fn):
      tree = flax.core.unfreeze(flax.core.freeze(good))
      fn(tree)
      return tree

    cases = (
        (lambda t: t["blocks_0"]["attn"]["to_q"].pop("kernel_scale"), r"Missing.*blocks_0/attn/to_q/kernel_scale"),
        (lambda t: t["blocks_1"]["ff"].__setitem__("extra", np.zeros(3)), r"Not in the model.*blocks_1/ff/extra"),
        (
            lambda t: t["blocks_0"]["ff"]["up_proj"].__setitem__("kernel_scale", np.ones(5, np.float32)),
            r"Wrong shape.*blocks_0/ff/up_proj/kernel_scale",
        ),
        (
            lambda t: t["blocks_0"]["attn"]["to_out"].__setitem__("kernel", np.zeros((_HIDDEN, _HIDDEN), np.float32)),
            r"Wrong dtype.*blocks_0/attn/to_out/kernel \(expected int8, got float32\)",
        ),
    )
    for fn, pattern in cases:
      with self.assertRaisesRegex(ValueError, pattern):
        check_transformer_param_tree(mutated(fn), abstract)
    # The unquantized float tree does not fit the quantized model.
    with self.assertRaisesRegex(ValueError, re.escape("blocks_0/attn/to_gate/kernel_scale")):
      check_transformer_param_tree(self.params, abstract)


if __name__ == "__main__":
  unittest.main()
