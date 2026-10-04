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

# CPU tests for krea2_transformer_scoped_vmem_limit_kib: key validation, the AOT
# meta entry, the compile options of exactly the DiT block executables (only on
# a TPU mesh) and compile_krea2 compiling with them. The CPU backend rejects the
# TPU option ("No such compile option"), so a compile that fails with it proves
# the option reached that program and a compile that succeeds proves it did not.

import functools
import types
import unittest
from unittest import mock

import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AbstractMesh, Mesh

from maxdiffusion import aot_cache, compile_krea2, generate_krea2
from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
from maxdiffusion.models.krea2.util import prepare_krea2_image_ids, prepare_krea2_text_ids
from maxdiffusion.pipelines.krea2 import krea2_pipeline
from maxdiffusion.pipelines.krea2.krea2_pipeline import (
    KREA2_PRELUDE_KEYS,
    KREA2_TEXT_CONTEXT_KEYS,
    FlaxKrea2Pipeline,
    resolve_transformer_scoped_vmem_limit_kib,
    transformer_compiler_options,
)

_KEY = "krea2_transformer_scoped_vmem_limit_kib"
_OPTION = "xla_tpu_scoped_vmem_limit_kib"
_B, _GRID_H, _GRID_W, _S_TXT = 1, 2, 2, 3


def _cpu_mesh():
  return Mesh(np.array(jax.devices()[:1]), ("data",))


def _fake_mesh(platform):
  return types.SimpleNamespace(devices=np.array([types.SimpleNamespace(platform=platform)]))


def _tiny_transformer():
  return Krea2Transformer2DModel(
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


def _config(**overrides):
  values = {
      "max_sequence_length": _S_TXT,
      "logical_axis_rules": (),
      "krea2_staged_transformer": False,
      "krea2_staged_donate_hidden_states": True,
      "activations_dtype": jnp.float32,
      "weights_dtype": jnp.float32,
      "seed": 0,
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


def _pipeline(transformer, **config_overrides):
  pipeline = FlaxKrea2Pipeline(
      transformer=transformer,
      vae=None,
      vae_cache=None,
      text_encoder=None,
      tokenizer=None,
      scheduler=None,
      config=_config(**config_overrides),
      mesh=_cpu_mesh(),
  )
  pipeline._setup_jit_functions()
  return pipeline


class ScopedVmemLimitConfigTest(unittest.TestCase):

  def test_accepted_values(self):
    self.assertEqual(resolve_transformer_scoped_vmem_limit_kib(types.SimpleNamespace()), 0)
    for value, expected in ((None, 0), (0, 0), (16384, 16384), (36864, 36864), (131072, 131072)):
      self.assertEqual(resolve_transformer_scoped_vmem_limit_kib(types.SimpleNamespace(**{_KEY: value})), expected)
    self.assertEqual(resolve_transformer_scoped_vmem_limit_kib(types.SimpleNamespace(**{_KEY: np.int64(65536)})), 65536)

  def test_rejected_values_name_the_key(self):
    for value in (1, -1, 16383, 131073, True, False, 36864.0, "36864", "36M"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, _KEY):
          resolve_transformer_scoped_vmem_limit_kib(types.SimpleNamespace(**{_KEY: value}))

  def test_pipeline_validates_at_setup(self):
    with self.assertRaisesRegex(ValueError, _KEY):
      _pipeline(_tiny_transformer(), **{_KEY: 4096})


class ScopedVmemAotMetaTest(unittest.TestCase):

  def test_default_adds_no_key(self):
    self.assertEqual(generate_krea2.transformer_scoped_vmem_aot_meta(0), {})

  def test_limit_is_recorded(self):
    self.assertEqual(generate_krea2.transformer_scoped_vmem_aot_meta(36864), {_KEY: 36864})
    self.assertEqual(generate_krea2.transformer_scoped_vmem_aot_meta(65536), {_KEY: 65536})


class TransformerCompilerOptionsTest(unittest.TestCase):

  def test_tpu_only_and_only_when_set(self):
    self.assertEqual(transformer_compiler_options(36864, _fake_mesh("tpu")), {_OPTION: 36864})
    self.assertIsNone(transformer_compiler_options(0, _fake_mesh("tpu")))
    self.assertIsNone(transformer_compiler_options(36864, _fake_mesh("cpu")))
    self.assertIsNone(transformer_compiler_options(36864, _cpu_mesh()))
    self.assertIsNone(transformer_compiler_options(36864, None))

  def test_mesh_platform(self):
    self.assertEqual(krea2_pipeline._mesh_platform(_cpu_mesh()), "cpu")
    self.assertEqual(krea2_pipeline._mesh_platform(_fake_mesh("tpu")), "tpu")
    self.assertIsNone(krea2_pipeline._mesh_platform(None))
    self.assertIsNone(krea2_pipeline._mesh_platform(AbstractMesh((1,), ("data",))))


def _unbox(params):
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x,
      params,
      is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
  )


class PipelineCompilerOptionsTest(unittest.TestCase):
  """Which executables get the option, on a mesh that reports "tpu" and on the real CPU mesh."""

  @classmethod
  def setUpClass(cls):
    cls.transformer = _tiny_transformer()
    s_img = _GRID_H * _GRID_W
    latents = jnp.array(np.random.RandomState(0).randn(_B, s_img, 16), dtype=jnp.float32)
    prompt_embeds = jnp.array(np.random.RandomState(1).randn(_B, _S_TXT, 3, 24), dtype=jnp.float32)
    text_mask = jnp.array([[True, True, False]])
    img_ids = prepare_krea2_image_ids(_B, _GRID_H, _GRID_W)
    txt_ids = prepare_krea2_text_ids(_B, _S_TXT)
    t_vec = jnp.full((_B,), 0.5, dtype=jnp.float32)
    cls.inputs = (latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec)
    cls.params = _unbox(
        cls.transformer.init(jax.random.PRNGKey(0), latents, prompt_embeds, t_vec, img_ids, txt_ids, text_mask)["params"]
    )

  def _tpu_pipeline(self, **overrides):
    with mock.patch.object(krea2_pipeline, "_mesh_platform", return_value="tpu"):
      return _pipeline(self.transformer, **overrides)

  def _step_inputs(self, pipeline):
    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = self.inputs
    text_params = {key: self.params[key] for key in KREA2_TEXT_CONTEXT_KEYS}
    text_hidden = pipeline._jitted_transformer_text_context(text_params, prompt_embeds, text_mask)
    return latents, text_hidden, text_mask, img_ids, txt_ids, t_vec

  def _assert_no_options(self, pipeline, names):
    for name in names:
      entry = getattr(pipeline, f"_jitted_{name}")
      self.assertIsNone(entry.compiler_options, name)

  def test_monolithic_step_gets_the_option_on_tpu(self):
    pipeline = self._tpu_pipeline(**{_KEY: 36864})
    self.assertEqual(pipeline._jitted_transformer_step.compiler_options, {_OPTION: 36864})
    self._assert_no_options(pipeline, ("qwen3_forward", "transformer_text_context", "vae_decode"))
    # The real compiles agree: the text context compiles with the CPU defaults,
    # the step program carries the TPU option the CPU backend rejects.
    step_inputs = self._step_inputs(pipeline)
    with self.assertRaisesRegex(Exception, _OPTION):
      pipeline._jitted_transformer_step(self.params, *step_inputs)

  def test_staged_block_gets_the_option_on_tpu(self):
    for donate in (True, False):
      with self.subTest(donate=donate):
        pipeline = self._tpu_pipeline(
            **{_KEY: 65536, "krea2_staged_transformer": True, "krea2_staged_donate_hidden_states": donate}
        )
        self.assertEqual(pipeline._jitted_transformer_block.compiler_options, {_OPTION: 65536})
        self.assertEqual(pipeline._jitted_transformer_block.donate_argnames, ("hidden_states",) if donate else ())
        self._assert_no_options(
            pipeline,
            ("qwen3_forward", "transformer_text_context", "transformer_prelude", "transformer_final", "vae_decode"),
        )
        prelude_params = {key: self.params[key] for key in KREA2_PRELUDE_KEYS}
        hidden_states, _, temb_mod, rotary_emb, attention_mask = pipeline._jitted_transformer_prelude(
            prelude_params, *self._step_inputs(pipeline)
        )
        with self.assertRaisesRegex(Exception, _OPTION):
          pipeline._jitted_transformer_block(
              self.params["blocks_0"], {}, hidden_states, temb_mod, rotary_emb, attention_mask
          )

  def test_default_limit_passes_nothing_on_tpu(self):
    self.assertIsNone(self._tpu_pipeline(**{_KEY: 0})._jitted_transformer_step.compiler_options)
    staged = self._tpu_pipeline(**{_KEY: 0, "krea2_staged_transformer": True})
    self.assertIsNone(staged._jitted_transformer_block.compiler_options)

  def test_cpu_mesh_ignores_the_limit_and_runs(self):
    pipeline = _pipeline(self.transformer, **{_KEY: 36864})
    self.assertIsNone(pipeline._jitted_transformer_step.compiler_options)
    reference = _pipeline(self.transformer)
    step_inputs = self._step_inputs(reference)
    np.testing.assert_array_equal(
        np.asarray(pipeline._jitted_transformer_step(self.params, *step_inputs)),
        np.asarray(reference._jitted_transformer_step(self.params, *step_inputs)),
    )


@functools.partial(aot_cache.cached_jit, compiler_options={"xla_cpu_enable_fast_math": False})
def _cpu_optioned(w, x):
  return x @ w


@functools.partial(aot_cache.cached_jit, compiler_options={_OPTION: 36864})
def _tpu_optioned(w, x):
  return x @ w


@aot_cache.cached_jit
def _plain(w, x):
  return x @ w


class CompileExecutableOptionsTest(unittest.TestCase):
  """compile_krea2.compile_executable compiles with the entry's own options and records them."""

  def setUp(self):
    self.w = jax.ShapeDtypeStruct((8, 4), jnp.float32)
    self.x = jax.ShapeDtypeStruct((2, 8), jnp.float32)

  def test_options_are_compiled_and_recorded(self):
    record, outputs = compile_krea2.compile_executable("cpu_optioned", _cpu_optioned, (self.w,), (self.x,), 1)
    self.assertEqual(record["compiler_options"], {"xla_cpu_enable_fast_math": False})
    self.assertEqual(outputs.shape, (2, 4))
    record, _ = compile_krea2.compile_executable("plain", _plain, (self.w,), (self.x,), 1)
    self.assertIsNone(record["compiler_options"])

  def test_tpu_option_reaches_the_compile(self):
    with self.assertRaisesRegex(Exception, _OPTION):
      compile_krea2.compile_executable("tpu_optioned", _tpu_optioned, (self.w,), (self.x,), 1)


if __name__ == "__main__":
  unittest.main()
