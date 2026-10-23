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

# CPU tests for krea2_seq_len_tuning (per-sequence-length block_q and DiT
# scoped-VMEM limit): parsing and validation, the flash_custom block_q table,
# the VMEM estimate guard, the AOT meta key, the per-L cached_jit entries and
# their dispatch (monolithic and staged, under warmup mode and on threads), and
# that an empty table changes nothing. The tiny transformer and the "tpu"-mesh
# trick come from krea2_scoped_vmem_test: the CPU backend rejects the TPU
# scoped-VMEM option, so a call that fails with it ran an entry that carries it.

import dataclasses
import glob
import os
import tempfile
import threading
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import aot_cache, compile_krea2, generate_krea2, max_utils
from maxdiffusion.kernels import krea2_attention
from maxdiffusion.models import attention_flax
from maxdiffusion.models.krea2.util import prepare_krea2_image_ids, prepare_krea2_text_ids
from maxdiffusion.pipelines.krea2 import krea2_pipeline
from maxdiffusion.pipelines.krea2.krea2_pipeline import (
    KREA2_PRELUDE_KEYS,
    KREA2_TEXT_CONTEXT_KEYS,
    Krea2SeqLenTuning,
    SeqLenDispatch,
    resolve_seq_len_tuning,
    seq_len_block_q_table,
    seq_len_scoped_vmem_limit_kib,
    seq_len_tuning_block_q_warnings,
    seq_len_tuning_json,
)
from maxdiffusion.tests.krea2_scoped_vmem_test import _fake_mesh, _pipeline, _tiny_transformer, _unbox

_KEY = "krea2_seq_len_tuning"
_VMEM_KEY = "krea2_transformer_scoped_vmem_limit_kib"
_OPTION = "xla_tpu_scoped_vmem_limit_kib"
_V6E = "TPU v6 lite"
_B, _GRID_H, _GRID_W, _S_TXT = 1, 2, 2, 3
# The tiny pipeline's DiT length: 2x2 image tokens + 3 text tokens.
_L = _GRID_H * _GRID_W + _S_TXT
_STEP = "krea2_pipeline.transformer_step"
_BLOCK = "krea2_pipeline.transformer_block"


def _resolve(value):
  return resolve_seq_len_tuning(types.SimpleNamespace(**{_KEY: value}))


def _flash_config(**overrides):
  values = {"attention": "flash_custom", "flash_block_sizes": {}, "krea2_attention_kernel": "auto"}
  values.update(overrides)
  return types.SimpleNamespace(**values)


class SeqLenTuningConfigTest(unittest.TestCase):

  _TABLE = {
      "16640": {"block_q": 2560, "scoped_vmem_limit_kib": 65536},
      16512: {"block_q": 2560},
      "4224": {"scoped_vmem_limit_kib": 0},
  }
  _EXPECTED = (
      Krea2SeqLenTuning(4224, None, 0),
      Krea2SeqLenTuning(16512, 2560, None),
      Krea2SeqLenTuning(16640, 2560, 65536),
  )

  def test_empty_values(self):
    self.assertEqual(resolve_seq_len_tuning(types.SimpleNamespace()), ())
    for value in (None, {}, "", "  ", "''", '""', "{}", "'{}'", " { } "):
      with self.subTest(value=value):
        self.assertEqual(_resolve(value), ())

  def test_dict_and_json_string(self):
    self.assertEqual(_resolve(self._TABLE), self._EXPECTED)
    as_json = '{"16640": {"block_q": 2560, "scoped_vmem_limit_kib": 65536}, "16512": {"block_q": 2560}, ' + (
        '"4224": {"scoped_vmem_limit_kib": 0}}'
    )
    self.assertEqual(_resolve(as_json), self._EXPECTED)
    self.assertEqual(_resolve(f"'{as_json}'"), self._EXPECTED)
    # numpy ints, a None field (= not set), padded decimal keys.
    self.assertEqual(
        _resolve({" 16512 ": {"block_q": np.int64(2560), "scoped_vmem_limit_kib": None}}),
        (Krea2SeqLenTuning(16512, 2560, None),),
    )
    # Hashable (it is part of the setup, like the carrier).
    self.assertEqual(hash(_resolve(self._TABLE)), hash(self._EXPECTED))

  def test_canonical_json(self):
    self.assertEqual(
        seq_len_tuning_json(self._EXPECTED),
        '{"4224":{"scoped_vmem_limit_kib":0},"16512":{"block_q":2560},'
        '"16640":{"block_q":2560,"scoped_vmem_limit_kib":65536}}',
    )
    self.assertEqual(seq_len_tuning_json(()), "{}")
    # Same table, any spelling: same JSON.
    self.assertEqual(seq_len_tuning_json(_resolve(seq_len_tuning_json(self._EXPECTED))), seq_len_tuning_json(self._EXPECTED))

  def test_rejected_values_name_the_key(self):
    bad = (
        [1, 2],
        "[1]",
        "{not json",
        "16512",
        {0: {"block_q": 2560}},
        {-1: {"block_q": 2560}},
        {True: {"block_q": 2560}},
        {16512.0: {"block_q": 2560}},
        {"abc": {"block_q": 2560}},
        {"16512.5": {"block_q": 2560}},
        {"-16512": {"block_q": 2560}},
        {16512: {"block_q": 2560}, "16512": {"block_q": 1024}},
        {16512: 2560},
        {16512: None},
        {16512: {}},
        {16512: {"block_q": None}},
        {16512: {"blockq": 2560}},
        {16512: {"block_q": 2560, "vmem": 1}},
        {16512: {"block_q": 1000}},
        {16512: {"block_q": 384}},
        {16512: {"block_q": 0}},
        {16512: {"block_q": True}},
        {16512: {"block_q": 2560.0}},
        {16512: {"block_q": "2560"}},
        {16512: {"scoped_vmem_limit_kib": 1000}},
        {16512: {"scoped_vmem_limit_kib": 131073}},
        {16512: {"scoped_vmem_limit_kib": -1}},
        {16512: {"scoped_vmem_limit_kib": True}},
        {16512: {"scoped_vmem_limit_kib": "65536"}},
    )
    for value in bad:
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, _KEY):
          _resolve(value)

  def test_per_seq_len_helpers(self):
    tuning = _resolve(self._TABLE)
    self.assertEqual(seq_len_block_q_table(tuning), ((16512, 2560), (16640, 2560)))
    self.assertIsNone(seq_len_block_q_table(()))
    self.assertIsNone(seq_len_block_q_table(_resolve({4224: {"scoped_vmem_limit_kib": 65536}})))
    self.assertEqual(seq_len_scoped_vmem_limit_kib(tuning, 36864, 16640), 65536)
    self.assertEqual(seq_len_scoped_vmem_limit_kib(tuning, 36864, 16512), 36864)  # block_q only
    self.assertEqual(seq_len_scoped_vmem_limit_kib(tuning, 36864, 4224), 0)
    self.assertEqual(seq_len_scoped_vmem_limit_kib(tuning, 36864, 999), 36864)

  def test_preset_and_command_line_overrides(self):
    from maxdiffusion import pyconfig  # pylint: disable=import-outside-toplevel

    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    prev = (pyconfig._config, pyconfig.config)  # pylint: disable=protected-access
    self.addCleanup(lambda: setattr(pyconfig, "_config", prev[0]) or setattr(pyconfig, "config", prev[1]))
    preset = ((16512, 2560, 65536), (16640, 2560, 65536))
    cases = (
        ("base_krea2_turbo_v6e1.yml", (), preset),
        ("base_krea2_turbo_v6e1.yml", ("krea2_seq_len_tuning={}",), ()),
        ("base_krea2_turbo_v6e1.yml", ('krea2_seq_len_tuning={"4224": {"block_q": 1024}}',), ((4224, 1024, None),)),
        ("base_krea2_turbo.yml", (), ()),
        ("base_krea2.yml", (), ()),
    )
    for name, overrides, expected in cases:
      with self.subTest(config=name, overrides=overrides):
        with tempfile.TemporaryDirectory() as out:
          pyconfig.initialize(
              [
                  None,
                  os.path.join(config_dir, name),
                  "run_name=t",
                  f"output_dir={out}/",
                  "skip_jax_distributed_system=True",
                  *overrides,
              ]
          )
        self.assertEqual(resolve_seq_len_tuning(pyconfig.config), tuple(Krea2SeqLenTuning(*e) for e in expected))


class SeqLenBlockQTest(unittest.TestCase):
  """The flash_custom block_q table: lookup, precedence, plumbing through the carrier."""

  _TABLE = ((16512, 2560), (16640, 2560))

  def _auto(self, seq_len, kernel="auto"):
    return krea2_attention.select_krea2_block_sizes(
        seq_len, user=max_utils.CustomFlashBlockSizes(kernel=kernel), device_kind=_V6E
    )

  def test_table_replaces_the_automatic_choice_at_its_length_only(self):
    carrier = max_utils.CustomFlashBlockSizes(kernel="auto", block_q_by_seq_len=self._TABLE)
    for seq_len in (16512, 16640):
      sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=carrier, device_kind=_V6E)
      # Only block_q changes, uncapped (the hybrid automatic block_q never exceeds 2048).
      self.assertEqual(sizes, dataclasses.replace(self._auto(seq_len), block_q=2560))
      self.assertEqual(sizes.variant, "hybrid")
    for seq_len in (4224, 16256, 15808, 16513):
      with self.subTest(seq_len=seq_len):
        self.assertEqual(
            krea2_attention.select_krea2_block_sizes(seq_len, user=carrier, device_kind=_V6E), self._auto(seq_len)
        )
    # The flash variant reads it too; a dict table works like the carrier's tuple.
    flash = krea2_attention.select_krea2_block_sizes(
        16512, user={"kernel": "flash", "block_q_by_seq_len": {16512: 3072}}, device_kind=_V6E
    )
    self.assertEqual((flash.variant, flash.block_q), ("flash", 3072))

  def test_explicit_block_q_wins(self):
    carrier = max_utils.CustomFlashBlockSizes(kernel="auto", block_q=1024, block_q_by_seq_len=self._TABLE)
    self.assertEqual(krea2_attention.select_krea2_block_sizes(16512, user=carrier, device_kind=_V6E).block_q, 1024)

  def test_seq_len_block_q(self):
    self.assertIsNone(krea2_attention.seq_len_block_q(None, 16512))
    self.assertIsNone(krea2_attention.seq_len_block_q(max_utils.CustomFlashBlockSizes(), 16512))
    self.assertIsNone(krea2_attention.seq_len_block_q({"block_q_by_seq_len": ()}, 16512))
    self.assertEqual(krea2_attention.seq_len_block_q({"block_q_by_seq_len": self._TABLE}, 16640), 2560)
    self.assertIsNone(krea2_attention.seq_len_block_q({"block_q_by_seq_len": self._TABLE}, 4224))

  def test_carrier_reader_passes_the_table(self):
    read = attention_flax._read_custom_block_sizes  # pylint: disable=protected-access
    self.assertIsNone(read(None)["block_q_by_seq_len"])
    self.assertIsNone(read({})["block_q_by_seq_len"])
    self.assertIsNone(read(max_utils.CustomFlashBlockSizes(kernel="auto"))["block_q_by_seq_len"])
    carrier = max_utils.CustomFlashBlockSizes(kernel="auto", block_q_by_seq_len=self._TABLE)
    self.assertEqual(read(carrier)["block_q_by_seq_len"], self._TABLE)

  def test_krea2_flash_block_sizes(self):
    flash_block_sizes = generate_krea2.krea2_flash_block_sizes
    # No table (missing key, {} or a table without block_q): exactly the carrier of before.
    for extra in ({}, {_KEY: {}}, {_KEY: "{}"}, {_KEY: {16512: {"scoped_vmem_limit_kib": 65536}}}):
      with self.subTest(extra=extra):
        carrier = flash_block_sizes(_flash_config(**extra))
        self.assertEqual(carrier, max_utils.CustomFlashBlockSizes(kernel="auto"))
        self.assertEqual(hash(carrier), hash(max_utils.CustomFlashBlockSizes(kernel="auto")))
        self.assertIsNone(carrier.block_q_by_seq_len)
    table = {"16640": {"block_q": 2560}, 16512: {"block_q": 2560, "scoped_vmem_limit_kib": 65536}}
    carrier = flash_block_sizes(_flash_config(**{_KEY: table}))
    self.assertEqual(carrier, max_utils.CustomFlashBlockSizes(kernel="auto", block_q_by_seq_len=self._TABLE))
    hash(carrier)  # still a valid static (hashable) carrier
    # With configured sizes the table rides along with them.
    sized = flash_block_sizes(_flash_config(**{_KEY: table, "flash_block_sizes": {"block_q": 1024, "block_kv": 2048}}))
    self.assertEqual((sized.block_q, sized.block_kv, sized.block_q_by_seq_len), (1024, 2048, self._TABLE))
    # Other attention kernels get their carrier unchanged.
    self.assertIsNone(flash_block_sizes(_flash_config(attention="dot_product", **{_KEY: table})))
    with self.assertRaisesRegex(ValueError, _KEY):
      flash_block_sizes(_flash_config(**{_KEY: {16512: {"block_q": 1000}}}))


class BlockQVmemGuardTest(unittest.TestCase):
  """The startup warning for table block_q values whose VMEM estimate exceeds the budget."""

  def test_estimate(self):
    fits = krea2_attention.block_q_vmem_estimate_fits
    # Hybrid on v6e with its default kv sizes: block_q 2560 needs more than the 36 MiB budget, fits 64 MiB.
    self.assertTrue(fits("hybrid", _V6E, 2560, 2048, 2048, vmem_limit_bytes=65536 * 1024))
    self.assertFalse(fits("hybrid", _V6E, 2560, 2048, 2048, vmem_limit_bytes=36864 * 1024))
    self.assertFalse(fits("hybrid", _V6E, 2560, 2048, 2048))  # default 32 MiB
    self.assertTrue(fits("hybrid", _V6E, 2048, 2048, 2048))  # the automatic choice's cap fits
    self.assertIsNone(fits("hybrid", None, 2560, 2048, 2048))  # no calibrated budget
    self.assertTrue(fits("hybrid", None, 2560, 2048, 2048, vmem_limit_bytes=65536 * 1024))
    self.assertTrue(fits("flash", _V6E, 2560, 2048, 1024))
    self.assertFalse(fits("flash", _V6E, 8192, 2048, 1024))
    self.assertTrue(fits("flash", _V6E, 8192, 2048, 1024, vmem_limit_bytes=96 * 1024 * 1024))
    self.assertIsNone(fits("flash", _V6E, 2560, 2048, 1024, dtype=jnp.float32))
    self.assertIsNone(fits("flash", "cpu", 2560, 2048, 1024))
    with self.assertRaisesRegex(ValueError, "variant"):
      fits("splash", _V6E, 2560, 2048, 1024)

  def _warnings(self, table, global_kib=36864, carrier=None, device_kind=_V6E, mesh=None, **kwargs):
    return seq_len_tuning_block_q_warnings(
        _resolve(table),
        global_kib,
        carrier if carrier is not None else max_utils.CustomFlashBlockSizes(kernel="auto"),
        device_kind,
        mesh if mesh is not None else _fake_mesh("tpu"),
        **kwargs,
    )

  def test_lengths_below_flash_min_seq_length_are_not_checked(self):
    # Below flash_min_seq_length the DiT attention is dot-product: the table's block_q never reaches a kernel.
    table = {4224: {"block_q": 2560}}
    self.assertEqual(len(self._warnings(table)), 1)  # flash_custom at 4224: 36 MiB is too small for 2560
    with mock.patch.object(krea2_pipeline.max_logging, "log") as log:
      self.assertEqual(self._warnings(table, flash_min_seq_length=8192), [])
    self.assertTrue(any("unused" in str(call.args[0]) and "8192" in str(call.args[0]) for call in log.call_args_list))
    # Lengths at or above the threshold are still checked.
    self.assertEqual(len(self._warnings({16512: {"block_q": 2560}}, flash_min_seq_length=8192)), 1)
    self.assertEqual(len(self._warnings({4224: {"block_q": 2560}}, flash_min_seq_length=4224)), 1)

  def test_warns_only_when_the_estimate_exceeds_the_limit_at_that_length(self):
    self.assertEqual(self._warnings({16512: {"block_q": 2560, "scoped_vmem_limit_kib": 65536}}), [])
    (warning,) = self._warnings({16512: {"block_q": 2560}})  # the global 36 MiB applies at 16512
    self.assertIn("16512", warning)
    self.assertIn("36864 KiB", warning)
    self.assertEqual(len(self._warnings({16512: {"block_q": 2560, "scoped_vmem_limit_kib": 0}})), 1)  # default limit
    self.assertEqual(self._warnings({16512: {"block_q": 2560}}, global_kib=65536), [])
    self.assertEqual(self._warnings({16512: {"scoped_vmem_limit_kib": 0}}), [])  # no block_q, nothing to check
    # The carrier's vmem_limit_bytes is the kernel's limit.
    carrier = max_utils.CustomFlashBlockSizes(kernel="auto", vmem_limit_bytes=96 * 1024 * 1024)
    self.assertEqual(self._warnings({16512: {"block_q": 2560}}, carrier=carrier), [])
    # An explicit block_q overrides the table: nothing to check.
    carrier = max_utils.CustomFlashBlockSizes(kernel="auto", block_q=1024)
    self.assertEqual(self._warnings({16512: {"block_q": 2560}}, carrier=carrier), [])
    # No calibrated chip, or a mesh that does not get the option (the chip default applies).
    self.assertEqual(self._warnings({16512: {"block_q": 2560}}, device_kind=None), [])
    self.assertEqual(
        len(self._warnings({16512: {"block_q": 2560, "scoped_vmem_limit_kib": 65536}}, mesh=_fake_mesh("cpu"))), 1
    )
    # The flash variant uses its own (smaller) estimate.
    self.assertEqual(self._warnings({16512: {"block_q": 2560}}, carrier=max_utils.CustomFlashBlockSizes(kernel="flash")), [])

  def test_preset_table_passes_the_guard(self):
    preset = {
        "16512": {"block_q": 2560, "scoped_vmem_limit_kib": 65536},
        "16640": {"block_q": 2560, "scoped_vmem_limit_kib": 65536},
    }
    self.assertEqual(self._warnings(preset), [])


class SeqLenTuningAotMetaTest(unittest.TestCase):

  def test_no_table_adds_no_key(self):
    self.assertEqual(generate_krea2.seq_len_tuning_aot_meta(()), {})
    self.assertEqual(generate_krea2.seq_len_tuning_aot_meta(resolve_seq_len_tuning(types.SimpleNamespace())), {})
    self.assertEqual(generate_krea2.seq_len_tuning_aot_meta(_resolve("{}")), {})

  def test_table_is_recorded_canonically(self):
    meta = generate_krea2.seq_len_tuning_aot_meta(
        _resolve({"16640": {"block_q": 2560}, 16512: {"scoped_vmem_limit_kib": 65536}})
    )
    self.assertEqual(meta, {_KEY: '{"16512":{"scoped_vmem_limit_kib":65536},"16640":{"block_q":2560}}'})
    same = generate_krea2.seq_len_tuning_aot_meta(
        _resolve('{"16512": {"scoped_vmem_limit_kib": 65536}, "16640": {"block_q": 2560}}')
    )
    self.assertEqual(meta, same)


class CompileReportBlockQTest(unittest.TestCase):

  def test_attention_block_q_at(self):
    table = ((16512, 2560),)
    transformer = types.SimpleNamespace(
        attention_kernel="flash_custom",
        flash_min_seq_length=0,
        flash_block_sizes=max_utils.CustomFlashBlockSizes(kernel="hybrid", block_q_by_seq_len=table),
        mesh=None,
    )
    config = types.SimpleNamespace(activations_dtype=jnp.bfloat16)
    self.assertEqual(compile_krea2.attention_block_q_at(transformer, 16512, config), (2560, "krea2_seq_len_tuning"))
    block_q, source = compile_krea2.attention_block_q_at(transformer, 4224, config)
    self.assertEqual(source, "automatic")
    self.assertEqual(block_q, krea2_attention.select_krea2_block_sizes(4224, user={"kernel": "hybrid"}).block_q)
    transformer.flash_block_sizes = max_utils.CustomFlashBlockSizes(kernel="hybrid", block_q=1024, block_q_by_seq_len=table)
    self.assertEqual(compile_krea2.attention_block_q_at(transformer, 16512, config), (1024, "flash_block_sizes"))
    transformer.attention_kernel = "dot_product"
    self.assertEqual(compile_krea2.attention_block_q_at(transformer, 16512, config)[0], None)


class _RegistryIsolation:
  """Drops the cached_jit entries a test registered and puts the AOT cache back into its never-installed state."""

  def _isolate_registry(self):
    registered = len(aot_cache._REGISTRY)

    def restore():
      aot_cache.wait_for_loads()
      state = aot_cache._STATE
      state.enabled = False
      state.lazy_load = False
      state.warmup_only = False
      state.cache_dir = ""
      state.fingerprint = ""
      state.mesh = None
      del aot_cache._REGISTRY[registered:]
      for entry in aot_cache._REGISTRY:
        with entry._lock:
          entry._compiled.clear()
          entry._out_specs.clear()
          entry._pending.clear()
          entry._adapters.clear()
          entry._on_disk.clear()
          entry._lazy_tried.clear()

    self.addCleanup(restore)


class SeqLenDispatchPipelineTest(_RegistryIsolation, unittest.TestCase):
  """Per-L DiT entries: names, options, the dispatch by static shapes, and the empty-table pin."""

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

  def setUp(self):
    self._isolate_registry()

  def _build(self, tpu=True, **overrides):
    """(pipeline, [(name, compiler_options, donate_argnames)] of the entries it registered)."""
    registered = len(aot_cache._REGISTRY)
    if tpu:
      with mock.patch.object(krea2_pipeline, "_mesh_platform", return_value="tpu"):
        pipeline = _pipeline(self.transformer, **overrides)
    else:
      pipeline = _pipeline(self.transformer, **overrides)
    new = [(e.name, e.compiler_options, e.donate_argnames) for e in aot_cache._REGISTRY[registered:]]
    return pipeline, new

  def _step_inputs(self, pipeline):
    latents, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = self.inputs
    text_params = {key: self.params[key] for key in KREA2_TEXT_CONTEXT_KEYS}
    text_hidden = pipeline._jitted_transformer_text_context(text_params, prompt_embeds, text_mask)
    return latents, text_hidden, text_mask, img_ids, txt_ids, t_vec

  def test_dispatch_key_is_the_kernel_sequence_length(self):
    # prepare_inputs builds [image | text]: the blocks see latents + text_hidden tokens.
    pipeline, _ = self._build(tpu=False)
    latents, text_hidden, text_mask, img_ids, txt_ids, t_vec = self._step_inputs(pipeline)
    prelude_params = {key: self.params[key] for key in KREA2_PRELUDE_KEYS}
    hidden_states = self.transformer.apply(
        {"params": prelude_params},
        latents,
        text_hidden,
        t_vec,
        img_ids,
        txt_ids,
        text_mask,
        method=self.transformer.prepare_inputs,
    )[0]
    self.assertEqual(hidden_states.shape[1], latents.shape[1] + text_hidden.shape[1])
    self.assertEqual(hidden_states.shape[1], _L)

  def test_setup_passes_flash_min_seq_length_to_the_guard(self):
    transformer = self.transformer.clone(attention_kernel="flash_custom", flash_min_seq_length=8192)
    with mock.patch.object(krea2_pipeline, "seq_len_tuning_block_q_warnings", return_value=[]) as guard:
      _pipeline(transformer, **{_VMEM_KEY: 36864, _KEY: {4224: {"block_q": 2560}}})
    guard.assert_called_once()
    self.assertEqual(guard.call_args.kwargs["flash_min_seq_length"], 8192)
    # The real guard at that threshold: no warning for L 4224.
    self.assertEqual(
        seq_len_tuning_block_q_warnings(
            _resolve({4224: {"block_q": 2560}}),
            36864,
            max_utils.CustomFlashBlockSizes(kernel="auto"),
            _V6E,
            _fake_mesh("tpu"),
            flash_min_seq_length=8192,
        ),
        [],
    )

  def test_empty_table_is_todays_setup(self):
    for tpu, global_kib, staged in ((True, 36864, False), (True, 0, False), (False, 36864, False), (True, 36864, True)):
      with self.subTest(tpu=tpu, global_kib=global_kib, staged=staged):
        base = {_VMEM_KEY: global_kib, "krea2_staged_transformer": staged}
        reference, reference_entries = self._build(tpu=tpu, **base)
        names = [name for name, _, _ in reference_entries]
        if staged:
          expected_names = [
              "krea2_pipeline.qwen3_forward",
              "krea2_pipeline.transformer_text_context",
              "krea2_pipeline.transformer_prelude",
              _BLOCK,
              "krea2_pipeline.transformer_final",
          ]
        else:
          expected_names = ["krea2_pipeline.qwen3_forward", "krea2_pipeline.transformer_text_context", _STEP]
        self.assertEqual(names, expected_names)
        dit = reference._jitted_transformer_block if staged else reference._jitted_transformer_step
        self.assertIsInstance(dit, aot_cache._AotEntry)  # pylint: disable=protected-access
        expected_options = {_OPTION: global_kib} if tpu and global_kib else None
        self.assertEqual(dit.compiler_options, expected_options)
        # No table, {} / "{}" / "''", a block_q-only table, a limit equal to the global one: the same entries.
        for table in (None, {}, "{}", "''", {_L: {"block_q": 1024}}, {_L: {"scoped_vmem_limit_kib": global_kib}}):
          pipeline, entries = self._build(tpu=tpu, **base, **({} if table is None else {_KEY: table}))
          self.assertEqual(entries, reference_entries, table)
          kind = "transformer_block" if staged else "transformer_step"
          self.assertIs(pipeline.dit_entry_for_seq_len(kind, _L), getattr(pipeline, f"_jitted_{kind}"))
    # Off a TPU every limit means "no option": a table of limits builds no extra entry either.
    _, reference_entries = self._build(tpu=False, **{_VMEM_KEY: 36864})
    _, entries = self._build(tpu=False, **{_VMEM_KEY: 36864, _KEY: {_L: {"scoped_vmem_limit_kib": 65536}}})
    self.assertEqual(entries, reference_entries)

  def test_monolithic_step_dispatches_by_seq_len(self):
    reference, _ = self._build(tpu=False)
    step_inputs = self._step_inputs(reference)
    expected = np.asarray(reference._jitted_transformer_step(self.params, *step_inputs))

    # Global default, 64 MiB at this L: the call runs the optioned entry (the CPU backend rejects its option).
    pipeline, entries = self._build(**{_VMEM_KEY: 0, _KEY: {_L: {"scoped_vmem_limit_kib": 65536}}})
    self.assertEqual(
        [(name, options) for name, options, _ in entries if "transformer_step" in name],
        [(_STEP, None), (f"{_STEP}__svmem65536", {_OPTION: 65536})],
    )
    self.assertIsInstance(pipeline._jitted_transformer_step, SeqLenDispatch)
    self.assertEqual(pipeline.dit_entry_for_seq_len("transformer_step", _L).name, f"{_STEP}__svmem65536")
    self.assertEqual(pipeline.dit_entry_for_seq_len("transformer_step", _L + 1).name, _STEP)
    with self.assertRaisesRegex(Exception, _OPTION):
      pipeline._jitted_transformer_step(self.params, *step_inputs)
    # Keyword arguments dispatch the same way.
    latents, text_hidden, text_mask, img_ids, txt_ids, t_vec = step_inputs
    with self.assertRaisesRegex(Exception, _OPTION):
      pipeline._jitted_transformer_step(
          self.params,
          latents=latents,
          text_hidden=text_hidden,
          text_mask=text_mask,
          img_ids=img_ids,
          txt_ids=txt_ids,
          t_vec=t_vec,
      )

    # The table entry is for another L: this call runs the global (option-free) entry.
    pipeline, _ = self._build(**{_VMEM_KEY: 0, _KEY: {_L + 1: {"scoped_vmem_limit_kib": 65536}}})
    np.testing.assert_array_equal(np.asarray(pipeline._jitted_transformer_step(self.params, *step_inputs)), expected)

    # Global 36 MiB, the compiler's default at this L: "__svmem0" without options runs; other L keep 36 MiB.
    pipeline, entries = self._build(**{_VMEM_KEY: 36864, _KEY: {_L: {"scoped_vmem_limit_kib": 0}}})
    self.assertEqual(
        [(name, options) for name, options, _ in entries if "transformer_step" in name],
        [(_STEP, {_OPTION: 36864}), (f"{_STEP}__svmem0", None)],
    )
    np.testing.assert_array_equal(np.asarray(pipeline._jitted_transformer_step(self.params, *step_inputs)), expected)
    self.assertEqual(pipeline.dit_entry_for_seq_len("transformer_step", 16512).compiler_options, {_OPTION: 36864})
    with self.assertRaisesRegex(ValueError, "transformer_block"):
      pipeline.dit_entry_for_seq_len("transformer_block", _L)

  def test_one_entry_per_distinct_limit(self):
    table = {
        _L: {"scoped_vmem_limit_kib": 65536},
        16512: {"scoped_vmem_limit_kib": 65536, "block_q": 2560},
        16640: {"scoped_vmem_limit_kib": 49152},
        4224: {"scoped_vmem_limit_kib": 36864},  # the global value: the global entry
        4608: {"block_q": 1024},  # no limit: the global entry
    }
    pipeline, entries = self._build(**{_VMEM_KEY: 36864, _KEY: table})
    self.assertEqual(
        [(name, options) for name, options, _ in entries if "transformer_step" in name],
        [(_STEP, {_OPTION: 36864}), (f"{_STEP}__svmem65536", {_OPTION: 65536}), (f"{_STEP}__svmem49152", {_OPTION: 49152})],
    )
    entry = pipeline.dit_entry_for_seq_len
    self.assertIs(entry("transformer_step", _L), entry("transformer_step", 16512))
    self.assertEqual(entry("transformer_step", 16640).name, f"{_STEP}__svmem49152")
    for seq_len in (4224, 4608, 99):
      self.assertEqual(entry("transformer_step", seq_len).name, _STEP)

  def test_staged_block_dispatches_by_seq_len(self):
    for donate in (True, False):
      with self.subTest(donate=donate):
        base = {_VMEM_KEY: 0, "krea2_staged_transformer": True, "krea2_staged_donate_hidden_states": donate}
        reference, _ = self._build(tpu=False, **base)
        step_inputs = self._step_inputs(reference)
        expected = np.asarray(reference._jitted_transformer_step(self.params, *step_inputs))

        pipeline, entries = self._build(**base, **{_KEY: {_L: {"scoped_vmem_limit_kib": 65536}}})
        donated = ("hidden_states",) if donate else ()
        self.assertEqual(
            [e for e in entries if "transformer_block" in e[0]],
            [(_BLOCK, None, donated), (f"{_BLOCK}__svmem65536", {_OPTION: 65536}, donated)],
        )
        self.assertIsInstance(pipeline._jitted_transformer_block, SeqLenDispatch)
        # The staged step picks the block entry from the prelude's hidden states.
        with self.assertRaisesRegex(Exception, _OPTION):
          pipeline._jitted_transformer_step(self.params, *step_inputs)
        prelude_params = {key: self.params[key] for key in KREA2_PRELUDE_KEYS}
        hidden_states, _, temb_mod, rotary_emb, attention_mask = pipeline._jitted_transformer_prelude(
            prelude_params, *step_inputs
        )
        with self.assertRaisesRegex(Exception, _OPTION):
          pipeline._jitted_transformer_block(
              self.params["blocks_0"], {}, hidden_states, temb_mod, rotary_emb, attention_mask
          )

        pipeline, _ = self._build(**base, **{_KEY: {_L + 1: {"scoped_vmem_limit_kib": 65536}}})
        np.testing.assert_array_equal(np.asarray(pipeline._jitted_transformer_step(self.params, *step_inputs)), expected)

  def test_warmup_mode_on_threads_saves_under_the_per_seq_len_name(self):
    # Global 36 MiB (rejected by CPU compiles), the compiler's default at this L: only "__svmem0" can compile here.
    pipeline, _ = self._build(**{_VMEM_KEY: 36864, _KEY: {_L: {"scoped_vmem_limit_kib": 0}}})
    step_inputs = self._step_inputs(pipeline)
    with tempfile.TemporaryDirectory() as cache_dir:
      aot_cache.install(cache_dir, {"test": "krea2_seq_len_tuning"}, krea2_pipeline_mesh())
      aot_cache.wait_for_loads()
      outputs, errors = [], []

      def call():
        try:
          outputs.append(np.asarray(pipeline._jitted_transformer_step(self.params, *step_inputs)))
        except Exception as e:  # noqa: BLE001 - re-raised below on the main thread
          errors.append(e)

      with aot_cache.warmup_mode():
        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
          thread.start()
        for thread in threads:
          thread.join(timeout=600)
      self.assertEqual(errors, [])
      self.assertEqual(len(outputs), 3)
      for out in outputs:
        self.assertEqual(out.shape, (_B, _GRID_H * _GRID_W, 16))
        self.assertFalse(np.any(out))  # warmup zeros: nothing executed
      self.assertGreaterEqual(aot_cache.save_pending(), 1)
      self.assertEqual(len(glob.glob(os.path.join(cache_dir, f"{_STEP}__svmem0-*.aotx"))), 1)
      self.assertEqual(glob.glob(os.path.join(cache_dir, f"{_STEP}-*.aotx")), [])

      # A later run (fresh entries, lazy install) loads that file into the per-L entry and runs it.
      run, _ = self._build(**{_VMEM_KEY: 36864, _KEY: {_L: {"scoped_vmem_limit_kib": 0}}})
      aot_cache.install(cache_dir, {"test": "krea2_seq_len_tuning"}, krea2_pipeline_mesh(), lazy_load=True)
      run_inputs = self._step_inputs(run)
      out = np.asarray(run._jitted_transformer_step(self.params, *run_inputs))
      self.assertEqual(len(run.dit_entry_for_seq_len("transformer_step", _L)._on_disk), 1)
      self.assertEqual(out.shape, (_B, _GRID_H * _GRID_W, 16))


def krea2_pipeline_mesh():
  return jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))


if __name__ == "__main__":
  unittest.main()
