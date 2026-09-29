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

# CPU tests for the Krea 2 quantized-weight cache (krea2_weight_cache_dir): the
# on-disk round trip, the quantized text encoder and W8A8 transformer trees
# (bit-identical leaves and forward outputs), fingerprints, misses, saving and
# the generate_krea2 helpers around it (float32 activations throughout).

import collections
import importlib.metadata
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import flax
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import linen as nn

from maxdiffusion.generate_krea2 import load_or_build_host_params, resolve_weight_cache_dirs, save_host_params
from maxdiffusion.models.krea2 import weight_cache
from maxdiffusion.models.krea2.text_encoder_quant import (
    KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION,
    quantize_text_encoder_model,
    quantize_text_encoder_params,
)
from maxdiffusion.models.krea2.transformer_quant import (
    KREA2_DEFAULT_QUANT_TARGETS,
    KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION,
    check_transformer_param_tree,
    quantize_transformer_params,
)
from maxdiffusion.models.krea2.util import KREA2_ROPE_PERMUTATION_REVISION, permute_rope_weights_to_rotate_half
from maxdiffusion.models.krea2.weight_cache import (
    KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION,
    KREA2_TRANSFORMER_WEIGHT_BUILD_REVISION,
    KREA2_WEIGHT_CACHE_FORMAT,
    WeightCacheSpec,
    component_dir,
    list_source_files,
    load_component,
    save_component,
    text_encoder_weight_cache_meta,
    transformer_weight_cache_meta,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Model
from maxdiffusion.tests.krea2_text_encoder_quant_test import (
    _abstract_quantized,
    _config as _qwen_config,
    _init_params as _qwen_init_params,
)
from maxdiffusion.tests.krea2_transformer_quant_pipeline_test import (
    _abstract_params,
    _model_args,
    _random_float_params,
    _tiny_transformer,
)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
_META = {"model": "krea/krea-2", "snapshot": "abc123", "weights_dtype": "bfloat16", "quant_targets": ["to_q"]}


def _abstract(tree):
  return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(np.shape(x), np.asarray(x).dtype), tree)


def _mixed_tree(seed=0):
  rng = np.random.RandomState(seed)
  return {
      "block": {
          "kernel": rng.randint(-127, 128, size=(7, 3)).astype(np.int8),
          "scale": rng.randn(3).astype(jnp.bfloat16),
      },
      "bias": rng.randn(33).astype(np.float32),
      "scalar": np.asarray(rng.randn(), np.float32),
      "empty": np.zeros((0, 4), jnp.bfloat16),
      # A non-contiguous view, like the checkpoint loader's transposed kernels.
      "view": rng.randn(6, 5).astype(np.float32).T,
  }


def _log_lines(log_mock):
  return [call.args[0] for call in log_mock.call_args_list if str(call.args[0]).startswith("[weight cache]")]


class _CacheTestCase(unittest.TestCase):
  """A temporary cache directory plus comparison and logging helpers."""

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.cache_dir = os.path.join(self._tmp.name, "cache")

  def tearDown(self):
    self._tmp.cleanup()

  def assert_bit_identical(self, got, expected):
    self.assertEqual(jax.tree_util.tree_structure(got), jax.tree_util.tree_structure(expected))
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(expected)):
      a, b = np.asarray(a), np.asarray(b)
      self.assertEqual(a.dtype, b.dtype)
      self.assertEqual(a.shape, b.shape)
      self.assertEqual(np.ascontiguousarray(a).tobytes(), np.ascontiguousarray(b).tobytes())

  def load(self, abstract, **kwargs):
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      result = load_component(self.cache_dir, "transformer", _META, abstract, **kwargs)
    return result, _log_lines(log)


class WeightCacheRoundTripTest(_CacheTestCase):

  def test_mixed_tree_round_trip(self):
    tree = _mixed_tree()
    table = np.arange(12, dtype=np.float32).reshape(3, 4).astype(jnp.bfloat16)
    source_files = [["model.safetensors", 123]]
    saved = save_component(self.cache_dir, "transformer", _META, tree, extras={"table": table}, source_files=source_files)
    self.assertEqual(saved, component_dir(self.cache_dir, "transformer", _META))
    self.assertEqual(sorted(os.listdir(saved)), ["meta.json", "weights.bin"])
    with open(os.path.join(saved, "meta.json"), encoding="utf-8") as f:
      header = json.load(f)
    self.assertEqual(header["format"], KREA2_WEIGHT_CACHE_FORMAT)
    self.assertEqual(header["byteorder"], "little")
    self.assertEqual(header["total_bytes"], os.path.getsize(os.path.join(saved, "weights.bin")))
    for entry in header["index"] + header["extras"]:
      self.assertEqual(entry["offset"] % 64, 0)

    (loaded, extras), lines = self.load(_abstract(tree), extra_names=("table",), source_files=source_files)
    self.assertEqual(len(lines), 1)
    self.assertIn("loaded", lines[0])
    self.assert_bit_identical(loaded, tree)
    self.assert_bit_identical(extras, {"table": table})
    for leaf in jax.tree_util.tree_leaves(loaded) + [extras["table"]]:
      self.assertIsInstance(leaf, np.ndarray)
      self.assertTrue(leaf.flags.writeable)
    self.assertEqual(loaded["scalar"].shape, ())
    np.testing.assert_array_equal(loaded["view"], tree["view"])

  def test_frozen_dict_abstract_gives_plain_dicts(self):
    tree = _mixed_tree()
    save_component(self.cache_dir, "transformer", _META, tree)
    (loaded, extras), _ = self.load(flax.core.freeze(_abstract(tree)))
    self.assertIsInstance(loaded, dict)
    self.assertIsInstance(loaded["block"], dict)
    self.assertEqual(extras, {})
    self.assert_bit_identical(loaded, tree)

  def test_list_source_files(self):
    source = os.path.join(self._tmp.name, "transformer")
    os.makedirs(source)
    self.assertIsNone(list_source_files(source))
    self.assertIsNone(list_source_files(os.path.join(self._tmp.name, "missing")))
    for name, size in (("b.safetensors", 5), ("a.safetensors", 3), ("config.json", 7)):
      with open(os.path.join(source, name), "wb") as f:
        f.write(b"x" * size)
    self.assertEqual(list_source_files(source), [["a.safetensors", 3], ["b.safetensors", 5]])
    # A file instead of a directory is a confirmed absence too.
    self.assertIsNone(list_source_files(os.path.join(source, "config.json")))


class ListSourceFilesFailedScanTest(_CacheTestCase):
  """A failed checkpoint scan is not "no checkpoint": its list never matches a stored one."""

  def setUp(self):
    super().setUp()
    self.source = os.path.join(self._tmp.name, "snapshot")
    os.makedirs(self.source)

  def _symlink(self, name):
    os.symlink(os.path.join(self._tmp.name, "blobs", name), os.path.join(self.source, name))

  def test_dangling_symlink_is_kept_without_a_size(self):
    self._symlink("a.safetensors")
    with open(os.path.join(self.source, "b.safetensors"), "wb") as f:
      f.write(b"x" * 5)
    self.assertEqual(list_source_files(self.source), [["a.safetensors", None], ["b.safetensors", 5]])

  def test_all_entries_dangling_give_a_list(self):
    self._symlink("a.safetensors")
    self._symlink("b.safetensors")
    self.assertEqual(list_source_files(self.source), [["a.safetensors", None], ["b.safetensors", None]])

  def test_unlistable_directory(self):
    with mock.patch.object(weight_cache.os, "listdir", side_effect=PermissionError("denied")):
      self.assertEqual(list_source_files(self.source), [["<unreadable>", None]])

  def test_failed_scan_misses_against_real_sizes(self):
    tree = _mixed_tree()
    source_files = [["a.safetensors", 3], ["b.safetensors", 5]]
    self.assertIsNotNone(save_component(self.cache_dir, "transformer", _META, tree, source_files=source_files))
    for scan in ([["<unreadable>", None]], [["a.safetensors", None], ["b.safetensors", 5]]):
      with self.subTest(scan=scan):
        result, lines = self.load(_abstract(tree), source_files=scan)
        self.assertIsNone(result)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("source checkpoint changed", lines[0])
    self.assertIsNotNone(self.load(_abstract(tree), source_files=source_files)[0])

  def test_failed_scan_misses_against_an_equal_failed_scan(self):
    tree = _mixed_tree()
    scan = [["a.safetensors", None]]
    self.assertIsNotNone(save_component(self.cache_dir, "transformer", _META, tree, source_files=scan))
    result, lines = self.load(_abstract(tree), source_files=scan)
    self.assertIsNone(result)
    self.assertIn("source checkpoint changed", lines[0])


class WeightCacheTextEncoderTest(_CacheTestCase):
  """The qwix int8 tree with the embedding table on the host, as generate_krea2 builds it."""

  def setUp(self):
    super().setUp()
    self.config = _qwen_config()
    self.model = FlaxQwen3Model(self.config)
    self.qmodel = quantize_text_encoder_model(self.model, 32)
    rng = np.random.RandomState(0)
    self.ids = jnp.asarray(rng.randint(0, 100, size=(2, 12)), dtype=jnp.int32)
    self.mask = jnp.asarray(np.concatenate([np.ones((2, 9)), np.zeros((2, 3))], axis=1), dtype=jnp.int32)
    params = jax.tree_util.tree_map(np.asarray, _qwen_init_params(self.model, self.ids, self.mask))
    self.table = params.pop("embed_tokens")["embedding"]
    self.float_params = params
    # Boxed (LogicallyPartitioned leaves), like main's abstract_qwen3_vars["params"].
    self.abstract = _abstract_quantized(self.qmodel, self.ids, self.mask, embeds_dim=self.config.hidden_size)
    boxed = jax.tree_util.tree_leaves(self.abstract, is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned))
    self.assertTrue(any(isinstance(leaf, nn.LogicallyPartitioned) for leaf in boxed))

  def _forward(self, params):
    embeds = jnp.asarray(self.table[np.asarray(self.ids)])
    position_ids = jnp.clip(jnp.cumsum(self.mask, axis=-1) - 1, 0, None)
    return self.qmodel.apply({"params": params}, None, self.mask, inputs_embeds=embeds, position_ids=position_ids)

  def test_round_trip_is_bit_identical_with_identical_forward(self):
    fresh = quantize_text_encoder_params(self.float_params, self.abstract)
    save_component(self.cache_dir, "text_encoder", _META, fresh, extras={"embedding_table": self.table})
    with mock.patch.object(weight_cache.max_logging, "log"):
      loaded, extras = load_component(
          self.cache_dir, "text_encoder", _META, self.abstract, extra_names=("embedding_table",), check_dtypes=False
      )
    self.assert_bit_identical(loaded, fresh)
    self.assert_bit_identical(extras["embedding_table"], self.table)
    ref_last, ref_all = self._forward(fresh)
    got_last, got_all = self._forward(loaded)
    for got, ref in zip(got_all + [got_last], ref_all + [ref_last]):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_compute_dtype_scales_need_check_dtypes_off(self):
    # Production stores bf16 scales while the abstract tree's scales are float32.
    fresh = quantize_text_encoder_params(self.float_params, self.abstract, scale_dtype=jnp.bfloat16)
    save_component(self.cache_dir, "text_encoder", _META, fresh)
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      self.assertIsNone(load_component(self.cache_dir, "text_encoder", _META, self.abstract))
      self.assertIn("dtype", _log_lines(log)[0])
      loaded, _ = load_component(self.cache_dir, "text_encoder", _META, self.abstract, check_dtypes=False)
    self.assert_bit_identical(loaded, fresh)
    scale = loaded["layers_0"]["mlp"]["down_proj"]["kernel"].array.scale
    self.assertEqual(scale.dtype, jnp.bfloat16)


class WeightCacheTransformerTest(_CacheTestCase):
  """The W8A8 tree after the rotate-half permutation, as generate_krea2 builds it."""

  def test_round_trip_is_bit_identical_with_identical_forward(self):
    model = _tiny_transformer(rope_layout="rotate_half")
    abstract = _abstract_params(model)
    params = permute_rope_weights_to_rotate_half(
        _random_float_params(),
        num_heads=model.num_attention_heads,
        num_kv_heads=model.num_key_value_heads,
        head_dim=model.attention_head_dim,
    )
    fresh = quantize_transformer_params(params, KREA2_DEFAULT_QUANT_TARGETS, scale_dtype=jnp.float32)
    check_transformer_param_tree(fresh, abstract)
    save_component(self.cache_dir, "transformer", _META, fresh)
    (loaded, _), _ = self.load(abstract, check_dtypes=True)
    check_transformer_param_tree(loaded, abstract)
    self.assert_bit_identical(loaded, fresh)
    self.assertEqual(loaded["blocks_0"]["attn"]["to_q"]["kernel"].dtype, np.int8)
    expected = model.apply({"params": fresh}, *_model_args()).sample
    got = model.apply({"params": loaded}, *_model_args()).sample
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


class WeightCacheFingerprintTest(unittest.TestCase):

  def test_every_meta_key_changes_the_directory(self):
    config = types.SimpleNamespace(pretrained_model_name_or_path="krea/krea-2", weights_dtype=jnp.bfloat16)
    meta = transformer_weight_cache_meta(
        config, "/hf/snapshots/abc123", "w8a8", ("to_q", "to_out"), "rotate_half", 48, 12, 128
    )
    base = component_dir("/c", "transformer", meta)
    self.assertEqual(base, component_dir("/c", "transformer", dict(reversed(list(meta.items())))))
    self.assertRegex(os.path.basename(base), r"^transformer-[0-9a-f]{12}$")
    changed = {
        "model": "other/model",
        "snapshot": "def456",
        "weights_dtype": "float32",
        "quantization": "other",
        "quant_targets": ["to_q"],
        "weight_quant_revision": meta["weight_quant_revision"] + 1,
        "rope_layout": "interleaved",
        "rope_permutation_revision": meta["rope_permutation_revision"] + 1,
        "num_attention_heads": 24,
        "num_key_value_heads": 6,
        "attention_head_dim": 64,
        "build_revision": meta["build_revision"] + 1,
    }
    self.assertEqual(set(changed), set(meta))
    for key, value in changed.items():
      self.assertNotEqual(component_dir("/c", "transformer", {**meta, key: value}), base, key)
    self.assertNotEqual(component_dir("/c", "text_encoder", meta), base)

  def test_every_text_encoder_meta_key_changes_the_directory(self):
    config = types.SimpleNamespace(pretrained_model_name_or_path="krea/krea-2", weights_dtype=jnp.bfloat16)
    meta = text_encoder_weight_cache_meta(config, "/hf/snapshots/abc123", "int8", 128, True, jnp.bfloat16)
    base = component_dir("/c", "text_encoder", meta)
    self.assertEqual(base, component_dir("/c", "text_encoder", dict(reversed(list(meta.items())))))
    changed = {
        "model": "other/model",
        "snapshot": "def456",
        "weights_dtype": "float32",
        "quantization": "other",
        "tile_size": 64,
        "embed_on_host": False,
        "scale_dtype": "float32",
        "qwix": "0.0.0",
        "weight_quant_revision": meta["weight_quant_revision"] + 1,
        "build_revision": meta["build_revision"] + 1,
    }
    self.assertEqual(set(changed), set(meta))
    for key, value in changed.items():
      self.assertNotEqual(component_dir("/c", "text_encoder", {**meta, key: value}), base, key)


class WeightCacheMissTest(_CacheTestCase):

  def setUp(self):
    super().setUp()
    self.tree = _mixed_tree()
    self.abstract = _abstract(self.tree)
    self.source_files = [["model.safetensors", 123]]
    self.dir = save_component(
        self.cache_dir,
        "transformer",
        _META,
        self.tree,
        extras={"table": np.ones(3, np.float32)},
        source_files=self.source_files,
    )
    self.assertIsNotNone(self.dir)

  def _header(self):
    with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
      return json.load(f)

  def _write_header(self, header):
    with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
      json.dump(header, f)

  def assert_miss(self, reason, abstract=None, **kwargs):
    result, lines = self.load(self.abstract if abstract is None else abstract, **kwargs)
    self.assertIsNone(result)
    self.assertEqual(len(lines), 1, lines)
    self.assertIn("miss", lines[0])
    self.assertIn(reason, lines[0])

  def assert_hit(self, abstract=None, **kwargs):
    result, lines = self.load(self.abstract if abstract is None else abstract, **kwargs)
    self.assertIsNotNone(result, lines)
    self.assert_bit_identical(result[0], self.tree)
    return lines

  def test_valid_cache_hits(self):
    lines = self.assert_hit(source_files=self.source_files)
    self.assertEqual(len(lines), 1)

  def test_no_directory(self):
    os.rename(self.dir, self.dir + "-renamed")
    self.assert_miss("no cache directory")

  def test_temporary_directory_only(self):
    os.rename(self.dir, f"{self.dir}.tmp-123-abcd")
    self.assert_miss("no cache directory")

  def test_meta_json_missing(self):
    os.remove(os.path.join(self.dir, "meta.json"))
    self.assert_miss("no meta.json")

  def test_meta_json_not_json(self):
    with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
      f.write("{not json")
    self.assert_miss("unreadable meta.json")

  def test_meta_json_malformed(self):
    header = self._header()
    del header["index"][0]["offset"]
    self._write_header(header)
    self.assert_miss("malformed meta.json")

  def test_other_format(self):
    header = self._header()
    header["format"] = KREA2_WEIGHT_CACHE_FORMAT + 1
    self._write_header(header)
    self.assert_miss("format")

  def test_other_component(self):
    header = self._header()
    header["component"] = "text_encoder"
    self._write_header(header)
    self.assert_miss("component")

  def test_stored_meta_differs_under_the_same_name(self):
    header = self._header()
    header["meta"]["model"] = "other/model"
    self._write_header(header)
    self.assert_miss("fingerprint inputs differ")

  def test_truncated_weights(self):
    path = os.path.join(self.dir, "weights.bin")
    os.truncate(path, os.path.getsize(path) - 1)
    self.assert_miss("weights.bin has")

  def test_appended_bytes(self):
    with open(os.path.join(self.dir, "weights.bin"), "ab") as f:
      f.write(b"\0")
    self.assert_miss("weights.bin has")

  def test_weights_missing(self):
    os.remove(os.path.join(self.dir, "weights.bin"))
    self.assert_miss("no weights.bin")

  def test_index_entry_beyond_the_file(self):
    header = self._header()
    header["index"][-1]["offset"] = header["total_bytes"]
    self._write_header(header)
    self.assert_miss("outside weights.bin")

  def test_duplicate_path(self):
    original = self._header()
    for what in ("index", "extras"):
      with self.subTest(what):
        header = json.loads(json.dumps(original))
        header[what].append(dict(header[what][0]))
        self._write_header(header)
        self.assert_miss(f"duplicate path {header[what][0]['path']} in {what}")

  def test_offsets_swapped_between_equal_shaped_entries(self):
    # Without the canonical layout check both leaves would read the other's bytes.
    tree = {"a": np.full(4, 1.0, np.float32), "b": np.full(4, 2.0, np.float32)}
    self.assertIsNotNone(save_component(self.cache_dir, "transformer", _META, tree))
    header = self._header()
    first, second = header["index"]
    first["offset"], second["offset"] = second["offset"], first["offset"]
    self._write_header(header)
    self.assert_miss("canonical layout", abstract=_abstract(tree))

  def test_offset_shifted_by_one_alignment_unit(self):
    # A gap before the last leaf, with total_bytes and the file size adjusted to match.
    header = self._header()
    header["extras"][-1]["offset"] += 64
    header["total_bytes"] += 64
    self._write_header(header)
    with open(os.path.join(self.dir, "weights.bin"), "ab") as f:
      f.write(b"\0" * 64)
    self.assert_miss("canonical layout", extra_names=("table",))

  def test_malformed_numbers_types_and_dtypes(self):
    def set_shape(header):
      header["index"][0]["shape"] = [-1]
      header["index"][0]["nbytes"] = -4

    cases = {
        "negative shape and nbytes": set_shape,
        "float offset": lambda header: header["index"][0].update(offset=float(header["index"][0]["offset"])),
        "string nbytes": lambda header: header["index"][0].update(nbytes=str(header["index"][0]["nbytes"])),
        "bool shape dimension": lambda header: header["index"][0].update(shape=[True] * len(header["index"][0]["shape"])),
        "float total_bytes": lambda header: header.update(total_bytes=float(header["total_bytes"])),
        "unknown dtype": lambda header: header["index"][0].update(dtype="float42"),
        "object dtype": lambda header: header["index"][0].update(dtype="object"),
        "index not a list": lambda header: header.update(index={"a": 1}),
        "entry not a dict": lambda header: header["index"].__setitem__(0, "bias"),
        "path not a string": lambda header: header["index"][0].update(path=1),
        "header not an object": None,
    }
    original = self._header()
    for name, change in cases.items():
      with self.subTest(name):
        header = json.loads(json.dumps(original))
        if change is None:
          header = [header]
        else:
          change(header)
        self._write_header(header)
        self.assert_miss("malformed meta.json")

  def test_failed_allocation(self):
    with mock.patch.object(weight_cache.np, "empty", side_effect=MemoryError("out of memory")):
      self.assert_miss("allocation failed")

  def test_missing_and_unexpected_paths(self):
    self.assert_miss("not cached", abstract={**self.abstract, "new": jax.ShapeDtypeStruct((2,), np.float32)})
    self.assert_miss("not in the model", abstract={k: v for k, v in self.abstract.items() if k != "bias"})

  def test_shape_mismatch(self):
    self.assert_miss("shape", abstract={**self.abstract, "bias": jax.ShapeDtypeStruct((34,), np.float32)})

  def test_dtype_mismatch(self):
    abstract = {**self.abstract, "bias": jax.ShapeDtypeStruct((33,), jnp.bfloat16)}
    self.assert_miss("dtype", abstract=abstract, check_dtypes=True)
    self.assert_hit(abstract=abstract, check_dtypes=False)

  def test_relaxed_dtype_check(self):
    # check_dtypes=False: a floating abstract leaf takes any floating width, other leaves their exact dtype.
    scale_f32 = {**self.abstract, "block": {**self.abstract["block"], "scale": jax.ShapeDtypeStruct((3,), np.float32)}}
    self.assert_hit(abstract=scale_f32, check_dtypes=False)
    kernel_f32 = {**self.abstract, "block": {**self.abstract["block"], "kernel": jax.ShapeDtypeStruct((7, 3), np.float32)}}
    self.assert_miss("dtype", abstract=kernel_f32, check_dtypes=False)
    # A stored uint8 kernel for the model's int8 kernel.
    tree = {**self.tree, "block": {**self.tree["block"], "kernel": self.tree["block"]["kernel"].view(np.uint8)}}
    self.assertIsNotNone(save_component(self.cache_dir, "transformer", _META, tree))
    self.assert_miss("dtype", check_dtypes=False)

  def test_missing_extra(self):
    self.assert_miss("no extra array 'other'", extra_names=("table", "other"))

  def test_source_files_differ(self):
    self.assert_miss("source checkpoint changed", source_files=[["model.safetensors", 124]])

  def test_no_source_files_trusts_the_cache(self):
    lines = self.assert_hit(source_files=None)
    self.assertEqual(len(lines), 2)
    self.assertIn("trusted", lines[0])

  def test_short_read(self):
    with mock.patch.object(weight_cache, "_read_into", side_effect=weight_cache._ShortRead("3 bytes missing")):
      self.assert_miss("read failed")

  def test_thread_pool_cannot_start(self):
    with (
        mock.patch.object(weight_cache, "ThreadPoolExecutor", side_effect=RuntimeError("can't start new thread")),
        mock.patch.object(weight_cache.os, "close", wraps=os.close) as close,
    ):
      self.assert_miss("can't start new thread")
    # The weights file descriptor is closed although no read ran.
    close.assert_called_once()

  def test_meta_json_read_out_of_memory(self):
    with mock.patch.object(weight_cache.json, "load", side_effect=MemoryError()):
      self.assert_miss("unreadable meta.json")

  def test_weights_open_fails(self):
    with mock.patch.object(weight_cache.os, "open", side_effect=OSError("I/O error")):
      self.assert_miss("read failed")

  def test_resource_errors_in_every_stage(self):
    cases = {
        "weight_cache_fingerprint": ("header read failed", MemoryError()),
        "_stored_leaves": ("header read failed", MemoryError()),
        "_match_leaves": ("index check failed", MemoryError()),
        "_read_arrays": ("read failed", RuntimeError("cannot schedule new futures after shutdown")),
    }
    for name, (reason, error) in cases.items():
      with self.subTest(name), mock.patch.object(weight_cache, name, side_effect=error):
        self.assert_miss(reason)
    with mock.patch.object(weight_cache.jax.tree_util, "tree_unflatten", side_effect=MemoryError()):
      self.assert_miss("tree reconstruction failed")

  def test_abstract_tree_errors_raise(self):
    # A broken abstract tree is a programming error, not a miss.
    with mock.patch.object(weight_cache.jax.tree_util, "tree_flatten_with_path", side_effect=TypeError("bad tree")):
      with mock.patch.object(weight_cache.max_logging, "log"), self.assertRaises(TypeError):
        load_component(self.cache_dir, "transformer", _META, self.abstract)


class WeightCacheSaveTest(_CacheTestCase):

  def _save(self, tree, **kwargs):
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      result = save_component(self.cache_dir, "transformer", _META, tree, **kwargs)
    return result, _log_lines(log)

  def _leftovers(self):
    return sorted(os.listdir(self.cache_dir)) if os.path.isdir(self.cache_dir) else []

  def test_replaces_an_invalid_directory(self):
    tree = _mixed_tree()
    directory, _ = self._save(tree)
    os.truncate(os.path.join(directory, "weights.bin"), 10)
    result, _ = self.load(_abstract(tree))
    self.assertIsNone(result)
    self.assertEqual(self._save(tree)[0], directory)
    (loaded, _), _ = self.load(_abstract(tree))
    self.assert_bit_identical(loaded, tree)
    self.assertEqual(self._leftovers(), [os.path.basename(directory)])

  def test_replaces_a_directory_with_other_leaves(self):
    # Same fingerprint, but the leaves no longer match the model (a miss on load).
    old = {"bias": np.zeros(4, np.float32)}
    directory, _ = self._save(old)
    tree = _mixed_tree()
    self.assertIsNone(self.load(_abstract(tree))[0])
    self.assertEqual(self._save(tree)[0], directory)
    (loaded, _), _ = self.load(_abstract(tree))
    self.assert_bit_identical(loaded, tree)

  @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file permissions")
  def test_replaces_a_directory_with_unreadable_weights(self):
    tree = _mixed_tree()
    directory, _ = self._save(tree)
    os.chmod(os.path.join(directory, "weights.bin"), 0o000)
    result, lines = self.load(_abstract(tree))
    self.assertIsNone(result)
    self.assertEqual(len(lines), 1, lines)
    self.assertIn("read failed", lines[0])
    self.assertEqual(self._save(tree)[0], directory)
    (loaded, _), _ = self.load(_abstract(tree))
    self.assert_bit_identical(loaded, tree)
    self.assertEqual(self._leftovers(), [os.path.basename(directory)])

  def test_keeps_a_valid_directory(self):
    directory, _ = self._save(_mixed_tree(seed=0))
    with open(os.path.join(directory, "weights.bin"), "rb") as f:
      before = f.read()
    result, lines = self._save(_mixed_tree(seed=1))
    self.assertEqual(result, directory)
    self.assertIn("kept", lines[0])
    with open(os.path.join(directory, "weights.bin"), "rb") as f:
      self.assertEqual(f.read(), before)
    self.assertEqual(self._leftovers(), [os.path.basename(directory)])

  def test_insufficient_free_space_writes_nothing(self):
    usage = collections.namedtuple("usage", "total used free")(10**12, 10**12, 1024)
    with mock.patch.object(weight_cache.shutil, "disk_usage", return_value=usage):
      result, lines = self._save(_mixed_tree())
    self.assertIsNone(result)
    self.assertEqual(len(lines), 1)
    self.assertIn("free", lines[0])
    self.assertEqual(self._leftovers(), [])

  def test_write_error_leaves_nothing(self):
    with mock.patch.object(weight_cache.os, "fsync", side_effect=OSError("disk full")):
      result, lines = self._save(_mixed_tree())
    self.assertIsNone(result)
    self.assertIn("disk full", lines[0])
    self.assertEqual(self._leftovers(), [])

  def test_memory_error_leaves_nothing(self):
    with mock.patch.object(weight_cache.np, "ascontiguousarray", side_effect=MemoryError()):
      result, lines = self._save(_mixed_tree())
    self.assertIsNone(result)
    self.assertIn("MemoryError", lines[0])
    self.assertEqual(self._leftovers(), [])

  def test_fingerprint_memory_error_leaves_nothing(self):
    with mock.patch.object(weight_cache, "weight_cache_fingerprint", side_effect=MemoryError()):
      result, lines = self._save(_mixed_tree())
    self.assertIsNone(result)
    self.assertEqual(len(lines), 1)
    # The directory name is unknown, so the warning names the cache directory.
    self.assertIn(f"could not save to {self.cache_dir}: MemoryError", lines[0])
    self.assertEqual(self._leftovers(), [])

  def test_uncreatable_cache_dir(self):
    blocker = os.path.join(self._tmp.name, "file")
    with open(blocker, "w", encoding="utf-8") as f:
      f.write("x")
    with mock.patch.object(weight_cache.max_logging, "log"):
      self.assertIsNone(save_component(os.path.join(blocker, "cache"), "transformer", _META, _mixed_tree()))

  @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores directory permissions")
  def test_unwritable_cache_dir(self):
    os.makedirs(self.cache_dir)
    os.chmod(self.cache_dir, 0o500)
    try:
      result, lines = self._save(_mixed_tree())
    finally:
      os.chmod(self.cache_dir, 0o700)
    self.assertIsNone(result)
    self.assertIn("could not save", lines[0])
    self.assertEqual(self._leftovers(), [])


class GenerateWeightCacheHelpersTest(_CacheTestCase):
  """The helpers generate_krea2.main uses to decide, read and write the cache."""

  def _config(self, cache_dir):
    return types.SimpleNamespace(krea2_weight_cache_dir=cache_dir)

  def test_cache_off(self):
    with mock.patch("maxdiffusion.generate_krea2.max_logging.log") as log:
      for config in (types.SimpleNamespace(), self._config(""), self._config("''"), self._config(None)):
        self.assertEqual(resolve_weight_cache_dirs(config, "w8a8", "int8", ()), ("", ""))
    log.assert_not_called()
    build = mock.Mock(return_value=(_mixed_tree(), None))
    trace = {}
    tree, extras, hit = load_or_build_host_params(None, None, build, trace, "transformer_cache_read")
    build.assert_called_once_with()
    self.assertFalse(hit)
    self.assertIsNone(extras)
    self.assertIsNone(save_host_params(None, tree, trace, "transformer_cache_write"))
    self.assertEqual(trace, {})
    self.assertFalse(os.path.exists(self.cache_dir))

  def test_decisions(self):
    config = self._config(self.cache_dir)
    self.assertEqual(resolve_weight_cache_dirs(config, "w8a8", "int8", ()), (self.cache_dir, self.cache_dir))
    with mock.patch("maxdiffusion.generate_krea2.max_logging.log") as log:
      self.assertEqual(resolve_weight_cache_dirs(config, "w8a8", "int8", (("a", 1.0),)), ("", self.cache_dir))
    self.assertIn("LoRA", log.call_args.args[0])
    with mock.patch("maxdiffusion.generate_krea2.max_logging.log") as log:
      self.assertEqual(resolve_weight_cache_dirs(config, "", "int8", ()), ("", self.cache_dir))
      self.assertEqual(resolve_weight_cache_dirs(config, "w8a8", "", ()), (self.cache_dir, ""))
    self.assertEqual(len(log.call_args_list), 2)
    for call in log.call_args_list:
      self.assertIn("nothing to cache", call.args[0])

  def test_miss_builds_once_and_saves_then_hit_skips_the_build(self):
    tree = _mixed_tree()
    table = np.arange(6, dtype=np.float32)
    spec = WeightCacheSpec(self.cache_dir, "text_encoder", _META, [["model.safetensors", 1]])
    build = mock.Mock(return_value=(tree, {"embedding_table": table}))
    trace = {}
    with mock.patch.object(weight_cache.max_logging, "log"):
      got, extras, hit = load_or_build_host_params(
          spec, _abstract(tree), build, trace, "qwen_cache_read", extra_names=("embedding_table",)
      )
      build.assert_called_once_with()
      self.assertFalse(hit)
      self.assertIs(got, tree)
      self.assertIsNotNone(save_host_params(spec, got, trace, "qwen_cache_write", extras=extras))
    self.assertEqual(set(trace), {"qwen_cache_read", "qwen_cache_write"})

    # The build stands for the checkpoint read, the permutation and the quantization.
    trace = {}
    build = mock.Mock(side_effect=AssertionError("the build must not run on a hit"))
    with mock.patch.object(weight_cache.max_logging, "log"):
      got, extras, hit = load_or_build_host_params(
          spec, _abstract(tree), build, trace, "qwen_cache_read", extra_names=("embedding_table",)
      )
    build.assert_not_called()
    self.assertTrue(hit)
    self.assert_bit_identical(got, tree)
    self.assert_bit_identical(extras["embedding_table"], table)
    self.assertEqual(set(trace), {"qwen_cache_read"})

  def test_lora_bypasses_only_the_transformer_cache(self):
    transformer_dir, text_encoder_dir = resolve_weight_cache_dirs(
        self._config(self.cache_dir), "w8a8", "int8", (("adapter", 1.0),)
    )
    specs = {
        "transformer": WeightCacheSpec(transformer_dir, "transformer", _META, None) if transformer_dir else None,
        "text_encoder": WeightCacheSpec(text_encoder_dir, "text_encoder", _META, None) if text_encoder_dir else None,
    }
    self.assertIsNone(specs["transformer"])
    trace = {}
    tree = _mixed_tree()
    with mock.patch.object(weight_cache.max_logging, "log"):
      for component, spec in specs.items():
        build = mock.Mock(return_value=(tree, None))
        got, _, hit = load_or_build_host_params(spec, _abstract(tree), build, trace, f"{component}_cache_read")
        build.assert_called_once_with()
        self.assertFalse(hit)
        save_host_params(spec, got, trace, f"{component}_cache_write")
    self.assertEqual(set(trace), {"text_encoder_cache_read", "text_encoder_cache_write"})
    self.assertEqual([name.split("-")[0] for name in os.listdir(self.cache_dir)], ["text_encoder"])


class WeightCacheMetaTest(unittest.TestCase):

  def setUp(self):
    self.config = types.SimpleNamespace(pretrained_model_name_or_path="krea/krea-2", weights_dtype=jnp.bfloat16)

  def test_transformer_meta_keys(self):
    meta = transformer_weight_cache_meta(
        self.config, "/hf/snapshots/abc123/", "w8a8", ("to_q", "to_out"), "rotate_half", 48, 12, 128
    )
    self.assertEqual(
        meta,
        {
            "model": "krea/krea-2",
            "snapshot": "abc123",
            "weights_dtype": "bfloat16",
            "quantization": "w8a8",
            "quant_targets": ["to_q", "to_out"],
            "weight_quant_revision": KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION,
            "rope_layout": "rotate_half",
            "rope_permutation_revision": KREA2_ROPE_PERMUTATION_REVISION,
            "num_attention_heads": 48,
            "num_key_value_heads": 12,
            "attention_head_dim": 128,
            "build_revision": KREA2_TRANSFORMER_WEIGHT_BUILD_REVISION,
        },
    )

  def test_text_encoder_meta_keys(self):
    meta = text_encoder_weight_cache_meta(self.config, "/hf/snapshots/abc123", "int8", 128, True, jnp.bfloat16)
    self.assertEqual(
        meta,
        {
            "model": "krea/krea-2",
            "snapshot": "abc123",
            "weights_dtype": "bfloat16",
            "quantization": "int8",
            "tile_size": 128,
            "embed_on_host": True,
            "scale_dtype": "bfloat16",
            "qwix": importlib.metadata.version("qwix"),
            "weight_quant_revision": KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION,
            "build_revision": KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION,
        },
    )

  def test_configs_define_the_key_off(self):
    for name in ("base_krea2.yml", "base_krea2_turbo.yml", "base_krea2_turbo_v6e1.yml"):
      with open(os.path.join(_CONFIG_DIR, name), encoding="utf-8") as f:
        self.assertEqual(yaml.safe_load(f)["krea2_weight_cache_dir"], "", name)


if __name__ == "__main__":
  unittest.main()
