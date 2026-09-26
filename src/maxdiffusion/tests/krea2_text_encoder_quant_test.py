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

# CPU tests for the Krea 2 text encoder residency options: int8 weight-only
# quantization (qwix PTQ) and the host-side embedding lookup (inputs_embeds).
# XLA:CPU cannot run bf16 vector math on some hosts, so activations are f32.

import types
import unittest
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion.models.krea2.text_encoder_quant import (
    TEXT_ENCODER_QUANT_MODULE_PATH,
    quantize_text_encoder_model,
    quantize_text_encoder_params,
    resolve_text_encoder_quantization,
    tree_nbytes,
)
from maxdiffusion.models.krea2.util import (
    KREA2_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
    KREA2_PROMPT_TEMPLATE_START_IDX,
    KREA2_PROMPT_TEMPLATE_SUFFIX,
    KREA2_TEXT_ENCODER_SELECT_LAYERS,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model
from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline

_TILE = 32
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _config(num_layers=2):
  return FlaxQwen3Config(
      vocab_size=100,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=num_layers,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      max_position_embeddings=128,
      dtype=jnp.float32,
  )


def _init_params(model, ids, mask):
  params = nn.unbox(model.init(jax.random.PRNGKey(0), ids, mask)["params"])
  # Give the norms non-trivial scales so a dropped/renamed norm would show up.
  return jax.tree_util.tree_map_with_path(
      lambda path, x: x * (1.0 + 0.1 * jnp.arange(x.shape[-1]) / x.shape[-1]) if "norm" in jax.tree_util.keystr(path) else x,
      params,
  )


def _abstract_quantized(qmodel, ids, mask, embeds_dim=None):
  def init():
    if embeds_dim is not None:
      embeds = jnp.zeros(ids.shape + (embeds_dim,), jnp.float32)
      return qmodel.init(jax.random.PRNGKey(0), None, mask, inputs_embeds=embeds)
    return qmodel.init(jax.random.PRNGKey(0), ids, mask)

  return jax.eval_shape(init)["params"]


def _rel_err(a, b):
  a = np.asarray(a, np.float64)
  b = np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _is_quantized(x):
  from qwix._src.providers.ptq import WithAux

  return isinstance(x, WithAux)


class Krea2TextEncoderQuantTest(unittest.TestCase):

  def setUp(self):
    self.config = _config()
    self.model = FlaxQwen3Model(self.config)
    rng = np.random.RandomState(0)
    self.ids = jnp.asarray(rng.randint(0, 100, size=(2, 12)), dtype=jnp.int32)
    self.mask = jnp.asarray(np.concatenate([np.ones((2, 9)), np.zeros((2, 3))], axis=1), dtype=jnp.int32)
    self.position_ids = jnp.clip(jnp.cumsum(self.mask, axis=-1) - 1, 0, None)
    self.params = _init_params(self.model, self.ids, self.mask)
    self.qmodel = quantize_text_encoder_model(self.model, _TILE)

  def _forward(self, model, params, **inputs):
    return model.apply({"params": params}, attention_mask=self.mask, position_ids=self.position_ids, **inputs)

  def test_module_path_matches_only_projections(self):
    import re

    for proj in _PROJECTIONS:
      group = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
      self.assertTrue(re.fullmatch(TEXT_ENCODER_QUANT_MODULE_PATH, f"layers_7/{group}/{proj}"))
    for path in ("embed_tokens", "norm", "layers_0/self_attn/q_norm", "layers_0/input_layernorm", "layers_0/mlp"):
      self.assertIsNone(re.fullmatch(TEXT_ENCODER_QUANT_MODULE_PATH, path))

  def test_quantized_forward_close_to_float(self):
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask)
    qparams = quantize_text_encoder_params(self.params, abstract)
    ref_last, ref_all = self._forward(self.model, self.params, input_ids=self.ids)
    q_last, q_all = self._forward(self.qmodel, qparams, input_ids=self.ids)
    self.assertEqual(len(q_all), len(ref_all))
    np.testing.assert_array_equal(np.asarray(q_all[0]), np.asarray(ref_all[0]))  # embedding is not quantized
    for q, ref in zip(q_all[1:] + [q_last], ref_all[1:] + [ref_last]):
      err = _rel_err(q, ref)
      self.assertLess(err, 5e-2)
      self.assertGreater(err, 0.0)  # the projections really are quantized

  def test_quantized_tree_structure_and_bytes(self):
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask)
    host_params = jax.tree_util.tree_map(np.asarray, self.params)
    host_params = jax.tree_util.tree_map_with_path(
        lambda path, x: x if "norm" in jax.tree_util.keystr(path) else x.astype(jnp.bfloat16), host_params
    )
    qparams = quantize_text_encoder_params(host_params, abstract, scale_dtype=jnp.bfloat16)
    ref_kernel_bytes, q_kernel_bytes = 0, 0
    for i in range(self.config.num_hidden_layers):
      for group, projections in (("self_attn", _PROJECTIONS[:4]), ("mlp", _PROJECTIONS[4:])):
        for proj in projections:
          leaf = qparams[f"layers_{i}"][group][proj]["kernel"]
          self.assertTrue(_is_quantized(leaf))
          self.assertEqual(leaf.array.qvalue.dtype, np.int8)
          self.assertEqual(leaf.array.scale.dtype, jnp.bfloat16)
          ref = host_params[f"layers_{i}"][group][proj]["kernel"]
          self.assertEqual(leaf.array.qvalue.shape, ref.shape)
          self.assertEqual(leaf.array.scale.shape, (ref.shape[0] // _TILE, ref.shape[1]))
          ref_kernel_bytes += ref.nbytes
          q_kernel_bytes += tree_nbytes(leaf)
      # Norms keep their loaded dtype and values.
      for norm in ("input_layernorm", "post_attention_layernorm"):
        got = qparams[f"layers_{i}"][norm]["weight"]
        np.testing.assert_array_equal(np.asarray(got), host_params[f"layers_{i}"][norm]["weight"])
        self.assertEqual(got.dtype, np.float32)
    self.assertLess(q_kernel_bytes, 0.6 * ref_kernel_bytes)
    self.assertFalse(_is_quantized(qparams["embed_tokens"]["embedding"]))
    self.assertEqual(qparams["embed_tokens"]["embedding"].dtype, jnp.bfloat16)

  def test_inputs_embeds_matches_input_ids(self):
    table = np.asarray(self.params["embed_tokens"]["embedding"])
    params_no_embed = {k: v for k, v in self.params.items() if k != "embed_tokens"}
    embeds = jnp.asarray(table[np.asarray(self.ids)])
    ref_last, ref_all = self._forward(self.model, self.params, input_ids=self.ids)
    last, all_hidden = self._forward(self.model, params_no_embed, inputs_embeds=embeds)
    for got, ref in zip(all_hidden + [last], ref_all + [ref_last]):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    with self.assertRaises(ValueError):
      self._forward(self.model, self.params, input_ids=self.ids, inputs_embeds=embeds)

  def test_embeds_init_has_no_embedding_table(self):
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask, embeds_dim=self.config.hidden_size)
    self.assertNotIn("embed_tokens", abstract)
    self.assertIn("norm", abstract)
    self.assertTrue(_is_quantized(abstract["layers_0"]["mlp"]["down_proj"]["kernel"]))
    self.assertFalse(_is_quantized(abstract["layers_0"]["self_attn"]["q_norm"]["weight"]))

    # The quantized model run from embeddings matches the one run from ids.
    abstract_ids = _abstract_quantized(self.qmodel, self.ids, self.mask)
    qparams = quantize_text_encoder_params(self.params, abstract_ids)
    qparams_no_embed = quantize_text_encoder_params({k: v for k, v in self.params.items() if k != "embed_tokens"}, abstract)
    embeds = jnp.asarray(np.asarray(self.params["embed_tokens"]["embedding"])[np.asarray(self.ids)])
    _, ref_all = self._forward(self.qmodel, qparams, input_ids=self.ids)
    _, got_all = self._forward(self.qmodel, qparams_no_embed, inputs_embeds=embeds)
    for got, ref in zip(got_all, ref_all):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_matching_tree_quantizes_with_boxed_and_unboxed_abstract(self):
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask)
    self.assertIsInstance(abstract["embed_tokens"]["embedding"], nn.LogicallyPartitioned)  # boxed
    host_params = jax.tree_util.tree_map(np.asarray, self.params)
    boxed = quantize_text_encoder_params(host_params, abstract)
    unboxed = quantize_text_encoder_params(host_params, nn.unbox(abstract))
    self.assertEqual(jax.tree_util.tree_structure(boxed), jax.tree_util.tree_structure(unboxed))
    for a, b in zip(jax.tree_util.tree_leaves(boxed), jax.tree_util.tree_leaves(unboxed)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    self.assertTrue(_is_quantized(unboxed["layers_1"]["self_attn"]["k_proj"]["kernel"]))

  def test_accepts_frozen_dict_host_and_abstract_trees(self):
    import flax

    abstract = nn.unbox(_abstract_quantized(self.qmodel, self.ids, self.mask))
    host_params = jax.tree_util.tree_map(np.asarray, self.params)
    expected = quantize_text_encoder_params(host_params, abstract)
    frozen_host = flax.core.freeze(host_params)
    frozen_abstract = flax.core.freeze(abstract)
    self.assertIsInstance(frozen_host, flax.core.FrozenDict)
    self.assertIsInstance(frozen_abstract, flax.core.FrozenDict)
    for host, abs_tree in ((frozen_host, frozen_abstract), (frozen_host, abstract), (host_params, frozen_abstract)):
      got = quantize_text_encoder_params(host, abs_tree)
      self.assertEqual(jax.tree_util.tree_structure(got), jax.tree_util.tree_structure(expected))
      for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    self.assertTrue(_is_quantized(got["layers_0"]["mlp"]["down_proj"]["kernel"]))

  def test_extra_nested_param_raises(self):
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask)
    host_params = jax.tree_util.tree_map(np.asarray, self.params)
    host_params["layers_0"]["mlp"]["extra_proj"] = {"kernel": np.zeros((64, 64), np.float32)}
    with self.assertRaisesRegex(ValueError, "Not in the quantized model's parameter tree: layers_0/mlp/extra_proj/kernel"):
      quantize_text_encoder_params(host_params, abstract)

  def test_missing_nested_param_raises(self):
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask)
    host_params = jax.tree_util.tree_map(np.asarray, self.params)
    del host_params["layers_1"]["self_attn"]["k_proj"]["kernel"]
    del host_params["layers_0"]["post_attention_layernorm"]
    with self.assertRaises(ValueError) as ctx:
      quantize_text_encoder_params(host_params, nn.unbox(abstract))
    message = str(ctx.exception)
    self.assertIn("Missing from the loaded params:", message)
    self.assertIn("layers_1/self_attn/k_proj/kernel", message)
    self.assertIn("layers_0/post_attention_layernorm/weight", message)
    self.assertNotIn("krea2_text_embed_on_host", message)

  def test_embedding_table_with_host_embedding_model_raises(self):
    # Host-embedding mode: the model is initialized from embeddings, so it has no table.
    abstract = _abstract_quantized(self.qmodel, self.ids, self.mask, embeds_dim=self.config.hidden_size)
    with self.assertRaisesRegex(ValueError, r"embed_tokens/embedding.*krea2_text_embed_on_host=True"):
      quantize_text_encoder_params(self.params, abstract)
    # And the reverse: the model expects the table but the params lack it.
    abstract_ids = _abstract_quantized(self.qmodel, self.ids, self.mask)
    params_no_embed = {k: v for k, v in self.params.items() if k != "embed_tokens"}
    with self.assertRaisesRegex(ValueError, r"Missing.*embed_tokens/embedding.*krea2_text_embed_on_host=False"):
      quantize_text_encoder_params(params_no_embed, abstract_ids)

  def test_safe_param_shardings_replicates_indivisible_leaves(self):
    from jax.sharding import AbstractMesh, NamedSharding, PartitionSpec as P

    from maxdiffusion.models.krea2.text_encoder_quant import safe_param_shardings

    mesh = AbstractMesh((3, 2), ("fsdp", "tensor"))
    abstract = {
        "qvalue": jax.ShapeDtypeStruct((2560, 4096), jnp.int8),
        "scale": jax.ShapeDtypeStruct((20, 4096), jnp.bfloat16),
        "norm": jax.ShapeDtypeStruct((2560,), jnp.float32),
    }
    shardings = {
        "qvalue": NamedSharding(mesh, P(None, "tensor")),
        "scale": NamedSharding(mesh, P("fsdp", "tensor")),  # 20 % 3 != 0
        "norm": NamedSharding(mesh, P(("fsdp", "tensor"))),  # 2560 % 6 != 0
    }
    fixed = safe_param_shardings(abstract, shardings, mesh)
    self.assertEqual(fixed["qvalue"].spec, P(None, "tensor"))
    self.assertEqual(fixed["scale"].spec, P())
    self.assertEqual(fixed["norm"].spec, P())

  def test_resolve_config(self):
    self.assertEqual(resolve_text_encoder_quantization(types.SimpleNamespace()), ("", 128, False))
    cfg = types.SimpleNamespace(
        krea2_text_encoder_quantization="int8", krea2_text_encoder_quant_tile_size=64, krea2_text_embed_on_host=True
    )
    self.assertEqual(resolve_text_encoder_quantization(cfg), ("int8", 64, True))
    with self.assertRaises(ValueError):
      resolve_text_encoder_quantization(types.SimpleNamespace(krea2_text_encoder_quantization="int4"))


def _set_model_specific_special_tokens(message, exc_type=AttributeError):
  """Raises from a frame named like the transformers method that trips on the
  Krea 2 `extra_special_tokens` list."""
  raise exc_type(message)


def _raise_elsewhere(message, exc_type=AttributeError):
  raise exc_type(message)


_LIST_KEYS_ERROR = "'list' object has no attribute 'keys'"


class Krea2TokenizerLoadTest(unittest.TestCase):

  def test_retries_without_extra_special_tokens(self):
    from maxdiffusion.models.krea2.util import load_krea2_tokenizer

    sentinel = object()
    calls = []

    def fake_from_pretrained(path, **kwargs):
      calls.append((path, kwargs))
      if kwargs.get("extra_special_tokens") != {}:
        _set_model_specific_special_tokens(_LIST_KEYS_ERROR)
      return sentinel

    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
      with mock.patch("maxdiffusion.max_logging.log") as log:
        tokenizer = load_krea2_tokenizer("/snap/tokenizer", "/snap")
    self.assertIs(tokenizer, sentinel)
    self.assertEqual(log.call_count, 1)
    # The special-tokens error skips the subfolder fallback and retries the same path once.
    self.assertEqual(len(calls), 2)
    self.assertEqual(
        calls,
        [
            ("/snap/tokenizer", {"local_files_only": True}),
            ("/snap/tokenizer", {"local_files_only": True, "extra_special_tokens": {}}),
        ],
    )

  def test_persistent_extra_special_tokens_error_reraises_after_one_retry(self):
    from maxdiffusion.models.krea2.util import load_krea2_tokenizer

    for snapshot_dir in ("/snap", None):
      calls = []

      def fake_from_pretrained(path, calls=calls, **kwargs):
        calls.append((path, kwargs))
        _set_model_specific_special_tokens(_LIST_KEYS_ERROR)

      with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
        with self.assertRaisesRegex(AttributeError, "'list' object has no attribute"):
          load_krea2_tokenizer("/snap/tokenizer", snapshot_dir)
      self.assertEqual(len(calls), 2, snapshot_dir)
      self.assertEqual(calls[1], ("/snap/tokenizer", {"local_files_only": True, "extra_special_tokens": {}}))

  def test_special_token_validation_error_is_not_retried(self):
    from maxdiffusion.models.krea2.util import load_krea2_tokenizer

    message = "Special token <x> has to be either str or AddedToken but got: <class 'int'>"
    sentinel = object()
    calls = []

    def fake_from_pretrained(path, **kwargs):
      calls.append((path, kwargs))
      if "subfolder" not in kwargs:
        _set_model_specific_special_tokens(message, TypeError)
      return sentinel

    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
      self.assertIs(load_krea2_tokenizer("/snap/tokenizer", "/snap"), sentinel)
    # Falls to the subfolder fallback, never retries with extra_special_tokens={}.
    self.assertEqual(
        calls,
        [
            ("/snap/tokenizer", {"local_files_only": True}),
            ("/snap", {"subfolder": "tokenizer", "local_files_only": True}),
        ],
    )

    calls.clear()
    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
      with self.assertRaisesRegex(TypeError, "has to be either str or AddedToken"):
        load_krea2_tokenizer("/snap/tokenizer")
    self.assertEqual(calls, [("/snap/tokenizer", {"local_files_only": True})])

  def test_list_attribute_error_requires_special_tokens_frame(self):
    from maxdiffusion.models.krea2.util import _is_extra_special_tokens_error, load_krea2_tokenizer

    def caught(fn, *args):
      try:
        fn(*args)
      except Exception as err:  # pylint: disable=broad-except
        return err
      raise AssertionError("expected an exception")

    self.assertTrue(_is_extra_special_tokens_error(caught(_set_model_specific_special_tokens, _LIST_KEYS_ERROR)))
    # Right message, wrong frame; right frame, wrong message; right frame and message, wrong type.
    self.assertFalse(_is_extra_special_tokens_error(caught(_raise_elsewhere, _LIST_KEYS_ERROR)))
    self.assertFalse(_is_extra_special_tokens_error(caught(_set_model_specific_special_tokens, "no attribute 'foo'")))
    self.assertFalse(
        _is_extra_special_tokens_error(caught(_set_model_specific_special_tokens, _LIST_KEYS_ERROR, TypeError))
    )
    self.assertFalse(_is_extra_special_tokens_error(AttributeError(_LIST_KEYS_ERROR)))  # no traceback

    calls = []

    def fake_from_pretrained(path, **kwargs):
      calls.append((path, kwargs))
      _raise_elsewhere(_LIST_KEYS_ERROR)

    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
      with self.assertRaisesRegex(AttributeError, "'list' object"):
        load_krea2_tokenizer("/snap/tokenizer")
    self.assertEqual(calls, [("/snap/tokenizer", {"local_files_only": True})])

  def test_other_errors_fall_back_to_snapshot_subfolder(self):
    from maxdiffusion.models.krea2.util import load_krea2_tokenizer

    sentinel = object()
    calls = []

    def fake_from_pretrained(path, **kwargs):
      calls.append((path, kwargs))
      if "subfolder" not in kwargs:
        raise OSError("no tokenizer files")
      return sentinel

    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
      self.assertIs(load_krea2_tokenizer("/snap/tokenizer", "/snap"), sentinel)
    self.assertEqual(calls[-1], ("/snap", {"subfolder": "tokenizer", "local_files_only": True}))

    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=ValueError("unrelated")):
      with self.assertRaisesRegex(ValueError, "unrelated"):
        load_krea2_tokenizer("/snap/tokenizer")


class _FakeTokenizer:
  """Deterministic stand-in for the Qwen tokenizer: one token per character."""

  def __call__(self, texts, truncation=False, padding=None, max_length=None, return_tensors="np"):
    del truncation, return_tensors
    # The chat-template suffix is 5 tokens in the real tokenizer.
    rows = [
        list(range(1, KREA2_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS + 1))
        if t == KREA2_PROMPT_TEMPLATE_SUFFIX
        else [(ord(c) % 97) + 1 for c in t]
        for t in texts
    ]
    if max_length is not None:
      rows = [r[:max_length] for r in rows]
    length = max_length if padding == "max_length" else max(len(r) for r in rows)
    ids = np.zeros((len(rows), length), np.int64)
    mask = np.zeros((len(rows), length), np.int64)
    for i, r in enumerate(rows):
      ids[i, : len(r)] = r
      mask[i, : len(r)] = 1
    return {"input_ids": ids, "attention_mask": mask}


class Krea2PipelineEmbeddingTableTest(unittest.TestCase):
  """encode_prompt with a host embedding table matches the in-model lookup."""

  def test_encode_prompt_with_host_table(self):
    num_layers = max(KREA2_TEXT_ENCODER_SELECT_LAYERS) + 1
    config = _config(num_layers=num_layers)
    model = FlaxQwen3Model(config)
    max_sequence_length = 16
    seq = max_sequence_length + KREA2_PROMPT_TEMPLATE_START_IDX
    ids = jnp.zeros((1, seq), jnp.int32)
    params = _init_params(model, ids, jnp.ones_like(ids))
    table = np.asarray(params["embed_tokens"]["embedding"])
    params_no_embed = {k: v for k, v in params.items() if k != "embed_tokens"}

    def pipeline(text_encoder, table=None):
      pipe_config = types.SimpleNamespace(
          max_sequence_length=max_sequence_length,
          logical_axis_rules=(),
          krea2_staged_transformer=False,
          krea2_text_compaction_multiple=0,
      )
      return FlaxKrea2Pipeline(
          transformer=types.SimpleNamespace(attention_kernel="dot_product", num_layers=1),
          vae=None,
          vae_cache=None,
          text_encoder=text_encoder,
          tokenizer=_FakeTokenizer(),
          scheduler=None,
          config=pipe_config,
          mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
          text_embedding_table=table,
      )

    prompts = ["a fox", "a much longer prompt about snow"]
    ref_pipe = pipeline(model)
    ref_pipe._setup_jit_functions()
    ref_embeds, ref_mask = ref_pipe.encode_prompt(prompts, params)
    host_pipe = pipeline(model, table)
    host_pipe._setup_jit_functions()
    got_embeds, got_mask = host_pipe.encode_prompt(prompts, params_no_embed)
    self.assertEqual(got_embeds.shape, (2, max_sequence_length, len(KREA2_TEXT_ENCODER_SELECT_LAYERS), 64))
    np.testing.assert_array_equal(np.asarray(got_mask), np.asarray(ref_mask))
    np.testing.assert_array_equal(np.asarray(got_embeds), np.asarray(ref_embeds))

    # int8 + host table: close to the float reference.
    qmodel = quantize_text_encoder_model(model, _TILE)
    embeds = jnp.zeros((1, seq, config.hidden_size), jnp.float32)
    abstract = jax.eval_shape(lambda: qmodel.init(jax.random.PRNGKey(0), None, jnp.ones_like(ids), inputs_embeds=embeds))
    qparams = quantize_text_encoder_params(params_no_embed, abstract["params"])
    q_pipe = pipeline(qmodel, table)
    q_pipe._setup_jit_functions()
    q_embeds, _ = q_pipe.encode_prompt(prompts, qparams)
    valid = np.asarray(ref_mask)
    self.assertLess(_rel_err(np.asarray(q_embeds)[valid], np.asarray(ref_embeds)[valid]), 5e-2)


if __name__ == "__main__":
  unittest.main()
