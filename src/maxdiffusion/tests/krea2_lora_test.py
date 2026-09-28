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

# CPU-runnable unit tests for Krea 2 inference-time LoRA loading.

import os
import unittest
from contextlib import ExitStack

import flax
import flax.linen as nn
import pytest
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from maxdiffusion.loaders.krea2_lora_pipeline import (
    Krea2LoraLoaderMixin,
    apply_diff_updates,
    insert_lora_params,
    make_lora_compile_spec,
    maybe_load_krea2_lora,
)
from maxdiffusion.models.krea2.lora_util import (
    DIFFUSERS_LORA_METADATA_KEY,
    convert_krea2_lora_to_flax,
    krea2_torch_path_to_flax_path,
    normalize_krea2_lora_state_dict,
    parse_lora_metadata,
)
from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel, Krea2TransformerBlock
from maxdiffusion.models.krea2.util import prepare_krea2_image_ids, prepare_krea2_text_ids
from maxdiffusion.pipelines.krea2.krea2_pipeline import (
    build_staged_block_lora_params,
    index_staged_block_lora_specs,
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
      num_attention_heads=4,  # hidden_size = 32
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


def _tiny_inputs(batch_size=1, s_img=4, s_txt=3):
  hs = jnp.array(np.random.RandomState(0).randn(batch_size, s_img, 16), dtype=jnp.float32)
  ehs = jnp.array(np.random.RandomState(1).randn(batch_size, s_txt, 3, 24), dtype=jnp.float32)
  t = jnp.full((batch_size,), 0.5)
  img_ids = prepare_krea2_image_ids(batch_size, 2, 2)
  txt_ids = prepare_krea2_text_ids(batch_size, s_txt)
  mask = jnp.ones((batch_size, s_txt), dtype=jnp.bool_)
  return hs, ehs, t, img_ids, txt_ids, mask


def _rand(rng, *shape):
  return rng.randn(*shape).astype(np.float32)


class NormalizeStateDictTest(unittest.TestCase):
  """Key normalization across kohya / ComfyUI / diffusers-PEFT formats."""

  def test_all_three_formats_and_skips(self):
    rng = np.random.RandomState(0)
    state_dict = {
        # kohya underscore style + alpha
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
        # ComfyUI dotted style, diffusers `ff.gate` naming
        "diffusion_model.transformer_blocks.1.ff.gate.lora_down.weight": _rand(rng, 2, 32),
        "diffusion_model.transformer_blocks.1.ff.gate.lora_up.weight": _rand(rng, 64, 2),
        # diffusers/PEFT style with to_out.0
        "transformer.transformer_blocks.1.attn.to_out.0.lora_A.weight": _rand(rng, 2, 32),
        "transformer.transformer_blocks.1.attn.to_out.0.lora_B.weight": _rand(rng, 32, 2),
        # text-encoder key -> skipped
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": _rand(rng, 2, 8),
        # unknown module -> unmatched
        "lora_unet_bogus_module.lora_down.weight": _rand(rng, 2, 8),
        # unknown suffix -> unmatched
        "some_random_tensor": _rand(rng, 3),
    }
    modules, skipped_te, unmatched = normalize_krea2_lora_state_dict(state_dict)

    entry = modules["transformer_blocks.0.attn.to_q"]
    self.assertEqual(entry["down"].shape, (2, 32))
    self.assertEqual(entry["up"].shape, (32, 2))
    self.assertEqual(entry["alpha"], 4.0)
    self.assertIn("transformer_blocks.1.ff.gate", modules)
    self.assertIn("transformer_blocks.1.attn.to_out.0", modules)
    self.assertEqual(skipped_te, ["lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"])
    self.assertEqual(sorted(unmatched), ["lora_unet_bogus_module.lora_down.weight", "some_random_tensor"])

  def test_diff_and_diff_b_keys(self):
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_transformer_blocks_0_norm1.diff": _rand(rng, 32),
        "lora_unet_transformer_blocks_0_scale_shift_table.diff": _rand(rng, 6, 32),
        "lora_unet_img_in.diff_b": _rand(rng, 32),
    }
    modules, _, unmatched = normalize_krea2_lora_state_dict(state_dict)
    self.assertEqual(unmatched, [])
    self.assertIn("diff", modules["transformer_blocks.0.norm1"])
    self.assertIn("diff", modules["transformer_blocks.0.scale_shift_table"])
    self.assertIn("diff_b", modules["img_in"])

  def test_kohya_to_out_0_keys_map_to_output_projection(self):
    """Regression test: kohya flattens diffusers' to_out.0 container into
    to_out_0, which must land on the Flax to_out module, not in unmatched."""
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_transformer_blocks_0_attn_to_out_0.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_out_0.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_text_fusion_refiner_blocks_0_attn_to_out_0.lora_down.weight": _rand(rng, 2, 24),
        "lora_unet_text_fusion_refiner_blocks_0_attn_to_out_0.lora_up.weight": _rand(rng, 24, 2),
    }
    modules, _, unmatched = normalize_krea2_lora_state_dict(state_dict)
    self.assertEqual(unmatched, [])
    flat_lora, ranks, _, _ = convert_krea2_lora_to_flax(state_dict, "test", weights_dtype=jnp.float32)
    self.assertIn(("blocks_0", "attn", "to_out"), ranks)
    self.assertIn(("text_fusion", "refiner_blocks_0", "attn", "to_out"), ranks)
    self.assertIn(("blocks_0", "attn", "to_out", "lora-test", "down", "kernel"), flat_lora)
    self.assertEqual(len(modules), 2)

  def test_fused_qkv_raises(self):
    state_dict = {"lora_unet_transformer_blocks_0_attn_qkv.lora_down.weight": np.zeros((2, 32), np.float32)}
    with self.assertRaises(NotImplementedError):
      normalize_krea2_lora_state_dict(state_dict)

  def test_torch_tensors_are_converted(self):
    torch = pytest.importorskip("torch")

    state_dict = {
        "lora_unet_img_in.lora_down.weight": torch.randn(2, 16, dtype=torch.bfloat16),
        "lora_unet_img_in.lora_up.weight": torch.randn(32, 2, dtype=torch.bfloat16),
        "lora_unet_img_in.alpha": torch.tensor(2.0),
    }
    modules, _, unmatched = normalize_krea2_lora_state_dict(state_dict)
    self.assertEqual(unmatched, [])
    self.assertIsInstance(modules["img_in"]["down"], np.ndarray)
    self.assertEqual(modules["img_in"]["down"].dtype, np.float32)
    self.assertEqual(modules["img_in"]["alpha"], 2.0)


class TorchPathToFlaxPathTest(unittest.TestCase):

  def test_mappings(self):
    cases = {
        "transformer_blocks.3.attn.to_q": ("blocks_3", "attn", "to_q"),
        "transformer_blocks.3.attn.to_out.0": ("blocks_3", "attn", "to_out"),
        "transformer_blocks.3.attn.to_out_0": ("blocks_3", "attn", "to_out"),
        "transformer_blocks.3.attn.to_out": ("blocks_3", "attn", "to_out"),
        "transformer_blocks.3.ff.gate": ("blocks_3", "ff", "gate_proj"),
        "transformer_blocks.3.ff.down_proj": ("blocks_3", "ff", "down_proj"),
        "transformer_blocks.3.norm2": ("blocks_3", "norm2"),
        "transformer_blocks.3.scale_shift_table": ("blocks_3", "scale_shift_table"),
        "text_fusion.layerwise_blocks.1.attn.to_v": ("text_fusion", "layerwise_blocks_1", "attn", "to_v"),
        "text_fusion.refiner_blocks.0.ff.up": ("text_fusion", "refiner_blocks_0", "ff", "up_proj"),
        "text_fusion.projector": ("text_fusion", "projector"),
        "img_in": ("img_in",),
        "time_mod_proj": ("time_mod_proj",),
        "time_embed.linear_1": ("time_embed", "linear_1"),
        "txt_in.linear_2": ("txt_in", "linear_2"),
        "txt_in.norm": ("txt_in", "norm"),
        "final_layer.linear": ("final_layer", "linear"),
        "final_layer.scale_shift_table": ("final_layer", "scale_shift_table"),
    }
    for torch_path, expected in cases.items():
      self.assertEqual(krea2_torch_path_to_flax_path(torch_path), expected, torch_path)

  def test_invalid_paths(self):
    for bad in (
        "bogus_module",
        "transformer_blocks.x.attn.to_q",
        "transformer_blocks.0.attn.to_w",
        "time_embed.norm",
        "text_fusion.other_blocks.0.attn.to_q",
    ):
      self.assertIsNone(krea2_torch_path_to_flax_path(bad), bad)


class ConverterTest(unittest.TestCase):

  def _kohya_state_dict(self):
    rng = np.random.RandomState(7)
    return {
        # blocks_0.attn.to_q: hidden 32 -> 32
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
        # blocks_1.ff.down_proj: 64 -> 32 (ComfyUI dotted)
        "diffusion_model.transformer_blocks.1.ff.down.lora_down.weight": _rand(rng, 2, 64),
        "diffusion_model.transformer_blocks.1.ff.down.lora_up.weight": _rand(rng, 32, 2),
        # text_fusion refiner attn.to_k: 24 -> 24 (kv heads 4 * head_dim 6)
        "lora_unet_text_fusion_refiner_blocks_0_attn_to_k.lora_down.weight": _rand(rng, 2, 24),
        "lora_unet_text_fusion_refiner_blocks_0_attn_to_k.lora_up.weight": _rand(rng, 24, 2),
        # img_in: 16 -> 32 (PEFT keys, no alpha)
        "transformer.img_in.lora_A.weight": _rand(rng, 2, 16),
        "transformer.img_in.lora_B.weight": _rand(rng, 32, 2),
        # diffs
        "lora_unet_transformer_blocks_0_norm1.diff": _rand(rng, 32),
        "lora_unet_transformer_blocks_0_scale_shift_table.diff": _rand(rng, 6, 32),
        "lora_unet_img_in.diff_b": _rand(rng, 32),
        # full-weight diff on a Dense (torch layout (out, in) -> transposed)
        "lora_unet_final_layer_linear.diff": _rand(rng, 16, 32),
    }

  def test_converted_paths_match_model_tree(self):
    state_dict = self._kohya_state_dict()
    flat_lora, ranks, alphas, diffs = convert_krea2_lora_to_flax(state_dict, "test", weights_dtype=jnp.float32)

    model = _tiny_model()
    base_params = _unbox(model.init(jax.random.PRNGKey(0), *_tiny_inputs())["params"])
    flat_base = flatten_dict(flax.core.unfreeze(base_params))

    self.assertEqual(len(ranks), 4)
    for module_path, rank in ranks.items():
      base_kernel = flat_base[(*module_path, "kernel")]
      down = flat_lora[(*module_path, "lora-test", "down", "kernel")]
      up = flat_lora[(*module_path, "lora-test", "up", "kernel")]
      # down: (in, r), up: (r, out) against the base (in, out) kernel.
      self.assertEqual(down.shape, (base_kernel.shape[0], rank))
      self.assertEqual(up.shape, (rank, base_kernel.shape[1]))

    self.assertEqual(alphas[("blocks_0", "attn", "to_q")], 4.0)
    self.assertIsNone(alphas[("img_in",)])

    # Values are transposed torch tensors.
    np.testing.assert_allclose(
        np.asarray(flat_lora[("blocks_0", "attn", "to_q", "lora-test", "down", "kernel")]),
        state_dict["lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight"].T,
    )

    # Diff targets exist in the base tree with matching shapes (kernel diffs transposed).
    for path, delta in diffs.items():
      self.assertIn(path, flat_base, path)
      self.assertEqual(tuple(np.asarray(delta).shape), tuple(flat_base[path].shape), path)

  def test_rank_larger_than_out_features_is_supported(self):
    rng = np.random.RandomState(0)
    state_dict = {
        # The real Krea 2 text-fusion projector maps 12 -> 1 and can be
        # targeted by full-coverage adapters whose rank is greater than 1.
        "lora_unet_text_fusion_projector.lora_down.weight": _rand(rng, 2, 3),
        "lora_unet_text_fusion_projector.lora_up.weight": _rand(rng, 1, 2),
    }
    flat_lora, ranks, _, _ = convert_krea2_lora_to_flax(state_dict, "test")
    path = ("text_fusion", "projector")
    self.assertEqual(ranks[path], 2)
    self.assertEqual(flat_lora[(*path, "lora-test", "down", "kernel")].shape, (3, 2))
    self.assertEqual(flat_lora[(*path, "lora-test", "up", "kernel")].shape, (2, 1))

  def test_mismatched_rank_dimensions_are_skipped(self):
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_text_fusion_projector.lora_down.weight": _rand(rng, 2, 3),
        "lora_unet_text_fusion_projector.lora_up.weight": _rand(rng, 1, 4),
    }
    flat_lora, ranks, _, _ = convert_krea2_lora_to_flax(state_dict, "test")
    self.assertEqual(flat_lora, {})
    self.assertEqual(ranks, {})


class ParseLoraMetadataTest(unittest.TestCase):

  def test_diffusers_prefixed_metadata(self):
    import json

    raw = json.dumps(
        {"transformer.r": 8, "transformer.lora_alpha": 32, "transformer.use_rslora": False, "text_encoder.lora_alpha": 7}
    )
    meta = parse_lora_metadata({DIFFUSERS_LORA_METADATA_KEY: raw})
    self.assertEqual(meta["lora_alpha"], 32.0)
    self.assertFalse(meta["use_rslora"])

  def test_flat_metadata_and_alpha_pattern(self):
    import json

    raw = json.dumps({"r": 4, "lora_alpha": 16, "use_rslora": True, "alpha_pattern": {"to_q": 8}})
    meta = parse_lora_metadata({DIFFUSERS_LORA_METADATA_KEY: raw})
    self.assertEqual(meta["lora_alpha"], 16.0)
    self.assertTrue(meta["use_rslora"])
    self.assertEqual(meta["alpha_pattern"], {"to_q": 8.0})

  def test_kohya_ss_network_alpha_fallback(self):
    meta = parse_lora_metadata({"ss_network_alpha": "16.0", "ss_network_dim": "32"})
    self.assertEqual(meta["lora_alpha"], 16.0)

  def test_adapter_config_takes_precedence(self):
    import json

    raw = json.dumps({"lora_alpha": 16})
    meta = parse_lora_metadata({DIFFUSERS_LORA_METADATA_KEY: raw}, adapter_config={"lora_alpha": 64, "use_rslora": True})
    self.assertEqual(meta["lora_alpha"], 64.0)
    self.assertTrue(meta["use_rslora"])

  def test_empty_sources(self):
    meta = parse_lora_metadata({}, None)
    self.assertIsNone(meta["lora_alpha"])
    self.assertFalse(meta["use_rslora"])


class MetadataAlphaConverterTest(unittest.TestCase):
  """Default alpha resolution for PEFT/diffusers adapters without .alpha tensors."""

  def _peft_state_dict(self, rng):
    return {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": _rand(rng, 8, 32),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": _rand(rng, 32, 8),
        "transformer.transformer_blocks.0.ff.gate.lora_A.weight": _rand(rng, 8, 32),
        "transformer.transformer_blocks.0.ff.gate.lora_B.weight": _rand(rng, 64, 8),
    }

  def test_metadata_default_alpha_applied(self):
    lora_meta = {"lora_alpha": 32.0, "use_rslora": False, "alpha_pattern": {}}
    _, _, alphas, _ = convert_krea2_lora_to_flax(
        self._peft_state_dict(np.random.RandomState(0)), "test", lora_meta=lora_meta
    )
    self.assertEqual(alphas[("blocks_0", "attn", "to_q")], 32.0)
    self.assertEqual(alphas[("blocks_0", "ff", "gate_proj")], 32.0)

  def test_rslora_scaling(self):
    # LoRALinearLayer multiplies by network_alpha / rank; rslora needs
    # alpha / sqrt(rank), i.e. network_alpha = alpha * sqrt(rank).
    lora_meta = {"lora_alpha": 32.0, "use_rslora": True, "alpha_pattern": {}}
    _, ranks, alphas, _ = convert_krea2_lora_to_flax(
        self._peft_state_dict(np.random.RandomState(0)), "test", lora_meta=lora_meta
    )
    rank = ranks[("blocks_0", "attn", "to_q")]
    self.assertEqual(rank, 8)
    self.assertAlmostEqual(alphas[("blocks_0", "attn", "to_q")], 32.0 * np.sqrt(8), places=5)

  def test_alpha_pattern_overrides_default(self):
    lora_meta = {"lora_alpha": 32.0, "use_rslora": False, "alpha_pattern": {"to_q": 8.0}}
    _, _, alphas, _ = convert_krea2_lora_to_flax(
        self._peft_state_dict(np.random.RandomState(0)), "test", lora_meta=lora_meta
    )
    self.assertEqual(alphas[("blocks_0", "attn", "to_q")], 8.0)
    self.assertEqual(alphas[("blocks_0", "ff", "gate_proj")], 32.0)

  def test_alpha_pattern_uses_peft_regex_semantics(self):
    lora_meta = {
        "lora_alpha": 32.0,
        "use_rslora": False,
        "alpha_pattern": {r"^transformer_blocks\.\d+\.attn\.to_q": 8.0, "q": 4.0},
    }
    _, _, alphas, _ = convert_krea2_lora_to_flax(
        self._peft_state_dict(np.random.RandomState(0)), "test", lora_meta=lora_meta
    )
    self.assertEqual(alphas[("blocks_0", "attn", "to_q")], 8.0)
    self.assertEqual(alphas[("blocks_0", "ff", "gate_proj")], 32.0)

  def test_explicit_alpha_tensor_wins_over_metadata(self):
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
    }
    lora_meta = {"lora_alpha": 32.0, "use_rslora": True, "alpha_pattern": {}}
    _, _, alphas, _ = convert_krea2_lora_to_flax(state_dict, "test", lora_meta=lora_meta)
    self.assertEqual(alphas[("blocks_0", "attn", "to_q")], 4.0)


class LoraCompileSpecTest(unittest.TestCase):

  def test_compile_constants_change_the_spec(self):
    path = ("blocks_0", "attn", "to_q")
    ranks = {path: 8}
    diffs = {("blocks_0", "norm1", "weight"): np.zeros((32,), np.float32)}

    base = make_lora_compile_spec("style", 1.0, ranks, {path: 16.0}, diffs)
    self.assertNotEqual(base, make_lora_compile_spec("style", 0.5, ranks, {path: 16.0}, diffs))
    self.assertNotEqual(base, make_lora_compile_spec("style", 1.0, ranks, {path: 32.0}, diffs))
    self.assertNotEqual(base, make_lora_compile_spec("style", 1.0, {path: 4}, {path: 16.0}, diffs))

  def test_tensor_values_do_not_participate_in_the_spec(self):
    path = ("blocks_0", "attn", "to_q")
    spec = make_lora_compile_spec("style", 1.0, {path: 8}, {path: None}, {})
    self.assertEqual(spec, ("style", 1.0, (("blocks_0.attn.to_q", 8, None),), ()))

  def test_no_adapter_keeps_the_existing_return_contract(self):
    class FakeConfig:
      lora_config = {"lora_model_name_or_path": []}

    self.assertEqual(len(maybe_load_krea2_lora(FakeConfig(), jnp.float32)), 3)
    *_, compile_spec = maybe_load_krea2_lora(FakeConfig(), jnp.float32, return_compile_spec=True)
    self.assertEqual(compile_spec, ())


class LoraStateDictLoadingTest(unittest.TestCase):
  """File resolution and metadata round-trips through lora_state_dict."""

  def _save_adapter(self, tmpdir, name="adapter.safetensors", metadata=None):
    import torch
    from safetensors.torch import save_file

    sd = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": torch.randn(2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": torch.randn(32, 2),
    }
    path = os.path.join(tmpdir, name)
    save_file(sd, path, metadata=metadata)
    return path

  def test_file_metadata_round_trip(self):
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
      raw = json.dumps({"transformer.lora_alpha": 32, "transformer.r": 2})
      path = self._save_adapter(tmpdir, metadata={DIFFUSERS_LORA_METADATA_KEY: raw})
      state_dict, sft_metadata, adapter_config = Krea2LoraLoaderMixin.lora_state_dict(path)
      self.assertEqual(len(state_dict), 2)
      self.assertEqual(parse_lora_metadata(sft_metadata, adapter_config)["lora_alpha"], 32.0)

  def test_directory_without_weight_name_picks_single_file(self):
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
      self._save_adapter(tmpdir)
      with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
        json.dump({"r": 2, "lora_alpha": 16, "use_rslora": False}, f)
      state_dict, _, adapter_config = Krea2LoraLoaderMixin.lora_state_dict(tmpdir)
      self.assertEqual(len(state_dict), 2)
      self.assertEqual(adapter_config["lora_alpha"], 16)

  def test_directory_with_multiple_files_requires_weight_name(self):
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
      self._save_adapter(tmpdir, "a.safetensors")
      self._save_adapter(tmpdir, "b.safetensors")
      with self.assertRaisesRegex(ValueError, "weight_name"):
        Krea2LoraLoaderMixin.lora_state_dict(tmpdir)
      state_dict, _, _ = Krea2LoraLoaderMixin.lora_state_dict(tmpdir, weight_name="b.safetensors")
      self.assertEqual(len(state_dict), 2)

  def test_hub_options_and_nested_adapter_config_are_used(self):
    import json
    import sys
    import tempfile
    import types
    from unittest import mock

    class EntryNotFoundError(Exception):
      pass

    download_calls = []
    with tempfile.TemporaryDirectory() as tmpdir:
      config_path = os.path.join(tmpdir, "adapter_config.json")
      with open(config_path, "w") as f:
        json.dump({"lora_alpha": 16}, f)

      def fake_hf_hub_download(
          repo_id,
          filename,
          *,
          subfolder=None,
          revision=None,
          cache_dir=None,
          force_download=False,
          local_files_only=False,
          token=None,
          user_agent=None,
          proxies=None,
          resume_download=None,
      ):
        download_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "subfolder": subfolder,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "local_files_only": local_files_only,
                "token": token,
                "user_agent": user_agent,
                "proxies": proxies,
                "resume_download": resume_download,
            }
        )
        return config_path if filename == "adapter_config.json" else os.path.join(tmpdir, "adapter.safetensors")

      fake_hub = types.ModuleType("huggingface_hub")
      fake_hub.hf_hub_download = fake_hf_hub_download
      fake_utils = types.ModuleType("huggingface_hub.utils")
      fake_utils.EntryNotFoundError = EntryNotFoundError
      with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.utils": fake_utils}):
        with mock.patch(
            "maxdiffusion.loaders.krea2_lora_pipeline._load_safetensors_with_metadata",
            return_value=({"weight": object()}, {"metadata": "present"}),
        ):
          _, metadata, adapter_config = Krea2LoraLoaderMixin.lora_state_dict(
              "org/repo",
              weight_name="nested/adapter.safetensors",
              subfolder="collection",
              revision="v1",
              cache_dir="/cache",
              force_download=True,
              local_files_only=True,
              use_auth_token="secret",
              proxies={"https": "proxy"},
              resume_download=True,
          )

    self.assertEqual(metadata, {"metadata": "present"})
    self.assertEqual(adapter_config["lora_alpha"], 16)
    self.assertEqual(download_calls[0]["filename"], "nested/adapter.safetensors")
    self.assertEqual(download_calls[0]["subfolder"], "collection")
    self.assertEqual(download_calls[1]["filename"], "adapter_config.json")
    self.assertEqual(download_calls[1]["subfolder"], "collection/nested")
    for call in download_calls:
      self.assertEqual(call["revision"], "v1")
      self.assertEqual(call["cache_dir"], "/cache")
      self.assertTrue(call["force_download"])
      self.assertTrue(call["local_files_only"])
      self.assertEqual(call["token"], "secret")
      self.assertEqual(call["proxies"], {"https": "proxy"})
      self.assertTrue(call["resume_download"])

  def test_hub_auto_selection_uses_cached_snapshot_when_offline(self):
    import json
    import sys
    import tempfile
    import types
    from unittest import mock

    class EntryNotFoundError(Exception):
      pass

    snapshot_calls = []
    with tempfile.TemporaryDirectory() as tmpdir:
      adapter_dir = os.path.join(tmpdir, "adapters")
      os.makedirs(adapter_dir)
      adapter_path = os.path.join(adapter_dir, "adapter.safetensors")
      with open(adapter_path, "w"):
        pass
      with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump({"lora_alpha": 24}, f)

      def fake_hf_hub_download(
          repo_id,
          filename,
          *,
          subfolder=None,
          revision=None,
          cache_dir=None,
          force_download=False,
          local_files_only=False,
          token=None,
          user_agent=None,
      ):
        raise AssertionError("offline auto-selection must not call hf_hub_download")

      class FakeHfApi:

        def snapshot_download(
            self,
            repo_id,
            *,
            revision=None,
            cache_dir=None,
            force_download=False,
            local_files_only=False,
            token=None,
            allow_patterns=None,
        ):
          snapshot_calls.append(
              {
                  "repo_id": repo_id,
                  "revision": revision,
                  "cache_dir": cache_dir,
                  "force_download": force_download,
                  "local_files_only": local_files_only,
                  "token": token,
                  "allow_patterns": allow_patterns,
              }
          )
          return tmpdir

        def list_repo_files(self, *args, **kwargs):
          raise AssertionError("offline auto-selection must not list remote files")

      fake_hub = types.ModuleType("huggingface_hub")
      fake_hub.hf_hub_download = fake_hf_hub_download
      fake_hub.HfApi = FakeHfApi
      fake_utils = types.ModuleType("huggingface_hub.utils")
      fake_utils.EntryNotFoundError = EntryNotFoundError
      with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.utils": fake_utils}):
        with mock.patch(
            "maxdiffusion.loaders.krea2_lora_pipeline._load_safetensors_with_metadata",
            return_value=({"weight": object()}, {}),
        ) as load_mock:
          _, _, adapter_config = Krea2LoraLoaderMixin.lora_state_dict(
              "org/repo",
              subfolder="adapters",
              revision="v2",
              cache_dir="/cache",
              local_files_only=True,
              token="secret",
          )

    load_mock.assert_called_once_with(adapter_path)
    self.assertEqual(adapter_config["lora_alpha"], 24)
    self.assertEqual(snapshot_calls[0]["repo_id"], "org/repo")
    self.assertEqual(snapshot_calls[0]["revision"], "v2")
    self.assertEqual(snapshot_calls[0]["cache_dir"], "/cache")
    self.assertTrue(snapshot_calls[0]["local_files_only"])
    self.assertEqual(snapshot_calls[0]["token"], "secret")

  def test_incompatible_adapter_fails_loudly(self):
    import tempfile

    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "te_only.safetensors")
      save_file({"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.randn(2, 8)}, path)

      class FakeConfig:
        lora_config = {
            "lora_model_name_or_path": [path],
            "weight_name": [os.path.basename(path)],
            "adapter_name": ["broken"],
            "scale": [1.0],
            "from_pt": ["true"],
        }

      with self.assertRaisesRegex(ValueError, "no keys applicable"):
        maybe_load_krea2_lora(FakeConfig(), jnp.float32)


class _DenseHost(nn.Module):
  """Minimal host so the intercepted Dense has a non-empty module path."""

  features: int = 8

  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.features, use_bias=False, name="proj")(x)


class _TwoArgProj(nn.Module):
  """Stand-in for a W8A8 projection: called as `proj(inputs, quantized_inputs)`
  and exposing the fields the Krea 2 LoRA interceptor reads. The base product
  uses the second argument, so the test can tell which input feeds LoRA."""

  features: int
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, inputs, quantized_inputs=None):
    kernel = self.param("kernel", nn.initializers.lecun_normal(), (inputs.shape[-1], self.features), self.param_dtype)
    if quantized_inputs is None:
      return jnp.matmul(inputs, kernel)
    x_q, x_scale = quantized_inputs
    return jnp.matmul(x_q * x_scale, kernel)


class _TwoArgHost(nn.Module):
  features: int = 8
  use_kwargs: bool = False

  @nn.compact
  def __call__(self, x, quantized):
    proj = _TwoArgProj(self.features, name="proj")
    if self.use_kwargs:
      return proj(inputs=x, quantized_inputs=quantized)
    return proj(x, quantized)


class InterceptorTest(unittest.TestCase):

  def test_interceptor_feeds_only_the_first_argument_of_a_two_argument_projection(self):
    rank, alpha, scale = 2, 4.0, 0.7
    rng = np.random.RandomState(0)
    x = jnp.asarray(_rand(rng, 3, 5))
    # Deliberately unrelated to x: the base product reads it, LoRA must not.
    quantized = (jnp.asarray(_rand(rng, 3, 5)), jnp.full((3, 1), 0.5, jnp.float32))
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(
        {("proj",): rank}, {("proj",): alpha}, "test", scale=scale
    )

    for use_kwargs in (False, True):
      with self.subTest(use_kwargs=use_kwargs):
        host = _TwoArgHost(use_kwargs=use_kwargs)
        with nn.intercept_methods(interceptor):
          params = flax.core.unfreeze(host.init(jax.random.PRNGKey(0), x, quantized)["params"])

        flat = flatten_dict(params)
        self.assertEqual(flat[("proj", "lora-test", "down", "kernel")].shape, (5, rank))
        self.assertEqual(flat[("proj", "lora-test", "up", "kernel")].shape, (rank, 8))
        down = _rand(rng, 5, rank)
        up = _rand(rng, rank, 8)
        flat[("proj", "lora-test", "down", "kernel")] = jnp.asarray(down)
        flat[("proj", "lora-test", "up", "kernel")] = jnp.asarray(up)
        params = unflatten_dict(flat)

        with nn.intercept_methods(interceptor):
          out = host.apply({"params": params}, x, quantized)

        kernel = flat[("proj", "kernel")]
        base = jnp.matmul(quantized[0] * quantized[1], kernel)
        expected = base + scale * (alpha / rank) * jnp.matmul(jnp.matmul(x, jnp.asarray(down)), jnp.asarray(up))
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_interceptor_allows_rank_wider_than_output(self):
    host = _DenseHost(features=1)
    x = jnp.array(np.random.RandomState(0).randn(3, 3), dtype=jnp.float32)
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(
        {("proj",): 2}, {("proj",): 2.0}, "wide", scale=1.0
    )

    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(host.init(jax.random.PRNGKey(0), x)["params"])
      output = host.apply({"params": params}, x)

    flat = flatten_dict(params)
    self.assertEqual(flat[("proj", "lora-wide", "down", "kernel")].shape, (3, 2))
    self.assertEqual(flat[("proj", "lora-wide", "up", "kernel")].shape, (2, 1))
    self.assertEqual(output.shape, (3, 1))

  def test_interceptor_matches_manual_lora_math(self):
    rank, alpha, scale = 2, 4.0, 0.7
    host = _DenseHost()
    x = jnp.array(np.random.RandomState(0).randn(3, 5), dtype=jnp.float32)

    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(
        {("proj",): rank}, {("proj",): alpha}, "test", scale=scale
    )
    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(host.init(jax.random.PRNGKey(0), x)["params"])

    rng = np.random.RandomState(1)
    flat = flatten_dict(params)
    self.assertIn(("proj", "lora-test", "down", "kernel"), flat)
    down = _rand(rng, 5, rank)
    up = _rand(rng, rank, 8)
    flat[("proj", "lora-test", "down", "kernel")] = jnp.asarray(down)
    flat[("proj", "lora-test", "up", "kernel")] = jnp.asarray(up)
    params = unflatten_dict(flat)

    with nn.intercept_methods(interceptor):
      out = host.apply({"params": params}, x)

    kernel = flat[("proj", "kernel")]
    expected = jnp.matmul(x, kernel) + scale * (alpha / rank) * (
        jnp.matmul(jnp.matmul(x, jnp.asarray(down)), jnp.asarray(up))
    )
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

  def test_zero_scale_is_inert(self):
    host = _DenseHost()
    x = jnp.array(np.random.RandomState(0).randn(3, 5), dtype=jnp.float32)
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor({("proj",): 2}, {("proj",): None}, "test", scale=0.0)
    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(host.init(jax.random.PRNGKey(0), x)["params"])
    flat = flatten_dict(params)
    flat[("proj", "lora-test", "down", "kernel")] = jnp.ones((5, 2))
    flat[("proj", "lora-test", "up", "kernel")] = jnp.ones((2, 8))
    params = unflatten_dict(flat)

    with nn.intercept_methods(interceptor):
      out_lora = host.apply({"params": params}, x)
    base_out = host.apply({"params": {"proj": {"kernel": params["proj"]["kernel"]}}}, x)
    np.testing.assert_array_equal(np.asarray(out_lora), np.asarray(base_out))


class EndToEndTinyModelTest(unittest.TestCase):

  def _lora_state_dict(self, rng):
    return {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(2.0),
    }

  def test_lora_changes_model_output(self):
    model = _tiny_model()
    inputs = _tiny_inputs()
    flat_lora, ranks, alphas, _ = convert_krea2_lora_to_flax(
        self._lora_state_dict(np.random.RandomState(3)), "style", weights_dtype=jnp.float32
    )
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, "style", scale=1.0)

    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(_unbox(model.init(jax.random.PRNGKey(0), *inputs)["params"]))
    params = insert_lora_params(params, flat_lora)

    base_out = model.apply({"params": params}, *inputs).sample
    with nn.intercept_methods(interceptor):
      lora_out = model.apply({"params": params}, *inputs).sample

    self.assertEqual(base_out.shape, lora_out.shape)
    self.assertTrue(np.any(np.abs(np.asarray(base_out) - np.asarray(lora_out)) > 1e-6))

  def test_eval_shape_tree_contains_lora_leaves(self):
    """Regression test for the sharding/param tree mismatch: the abstract tree
    produced under the interceptor must contain every converted LoRA leaf."""
    model = _tiny_model()
    inputs = _tiny_inputs()
    flat_lora, ranks, alphas, _ = convert_krea2_lora_to_flax(
        self._lora_state_dict(np.random.RandomState(3)), "style", weights_dtype=jnp.float32
    )
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, "style", scale=1.0)

    with nn.intercept_methods(interceptor):
      abstract_vars = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), *inputs))

    flat_abstract = flatten_dict(flax.core.unfreeze(_unbox(abstract_vars["params"])))
    for path, value in flat_lora.items():
      self.assertIn(path, flat_abstract, path)
      self.assertEqual(tuple(flat_abstract[path].shape), tuple(value.shape), path)

  def test_staged_matches_multiple_adapters_with_nonuniform_ranks(self):
    model = _tiny_model()
    inputs = _tiny_inputs()
    rng = np.random.RandomState(7)
    style_state = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
        "lora_unet_transformer_blocks_1_ff_down.lora_down.weight": _rand(rng, 3, 64),
        "lora_unet_transformer_blocks_1_ff_down.lora_up.weight": _rand(rng, 32, 3),
    }
    detail_state = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 1, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 1),
        "lora_unet_transformer_blocks_1_attn_to_out.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_1_attn_to_out.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_1_attn_to_out.alpha": np.float32(2.0),
    }
    adapters = []
    for name, scale, state_dict in (
        ("style", 0.7, style_state),
        ("detail", 0.25, detail_state),
    ):
      flat_lora, ranks, alphas, diffs = convert_krea2_lora_to_flax(
          state_dict, name, weights_dtype=jnp.float32
      )
      adapters.append(
          (
              flat_lora,
              Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, name, scale),
              make_lora_compile_spec(name, scale, ranks, alphas, diffs),
          )
      )

    with ExitStack() as stack:
      for _, interceptor, _ in adapters:
        stack.enter_context(nn.intercept_methods(interceptor))
      params = flax.core.unfreeze(_unbox(model.init(jax.random.PRNGKey(0), *inputs)["params"]))
    for flat_lora, _, _ in adapters:
      params = insert_lora_params(params, flat_lora)

    with ExitStack() as stack:
      for _, interceptor, _ in adapters:
        stack.enter_context(nn.intercept_methods(interceptor))
      expected = model.apply({"params": params}, *inputs).sample

    hs, ehs, timestep, img_ids, txt_ids, mask = inputs
    text_hidden = model.apply(
        {"params": {key: params[key] for key in ("text_fusion", "txt_in")}},
        ehs,
        mask,
        method=model.encode_text_context,
    )
    prelude_keys = ("img_in", "time_embed", "time_mod_proj")
    hidden, temb, temb_mod, rotary_emb, attention_mask = model.apply(
        {"params": {key: params[key] for key in prelude_keys}},
        hs,
        text_hidden,
        timestep,
        img_ids,
        txt_ids,
        mask,
        method=model.prepare_inputs,
    )
    block_specs = index_staged_block_lora_specs(tuple(item[2] for item in adapters), model.num_layers)
    block = Krea2TransformerBlock(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_kv_heads=2,
    )
    for block_idx in range(model.num_layers):
      block_params = params[f"blocks_{block_idx}"]
      explicit_lora = build_staged_block_lora_params(block_params, block_specs[block_idx])
      hidden = block.apply(
          {"params": block_params},
          hidden,
          temb_mod=temb_mod,
          image_rotary_emb=rotary_emb,
          attention_mask=attention_mask,
          lora_params=explicit_lora,
      )
    actual = model.apply(
        {"params": {"final_layer": params["final_layer"]}},
        hidden,
        temb,
        hs.shape[1],
        method=model.finalize_output,
    )

    block_0_lora = build_staged_block_lora_params(params["blocks_0"], block_specs[0])
    self.assertEqual(len(block_0_lora["attn"]["to_q"]), 2)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_insert_lora_params_rejects_unknown_or_mismatched(self):
    model = _tiny_model()
    inputs = _tiny_inputs()
    params = flax.core.unfreeze(_unbox(model.init(jax.random.PRNGKey(0), *inputs)["params"]))
    with self.assertRaises(ValueError):
      insert_lora_params(params, {("blocks_0", "attn", "to_q", "lora-x", "down", "kernel"): jnp.zeros((32, 2))})


class DiffMergeTest(unittest.TestCase):

  def test_diff_and_diff_b_merge_with_scale(self):
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_transformer_blocks_0_norm1.diff": _rand(rng, 32),
        "lora_unet_final_layer_linear.diff": _rand(rng, 16, 32),
        "lora_unet_final_layer_linear.diff_b": _rand(rng, 16),
    }
    _, _, _, diffs = convert_krea2_lora_to_flax(state_dict, "test")

    params = {
        "blocks_0": {"norm1": {"weight": jnp.zeros((32,), jnp.float32)}},
        "final_layer": {"linear": {"kernel": jnp.ones((32, 16), jnp.float32), "bias": jnp.zeros((16,), jnp.float32)}},
    }
    merged = apply_diff_updates(params, diffs, scale=0.5)

    np.testing.assert_allclose(
        np.asarray(merged["blocks_0"]["norm1"]["weight"]),
        0.5 * state_dict["lora_unet_transformer_blocks_0_norm1.diff"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(merged["final_layer"]["linear"]["kernel"]),
        1.0 + 0.5 * state_dict["lora_unet_final_layer_linear.diff"].T,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(merged["final_layer"]["linear"]["bias"]),
        0.5 * state_dict["lora_unet_final_layer_linear.diff_b"],
        rtol=1e-6,
    )

  def test_all_missing_diff_targets_fail_loudly(self):
    params = {"final_layer": {"linear": {"kernel": jnp.zeros((2, 2), jnp.float32)}}}
    diffs = {("blocks_999", "attn", "to_q", "kernel"): jnp.zeros((2, 2), jnp.float32)}
    with self.assertRaisesRegex(ValueError, "None of the LoRA diff targets exist"):
      apply_diff_updates(params, diffs)

  def test_same_size_wrong_shape_diff_is_rejected(self):
    params = {"final_layer": {"linear": {"kernel": jnp.zeros((2, 3), jnp.float32)}}}
    diffs = {("final_layer", "linear", "kernel"): jnp.zeros((3, 2), jnp.float32)}
    with self.assertRaisesRegex(ValueError, r"has shape \(3, 2\).*expects \(2, 3\)"):
      apply_diff_updates(params, diffs)


if __name__ == "__main__":
  unittest.main()
