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

# CPU tests for Krea 2 hybrid sampling: the first k denoise steps run a second
# transformer tree E with CFG, the rest the main tree M (krea2_pipeline
# hybrid_* kwargs), the per-request text encoder offload, and the runtime side
# (loading E from the weight cache, request validation, evict + re-place).
# Tiny transformer, stub text encoder and VAE; no model and no checkpoint.

import gc
import tempfile
import types
import unittest
import weakref
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import generate_krea2
from maxdiffusion.generate_krea2 import Krea2Runtime, StartupTimeline
from maxdiffusion.models.krea2 import weight_cache
from maxdiffusion.pipelines.krea2 import krea2_pipeline
from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline, free_params, place_params
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.tests.krea2_pipeline_residency_test import (
    _B,
    _GRID_H,
    _GRID_W,
    _S_TXT,
    _config,
    _inputs,
    _mesh,
    _tiny_transformer,
    _unbox,
)

_PROMPT = "a fox"
_STEPS = 8  # the Turbo schedule


def _scheduler():
  return FlaxFlowMatchScheduler(
      num_train_timesteps=1000,
      shift=1.0,
      use_dynamic_shifting=True,
      time_shift_type="exponential",
  )


def _turbo_grid(scheduler):
  """The 8 Turbo sigmas exactly as the pipeline's is_distilled branch computes them."""
  state = scheduler.set_timesteps_ltx2(
      state=scheduler.create_state(),
      num_inference_steps=_STEPS,
      shift=1.15,
      sigmas=jnp.linspace(1.0, 1.0 / _STEPS, _STEPS, dtype=jnp.float32),
  )
  return state.sigmas


class _SegmentScheduler:
  """Minimal port of the qmix gen_mix SegmentScheduler (the experiment's two-call handoff).

  Runs one pipeline call over a sigma segment: a terminal segment has n points
  and ends at 0 through the real step's final branch; a non-terminal one has
  n + 1 points and its last step ends at the held sigma_end (the real step on
  the (n + 1)-point state). prev_sample after each step count in `capture_at`
  is kept.
  """

  def __init__(self, real, sigmas, terminal, capture_at=()):
    pts = jnp.asarray(sigmas, dtype=jnp.float32)
    self._real = real
    self.terminal = terminal
    self.n = int(pts.shape[0]) if terminal else int(pts.shape[0]) - 1
    ntt = real.config.num_train_timesteps
    self.sigmas = pts[: self.n]
    self.timesteps = self.sigmas * ntt
    self._ext = None if terminal else (pts, pts * ntt)
    self.capture_at = set(capture_at)
    self.captured = {}
    self.steps_done = 0

  def __getattr__(self, name):
    return getattr(object.__getattribute__(self, "_real"), name)

  def create_state(self):
    return self._real.create_state()

  def set_timesteps_ltx2(self, state, num_inference_steps=100, **kwargs):
    del kwargs  # shift / sigmas: the segment's own grid
    assert num_inference_steps == self.n, (num_inference_steps, self.n)
    return state.replace(
        sigmas=self.sigmas,
        timesteps=self.timesteps,
        linear_timesteps_weights=None,
        training=False,
        num_inference_steps=self.n,
    )

  def step(self, state, model_output, timestep, sample, **kwargs):
    use = state if self.terminal else state.replace(sigmas=self._ext[0], timesteps=self._ext[1])
    prev, _ = self._real.step(state=use, model_output=model_output, timestep=timestep, sample=sample, **kwargs)
    self.steps_done += 1
    if self.steps_done in self.capture_at:
      self.captured[self.steps_done] = prev
    return prev, state


def _host_numpy(tree):
  return jax.tree_util.tree_map(np.asarray, tree)


def _replicated(tree):
  sharding = NamedSharding(_mesh(), P())
  return jax.tree_util.tree_map(lambda _: sharding, tree)


def _all_deleted(tree):
  return all(leaf.is_deleted() for leaf in jax.tree_util.tree_leaves(tree))


def _none_deleted(tree):
  return not any(leaf.is_deleted() for leaf in jax.tree_util.tree_leaves(tree))


class Krea2HybridPipelineTest(unittest.TestCase):
  """`FlaxKrea2Pipeline.__call__` with the hybrid_* and offload_text_encoder kwargs."""

  def setUp(self):
    self.transformer = _tiny_transformer(in_channels=64)
    self.latents = np.random.RandomState(0).randn(_B, _GRID_H * _GRID_W, 64).astype(np.float32)
    _, prompt_embeds, text_mask, img_ids, txt_ids, t_vec = _inputs()

    def init(seed):
      params = self.transformer.init(
          jax.random.PRNGKey(seed), jnp.asarray(self.latents), prompt_embeds, t_vec, img_ids, txt_ids, text_mask
      )["params"]
      return _host_numpy(_unbox(params))

    # Host trees (numpy) as the runtime keeps them; M = main tree, E = the early tree.
    self.host_m = init(0)
    self.host_e = init(1)
    self.host_qwen3 = {"embed": np.ones((4, 24), np.float32)}
    self.shardings = {"transformer": _replicated(self.host_m), "text_encoder": _replicated(self.host_qwen3)}
    self.m = place_params(self.host_m, self.shardings["transformer"])
    self.e = place_params(self.host_e, self.shardings["transformer"])
    self.qwen3 = place_params(self.host_qwen3, self.shardings["text_encoder"])
    # Different embeddings per prompt text: with equal ones CFG would be a bitwise no-op.
    rng = np.random.RandomState(1)
    self.embeds = {
        _PROMPT: rng.randn(_B, _S_TXT, 3, 24).astype(np.float32),
        "": rng.randn(_B, _S_TXT, 3, 24).astype(np.float32),
        "blurry": rng.randn(_B, _S_TXT, 3, 24).astype(np.float32),
    }
    self.text_mask = np.array([[True, True, False]])

  def _pipeline(self, offload_components=(), scheduler=None, param_shardings="default", **config_overrides):
    pipeline = FlaxKrea2Pipeline(
        transformer=self.transformer,
        vae=types.SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
        vae_cache=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler or _scheduler(),
        config=_config(**{"is_distilled": True, **config_overrides}),
        mesh=_mesh(),
        offload_components=offload_components,
        param_shardings=self.shardings if param_shardings == "default" else param_shardings,
    )
    pipeline._setup_jit_functions()
    seen = {"encoded": [], "steps": [], "qwen3_params": []}

    def fake_encode_prompt(prompts, qwen3_params):
      seen["encoded"].append(prompts[0])
      seen["qwen3_params"].append(qwen3_params)
      # A fresh array per call: nothing outside the pipeline keeps the embeddings alive.
      return jnp.array(self.embeds[prompts[0]]), jnp.array(self.text_mask)

    def fake_vae_decode(graphdef, state, rest_of_state, latents_5d):
      seen["latents_5d"] = np.asarray(latents_5d)
      return jnp.zeros((_B, 1, 32, 32, 3))

    real_step = pipeline._jitted_transformer_step

    def spy_step(params, latents, text_hidden, text_mask, img_ids, txt_ids, t_vec):
      seen["steps"].append({"params": params, "text_hidden": np.asarray(text_hidden), "t": float(t_vec[0])})
      if "on_step" in seen:
        seen["on_step"]()
      return real_step(params, latents, text_hidden, text_mask, img_ids, txt_ids, t_vec)

    pipeline.encode_prompt = fake_encode_prompt
    pipeline._jitted_vae_decode = fake_vae_decode
    pipeline._jitted_transformer_step = spy_step
    return pipeline, seen

  def _call(self, pipeline, params=None, qwen3_params=None, latents=None, steps=_STEPS, **kwargs):
    with mock.patch.object(krea2_pipeline.nnx, "split", return_value=(None, None, None)):
      _, trace = pipeline(
          _PROMPT,
          self.m if params is None else params,
          self.qwen3 if qwen3_params is None else qwen3_params,
          height=32,
          width=32,
          num_inference_steps=steps,
          guidance_scale=kwargs.pop("guidance_scale", 0.0),
          do_classifier_free_guidance=kwargs.pop("do_classifier_free_guidance", False),
          latents=self.latents if latents is None else latents,
          save_outputs=False,
          **kwargs,
      )
    return trace

  def _run(self, *, pipeline_kwargs=None, **call_kwargs):
    pipeline, seen = self._pipeline(**(pipeline_kwargs or {}))
    trace = self._call(pipeline, **call_kwargs)
    return trace, seen, pipeline

  # -- hybrid_steps = 0 ------------------------------------------------------------------------------------------

  def test_hybrid_steps_zero_is_bitwise_the_default_call(self):
    default_trace, default_seen, _ = self._run()
    zero_trace, zero_seen, _ = self._run(hybrid_steps=0, hybrid_params=self.e, hybrid_guidance=3.0)
    np.testing.assert_array_equal(zero_seen["latents_5d"], default_seen["latents_5d"])
    self.assertEqual(set(zero_trace), set(default_trace))
    for key in ("seed", "text_tokens"):
      self.assertEqual(zero_trace[key], default_trace[key])
    for key in ("hybrid_steps", "hybrid_guidance", "hybrid_evicted", "negative_text_tokens"):
      self.assertNotIn(key, zero_trace)
    # Only M, only the prompt: no negative encode, no E step.
    self.assertEqual(zero_seen["encoded"], [_PROMPT])
    self.assertEqual(len(zero_seen["steps"]), _STEPS)
    self.assertTrue(all(step["params"] is self.m for step in zero_seen["steps"]))
    self.assertTrue(_none_deleted(self.e))

  def test_non_hybrid_path_frees_prompt_embeds_before_the_denoise_loop(self):
    # Today's point: the tapped-layer embeddings die right after the text context, before the first DiT step.
    for name, call_kwargs in (
        ("fast", {}),
        ("cfg", {"guidance_scale": 1.5, "do_classifier_free_guidance": True, "negative_prompt": "blurry"}),
        ("hybrid", {"hybrid_steps": 2, "hybrid_params": self.e}),
    ):
      with self.subTest(path=name):
        pipeline, seen = self._pipeline()
        refs = []
        real_context = pipeline._jitted_transformer_text_context

        def spy_context(text_params, prompt_embeds, text_mask, real_context=real_context, refs=refs):
          refs.append(weakref.ref(prompt_embeds))
          return real_context(text_params, prompt_embeds, text_mask)

        live_at_first_step = []

        def on_step(refs=refs, live=live_at_first_step):
          if not live:
            gc.collect()
            live.append([ref() is not None for ref in refs])

        pipeline._jitted_transformer_text_context = spy_context
        seen["on_step"] = on_step
        self._call(pipeline, **call_kwargs)
        expected_contexts = {"fast": 1, "cfg": 2, "hybrid": 3}[name]
        self.assertEqual(len(refs), expected_contexts)
        self.assertEqual(live_at_first_step, [[False] * expected_contexts])

  # -- the numerical contract --------------------------------------------------------------------------------

  def test_hybrid_matches_two_call_segment_composition_bitwise(self):
    _, fast_seen, _ = self._run()
    for k in (2, 3):
      with self.subTest(k=k):
        native_trace, native_seen, _ = self._run(hybrid_steps=k, hybrid_params=self.e, hybrid_guidance=1.5)
        self.assertEqual(native_trace["hybrid_steps"], k)
        self.assertEqual(native_trace["hybrid_guidance"], 1.5)
        self.assertEqual(native_trace["negative_text_tokens"], _S_TXT)

        # The experiment: call 1 = E with CFG g 1.5 and an empty negative prompt over T[0..k]
        # (non-terminal), call 2 = M without CFG from its latents over T[k..7] (terminal).
        real = _scheduler()
        grid = _turbo_grid(real)
        seg1 = _SegmentScheduler(real, grid[: k + 1], terminal=False, capture_at=(k,))
        pipeline1, _ = self._pipeline(scheduler=seg1)
        self._call(
            pipeline1,
            params=self.e,
            steps=k,
            guidance_scale=1.5,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        handoff = np.asarray(jax.device_get(seg1.captured[k]))
        self.assertEqual(handoff.dtype, np.float32)
        seg2 = _SegmentScheduler(real, grid[k:], terminal=True)
        pipeline2, composed_seen = self._pipeline(scheduler=seg2)
        self._call(pipeline2, params=self.m, steps=_STEPS - k, latents=handoff)
        self.assertEqual(seg2.steps_done, _STEPS - k)

        np.testing.assert_array_equal(native_seen["latents_5d"], composed_seen["latents_5d"])
        # And the hybrid really changed the image.
        self.assertFalse(np.array_equal(native_seen["latents_5d"], fast_seen["latents_5d"]))

  def test_early_steps_use_e_with_its_text_context_and_cfg_later_steps_m(self):
    k = 3
    _, seen, pipeline = self._run(hybrid_steps=k, hybrid_params=self.e, hybrid_guidance=1.5)
    self.assertEqual(seen["encoded"], [_PROMPT, ""])

    def context(tree, prompt):
      text_params = {key: tree[key] for key in krea2_pipeline.KREA2_TEXT_CONTEXT_KEYS}
      out = pipeline._jitted_transformer_text_context(
          text_params, jnp.array(self.embeds[prompt]), jnp.array(self.text_mask)
      )
      return np.asarray(out)

    e_prompt, e_negative, m_prompt = context(self.e, _PROMPT), context(self.e, ""), context(self.m, _PROMPT)
    self.assertFalse(np.array_equal(e_prompt, m_prompt))
    steps = seen["steps"]
    # CFG (two E steps) only before k, one M step after.
    self.assertEqual(len(steps), 2 * k + (_STEPS - k))
    for i in range(k):
      cond, uncond = steps[2 * i], steps[2 * i + 1]
      self.assertIs(cond["params"], self.e)
      self.assertIs(uncond["params"], self.e)
      self.assertEqual(cond["t"], uncond["t"])
      np.testing.assert_array_equal(cond["text_hidden"], e_prompt)
      np.testing.assert_array_equal(uncond["text_hidden"], e_negative)
    for step in steps[2 * k :]:
      self.assertIs(step["params"], self.m)
      np.testing.assert_array_equal(step["text_hidden"], m_prompt)
    # One sigma per step of the one Turbo loop: E's k steps then M's 8 - k, all distinct.
    timesteps = [steps[2 * i]["t"] for i in range(k)] + [step["t"] for step in steps[2 * k :]]
    self.assertEqual(len(set(timesteps)), _STEPS)
    self.assertEqual(timesteps, sorted(timesteps, reverse=True))

  def test_hybrid_negative_prompt_and_compacted_buckets(self):
    trace, seen, _ = self._run(
        pipeline_kwargs={"krea2_text_compaction_multiple": 2, "krea2_staged_transformer": False},
        hybrid_steps=2,
        hybrid_params=self.e,
        hybrid_negative_prompt="blurry",
        negative_prompt="ignored without the call's CFG",
    )
    self.assertEqual(seen["encoded"], [_PROMPT, "blurry"])
    # Two valid tokens, multiple 2.
    self.assertEqual((trace["text_tokens"], trace["negative_text_tokens"]), (2, 2))

  # -- evict ------------------------------------------------------------------------------------------------

  def test_evict_frees_e_after_its_steps_and_a_replaced_tree_gives_the_same_image(self):
    _, reference_seen, _ = self._run(hybrid_steps=2, hybrid_params=self.e)
    pipeline, seen = self._pipeline()
    e_steps_after_free = []

    def on_step():
      # After eviction no step may see a deleted E (and M stays resident).
      e_steps_after_free.append(_all_deleted(self.e))

    seen["on_step"] = on_step
    trace = self._call(pipeline, hybrid_steps=2, hybrid_params=self.e, hybrid_host_params=self.host_e, hybrid_evict=True)
    self.assertIs(trace["hybrid_evicted"], True)
    self.assertTrue(_all_deleted(self.e))
    self.assertTrue(_none_deleted(self.m))
    # 2 x 2 E steps with E live, then the M steps after the free.
    self.assertEqual(e_steps_after_free, [False] * 4 + [True] * (_STEPS - 2))
    np.testing.assert_array_equal(seen["latents_5d"], reference_seen["latents_5d"])
    # The host copy survives; the runtime re-places it and the next hybrid call is bitwise the same.
    self.assertIsInstance(jax.tree_util.tree_leaves(self.host_e)[0], np.ndarray)
    self.e = place_params(self.host_e, self.shardings["transformer"])
    _, again_seen, _ = self._run(hybrid_steps=2, hybrid_params=self.e, hybrid_host_params=self.host_e, hybrid_evict=True)
    np.testing.assert_array_equal(again_seen["latents_5d"], reference_seen["latents_5d"])

  def test_without_evict_e_stays_resident(self):
    trace, _, _ = self._run(hybrid_steps=3, hybrid_params=self.e, hybrid_host_params=self.host_e)
    self.assertNotIn("hybrid_evicted", trace)
    self.assertTrue(_none_deleted(self.e))

  # -- per-call text encoder offload ------------------------------------------------------------------------

  def test_offload_text_encoder_swaps_in_frees_and_matches_resident(self):
    resident_trace, resident_seen, _ = self._run()
    pipeline, seen = self._pipeline()
    freed = []

    def spy_free(tree, host_tree=None):
      freed.append(tree)
      return free_params(tree, host_tree)

    with mock.patch.object(krea2_pipeline, "free_params", side_effect=spy_free):
      trace = self._call(pipeline, qwen3_params=self.host_qwen3, offload_text_encoder=True)
    np.testing.assert_array_equal(seen["latents_5d"], resident_seen["latents_5d"])
    for key in ("text_encoder_swap_in", "text_encoder_swap_bytes"):
      self.assertNotIn(key, resident_trace)
      self.assertIn(key, trace)
    self.assertEqual(trace["text_encoder_swap_bytes"], 4 * 24 * 4)
    placed = seen["qwen3_params"][0]
    self.assertIsNot(placed, self.host_qwen3)
    self.assertEqual(len(freed), 1)
    self.assertIs(freed[0], placed)
    self.assertTrue(_all_deleted(placed))
    self.assertIsInstance(self.host_qwen3["embed"], np.ndarray)
    # The pipeline's own residency is untouched.
    self.assertEqual(pipeline.offload_components, frozenset())

  def test_offload_text_encoder_with_hybrid_encodes_both_prompts_from_one_placement(self):
    _, resident_seen, _ = self._run(hybrid_steps=2, hybrid_params=self.e)
    trace, seen, _ = self._run(
        qwen3_params=self.host_qwen3, offload_text_encoder=True, hybrid_steps=2, hybrid_params=self.e
    )
    np.testing.assert_array_equal(seen["latents_5d"], resident_seen["latents_5d"])
    self.assertEqual(len(seen["qwen3_params"]), 2)
    self.assertIs(seen["qwen3_params"][0], seen["qwen3_params"][1])
    self.assertTrue(_all_deleted(seen["qwen3_params"][0]))
    self.assertIn("text_encoder_swap_in", trace)

  # -- failures after a swap-in -----------------------------------------------------------------------------

  def _spy_swap_ins(self, pipeline):
    placed = {}
    real_swap_in = pipeline._swap_in

    def spy(component, host_params, trace):
      placed[component] = real_swap_in(component, host_params, trace)
      return placed[component]

    pipeline._swap_in = spy
    return placed

  def test_a_failing_encode_frees_the_swapped_in_text_encoder(self):
    for name, pipeline_kwargs, call_kwargs in (
        ("per call", {}, {"offload_text_encoder": True}),
        ("offload_components", {"offload_components": ("text_encoder",)}, {}),
    ):
      with self.subTest(path=name):
        pipeline, _ = self._pipeline(**pipeline_kwargs)
        placed = self._spy_swap_ins(pipeline)

        def failing_encode(prompts, qwen3_params):
          raise RuntimeError("encode failed")

        pipeline.encode_prompt = failing_encode
        with self.assertRaisesRegex(RuntimeError, "encode failed"):
          self._call(pipeline, qwen3_params=self.host_qwen3, **call_kwargs)
        self.assertTrue(_all_deleted(placed["text_encoder"]))
        self.assertIsInstance(self.host_qwen3["embed"], np.ndarray)

  def test_a_failing_step_frees_the_swapped_in_transformer(self):
    pipeline, seen = self._pipeline(offload_components=("transformer",))
    placed = self._spy_swap_ins(pipeline)

    def fail():
      raise RuntimeError("step failed")

    seen["on_step"] = fail
    with self.assertRaisesRegex(RuntimeError, "step failed"):
      self._call(pipeline, params=self.host_m)
    self.assertTrue(_all_deleted(placed["transformer"]))
    # A good call afterwards: placed again from the intact host tree, same image as a resident call.
    _, resident_seen, _ = self._run()
    pipeline, seen = self._pipeline(offload_components=("transformer",))
    self._call(pipeline, params=self.host_m)
    np.testing.assert_array_equal(seen["latents_5d"], resident_seen["latents_5d"])

  def test_success_path_frees_the_text_encoder_right_after_encoding(self):
    # Today's point: before the DiT runs (the first step sees it freed).
    pipeline, seen = self._pipeline()
    placed = self._spy_swap_ins(pipeline)
    at_first_step = []
    seen["on_step"] = lambda: at_first_step.append(_all_deleted(placed["text_encoder"]))
    self._call(pipeline, qwen3_params=self.host_qwen3, offload_text_encoder=True)
    self.assertTrue(at_first_step[0])

  # -- validation ------------------------------------------------------------------------------------------

  def test_validation_errors(self):
    cases = (
        ({"hybrid_steps": 2}, "hybrid_params"),
        ({"hybrid_steps": _STEPS, "hybrid_params": self.e}, "hybrid_steps"),
        ({"hybrid_steps": -1, "hybrid_params": self.e}, "hybrid_steps"),
        ({"hybrid_steps": True, "hybrid_params": self.e}, "hybrid_steps"),
        ({"hybrid_steps": 2.0, "hybrid_params": self.e}, "hybrid_steps"),
        (
            {"hybrid_steps": 2, "hybrid_params": self.e, "guidance_scale": 1.5, "do_classifier_free_guidance": True},
            "classifier-free guidance",
        ),
        ({"hybrid_steps": 2, "hybrid_params": self.e, "hybrid_evict": True}, "hybrid_host_params"),
    )
    for call_kwargs, pattern in cases:
      with self.subTest(kwargs=sorted(call_kwargs)):
        pipeline, seen = self._pipeline()
        with self.assertRaisesRegex(ValueError, pattern):
          self._call(pipeline, **call_kwargs)
        self.assertEqual(seen["encoded"], [])  # fails before any work
    self.assertTrue(_none_deleted(self.e))

    pipeline, _ = self._pipeline(is_distilled=False)
    with self.assertRaisesRegex(ValueError, "is_distilled"):
      self._call(pipeline, hybrid_steps=2, hybrid_params=self.e)
    pipeline, _ = self._pipeline(offload_components=("transformer",))
    with self.assertRaisesRegex(ValueError, "transformer"):
      self._call(pipeline, params=self.host_m, hybrid_steps=2, hybrid_params=self.e)
    pipeline, _ = self._pipeline(param_shardings={})
    with self.assertRaisesRegex(ValueError, "param_shardings"):
      self._call(pipeline, qwen3_params=self.host_qwen3, offload_text_encoder=True)
    pipeline, _ = self._pipeline(offload_components=("text_encoder",))
    with self.assertRaisesRegex(ValueError, "offload_text_encoder"):
      self._call(pipeline, qwen3_params=self.host_qwen3, offload_text_encoder=False)


# ---------------------------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------------------------


def _runtime_config(**overrides):
  values = {
      "num_inference_steps": 8,
      "guidance_scale": 0.0,
      "do_classifier_free_guidance": False,
      "negative_prompt": "",
      "batch_size": 1,
      "output_dir": "/tmp/out/",
      "output_name": "krea2.png",
      "max_sequence_length": 512,
      "is_distilled": True,
      "krea2_weight_cache_wait_s": 0,
      "krea2_hybrid_weight_cache_dir": "",
      "krea2_hybrid_model_id": "",
      "krea2_hybrid_max_steps": 3,
      "krea2_hybrid_guidance": 1.5,
      "krea2_hybrid_negative_prompt": "",
      "krea2_text_encoder_offload_min_pixels": 0,
      "krea2_hybrid_evict_min_pixels": 0,
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


class _FakePipeline:
  """Records each call; `on_call(kwargs)` may simulate what the real pipeline does to the trees."""

  def __init__(self, param_shardings=None):
    self.calls = []
    self.param_shardings = dict(param_shardings or {})
    self.offload_components = frozenset()
    self.on_call = None

  def __call__(self, **kwargs):
    self.calls.append(kwargs)
    trace = {"prompt_encoding": 0.1, "denoise_loop": 1.0, "vae_decode": 0.2, "text_tokens": 128, "seed": 1}
    if self.on_call is not None:
      self.on_call(kwargs, trace)
    return ["image"], trace


_MAIN_META = {"model": "/m/merged", "snapshot": "merged", "quantization": "w8a8", "build_revision": 1}


def _tiny_tree(offset=0.0):
  return {
      "blocks_0": {"kernel": np.arange(12, dtype=np.float32).reshape(3, 4) + offset},
      "txt_in": {"bias": np.full((5,), 2.0 + offset, np.float32)},
  }


def _abstract(tree):
  return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)


class Krea2HybridRuntimeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(generate_krea2.nn, "intercept_methods", side_effect=lambda interceptor: interceptor)
    patcher.start()
    self.addCleanup(patcher.stop)
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.cache_dir = tmp.name
    self.main_host = _tiny_tree()
    self.e_host = _tiny_tree(offset=10.0)
    self.qwen3_host = {"embed": np.ones((4, 6), np.float32)}
    self.shardings = {"transformer": _replicated(self.main_host), "text_encoder": _replicated(self.qwen3_host)}

  def _save_e(self, model_id="krea2-rawft3513-b0", tree=None):
    meta = generate_krea2.hybrid_transformer_cache_meta(_MAIN_META, model_id)
    saved = weight_cache.save_component(self.cache_dir, "transformer", meta, tree or self.e_host)
    self.assertIsNotNone(saved)
    return saved

  def _runtime(self, te_offload=False, **config):
    self.pipeline = _FakePipeline(self.shardings)
    qwen3 = place_params(self.qwen3_host, self.shardings["text_encoder"])
    return Krea2Runtime(
        config=_runtime_config(**config),
        pipeline=self.pipeline,
        params=place_params(self.main_host, self.shardings["transformer"]),
        qwen3_params=qwen3,
        lora_interceptors=(),
        default_prompts=["p"],
        default_latents=None,
        default_height=1024,
        default_width=1024,
        load_time=1.0,
        timeline=StartupTimeline(0.0),
        precompile_plan=[],
        text_compaction_multiple=128,
        qwen3_host_params=self.qwen3_host if te_offload else None,
        transformer_cache_meta=dict(_MAIN_META),
        transformer_abstract_params=_abstract(self.main_host),
        transformer_shardings=self.shardings["transformer"],
    )

  def _hybrid_runtime(self, **config):
    self._save_e()
    config = {"krea2_hybrid_weight_cache_dir": self.cache_dir, "krea2_hybrid_model_id": "krea2-rawft3513-b0", **config}
    runtime = self._runtime(**config)
    self.assertTrue(runtime.load_hybrid_params())
    return runtime

  # -- cache meta and config ------------------------------------------------------------------------------

  def test_hybrid_transformer_cache_meta(self):
    meta = generate_krea2.hybrid_transformer_cache_meta(_MAIN_META, "krea2-rawft3513-b0")
    self.assertEqual(meta, {**_MAIN_META, "model": "krea2-rawft3513-b0", "snapshot": "krea2-rawft3513-b0"})
    self.assertEqual(_MAIN_META["model"], "/m/merged")  # not mutated
    self.assertNotEqual(
        weight_cache.component_dir("/c", "transformer", meta), weight_cache.component_dir("/c", "transformer", _MAIN_META)
    )

  def test_resolve_hybrid_config_defaults_are_off(self):
    hybrid = generate_krea2.resolve_hybrid_config(_runtime_config())
    self.assertEqual(hybrid.cache_dir, "")
    self.assertEqual((hybrid.text_encoder_offload_min_pixels, hybrid.evict_min_pixels), (0, 0))
    # Quotes of a command-line override mean off; a config without the keys too.
    self.assertEqual(generate_krea2.resolve_hybrid_config(_runtime_config(krea2_hybrid_weight_cache_dir="''")).cache_dir, "")
    self.assertEqual(generate_krea2.resolve_hybrid_config(types.SimpleNamespace()).cache_dir, "")
    hybrid = generate_krea2.resolve_hybrid_config(
        _runtime_config(
            krea2_hybrid_weight_cache_dir="/w",
            krea2_hybrid_model_id="'e'",
            krea2_hybrid_max_steps=7,
            krea2_hybrid_guidance=2,
            krea2_text_encoder_offload_min_pixels=2000000,
            krea2_hybrid_evict_min_pixels=1,
        )
    )
    self.assertEqual(
        (hybrid.cache_dir, hybrid.model_id, hybrid.max_steps, hybrid.guidance), ("/w", "e", 7, 2.0)
    )
    self.assertIsInstance(hybrid.guidance, float)
    self.assertEqual((hybrid.text_encoder_offload_min_pixels, hybrid.evict_min_pixels), (2000000, 1))

  def test_resolve_hybrid_config_errors(self):
    on = {"krea2_hybrid_weight_cache_dir": "/w", "krea2_hybrid_model_id": "e"}
    for overrides, pattern in (
        ({"krea2_hybrid_weight_cache_dir": "/w"}, "krea2_hybrid_model_id"),
        ({**on, "krea2_hybrid_max_steps": 0}, "krea2_hybrid_max_steps"),
        ({**on, "krea2_hybrid_max_steps": 8}, "krea2_hybrid_max_steps"),
        ({**on, "krea2_hybrid_max_steps": True}, "krea2_hybrid_max_steps"),
        ({**on, "num_inference_steps": 3}, "krea2_hybrid_max_steps"),
        ({**on, "krea2_hybrid_guidance": -1.0}, "krea2_hybrid_guidance"),
        ({**on, "krea2_hybrid_guidance": float("nan")}, "krea2_hybrid_guidance"),
        ({**on, "krea2_hybrid_guidance": "1.5"}, "krea2_hybrid_guidance"),
        ({**on, "krea2_hybrid_negative_prompt": 3}, "krea2_hybrid_negative_prompt"),
        ({**on, "is_distilled": False}, "is_distilled"),
        ({**on, "guidance_scale": 3.5, "do_classifier_free_guidance": True}, "classifier-free guidance"),
        ({**on, "guidance_scale": 3.5, "do_classifier_free_guidance": None}, "classifier-free guidance"),
        ({"krea2_text_encoder_offload_min_pixels": -1}, "krea2_text_encoder_offload_min_pixels"),
        ({"krea2_text_encoder_offload_min_pixels": 1.5}, "krea2_text_encoder_offload_min_pixels"),
        ({"krea2_hybrid_evict_min_pixels": True}, "krea2_hybrid_evict_min_pixels"),
    ):
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(ValueError, pattern):
          generate_krea2.resolve_hybrid_config(_runtime_config(**overrides))
    config = _runtime_config(**on)
    with self.assertRaisesRegex(ValueError, "krea2_offload_components"):
      generate_krea2.resolve_hybrid_config(config, offload_components=("transformer",))
    with self.assertRaisesRegex(ValueError, "LoRA"):
      generate_krea2.resolve_hybrid_config(config, lora_configured=True)
    # Without the hybrid dir neither matters.
    generate_krea2.resolve_hybrid_config(_runtime_config(), offload_components=("transformer",), lora_configured=True)
    # A guidance scale with CFG switched off (or CFG on with scale 0) is no CFG: accepted.
    generate_krea2.resolve_hybrid_config(_runtime_config(**on, guidance_scale=3.5, do_classifier_free_guidance=False))
    generate_krea2.resolve_hybrid_config(_runtime_config(**on, guidance_scale=0.0, do_classifier_free_guidance=True))
    # Main CFG on / Raw config without the hybrid dir: not this function's business.
    generate_krea2.resolve_hybrid_config(_runtime_config(guidance_scale=3.5, do_classifier_free_guidance=True, is_distilled=False))

  # -- load_hybrid_params ----------------------------------------------------------------------------------

  def test_load_hybrid_params_from_a_saved_component(self):
    runtime = self._hybrid_runtime()
    self.assertEqual(runtime.hybrid_max_steps, 3)
    self.assertEqual(runtime.hybrid_guidance, 1.5)
    self.assertIn("hybrid_load", runtime.load_trace)
    self.assertIsNone(runtime.hybrid_host_params)  # no evict: no host copy
    for leaf, expected in zip(jax.tree_util.tree_leaves(runtime.hybrid_params), jax.tree_util.tree_leaves(self.e_host)):
      self.assertIsInstance(leaf, jax.Array)
      np.testing.assert_array_equal(np.asarray(leaf), expected)
    self.assertEqual(
        jax.tree_util.tree_structure(runtime.hybrid_params), jax.tree_util.tree_structure(runtime.params)
    )
    # Idempotent.
    placed = runtime.hybrid_params
    self.assertTrue(runtime.load_hybrid_params())
    self.assertIs(runtime.hybrid_params, placed)

  def test_load_keeps_a_host_copy_only_for_evict(self):
    runtime = self._hybrid_runtime(krea2_hybrid_evict_min_pixels=4000000)
    self.assertIsNotNone(runtime.hybrid_host_params)
    for leaf in jax.tree_util.tree_leaves(runtime.hybrid_host_params):
      self.assertIsInstance(leaf, np.ndarray)

  def test_load_without_the_dir_is_a_no_op(self):
    runtime = self._runtime()
    self.assertFalse(runtime.load_hybrid_params())
    self.assertIsNone(runtime.hybrid_params)
    self.assertEqual(runtime.hybrid_max_steps, 0)
    self.assertNotIn("hybrid_load", runtime.load_trace)

  def test_load_errors(self):
    runtime = self._runtime(krea2_hybrid_weight_cache_dir=self.cache_dir, krea2_hybrid_model_id="krea2-rawft3513-b0")
    expected_dir = weight_cache.component_dir(
        self.cache_dir, "transformer", generate_krea2.hybrid_transformer_cache_meta(_MAIN_META, "krea2-rawft3513-b0")
    )
    with self.assertRaisesRegex(RuntimeError, expected_dir):
      runtime.load_hybrid_params()
    self.assertEqual(runtime.hybrid_max_steps, 0)

    # Another model id is another directory.
    self._save_e(model_id="other")
    with self.assertRaisesRegex(RuntimeError, expected_dir):
      runtime.load_hybrid_params()

    # A stored tree that does not fit the runtime model.
    self._save_e(tree={"blocks_0": {"kernel": np.zeros((3, 5), np.float32)}, "txt_in": self.e_host["txt_in"]})
    with self.assertRaises((RuntimeError, ValueError)):
      runtime.load_hybrid_params()
    self.assertIsNone(runtime.hybrid_params)

    runtime = self._runtime(krea2_hybrid_weight_cache_dir=self.cache_dir)
    with self.assertRaisesRegex(ValueError, "krea2_hybrid_model_id"):
      runtime.load_hybrid_params()

  def test_load_waits_for_the_component_with_the_weight_cache_wait(self):
    runtime = self._hybrid_runtime()
    runtime.hybrid_params = None
    runtime.config.krea2_weight_cache_wait_s = 5
    with mock.patch.object(weight_cache, "wait_for_component", return_value=0.0) as wait:
      self.assertTrue(runtime.load_hybrid_params())
    wait.assert_called_once()
    self.assertEqual(wait.call_args.args[:2], (self.cache_dir, "transformer"))
    self.assertIn("hybrid_wait", runtime.load_trace)

  # -- generate ---------------------------------------------------------------------------------------------

  def test_generate_validates_hybrid_steps(self):
    runtime = self._runtime()
    for steps in (1, -1, True, 2.0, "2"):
      with self.subTest(steps=steps):
        with self.assertRaisesRegex(ValueError, "hybrid_steps"):
          runtime.generate(["a"], height=1024, width=1024, hybrid_steps=steps)
    runtime = self._hybrid_runtime()
    with self.assertRaisesRegex(ValueError, "hybrid_steps"):
      runtime.generate(["a"], height=1024, width=1024, hybrid_steps=4)
    self.assertEqual(self.pipeline.calls, [])

  def test_generate_passes_hybrid_kwargs_only_for_hybrid_requests(self):
    runtime = self._hybrid_runtime(krea2_hybrid_negative_prompt="ugly")
    _, trace = runtime.generate(["a"], height=1024, width=1024)
    default_keys = set(runtime.call_kwargs(height=1, width=1)) | {"prompt", "output_name"}
    self.assertEqual(set(self.pipeline.calls[0]), default_keys)
    self.assertNotIn("hybrid_steps", trace)

    runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    kwargs = self.pipeline.calls[1]
    self.assertIs(kwargs["hybrid_params"], runtime.hybrid_params)
    self.assertEqual(kwargs["hybrid_steps"], 2)
    self.assertEqual(kwargs["hybrid_guidance"], 1.5)
    self.assertEqual(kwargs["hybrid_negative_prompt"], "ugly")
    self.assertNotIn("hybrid_evict", kwargs)
    self.assertNotIn("offload_text_encoder", kwargs)
    self.assertIs(kwargs["params"], runtime.params)

  def test_text_encoder_per_request_offload(self):
    runtime = self._runtime(te_offload=True, krea2_text_encoder_offload_min_pixels=2048 * 2048)
    resident = runtime.qwen3_params

    # A large request: the device copy is freed, the pipeline gets the host tree to swap in.
    _, trace = runtime.generate(["a"], height=2048, width=2048)
    kwargs = self.pipeline.calls[0]
    self.assertIs(kwargs["offload_text_encoder"], True)
    self.assertIs(kwargs["qwen3_params"], self.qwen3_host)
    self.assertTrue(_all_deleted(resident))
    self.assertNotIn("text_encoder_replace_s", trace)
    # Another large one: nothing to free.
    runtime.generate(["a"], height=2048, width=2048)
    self.assertIs(self.pipeline.calls[1]["qwen3_params"], self.qwen3_host)

    # A small one: placed again from the host copy before the call.
    _, trace = runtime.generate(["a"], height=1024, width=1024)
    kwargs = self.pipeline.calls[2]
    self.assertNotIn("offload_text_encoder", kwargs)
    self.assertIs(kwargs["qwen3_params"], runtime.qwen3_params)
    self.assertTrue(_none_deleted(runtime.qwen3_params))
    np.testing.assert_array_equal(np.asarray(runtime.qwen3_params["embed"]), self.qwen3_host["embed"])
    self.assertGreaterEqual(trace["text_encoder_replace_s"], 0.0)
    _, trace = runtime.generate(["a"], height=1024, width=1024)
    self.assertNotIn("text_encoder_replace_s", trace)

  def test_text_encoder_is_placed_back_for_warmup_and_precompile(self):
    runtime = self._runtime(te_offload=True, krea2_text_encoder_offload_min_pixels=1)
    runtime.generate(["a"], height=1024, width=1024)
    self.assertIsNone(runtime.qwen3_params)
    with mock.patch.object(generate_krea2, "run_precompile", return_value=[]) as run:
      runtime.precompile([])
    self.assertTrue(_none_deleted(run.call_args.args[2]["qwen3_params"]))
    self.assertIsNotNone(runtime.qwen3_params)

  def test_text_encoder_state_after_a_failing_large_request(self):
    runtime = self._runtime(te_offload=True, krea2_text_encoder_offload_min_pixels=2048 * 2048)

    def fail(kwargs, trace):
      raise RuntimeError("boom")

    self.pipeline.on_call = fail
    with self.assertRaisesRegex(RuntimeError, "boom"):
      runtime.generate(["a"], height=2048, width=2048)
    # Freed before the call (the pipeline frees its own swap-in on failure): off the device, and the state says so.
    self.assertIsNone(runtime.qwen3_params)
    self.pipeline.on_call = None
    _, trace = runtime.generate(["a"], height=1024, width=1024)
    self.assertIn("text_encoder_replace_s", trace)
    self.assertTrue(_none_deleted(runtime.qwen3_params))

  def test_text_bucket_never_needs_the_text_encoder(self):
    # The encode pool's bucket lookup only tokenizes: it works while the text encoder is off the device.
    from maxdiffusion.tests.krea2_runtime_test import _WordTokenizer  # pylint: disable=import-outside-toplevel

    runtime = self._runtime(te_offload=True, krea2_text_encoder_offload_min_pixels=1)
    self.pipeline.tokenizer = _WordTokenizer()
    self.pipeline.text_compaction_multiple = 128
    runtime.generate(["a"], height=1024, width=1024)
    self.assertIsNone(runtime.qwen3_params)
    self.assertEqual(runtime.text_bucket(["a few words"]), 128)
    self.assertIsNone(runtime.qwen3_params)

  def test_without_the_knob_the_text_encoder_stays(self):
    runtime = self._runtime()
    resident = runtime.qwen3_params
    runtime.generate(["a"], height=2048, width=2048)
    self.assertIs(self.pipeline.calls[0]["qwen3_params"], resident)
    self.assertNotIn("offload_text_encoder", self.pipeline.calls[0])
    self.assertTrue(_none_deleted(resident))

  def _simulate_pipeline_evict(self, kwargs, trace):
    # What the real pipeline does after E's steps with hybrid_evict.
    if kwargs.get("hybrid_evict"):
      free_params(kwargs["hybrid_params"], kwargs["hybrid_host_params"])
      trace["hybrid_evicted"] = True

  def test_evict_policy_frees_and_replaces_e(self):
    runtime = self._hybrid_runtime(krea2_hybrid_evict_min_pixels=2048 * 2048)
    self.pipeline.on_call = self._simulate_pipeline_evict
    first = runtime.hybrid_params

    # Small hybrid request: no evict.
    runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    self.assertNotIn("hybrid_evict", self.pipeline.calls[0])
    self.assertIs(runtime.hybrid_params, first)

    # Large hybrid request: the pipeline frees E after its steps, the runtime places it again.
    _, trace = runtime.generate(["a"], height=2048, width=2048, hybrid_steps=3)
    kwargs = self.pipeline.calls[1]
    self.assertIs(kwargs["hybrid_evict"], True)
    self.assertIs(kwargs["hybrid_host_params"], runtime.hybrid_host_params)
    self.assertTrue(_all_deleted(first))
    self.assertTrue(_none_deleted(runtime.hybrid_params))
    self.assertIs(trace["hybrid_evicted"], True)
    self.assertGreaterEqual(trace["hybrid_replace_s"], 0.0)
    for leaf, expected in zip(jax.tree_util.tree_leaves(runtime.hybrid_params), jax.tree_util.tree_leaves(self.e_host)):
      np.testing.assert_array_equal(np.asarray(leaf), expected)

    # Large fast request: E is freed before the call (the pipeline call is today's) and placed again after.
    second = runtime.hybrid_params
    seen = {}
    self.pipeline.on_call = lambda kwargs, trace: seen.update(deleted=_all_deleted(second))
    _, trace = runtime.generate(["a"], height=2048, width=2048)
    self.assertTrue(seen["deleted"])
    self.assertNotIn("hybrid_params", self.pipeline.calls[2])
    self.assertIs(trace["hybrid_evicted"], True)
    self.assertIn("hybrid_replace_s", trace)
    self.assertTrue(_none_deleted(runtime.hybrid_params))

  def test_evict_replaces_e_even_when_the_pipeline_fails(self):
    runtime = self._hybrid_runtime(krea2_hybrid_evict_min_pixels=1)

    def fail_after_evict(kwargs, trace):
      self._simulate_pipeline_evict(kwargs, trace)
      raise RuntimeError("boom")

    self.pipeline.on_call = fail_after_evict
    with self.assertRaisesRegex(RuntimeError, "boom"):
      runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    self.assertTrue(_none_deleted(runtime.hybrid_params))

    # A failure before the eviction (E still whole): E is kept as it is, no second placement.
    def fail_early(kwargs, trace):
      raise RuntimeError("early")

    self.pipeline.on_call = fail_early
    before = runtime.hybrid_params
    with mock.patch.object(krea2_pipeline, "place_params", side_effect=AssertionError("placed again")):
      with self.assertRaisesRegex(RuntimeError, "early"):
        runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    self.assertIs(runtime.hybrid_params, before)
    self.assertTrue(_none_deleted(before))

    # A failure after part of E was freed: the rest is freed and E placed whole again.
    def fail_half_freed(kwargs, trace):
      jax.tree_util.tree_leaves(kwargs["hybrid_params"])[0].delete()
      raise RuntimeError("half")

    self.pipeline.on_call = fail_half_freed
    with self.assertRaisesRegex(RuntimeError, "half"):
      runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    self.assertTrue(_all_deleted(before))
    self.assertTrue(_none_deleted(runtime.hybrid_params))

  def test_a_failed_replacement_heals_on_the_next_request_that_needs_e(self):
    runtime = self._hybrid_runtime(krea2_hybrid_evict_min_pixels=2048 * 2048)
    runtime.hybrid_params = None  # what a failed placement after an evicting call leaves
    # A large fast request runs without E anyway: no placement.
    with mock.patch.object(krea2_pipeline, "place_params", side_effect=AssertionError("placed")):
      _, trace = runtime.generate(["a"], height=2048, width=2048)
    self.assertIsNone(runtime.hybrid_params)
    self.assertNotIn("hybrid_replace_s", trace)
    # A small hybrid request places E first.
    _, trace = runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    self.assertIn("hybrid_replace_s", trace)
    self.assertIs(self.pipeline.calls[-1]["hybrid_params"], runtime.hybrid_params)
    self.assertTrue(_none_deleted(runtime.hybrid_params))
    # Without a host copy there is nothing to heal from: a RuntimeError (not the request's ValueError).
    runtime.hybrid_params, runtime.hybrid_host_params = None, None
    calls = len(self.pipeline.calls)
    with self.assertRaisesRegex(RuntimeError, "not on the device"):
      runtime.generate(["a"], height=1024, width=1024, hybrid_steps=2)
    self.assertEqual(len(self.pipeline.calls), calls)

  # -- device memory ----------------------------------------------------------------------------------------

  def test_device_memory_gib(self):
    gib = 1024**3
    device = mock.Mock()
    device.memory_stats.return_value = {"bytes_in_use": 2 * gib, "peak_bytes_in_use": 3 * gib, "bytes_limit": 31 * gib}
    self.assertEqual(generate_krea2.device_memory_gib(device), {"in_use": 2.0, "peak": 3.0, "limit": 31.0})
    device.memory_stats.return_value = None  # the CPU backend
    self.assertEqual(generate_krea2.device_memory_gib(device), {"in_use": None, "peak": None, "limit": None})
    device.memory_stats.side_effect = RuntimeError("unsupported")
    self.assertEqual(generate_krea2.device_memory_gib(device), {"in_use": None, "peak": None, "limit": None})
    # The default device on this (CPU) test backend: never raises.
    self.assertEqual(set(generate_krea2.device_memory_gib()), {"in_use", "peak", "limit"})
    self.assertEqual(set(self._runtime().device_memory_gib()), {"in_use", "peak", "limit"})


if __name__ == "__main__":
  unittest.main()
