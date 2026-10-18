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

# CPU tests for the generate_krea2 resolution presets and precompile mode.
# No model and no checkpoint: configs are SimpleNamespaces and the pipeline is
# a fake callable.

import contextlib
import os
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import yaml

from maxdiffusion import aot_cache, max_logging
from maxdiffusion.generate_krea2 import (
    log_precompile_summary,
    resolve_generation_size,
    resolve_precompile_plan,
    resolve_precompile_workers,
    run_precompile,
)
from maxdiffusion.models.krea2.resolution_presets import (
    KREA2_ASPECT_RATIOS,
    list_krea2_resolutions,
    resolve_krea2_resolution,
)


def _config(**overrides):
  values = {
      "height": 1024,
      "width": 1024,
      "krea2_aspect_ratio": "",
      "krea2_image_size": "",
      "krea2_precompile": "",
      "krea2_precompile_text_tokens": [128],
      "aot_cache_dir": "/tmp/aot",
      "max_sequence_length": 512,
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


class ResolveGenerationSizeTest(unittest.TestCase):

  def test_without_presets_rounds_height_and_width(self):
    self.assertEqual(resolve_generation_size(_config()), (1024, 1024, "1024x1024 (height/width)"))
    height, width, description = resolve_generation_size(_config(height=1000, width=1500))
    self.assertEqual((height, width), (1008, 1504))
    self.assertEqual(description, "1504x1008 (height/width)")

  def test_missing_preset_keys_behave_like_empty(self):
    config = types.SimpleNamespace(height=520, width=776)
    self.assertEqual(resolve_generation_size(config)[:2], (528, 784))

  def test_ratio_only_defaults_to_1k(self):
    height, width, description = resolve_generation_size(_config(krea2_aspect_ratio="16:9"))
    self.assertEqual((height, width), (768, 1344))
    self.assertEqual(description, "preset 1k@16:9 -> 1344x768")

  def test_size_only_defaults_to_square(self):
    height, width, description = resolve_generation_size(_config(krea2_image_size="2k"))
    self.assertEqual((height, width), (2048, 2048))
    self.assertEqual(description, "preset 2k@1:1 -> 2048x2048")

  def test_both_keys_override_height_and_width(self):
    config = _config(height=512, width=512, krea2_aspect_ratio="9:21", krea2_image_size="2K")
    height, width, description = resolve_generation_size(config)
    self.assertEqual((height, width), (3072, 1344))
    self.assertEqual(description, "preset 2k@9:21 -> 1344x3072")

  def test_every_preset_matches_the_table(self):
    for resolution in list_krea2_resolutions():
      config = _config(krea2_aspect_ratio=resolution.aspect_ratio, krea2_image_size=resolution.image_size)
      self.assertEqual(resolve_generation_size(config)[:2], (resolution.height, resolution.width))

  def test_bad_values_raise(self):
    with self.assertRaisesRegex(ValueError, "16x9"):
      resolve_generation_size(_config(krea2_aspect_ratio="16x9"))
    with self.assertRaisesRegex(ValueError, "4k"):
      resolve_generation_size(_config(krea2_image_size="4k"))

  def test_unquoted_yaml_ratio_raises_with_quoting_hint(self):
    # YAML parses an unquoted 16:9 as the sexagesimal integer 969.
    with self.assertRaisesRegex(ValueError, "quoted string"):
      resolve_generation_size(_config(krea2_aspect_ratio=969))


class ResolvePrecompilePlanTest(unittest.TestCase):

  def test_empty_spec_gives_empty_plan(self):
    self.assertEqual(resolve_precompile_plan(_config(), 128), [])
    self.assertEqual(resolve_precompile_plan(_config(krea2_precompile="  "), 128), [])
    # Nothing to compile, so no cache directory is needed either.
    self.assertEqual(resolve_precompile_plan(_config(aot_cache_dir=""), 128), [])
    self.assertEqual(resolve_precompile_plan(types.SimpleNamespace(max_sequence_length=512), 128), [])

  def test_all_with_two_buckets_in_documented_order(self):
    config = _config(krea2_precompile="all", krea2_precompile_text_tokens=[256, 128])
    plan = resolve_precompile_plan(config, 128)
    self.assertEqual(len(plan), 44)
    expected = [(resolution, bucket) for resolution in list_krea2_resolutions() for bucket in (128, 256)]
    self.assertEqual(plan, expected)
    self.assertEqual(plan[0], (resolve_krea2_resolution("1:1", "1k"), 128))
    self.assertEqual(plan[1], (resolve_krea2_resolution("1:1", "1k"), 256))
    self.assertEqual(plan[-1], (resolve_krea2_resolution("9:21", "2k"), 256))
    self.assertEqual([r.aspect_ratio for r, _ in plan[: 2 * len(KREA2_ASPECT_RATIOS) : 2]], list(KREA2_ASPECT_RATIOS))

  def test_spec_order_is_kept(self):
    config = _config(krea2_precompile="2k@16:9, 1k@1:1")
    plan = resolve_precompile_plan(config, 128)
    self.assertEqual([(r.label, bucket) for r, bucket in plan], [("2k@16:9", 128), ("1k@1:1", 128)])

  def test_buckets_are_rounded_clipped_and_deduplicated(self):
    config = _config(krea2_precompile="1k@1:1", krea2_precompile_text_tokens=[600, 100, 128, 129, 512, 300])
    self.assertEqual([bucket for _, bucket in resolve_precompile_plan(config, 128)], [128, 256, 384, 512])
    # A maximum length that is not a multiple of the compaction multiple is the clip.
    config = _config(krea2_precompile="1k@1:1", krea2_precompile_text_tokens=[200, 300], max_sequence_length=300)
    self.assertEqual([bucket for _, bucket in resolve_precompile_plan(config, 128)], [256, 300])

  def test_empty_bucket_list_means_128(self):
    config = _config(krea2_precompile="1k@1:1", krea2_precompile_text_tokens=[])
    self.assertEqual([bucket for _, bucket in resolve_precompile_plan(config, 128)], [128])
    self.assertEqual([bucket for _, bucket in resolve_precompile_plan(config, 256)], [256])
    config = _config(krea2_precompile="1k@1:1", krea2_precompile_text_tokens=())
    self.assertEqual([bucket for _, bucket in resolve_precompile_plan(config, 128)], [128])

  def test_compaction_off_uses_the_full_length(self):
    config = _config(krea2_precompile="2k", krea2_precompile_text_tokens=[128, 256])
    plan = resolve_precompile_plan(config, 0)
    self.assertEqual(len(plan), len(KREA2_ASPECT_RATIOS))
    self.assertEqual({bucket for _, bucket in plan}, {512})

  def test_bad_buckets_raise(self):
    for tokens in ([0], [-128], [128, 0], [128.0], ["128"], [True]):
      with self.subTest(tokens=tokens):
        config = _config(krea2_precompile="all", krea2_precompile_text_tokens=tokens)
        with self.assertRaisesRegex(ValueError, "positive ints"):
          resolve_precompile_plan(config, 128)

  def test_bad_spec_raises(self):
    with self.assertRaisesRegex(ValueError, "3k"):
      resolve_precompile_plan(_config(krea2_precompile="3k"), 128)

  def test_non_empty_spec_needs_cache_dir(self):
    with self.assertRaisesRegex(ValueError, "aot_cache_dir"):
      resolve_precompile_plan(_config(krea2_precompile="all", aot_cache_dir=""), 128)


class _FakePipeline:
  """Records every call's kwargs and the warmup flag at call time.

  By default the trace reports the forced bucket as the one used (plus the
  negative prompt's with `cfg`); `trace_fn(kwargs)` replaces that trace.
  """

  def __init__(self, fail_at=None, cfg=False, trace_fn=None):
    self.calls = []
    self.fail_at = fail_at
    self.cfg = cfg
    self.trace_fn = trace_fn

  def __call__(self, **kwargs):
    self.calls.append((kwargs, aot_cache._STATE.warmup_only))
    if self.fail_at is not None and len(self.calls) == self.fail_at:
      raise RuntimeError("compile failed")
    if self.trace_fn is not None:
      return [], self.trace_fn(kwargs)
    trace = {"prompt_encoding": 0.1, "text_tokens": kwargs["min_text_tokens"]}
    if self.cfg:
      trace["negative_text_tokens"] = kwargs["min_text_tokens"]
    return [], trace


class RunPrecompileTest(unittest.TestCase):

  def setUp(self):
    # warmup_mode only sets the flag when the cache is enabled.
    patcher = mock.patch.object(aot_cache._STATE, "enabled", True)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.addCleanup(setattr, aot_cache._STATE, "warmup_only", False)
    self.plan = [
        (resolve_krea2_resolution("16:9", "2k"), 128),
        (resolve_krea2_resolution("16:9", "2k"), 256),
        (resolve_krea2_resolution("2:3", "1k"), 128),
    ]
    self.call_kwargs = {
        "params": "params",
        "qwen3_params": "qwen3_params",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 8,
        # Guidance off (as for Turbo): the fake pipeline reports no negative bucket.
        "guidance_scale": 0.0,
        "latents": "latents for 1024x1024 only",
        "output_dir": "output/",
    }
    self.saves = []

  def _save_pending(self):
    # Saving must happen outside the warmup context.
    self.saves.append(aot_cache._STATE.warmup_only)
    return 2

  def test_one_warmup_call_and_one_save_per_entry(self):
    pipeline = _FakePipeline()
    with mock.patch.object(aot_cache, "save_pending", side_effect=self._save_pending) as save:
      records = run_precompile(pipeline, self.plan, self.call_kwargs, ["a fox"])

    self.assertEqual(len(pipeline.calls), 3)
    self.assertEqual(save.call_count, 3)
    self.assertEqual(self.saves, [False, False, False])
    for (kwargs, warmup_only), (resolution, text_tokens) in zip(pipeline.calls, self.plan):
      self.assertTrue(warmup_only)
      self.assertEqual(kwargs["height"], resolution.height)
      self.assertEqual(kwargs["width"], resolution.width)
      self.assertEqual(kwargs["min_text_tokens"], text_tokens)
      self.assertIs(kwargs["save_outputs"], False)
      self.assertIsNone(kwargs["latents"])
      self.assertEqual(kwargs["prompt"], ["a fox"])
      self.assertEqual(kwargs["params"], "params")
      self.assertEqual(kwargs["num_inference_steps"], 8)
    self.assertEqual(pipeline.calls[0][0]["height"], 1536)
    self.assertEqual(pipeline.calls[0][0]["width"], 2688)
    # The caller's kwargs are not modified.
    self.assertEqual(self.call_kwargs["latents"], "latents for 1024x1024 only")
    self.assertEqual(self.call_kwargs["height"], 1024)
    self.assertFalse(aot_cache._STATE.warmup_only)

    self.assertEqual(len(records), 3)
    expected_keys = {"label", "height", "width", "image_tokens", "text_tokens", "requested_text_tokens"}
    expected_keys |= {"seconds", "saved"}
    for record in records:
      self.assertEqual(set(record), expected_keys)
      self.assertGreaterEqual(record["seconds"], 0.0)
      self.assertEqual(record["saved"], 2)
      self.assertEqual(record["text_tokens"], record["requested_text_tokens"])
    self.assertEqual(
        {k: records[0][k] for k in ("label", "height", "width", "image_tokens", "text_tokens")},
        {"label": "2k@16:9", "height": 1536, "width": 2688, "image_tokens": 16128, "text_tokens": 128},
    )
    self.assertEqual(records[1]["text_tokens"], 256)
    self.assertEqual(records[2]["label"], "1k@2:3")
    self.assertEqual((records[2]["height"], records[2]["width"]), (1248, 832))

    with mock.patch.object(max_logging, "log") as log:
      log_precompile_summary(records)
    lines = [call.args[0] for call in log.call_args_list]
    self.assertTrue(any(line.startswith("2k@16:9") and "2688x1536" in line for line in lines))
    self.assertTrue(any("3 entries, 6 executable(s) saved" in line for line in lines))

  def test_warmup_flag_is_reset_when_the_pipeline_raises(self):
    pipeline = _FakePipeline(fail_at=2)
    with mock.patch.object(aot_cache, "save_pending", side_effect=self._save_pending) as save:
      with self.assertRaisesRegex(RuntimeError, "compile failed"):
        run_precompile(pipeline, self.plan, self.call_kwargs, ["a fox"])
    self.assertFalse(aot_cache._STATE.warmup_only)
    # The first entry was saved before the second one failed.
    self.assertEqual(save.call_count, 1)
    self.assertEqual(self.saves, [False])

  def _guided_kwargs(self):
    return {**self.call_kwargs, "guidance_scale": 1.5, "do_classifier_free_guidance": True}

  def test_cfg_record_carries_the_negative_bucket(self):
    pipeline = _FakePipeline(cfg=True)
    with (
        mock.patch.object(aot_cache, "save_pending", side_effect=self._save_pending),
        mock.patch.object(max_logging, "log") as log,
    ):
      records = run_precompile(pipeline, self.plan, self._guided_kwargs(), ["a fox"])
    self.assertEqual([r["negative_text_tokens"] for r in records], [128, 256, 128])
    self.assertEqual([r["text_tokens"] for r in records], [128, 256, 128])
    lines = [call.args[0] for call in log.call_args_list]
    self.assertIn("[precompile 2/3] 2k@16:9 2688x1536 text 256 (negative 256)", lines[1])

  def _assert_stops_after_first_entry(self, pipeline, *patterns, call_kwargs=None):
    self.saves = []
    with mock.patch.object(aot_cache, "save_pending", side_effect=self._save_pending) as save:
      with self.assertRaises(RuntimeError) as ctx:
        run_precompile(pipeline, self.plan, call_kwargs or self.call_kwargs, ["a fox"])
    for pattern in patterns:
      self.assertIn(pattern, str(ctx.exception))
    # What the failing entry compiled was saved; the next entries never started.
    self.assertEqual(save.call_count, 1)
    self.assertEqual(self.saves, [False])
    self.assertEqual(len(pipeline.calls), 1)
    self.assertFalse(aot_cache._STATE.warmup_only)

  def test_larger_actual_bucket_raises_after_saving(self):
    # A prompt longer than the requested bucket makes the pipeline compile a larger one.
    pipeline = _FakePipeline(trace_fn=lambda kwargs: {"text_tokens": 256})
    self._assert_stops_after_first_entry(pipeline, "[precompile 1/3] 2k@16:9", "requested text bucket 128", "prompt 256")

  def test_larger_negative_bucket_raises_after_saving(self):
    pipeline = _FakePipeline(trace_fn=lambda kwargs: {"text_tokens": 128, "negative_text_tokens": 384})
    self._assert_stops_after_first_entry(pipeline, "2k@16:9", "requested text bucket 128", "negative prompt 384")

  def test_trace_without_text_tokens_raises(self):
    pipeline = _FakePipeline(trace_fn=lambda kwargs: {"prompt_encoding": 0.1})
    self._assert_stops_after_first_entry(pipeline, "2k@16:9", "'text_tokens'", "128")

  def test_guidance_requires_the_negative_bucket(self):
    # Explicitly on, and on through the pipeline's defaults (guidance_scale 4.5, switch unset).
    defaults = {k: v for k, v in self.call_kwargs.items() if k != "guidance_scale"}
    for call_kwargs in (self._guided_kwargs(), defaults):
      with self.subTest(call_kwargs={k: call_kwargs.get(k) for k in ("guidance_scale", "do_classifier_free_guidance")}):
        pipeline = _FakePipeline()  # reports text_tokens only
        self._assert_stops_after_first_entry(
            pipeline, "[precompile 1/3] 2k@16:9", "'negative_text_tokens'", call_kwargs=call_kwargs
        )

  def test_negative_bucket_is_optional_without_guidance(self):
    # Guidance off through the switch although guidance_scale > 0.
    call_kwargs = {**self.call_kwargs, "guidance_scale": 4.5, "do_classifier_free_guidance": False}
    for kwargs in (self.call_kwargs, call_kwargs):
      pipeline = _FakePipeline()
      with mock.patch.object(aot_cache, "save_pending", side_effect=self._save_pending):
        records = run_precompile(pipeline, self.plan, kwargs, ["a fox"])
      self.assertEqual(len(pipeline.calls), 3)
      self.assertTrue(all("negative_text_tokens" not in record for record in records))

  def test_empty_plan_does_nothing(self):
    pipeline = _FakePipeline()
    with mock.patch.object(aot_cache, "save_pending", side_effect=self._save_pending) as save:
      self.assertEqual(run_precompile(pipeline, [], self.call_kwargs, ["a fox"]), [])
    self.assertEqual(pipeline.calls, [])
    self.assertEqual(save.call_count, 0)


class _ThreadedFakePipeline:
  """A thread-safe fake pipeline for the parallel precompile path.

  Records each call's kwargs, thread and warmup flag; `gate(index, kwargs)`
  runs inside the call (block, sleep, raise), `trace_fn(kwargs)` replaces the
  trace like in _FakePipeline. `events` interleaves ("start"/"end", index) of
  calls with ("save", active calls) of the patched save.
  """

  def __init__(self, plan, gate=None, trace_fn=None, tokenizer=None):
    self.plan = plan
    self.gate = gate
    self.trace_fn = trace_fn
    self.tokenizer = tokenizer
    self.offload_components = frozenset()
    self.lock = threading.Lock()
    self.calls = []
    self.events = []
    self.active = 0
    self.setups = []

  def _setup_jit_functions(self):
    with self.lock:
      self.setups.append((threading.current_thread().name, len(self.calls)))

  def _index(self, kwargs):
    for index, (resolution, requested) in enumerate(self.plan, start=1):
      if (resolution.height, resolution.width, requested) == (kwargs["height"], kwargs["width"], kwargs["min_text_tokens"]):
        return index
    raise AssertionError(f"not a plan entry: {kwargs}")

  def __call__(self, **kwargs):
    index = self._index(kwargs)
    with self.lock:
      self.calls.append(
          {
              "index": index,
              "kwargs": kwargs,
              "thread": threading.current_thread().name,
              "warmup_only": aot_cache._STATE.warmup_only,
              "tokenizer": self.tokenizer,
          }
      )
      self.events.append(("start", index))
      self.active += 1
    try:
      if self.gate is not None:
        self.gate(index, kwargs)
      if self.trace_fn is not None:
        return [], self.trace_fn(kwargs)
      return [], {"prompt_encoding": 0.1, "text_tokens": kwargs["min_text_tokens"]}
    finally:
      with self.lock:
        self.active -= 1
        self.events.append(("end", index))

  def started(self):
    return sorted(call["index"] for call in self.calls)


class RunPrecompileParallelTest(unittest.TestCase):

  def setUp(self):
    patcher = mock.patch.object(aot_cache._STATE, "enabled", True)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.addCleanup(setattr, aot_cache._STATE, "warmup_only", False)
    self.plan = [
        (resolve_krea2_resolution("16:9", "2k"), 128),
        (resolve_krea2_resolution("16:9", "2k"), 256),
        (resolve_krea2_resolution("2:3", "1k"), 128),
        (resolve_krea2_resolution("1:1", "1k"), 128),
    ]
    self.call_kwargs = {
        "params": "params",
        "qwen3_params": "qwen3_params",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
        "latents": "latents for 1024x1024 only",
        "output_dir": "output/",
    }
    self.saves = []

  def _run(self, pipeline, plan=None, **kwargs):
    """run_precompile with the save and the log patched; returns (records or exception, log lines)."""

    def save_pending():
      with pipeline.lock:
        pipeline.events.append(("save", pipeline.active))
      self.saves.append(aot_cache._STATE.warmup_only)
      return 2

    with (
        mock.patch.object(aot_cache, "save_pending", side_effect=save_pending),
        mock.patch.object(aot_cache, "warmup_mode", wraps=aot_cache.warmup_mode) as self.warmup_mode,
        mock.patch.object(max_logging, "log") as log,
    ):
      try:
        result = run_precompile(pipeline, self.plan if plan is None else plan, self.call_kwargs, ["a fox"], **kwargs)
      except Exception as e:  # noqa: BLE001 - returned to the test
        result = e
    return result, [call.args[0] for call in log.call_args_list]

  def test_every_entry_runs_concurrently_and_records_keep_plan_order(self):
    # All four calls meet at the barrier (so they run at once), then finish in reverse plan order.
    barrier = threading.Barrier(4, timeout=60)

    def gate(index, kwargs):
      barrier.wait()
      time.sleep(0.05 * (len(self.plan) - index))

    pipeline = _ThreadedFakePipeline(self.plan, gate=gate)
    records, lines = self._run(pipeline, workers=4)

    self.assertIsInstance(records, list)
    self.assertEqual(pipeline.started(), [1, 2, 3, 4])
    self.assertEqual(
        [(r["label"], r["requested_text_tokens"], r["text_tokens"]) for r in records],
        [("2k@16:9", 128, 128), ("2k@16:9", 256, 256), ("1k@2:3", 128, 128), ("1k@1:1", 128, 128)],
    )
    for record in records:
      self.assertEqual(record["saved"], 2)
      self.assertGreaterEqual(record["seconds"], 0.0)
    for call in pipeline.calls:
      self.assertTrue(call["warmup_only"])
      self.assertTrue(call["thread"].startswith("krea2-precompile"))
      self.assertIs(call["kwargs"]["save_outputs"], False)
      self.assertIsNone(call["kwargs"]["latents"])
      self.assertEqual(call["kwargs"]["prompt"], ["a fox"])
    self.assertEqual(self.call_kwargs["latents"], "latents for 1024x1024 only")
    # One warmup context around the whole pool, left once every thread is done.
    self.assertEqual(self.warmup_mode.call_count, 1)
    self.assertFalse(aot_cache._STATE.warmup_only)
    # One save after every completed entry, while the flag is still set.
    self.assertEqual(self.saves, [True] * 4)
    ends = 0
    for event in pipeline.events:
      ends += event[0] == "end"
      if event[0] == "save":
        self.assertGreaterEqual(ends, 1)
    self.assertEqual(lines[0], "[precompile] parallel: 4 entries, 4 workers")
    entry_lines = lines[1:]
    self.assertEqual(len(entry_lines), 4)
    # Completion order (reverse), with the per-entry format of the sequential path.
    self.assertTrue(entry_lines[0].startswith("[precompile 4/4] 1k@1:1 1024x1024 text 128: "))
    self.assertTrue(entry_lines[3].startswith("[precompile 1/4] 2k@16:9 2688x1536 text 128: "))
    self.assertTrue(all(line.endswith("s, 2 executable(s) saved") for line in entry_lines))
    # The jitted entries were built once, on the calling thread, before any call.
    self.assertEqual(pipeline.setups, [(threading.current_thread().name, 0)])

  def test_workers_are_capped_by_the_plan(self):
    pipeline = _ThreadedFakePipeline(self.plan[:2])
    records, lines = self._run(pipeline, plan=self.plan[:2], workers=8)
    self.assertEqual(len(records), 2)
    self.assertEqual(lines[0], "[precompile] parallel: 2 entries, 2 workers")

  def test_one_entry_or_one_worker_takes_the_sequential_path(self):
    for plan, workers in ((self.plan[:1], 8), (self.plan, 1)):
      with self.subTest(entries=len(plan), workers=workers):
        pipeline = _ThreadedFakePipeline(plan)
        self.saves = []
        records, lines = self._run(pipeline, plan=plan, workers=workers)
        self.assertEqual(len(records), len(plan))
        self.assertFalse(any("parallel" in line or "workers" in line for line in lines))
        self.assertTrue(all(call["thread"] == threading.current_thread().name for call in pipeline.calls))
        self.assertEqual(self.saves, [False] * len(plan))
        self.assertEqual(pipeline.setups, [])

  def test_empty_plan_does_nothing(self):
    pipeline = _ThreadedFakePipeline([])
    records, lines = self._run(pipeline, plan=[], workers=4)
    self.assertEqual(records, [])
    self.assertEqual((pipeline.calls, lines, self.saves), ([], [], []))

  def _assert_sequential_fallback(self, pipeline, reason):
    records, lines = self._run(pipeline, workers=4)
    self.assertEqual(len(records), 4)
    self.assertEqual(lines[0], f"[precompile] 4 workers requested, compiling one entry after the other: {reason}")
    self.assertFalse(any("parallel:" in line for line in lines))
    self.assertEqual(len(lines), 5)
    self.assertTrue(all(call["thread"] == threading.current_thread().name for call in pipeline.calls))
    self.assertEqual(pipeline.started(), [1, 2, 3, 4])

  def test_aot_cache_off_falls_back_to_sequential(self):
    pipeline = _ThreadedFakePipeline(self.plan)
    with mock.patch.object(aot_cache._STATE, "enabled", False):
      self._assert_sequential_fallback(pipeline, "the AOT cache is off, so warmup calls execute")
    # Warmup mode is a no-op without the cache: the calls executed.
    self.assertFalse(any(call["warmup_only"] for call in pipeline.calls))

  def test_offloaded_components_fall_back_to_sequential(self):
    pipeline = _ThreadedFakePipeline(self.plan)
    pipeline.offload_components = frozenset({"transformer", "text_encoder"})
    self._assert_sequential_fallback(pipeline, "offloaded components text_encoder, transformer")

  def test_bucket_mismatch_cancels_waiting_entries_saves_and_raises(self):
    # Entry 1 compiles a larger bucket; entry 2 is still running when it fails; the next entry a free
    # worker may pick up runs slowly, so the remaining ones are still queued when the failure cancels them.
    entry_2_running = threading.Event()

    def gate(index, kwargs):
      if index == 1:
        self.assertTrue(entry_2_running.wait(timeout=60))
      elif index == 2:
        entry_2_running.set()
        time.sleep(0.3)
      else:
        time.sleep(0.5)

    def trace_fn(kwargs):
      if kwargs["min_text_tokens"] == 128 and kwargs["height"] == 1536:
        return {"text_tokens": 256}
      return {"text_tokens": kwargs["min_text_tokens"]}

    plan = self.plan + [(resolve_krea2_resolution("3:2", "1k"), 128), (resolve_krea2_resolution("4:3", "1k"), 128)]
    pipeline = _ThreadedFakePipeline(plan, gate=gate, trace_fn=trace_fn)
    error, lines = self._run(pipeline, plan=plan, workers=2)

    self.assertIsInstance(error, RuntimeError)
    # The sequential path's message.
    self.assertIn("[precompile 1/6] 2k@16:9 2688x1536: requested text bucket 128", str(error))
    self.assertIn("prompt 256", str(error))
    self.assertIn("2 executable(s) were saved", str(error))
    # Entries 1 and 2 ran, a free worker took at most one more; the rest were cancelled.
    started = pipeline.started()
    self.assertEqual(started[:2], [1, 2])
    self.assertLessEqual(len(started), 3)
    # Saved after the failing entry, and once more after every running entry had finished.
    self.assertGreaterEqual(len(self.saves), 2)
    self.assertEqual(pipeline.events[-1], ("save", 0))
    self.assertEqual(sum(event[0] == "end" for event in pipeline.events), len(started))
    self.assertEqual(self.warmup_mode.call_count, 1)
    self.assertFalse(aot_cache._STATE.warmup_only)

  def test_entry_exception_saves_and_raises(self):
    def gate(index, kwargs):
      if index == 3:
        raise RuntimeError("compile failed")

    pipeline = _ThreadedFakePipeline(self.plan, gate=gate)
    error, _ = self._run(pipeline, workers=2)
    self.assertIsInstance(error, RuntimeError)
    self.assertEqual(str(error), "compile failed")
    self.assertEqual(pipeline.events[-1], ("save", 0))
    self.assertFalse(aot_cache._STATE.warmup_only)

  def test_missing_negative_bucket_raises_like_the_sequential_path(self):
    pipeline = _ThreadedFakePipeline(self.plan)
    self.call_kwargs.update(guidance_scale=1.5, do_classifier_free_guidance=True)
    error, _ = self._run(pipeline, workers=2)
    self.assertIsInstance(error, RuntimeError)
    self.assertIn("'negative_text_tokens'", str(error))

  def test_call_context_is_entered_on_each_worker_thread(self):
    depth = threading.local()
    seen = []

    @contextlib.contextmanager
    def call_context():
      depth.value = getattr(depth, "value", 0) + 1
      try:
        yield
      finally:
        depth.value -= 1

    def gate(index, kwargs):
      seen.append((threading.current_thread().name, getattr(depth, "value", 0)))

    pipeline = _ThreadedFakePipeline(self.plan, gate=gate)
    records, _ = self._run(pipeline, workers=3, call_context=call_context)
    self.assertEqual(len(records), 4)
    self.assertEqual(len(seen), 4)
    for thread, value in seen:
      self.assertNotEqual(thread, threading.current_thread().name)
      self.assertEqual(value, 1)

  def test_each_worker_thread_gets_its_own_tokenizer_copy(self):
    tokenizer = {"vocab": ["a", "fox"]}
    pipeline = _ThreadedFakePipeline(self.plan, gate=lambda index, kwargs: time.sleep(0.05), tokenizer=tokenizer)
    records, _ = self._run(pipeline, workers=2)
    self.assertEqual(len(records), 4)
    by_thread = {}
    for call in pipeline.calls:
      self.assertIsNot(call["tokenizer"], tokenizer)
      self.assertEqual(call["tokenizer"], tokenizer)
      by_thread.setdefault(call["thread"], set()).add(id(call["tokenizer"]))
    # One copy per worker thread, reused for its next entries, never shared between threads.
    self.assertTrue(all(len(ids) == 1 for ids in by_thread.values()))
    self.assertEqual(len(set().union(*by_thread.values())), len(by_thread))
    self.assertIs(pipeline.tokenizer, tokenizer)


class PrecompileWorkersConfigTest(unittest.TestCase):

  _CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

  def test_values(self):
    self.assertEqual(resolve_precompile_workers(types.SimpleNamespace()), 1)
    for value, expected in ((None, 1), ("", 1), (1, 1), (8, 8)):
      self.assertEqual(resolve_precompile_workers(types.SimpleNamespace(krea2_precompile_workers=value)), expected)
    for value in (0, -1, "x", "8", True, 2.0):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "krea2_precompile_workers must be an integer >= 1"):
          resolve_precompile_workers(types.SimpleNamespace(krea2_precompile_workers=value))

  def test_configs_default_to_one(self):
    for name in ("base_krea2.yml", "base_krea2_turbo.yml", "base_krea2_turbo_v6e1.yml"):
      with open(os.path.join(self._CONFIG_DIR, name), encoding="utf-8") as f:
        config = yaml.safe_load(f)
      self.assertIs(type(config["krea2_precompile_workers"]), int, name)
      self.assertEqual(config["krea2_precompile_workers"], 1, name)

  def test_command_line_override_through_pyconfig(self):
    from maxdiffusion import pyconfig  # pylint: disable=import-outside-toplevel

    prev = (pyconfig._config, pyconfig.config)  # pylint: disable=protected-access
    self.addCleanup(lambda: setattr(pyconfig, "_config", prev[0]) or setattr(pyconfig, "config", prev[1]))
    for overrides, expected in (((), 1), (("krea2_precompile_workers=8",), 8)):
      with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as out:
        pyconfig.initialize(
            [
                None,
                os.path.join(self._CONFIG_DIR, "base_krea2_turbo_v6e1.yml"),
                "run_name=t",
                f"output_dir={out}/",
                "skip_jax_distributed_system=True",
                *overrides,
            ]
        )
        self.assertEqual(resolve_precompile_workers(pyconfig.config), expected)


if __name__ == "__main__":
  unittest.main()
