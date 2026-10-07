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

# CPU tests for generate_krea2.Krea2Runtime and the create_runtime/main split.
# No model and no checkpoint: the pipeline, tokenizer and interceptors are fakes.

import contextlib
import types
import unittest
from unittest import mock

import numpy as np

from maxdiffusion import generate_krea2
from maxdiffusion.generate_krea2 import KREA2_PRECOMPILE_PROMPT, Krea2Runtime, StartupTimeline
from maxdiffusion.models.krea2.resolution_presets import list_krea2_resolutions
from maxdiffusion.models.krea2.util import KREA2_PROMPT_TEMPLATE_PREFIX, KREA2_PROMPT_TEMPLATE_START_IDX
from maxdiffusion.pipelines.krea2.krea2_pipeline import compact_text_embeddings, tokenize_krea2_prompts


class _Interceptor:
  """Stands in for the context manager `nn.intercept_methods(interceptor)` returns."""

  def __init__(self):
    self.enters = 0
    self.exits = 0

  @property
  def active(self):
    return self.enters - self.exits

  def __enter__(self):
    self.enters += 1
    return self

  def __exit__(self, *exc):
    self.exits += 1
    return False


class _WordTokenizer:
  """One token per word after the template prefix (34 tokens); the suffix call gives 5 tokens."""

  def __call__(self, texts, truncation=False, padding=None, max_length=None, return_tensors=None):
    del truncation, return_tensors
    if max_length is None:  # the template suffix
      mask = np.ones((len(texts), 5), dtype=np.int64)
      return {"input_ids": mask.copy(), "attention_mask": mask}
    assert padding == "max_length"
    mask = np.zeros((len(texts), max_length), dtype=np.int64)
    for row, text in enumerate(texts):
      assert text.startswith(KREA2_PROMPT_TEMPLATE_PREFIX)
      words = len(text[len(KREA2_PROMPT_TEMPLATE_PREFIX) :].split())
      mask[row, : min(KREA2_PROMPT_TEMPLATE_START_IDX + words, max_length)] = 1
    return {"input_ids": mask.copy(), "attention_mask": mask}


class _FakePipeline:

  def __init__(self, interceptors, text_compaction_multiple=128):
    self.calls = []
    self.interceptors = interceptors
    self.tokenizer = _WordTokenizer()
    self.text_compaction_multiple = text_compaction_multiple
    self.output = object()
    self.on_call = None

  def __call__(self, **kwargs):
    self.calls.append({"kwargs": kwargs, "active": [i.active for i in self.interceptors]})
    if self.on_call is not None:
      self.on_call(kwargs)
    trace = {"prompt_encoding": 0.1, "denoise_loop": 1.0, "vae_decode": 0.2, "text_tokens": 128, "seed": 42}
    return [self.output], trace


def _config(**overrides):
  values = {
      "num_inference_steps": 8,
      "guidance_scale": 0.0,
      "do_classifier_free_guidance": False,
      "negative_prompt": "blurry",
      "batch_size": 1,
      "output_dir": "/tmp/out/",
      "output_name": "krea2.png",
      "max_sequence_length": 512,
      "enable_profiler": False,
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


class _RuntimeTestCase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.interceptors = (_Interceptor(), _Interceptor())
    patcher = mock.patch.object(generate_krea2.nn, "intercept_methods", side_effect=lambda interceptor: interceptor)
    self.intercept_methods = patcher.start()
    self.addCleanup(patcher.stop)

  def _runtime(self, text_compaction_multiple=128, **config):
    self.pipeline = _FakePipeline(self.interceptors, text_compaction_multiple)
    return Krea2Runtime(
        config=_config(**config),
        pipeline=self.pipeline,
        params="params",
        qwen3_params="qwen3_params",
        lora_interceptors=self.interceptors,
        default_prompts=["a configured prompt"],
        default_latents="default_latents",
        default_height=1024,
        default_width=1024,
        load_time=1.5,
        timeline=StartupTimeline(0.0),
        precompile_plan=[],
        text_compaction_multiple=text_compaction_multiple,
    )

  def _assert_entered_once_per_call(self, calls):
    for interceptor in self.interceptors:
      self.assertEqual((interceptor.enters, interceptor.exits), (calls, calls))
    for call in self.pipeline.calls:
      self.assertEqual(call["active"], [1] * len(self.interceptors))


class GenerateTest(_RuntimeTestCase):

  def test_generate_passes_the_request_and_enters_each_interceptor_once(self):
    runtime = self._runtime()
    outputs, trace = runtime.generate(["a fox"], height=768, width=1344, seed=9, output_type="pil")
    self.assertEqual(outputs, [self.pipeline.output])
    self.assertEqual(trace["seed"], 42)
    kwargs = self.pipeline.calls[0]["kwargs"]
    self.assertEqual(kwargs["prompt"], ["a fox"])
    self.assertEqual((kwargs["height"], kwargs["width"]), (768, 1344))
    self.assertEqual(kwargs["seed"], 9)
    self.assertEqual(kwargs["output_type"], "pil")
    self.assertIsNone(kwargs["latents"])
    self.assertEqual(kwargs["output_name"], "krea2.png")
    self.assertEqual(kwargs["params"], "params")
    self.assertEqual(kwargs["qwen3_params"], "qwen3_params")
    self.assertEqual(kwargs["num_inference_steps"], 8)
    self.assertEqual(kwargs["negative_prompt"], "blurry")
    self.assertEqual(kwargs["batch_size"], 1)
    self.assertEqual(kwargs["output_dir"], "/tmp/out/")
    self.assertNotIn("save_outputs", kwargs)
    self._assert_entered_once_per_call(1)

    runtime.generate(["b"], height=1024, width=1024, output_type="file", output_name="x.png", latents="lat")
    kwargs = self.pipeline.calls[1]["kwargs"]
    self.assertIsNone(kwargs["seed"])
    self.assertEqual((kwargs["output_type"], kwargs["output_name"], kwargs["latents"]), ("file", "x.png", "lat"))
    self._assert_entered_once_per_call(2)

  def test_interceptors_exit_when_the_pipeline_raises(self):
    runtime = self._runtime()

    def fail(_):
      raise RuntimeError("boom")

    self.pipeline.on_call = fail
    with self.assertRaisesRegex(RuntimeError, "boom"):
      runtime.generate(["a"], height=1024, width=1024)
    self._assert_entered_once_per_call(1)


class WarmupAndPrecompileTest(_RuntimeTestCase):

  def test_warmup_default_runs_in_warmup_mode_then_saves(self):
    runtime = self._runtime()
    events = []

    @contextlib.contextmanager
    def warmup_mode():
      events.append("warmup_enter")
      yield
      events.append("warmup_exit")

    self.pipeline.on_call = lambda kwargs: events.append("call")
    with (
        mock.patch.object(generate_krea2.aot_cache, "warmup_mode", side_effect=warmup_mode),
        mock.patch.object(generate_krea2.aot_cache, "save_pending", side_effect=lambda: events.append("save")),
    ):
      trace = runtime.warmup_default()
    self.assertEqual(events, ["warmup_enter", "call", "warmup_exit", "save"])
    self.assertEqual(trace["denoise_loop"], 1.0)
    kwargs = self.pipeline.calls[0]["kwargs"]
    self.assertEqual(kwargs["prompt"], ["a configured prompt"])
    self.assertEqual(kwargs["output_name"], "krea2_warmup.png")
    self.assertIs(kwargs["save_outputs"], False)
    self.assertEqual((kwargs["height"], kwargs["width"]), (1024, 1024))
    self.assertEqual(kwargs["latents"], "default_latents")
    self.assertIsNone(kwargs["seed"])
    self._assert_entered_once_per_call(1)

  def test_precompile_delegates_to_run_precompile(self):
    runtime = self._runtime(batch_size=2)
    plan = [(list_krea2_resolutions("1k")[0], 128)]
    seen = {}

    def fake_run_precompile(pipeline, run_plan, call_kwargs, prompts):
      seen.update(pipeline=pipeline, plan=run_plan, call_kwargs=call_kwargs, prompts=prompts)
      seen["active"] = [i.active for i in self.interceptors]
      return [{"label": "x"}]

    with mock.patch.object(generate_krea2, "run_precompile", side_effect=fake_run_precompile):
      records = runtime.precompile(plan)
    self.assertEqual(records, [{"label": "x"}])
    self.assertIs(seen["pipeline"], self.pipeline)
    self.assertIs(seen["plan"], plan)
    self.assertEqual(seen["prompts"], [KREA2_PRECOMPILE_PROMPT] * 2)
    self.assertEqual(seen["call_kwargs"]["negative_prompt"], "")
    self.assertIsNone(seen["call_kwargs"]["latents"])
    self.assertEqual(seen["call_kwargs"]["params"], "params")
    self.assertEqual(seen["active"], [1, 1])
    for interceptor in self.interceptors:
      self.assertEqual((interceptor.enters, interceptor.exits), (1, 1))


class ServingPlanTest(_RuntimeTestCase):

  def test_all_presets_times_rounded_buckets(self):
    runtime = self._runtime()
    plan = runtime.serving_plan("all", [128, 300])
    resolutions = list_krea2_resolutions()
    self.assertEqual(len(resolutions), 22)
    self.assertEqual(plan, [(resolution, bucket) for resolution in resolutions for bucket in (128, 384)])

  def test_subset_default_bucket_and_clipping(self):
    runtime = self._runtime()
    plan = runtime.serving_plan("2k@16:9, 1k", [])
    self.assertEqual(plan[0][0].label, "2k@16:9")
    self.assertEqual({bucket for _, bucket in plan}, {128})
    self.assertEqual(len(plan), 1 + len(list_krea2_resolutions("1k")))
    self.assertEqual({bucket for _, bucket in runtime.serving_plan("1k@1:1", [600, 512])}, {512})

  def test_without_compaction_the_full_length_is_the_only_bucket(self):
    runtime = self._runtime(text_compaction_multiple=0)
    self.assertEqual([bucket for _, bucket in runtime.serving_plan("1k@1:1", [128, 256])], [512])

  def test_rejects_bad_specs_and_buckets(self):
    runtime = self._runtime()
    for spec in ("3k", "1k@7:3", "1k@1:1@2k", "", "  ,  "):
      with self.subTest(spec=spec):
        with self.assertRaises(ValueError):
          runtime.serving_plan(spec, [128])
    for tokens in ([0], [-128], [True], ["128"], [1.5]):
      with self.subTest(tokens=tokens):
        with self.assertRaisesRegex(ValueError, "positive ints"):
          runtime.serving_plan("1k", tokens)


class TextBucketTest(_RuntimeTestCase):

  def _prompt(self, words):
    return " ".join(["w"] * words)

  def test_buckets_and_clipping(self):
    runtime = self._runtime()
    # Valid text tokens = words + the 5 suffix tokens (the 34 prefix tokens are dropped).
    for words, bucket in ((0, 128), (1, 128), (123, 128), (124, 256), (251, 256), (252, 384), (507, 512), (900, 512)):
      with self.subTest(words=words):
        self.assertEqual(runtime.text_bucket([self._prompt(words)]), bucket)
    self.assertEqual(runtime.text_bucket([self._prompt(10), self._prompt(200)]), 256)

  def test_matches_compact_text_embeddings(self):
    runtime = self._runtime()
    tokenizer = _WordTokenizer()
    for words in (3, 130, 300, 700):
      with self.subTest(words=words):
        prompts = [self._prompt(words)]
        _, mask, _ = tokenize_krea2_prompts(tokenizer, prompts, 512)
        text_mask = mask[:, KREA2_PROMPT_TEMPLATE_START_IDX:]
        embeds = np.zeros((1, text_mask.shape[1], 1, 1), dtype=np.float32)
        compact, _ = compact_text_embeddings(embeds, text_mask, 128)
        self.assertEqual(runtime.text_bucket(prompts), compact.shape[1])

  def test_without_compaction_and_for_tokens(self):
    runtime = self._runtime(text_compaction_multiple=0)
    self.assertEqual(runtime.text_bucket([self._prompt(3)]), 512)
    runtime = self._runtime()
    self.assertEqual(
        [runtime.text_bucket_for_tokens(n) for n in (0, 1, 128, 129, 512, 513, 10_000)],
        [128, 128, 128, 256, 512, 512, 512],
    )

  def test_uses_a_private_tokenizer_copy(self):
    runtime = self._runtime()
    runtime.text_bucket(["a"])
    runtime.text_bucket(["b"])
    # pylint: disable-next=protected-access
    self.assertIsNot(runtime._bucket_tokenizer, self.pipeline.tokenizer)
    self.assertIsInstance(runtime._bucket_tokenizer, _WordTokenizer)  # pylint: disable=protected-access


class MainTest(unittest.TestCase):

  def _fake_runtime(self, precompile_plan=()):
    events = []
    runtime = mock.Mock()
    runtime.config = _config()
    runtime.timeline = StartupTimeline(0.0)
    runtime.timed_phases = generate_krea2.KREA2_TIMED_PHASES
    runtime.precompile_plan = list(precompile_plan)
    runtime.default_prompts = ["p"]
    runtime.default_height, runtime.default_width = 1024, 768
    runtime.default_latents = None
    runtime.load_time = 2.0

    def warmup_default():
      events.append("warmup")
      return {"prompt_encoding": 1.0, "denoise_loop": 2.0, "vae_decode": 3.0}

    def generate(prompts, **kwargs):
      events.append(("generate", tuple(prompts), tuple(sorted(kwargs.items()))))
      return ["/tmp/out/krea2.png"], {"prompt_encoding": 0.1, "denoise_loop": 0.2, "vae_decode": 0.3}

    def precompile(plan):
      events.append(("precompile", tuple(plan)))
      return []

    runtime.warmup_default.side_effect = warmup_default
    runtime.generate.side_effect = generate
    runtime.precompile.side_effect = precompile
    return runtime, events

  def test_main_warms_up_then_generates_a_file(self):
    runtime, events = self._fake_runtime()
    with (
        mock.patch.object(generate_krea2, "create_runtime", return_value=runtime) as create,
        mock.patch.object(generate_krea2.max_utils, "Profiler") as profiler,
        mock.patch.object(generate_krea2.max_logging, "log") as log,
    ):
      generate_krea2.main(["prog", "cfg.yml", "prompt=x"])
    create.assert_called_once_with(["prog", "cfg.yml", "prompt=x"])
    profiler.assert_called_once()
    self.assertEqual(
        events,
        [
            "warmup",
            (
                "generate",
                ("p",),
                (
                    ("height", 1024),
                    ("latents", None),
                    ("output_name", "krea2.png"),
                    ("output_type", "file"),
                    ("width", 768),
                ),
            ),
        ],
    )
    runtime.precompile.assert_not_called()
    lines = [c.args[0] for c in log.call_args_list]
    self.assertIn("2) Cold-Start / Warmup Pass (XLA Compilation): 6.00 seconds", lines)
    self.assertIn("3) Main Warmed-Up Pass: 0.60 seconds", lines)
    self.assertEqual(lines[-1], "SUCCESS! Generation complete for 1 image(s)!")
    self.assertEqual(set(runtime.timeline.marks()), {"warmup_done", "saved"})

  def test_precompile_mode_returns_after_precompile(self):
    runtime, events = self._fake_runtime(precompile_plan=[("res", 128)])
    with (
        mock.patch.object(generate_krea2, "create_runtime", return_value=runtime),
        mock.patch.object(generate_krea2, "log_precompile_summary") as summary,
        mock.patch.object(generate_krea2.max_logging, "log") as log,
    ):
      generate_krea2.main(["prog"])
    self.assertEqual(events, [("precompile", (("res", 128),))])
    summary.assert_called_once_with([])
    runtime.warmup_default.assert_not_called()
    runtime.generate.assert_not_called()
    self.assertEqual(
        log.call_args_list[-1].args[0], "SUCCESS! Precompile complete for 0 resolution/text combination(s)!"
    )

  def test_build_only_mode_returns_at_once(self):
    with mock.patch.object(generate_krea2, "create_runtime", return_value=None):
      self.assertIsNone(generate_krea2.main(["prog"]))


if __name__ == "__main__":
  unittest.main()
