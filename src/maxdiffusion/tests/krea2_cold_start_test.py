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

"""Cold-start pieces of Krea 2 generation: config keys, startup timeline, weight cache wait, tokenizer single
load, torch/transformers-free imports and the numpy VAE read (CPU only, no model files)."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import yaml

from maxdiffusion import generate_krea2, max_utils
from maxdiffusion.models.krea2 import weight_cache
from maxdiffusion.models.krea2.weight_cache import WeightCacheSpec, component_dir, wait_for_component

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_CONFIGS = ("base_krea2.yml", "base_krea2_turbo.yml", "base_krea2_turbo_v6e1.yml")


def _run_python(code):
  """Runs `code` in a fresh interpreter on the CPU backend; returns its stdout."""
  env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=_SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
  result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300, check=False)
  if result.returncode != 0:
    raise AssertionError(f"subprocess failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
  return result.stdout


class Krea2ColdStartConfigTest(unittest.TestCase):

  def _load_preset(self, *overrides):
    from maxdiffusion import pyconfig  # pylint: disable=import-outside-toplevel

    prev = (pyconfig._config, pyconfig.config)  # pylint: disable=protected-access
    self.addCleanup(lambda: setattr(pyconfig, "_config", prev[0]) or setattr(pyconfig, "config", prev[1]))
    with tempfile.TemporaryDirectory() as out:
      pyconfig.initialize(
          [
              None,
              os.path.join(_CONFIG_DIR, "base_krea2_turbo_v6e1.yml"),
              "run_name=t",
              f"output_dir={out}/",
              "skip_jax_distributed_system=True",
              *overrides,
          ]
      )
    return pyconfig.config

  def test_configs_define_the_keys_off(self):
    for name in _CONFIGS:
      with open(os.path.join(_CONFIG_DIR, name), encoding="utf-8") as f:
        config = yaml.safe_load(f)
      self.assertEqual(config["aot_cache_gcs"], "", name)
      self.assertIs(type(config["krea2_weight_cache_wait_s"]), int, name)
      self.assertEqual(config["krea2_weight_cache_wait_s"], 0, name)
      self.assertIs(type(config["krea2_model_wait_s"]), int, name)
      self.assertEqual(config["krea2_model_wait_s"], 0, name)

  def test_preset_unset_and_valid_values(self):
    config = self._load_preset()
    self.assertEqual(config.aot_cache_gcs, "")
    self.assertEqual(config.krea2_weight_cache_wait_s, 0)
    self.assertEqual(generate_krea2.resolve_aot_cache_gcs(config), "")
    self.assertEqual(generate_krea2.resolve_weight_cache_wait_s(config), 0)
    self.assertEqual(config.krea2_model_wait_s, 0)
    self.assertEqual(generate_krea2.resolve_model_wait_s(config), 0)

    config = self._load_preset(
        "aot_cache_gcs=gs://tpu-test-507316-krea2-use1/aot/",
        "krea2_weight_cache_wait_s=900",
        "aot_cache_dir=/x",
        "krea2_model_wait_s=120",
    )
    self.assertEqual(config.aot_cache_gcs, "gs://tpu-test-507316-krea2-use1/aot")
    self.assertEqual(config.krea2_weight_cache_wait_s, 900)
    # The v6e-1 preset loads lazily, so the URL is usable as is.
    self.assertEqual(generate_krea2.resolve_aot_cache_gcs(config), "gs://tpu-test-507316-krea2-use1/aot")
    self.assertEqual(generate_krea2.resolve_weight_cache_wait_s(config), 900)
    self.assertEqual(config.krea2_model_wait_s, 120)
    self.assertEqual(generate_krea2.resolve_model_wait_s(config), 120)

    self.assertEqual(self._load_preset("aot_cache_gcs=''").aot_cache_gcs, "")

  def test_preset_rejects_invalid_values(self):
    for override in (
        "aot_cache_gcs=s3://bucket/aot",
        "aot_cache_gcs=bucket/aot",
        "aot_cache_gcs=gs://Bucket/aot",
        "aot_cache_gcs=gs://bucket//aot",
        "krea2_weight_cache_wait_s=-1",
        "krea2_weight_cache_wait_s=abc",
        "krea2_weight_cache_wait_s=1.5",
        "krea2_model_wait_s=abc",
        "krea2_model_wait_s=1.5",
        "krea2_model_wait_s=True",
    ):
      with self.subTest(override=override):
        with self.assertRaises(ValueError):
          self._load_preset(override)
    # Only a negative value parses as an int on the command line and reaches the cold-start check.
    with self.assertRaisesRegex(ValueError, r"krea2_model_wait_s must be an integer >= 0 \(seconds, 0 = off\), got -1"):
      self._load_preset("krea2_model_wait_s=-1")

  def test_validate_model_wait_s(self):
    from maxdiffusion import pyconfig  # pylint: disable=import-outside-toplevel

    validate = pyconfig._validate_krea2_cold_start_keys  # pylint: disable=protected-access
    for value in (0, 1, 900):
      validate({"krea2_model_wait_s": value})
    validate({})
    for value in (-1, True, "5"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(
            ValueError, rf"krea2_model_wait_s must be an integer >= 0 \(seconds, 0 = off\), got {value!r}"
        ):
          validate({"krea2_model_wait_s": value})

  def test_resolve_aot_cache_gcs_needs_dir_and_lazy_load(self):
    url = "gs://krea2-bkt/aot"
    ok = types.SimpleNamespace(aot_cache_gcs=url, aot_cache_dir="/aot", aot_cache_lazy_load=True)
    self.assertEqual(generate_krea2.resolve_aot_cache_gcs(ok), url)
    self.assertEqual(generate_krea2.resolve_aot_cache_gcs(types.SimpleNamespace()), "")
    for config, pattern in (
        (types.SimpleNamespace(aot_cache_gcs=url, aot_cache_dir="", aot_cache_lazy_load=True), "aot_cache_dir"),
        (types.SimpleNamespace(aot_cache_gcs=url, aot_cache_dir="/aot", aot_cache_lazy_load=False), "lazy_load"),
        (types.SimpleNamespace(aot_cache_gcs="gs:/x", aot_cache_dir="/aot", aot_cache_lazy_load=True), "aot_cache_gcs"),
    ):
      with self.subTest(config=config):
        with self.assertRaisesRegex(ValueError, pattern):
          generate_krea2.resolve_aot_cache_gcs(config)

  def test_resolve_weight_cache_wait_s(self):
    self.assertEqual(generate_krea2.resolve_weight_cache_wait_s(types.SimpleNamespace()), 0)
    self.assertEqual(generate_krea2.resolve_weight_cache_wait_s(types.SimpleNamespace(krea2_weight_cache_wait_s=5)), 5)
    for value in (-1, True, 2.5, "10"):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          generate_krea2.resolve_weight_cache_wait_s(types.SimpleNamespace(krea2_weight_cache_wait_s=value))

  def test_resolve_model_wait_s(self):
    resolve = generate_krea2.resolve_model_wait_s
    self.assertEqual(resolve(types.SimpleNamespace()), 0)
    self.assertEqual(resolve(types.SimpleNamespace(krea2_model_wait_s=0)), 0)
    self.assertEqual(resolve(types.SimpleNamespace(krea2_model_wait_s=30)), 30)
    for value in (-1, True, 2.5, "10"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "krea2_model_wait_s must be an integer >= 0"):
          resolve(types.SimpleNamespace(krea2_model_wait_s=value))

  def test_gcs_without_lazy_load_fails_before_model_load(self):
    from maxdiffusion import pyconfig  # pylint: disable=import-outside-toplevel

    prev = (pyconfig._config, pyconfig.config)  # pylint: disable=protected-access
    self.addCleanup(lambda: setattr(pyconfig, "_config", prev[0]) or setattr(pyconfig, "config", prev[1]))
    with tempfile.TemporaryDirectory() as out:
      argv = [
          None,
          os.path.join(_CONFIG_DIR, "base_krea2_turbo_v6e1.yml"),
          f"output_dir={out}/",
          "skip_jax_distributed_system=True",
          f"aot_cache_dir={out}/aot",
          "aot_cache_lazy_load=False",
          "aot_cache_gcs=gs://krea2-bkt/aot",
      ]
      with (
          mock.patch.object(generate_krea2, "build_krea2_transformer") as build,
          mock.patch.object(generate_krea2, "create_device_mesh") as mesh,
          mock.patch.object(generate_krea2, "register_exit_timing"),
          self.assertRaisesRegex(ValueError, "aot_cache_lazy_load"),
      ):
        generate_krea2.main(argv)
      build.assert_not_called()
      mesh.assert_not_called()


class Krea2StartupTimelineTest(unittest.TestCase):

  def test_resolve_process_t0(self):
    resolve = generate_krea2.resolve_process_t0
    self.assertEqual(resolve(None, 100.0), (100.0, "module import"))
    self.assertEqual(resolve("", 100.0), (100.0, "module import"))
    self.assertEqual(resolve("98.25", 100.0), (98.25, "KREA2_PROCESS_T0"))
    self.assertEqual(resolve("100", 100.0), (100.0, "KREA2_PROCESS_T0"))
    for bad in ("100.5", "abc", "nan", "inf"):
      with self.subTest(bad=bad):
        t0, origin = resolve(bad, 100.0)
        self.assertEqual(t0, 100.0)
        self.assertIn("KREA2_PROCESS_T0 ignored", origin)

  def test_module_t0_is_taken_before_the_heavy_imports(self):
    self.assertIsInstance(generate_krea2._PROCESS_T0, float)  # pylint: disable=protected-access
    self.assertLessEqual(generate_krea2._PROCESS_T0, time.time())  # pylint: disable=protected-access
    with open(generate_krea2.__file__, encoding="utf-8") as f:
      source = f.read()
    self.assertLess(source.index("_PROCESS_T0 = time.time()"), source.index("import jax"))

  def test_lines(self):
    timeline = generate_krea2.StartupTimeline(100.0)
    for name, at in (
        ("imports", 104.0),
        ("config", 109.5),
        ("tpu_init", 110.0),
        ("shapes", 116.0),
        ("load_done", 121.0),
        ("saved", 125.25),
    ):
      timeline.mark(name, at=at)
    # In the parallel thread: listed, not part of the gaps (it would split the load gap).
    timeline.mark("tokenizer", at=118.0, parallel=True)
    timeline_line, gaps_line = timeline.lines()
    self.assertEqual(
        timeline_line,
        "[TIMING] Startup timeline (s since process start): imports=4.00 config=9.50 tpu_init=10.00 shapes=16.00 "
        "tokenizer=18.00 load_done=21.00 saved=25.25",
    )
    self.assertEqual(gaps_line, "[TIMING] Startup biggest gaps: shapes 6.00, config 5.50")
    with mock.patch.object(generate_krea2.max_logging, "log") as log:
      timeline.log()
    self.assertEqual([c.args[0] for c in log.call_args_list], [timeline_line, gaps_line])

  def test_model_dir_mark_between_mesh_and_shapes(self):
    order = generate_krea2.STARTUP_TIMELINE_ORDER
    self.assertEqual(order[order.index("mesh") + 1 : order.index("shapes")], ("model_dir",))
    # main() waits for the model directory after the mesh, before its first read of it.
    with open(generate_krea2.__file__, encoding="utf-8") as f:
      source = f.read()
    positions = [
        source.index('timeline.mark("mesh")'),
        source.index("wait_for_model_dir(config.pretrained_model_name_or_path, model_wait_s)"),
        source.index('timeline.mark("model_dir")'),
        source.index("repo_id = config.pretrained_model_name_or_path"),
    ]
    self.assertEqual(positions, sorted(positions))

  def test_serial_tokenizer_is_a_gap(self):
    timeline = generate_krea2.StartupTimeline(0.0)
    timeline.mark("imports", at=1.0)
    timeline.mark("load_done", at=2.0)
    timeline.mark("tokenizer", at=9.0)
    self.assertEqual(timeline.lines()[1], "[TIMING] Startup biggest gaps: tokenizer 7.00, imports 1.00")

  def test_exit_hook_registered_once(self):
    with (
        mock.patch.object(generate_krea2, "_EXIT_HOOK_T0", []),
        mock.patch.object(generate_krea2.atexit, "register") as register,
    ):
      generate_krea2.register_exit_timing(10.0)
      generate_krea2.register_exit_timing(20.0)
      register.assert_called_once_with(generate_krea2._log_exit_begins)  # pylint: disable=protected-access
      with (
          mock.patch.object(generate_krea2.time, "time", return_value=23.5),
          mock.patch.object(generate_krea2.max_logging, "log") as log,
      ):
        generate_krea2._log_exit_begins()  # pylint: disable=protected-access
      log.assert_called_once_with("[TIMING] exit begins at 3.50")


class Krea2WeightCacheWaitTest(unittest.TestCase):

  _META = {"component": "transformer", "x": 1}

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.cache_dir = self._tmp.name
    self.directory = component_dir(self.cache_dir, "transformer", self._META)

  def _complete(self, directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
      f.write("{}")

  def _fake_clock(self, on_sleep=None):
    now = [1000.0]
    sleeps = []

    def clock():
      return now[0]

    def sleep(seconds):
      sleeps.append(seconds)
      now[0] += seconds
      if on_sleep is not None:
        on_sleep(len(sleeps))

    return clock, sleep, sleeps

  def test_present_or_off_returns_at_once(self):
    clock, sleep, sleeps = self._fake_clock()
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      self.assertEqual(wait_for_component(self.cache_dir, "transformer", self._META, 0, sleep=sleep, clock=clock), 0.0)
      self._complete(self.directory)
      self.assertEqual(wait_for_component(self.cache_dir, "transformer", self._META, 30, sleep=sleep, clock=clock), 0.0)
    self.assertEqual(sleeps, [])
    log.assert_not_called()

  def test_appears_during_the_wait(self):
    clock, sleep, sleeps = self._fake_clock(on_sleep=lambda n: n == 3 and self._complete(self.directory))
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      waited = wait_for_component(self.cache_dir, "transformer", self._META, 30, sleep=sleep, clock=clock)
    self.assertEqual(waited, 3.0)
    self.assertEqual(sleeps, [1.0, 1.0, 1.0])
    self.assertEqual(
        [c.args[0] for c in log.call_args_list],
        [
            f"[weight cache] transformer: waiting up to 30 s for {self.directory} (pull in progress)",
            f"[weight cache] transformer: {self.directory} appeared after 3.0 s",
        ],
    )

  def test_a_directory_without_meta_json_is_not_complete(self):
    os.makedirs(self.directory)
    clock, sleep, _ = self._fake_clock()
    with mock.patch.object(weight_cache.max_logging, "log"):
      self.assertIsNone(wait_for_component(self.cache_dir, "transformer", self._META, 3, sleep=sleep, clock=clock))

  def test_timeout(self):
    clock, sleep, sleeps = self._fake_clock()
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      waited = wait_for_component(self.cache_dir, "transformer", self._META, 2.5, poll_s=1.0, sleep=sleep, clock=clock)
    self.assertIsNone(waited)
    self.assertEqual(sleeps, [1.0, 1.0, 0.5])
    self.assertIn("did not appear within 2.5 s", log.call_args_list[-1].args[0])

  def test_another_fingerprint_does_not_end_the_wait(self):
    # A pull writes directories in sorted-name order: an older fingerprint completing first is no reason to stop.
    stale = os.path.join(self.cache_dir, "transformer-000000000000")
    self._complete(stale)
    pulled = os.path.join(self.cache_dir, "transformer-111111111111")

    def on_sleep(n):
      if n == 2:
        self._complete(pulled)
      elif n == 5:
        self._complete(self.directory)

    clock, sleep, sleeps = self._fake_clock(on_sleep=on_sleep)
    self._complete(os.path.join(self.cache_dir, "text_encoder-222222222222"))
    os.makedirs(os.path.join(self.cache_dir, "transformer-333333333333.dl-tmp-7"))
    with mock.patch.object(weight_cache.max_logging, "log") as log:
      waited = wait_for_component(self.cache_dir, "transformer", self._META, 900, sleep=sleep, clock=clock)
    self.assertEqual(waited, 5.0)
    self.assertEqual(sleeps, [1.0] * 5)
    self.assertEqual(
        [c.args[0] for c in log.call_args_list],
        [
            f"[weight cache] transformer: waiting up to 900 s for {self.directory} (pull in progress)",
            f"[weight cache] transformer: {self.directory} appeared after 5.0 s",
        ],
    )

  def test_real_threads(self):
    timer = threading.Timer(0.3, self._complete, args=(self.directory,))
    timer.start()
    self.addCleanup(timer.cancel)
    with mock.patch.object(weight_cache.max_logging, "log"):
      waited = wait_for_component(self.cache_dir, "transformer", self._META, 10, poll_s=0.05)
    self.assertIsNotNone(waited)
    self.assertGreater(waited, 0.2)
    self.assertLess(waited, 5.0)

  def test_load_or_build_waits_only_with_the_knob(self):
    def build():
      return {"built": True}, None

    for wait_s, expect_wait in ((0, False), (7, True)):
      with self.subTest(wait_s=wait_s):
        cache = WeightCacheSpec(self.cache_dir, "transformer", self._META, [["a.safetensors", 1]], wait_s)
        trace = {}
        with (
            mock.patch.object(weight_cache, "wait_for_component", return_value=0.0) as wait,
            mock.patch.object(weight_cache, "load_component", return_value=None),
        ):
          tree, _, hit = generate_krea2.load_or_build_host_params(cache, {}, build, trace, "transformer_cache_read")
        self.assertEqual((tree, hit), ({"built": True}, False))
        self.assertIn("transformer_cache_read", trace)
        if expect_wait:
          wait.assert_called_once_with(self.cache_dir, "transformer", self._META, 7)
          self.assertIn("transformer_cache_wait", trace)
        else:
          wait.assert_not_called()
          self.assertNotIn("transformer_cache_wait", trace)

  def test_spec_defaults_to_no_wait(self):
    self.assertEqual(WeightCacheSpec("/c", "transformer", {}, None).wait_s, 0)


class Krea2ModelDirWaitTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.path = self._tmp.name

  def _mark(self, name=".krea2_fetch_slim"):
    with open(os.path.join(self.path, name), "w", encoding="utf-8") as f:
      f.write("")

  def _fake_clock(self, on_sleep=None):
    now = [1000.0]
    sleeps = []

    def clock():
      return now[0]

    def sleep(seconds):
      sleeps.append(seconds)
      now[0] += seconds
      if on_sleep is not None:
        on_sleep(len(sleeps))

    return clock, sleep, sleeps

  def test_markers(self):
    self.assertEqual(generate_krea2.MODEL_FETCH_MARKERS, (".krea2_fetch_slim", ".krea2_fetch_complete"))

  def test_off_or_marker_present_returns_at_once(self):
    clock, sleep, sleeps = self._fake_clock()
    with mock.patch.object(generate_krea2.max_logging, "log") as log:
      self.assertEqual(generate_krea2.wait_for_model_dir(self.path, 0, sleep=sleep, clock=clock), 0.0)
      missing = os.path.join(self.path, "not-yet")
      self.assertEqual(generate_krea2.wait_for_model_dir(missing, 0, sleep=sleep, clock=clock), 0.0)
      for name in generate_krea2.MODEL_FETCH_MARKERS:
        with self.subTest(name=name):
          self._mark(name)
          self.assertEqual(generate_krea2.wait_for_model_dir(self.path, 30, sleep=sleep, clock=clock), 0.0)
          os.remove(os.path.join(self.path, name))
    self.assertEqual(sleeps, [])
    log.assert_not_called()

  def test_appears_during_the_wait(self):
    clock, sleep, sleeps = self._fake_clock(on_sleep=lambda n: n == 3 and self._mark())
    with mock.patch.object(generate_krea2.max_logging, "log") as log:
      waited = generate_krea2.wait_for_model_dir(self.path, 30, sleep=sleep, clock=clock)
    self.assertEqual(waited, 1.5)
    self.assertEqual(sleeps, [0.5, 0.5, 0.5])
    self.assertEqual(
        [c.args[0] for c in log.call_args_list],
        [
            f"[model] waiting up to 30 s for a fetch marker in {self.path} (fetch in progress)",
            f"[model] {self.path}: fetch marker appeared after 1.5 s",
        ],
    )

  def test_either_marker_ends_the_wait(self):
    for name in generate_krea2.MODEL_FETCH_MARKERS:
      with self.subTest(name=name):
        clock, sleep, sleeps = self._fake_clock(on_sleep=lambda n, name=name: n == 2 and self._mark(name))
        with mock.patch.object(generate_krea2.max_logging, "log"):
          self.assertEqual(generate_krea2.wait_for_model_dir(self.path, 30, sleep=sleep, clock=clock), 1.0)
        self.assertEqual(sleeps, [0.5, 0.5])
        os.remove(os.path.join(self.path, name))

  def test_a_directory_without_a_marker_does_not_end_the_wait(self):
    # Files of the fetch in progress, a directory named like a marker: none of them is a marker file.
    os.makedirs(os.path.join(self.path, "text_encoder"))
    with open(os.path.join(self.path, "text_encoder", "config.json"), "w", encoding="utf-8") as f:
      f.write("{}")
    os.makedirs(os.path.join(self.path, ".krea2_fetch_complete"))
    clock, sleep, sleeps = self._fake_clock()
    with mock.patch.object(generate_krea2.max_logging, "log"):
      self.assertIsNone(generate_krea2.wait_for_model_dir(self.path, 3, sleep=sleep, clock=clock))
    self.assertEqual(sleeps, [0.5] * 6)

  def test_timeout(self):
    missing = os.path.join(self.path, "Krea-2-Turbo")
    clock, sleep, sleeps = self._fake_clock()
    with mock.patch.object(generate_krea2.max_logging, "log") as log:
      waited = generate_krea2.wait_for_model_dir(missing, 1.2, sleep=sleep, clock=clock)
    self.assertIsNone(waited)
    self.assertEqual([round(seconds, 9) for seconds in sleeps], [0.5, 0.5, 0.2])
    self.assertEqual(log.call_args_list[-1].args[0], f"[model] {missing}: no fetch marker within 1.2 s")

  def test_is_local_model_path(self):
    for path in ("/abs/x", "./x", "../x", "~/x"):
      with self.subTest(path=path):
        self.assertTrue(generate_krea2.is_local_model_path(path))
    cwd = os.getcwd()
    os.chdir(self.path)
    try:
      os.makedirs(os.path.join("models", "Krea-2-Turbo"))
      self.assertTrue(generate_krea2.is_local_model_path(os.path.join("models", "Krea-2-Turbo")))
      for path in ("krea/Krea-2-Turbo", "org/name", "Krea-2-Turbo"):
        with self.subTest(path=path):
          self.assertFalse(generate_krea2.is_local_model_path(path))
    finally:
      os.chdir(cwd)

  def test_hf_repo_id_is_not_waited_for(self):
    clock, sleep, sleeps = self._fake_clock()
    with mock.patch.object(generate_krea2.max_logging, "log") as log:
      self.assertEqual(generate_krea2.wait_for_model_dir("krea/Krea-2-Turbo", 600, sleep=sleep, clock=clock), 0.0)
    self.assertEqual(sleeps, [])
    self.assertEqual(log.call_count, 1)
    self.assertIn("ignored", log.call_args.args[0])
    self.assertIn("krea/Krea-2-Turbo", log.call_args.args[0])

  def test_real_threads(self):
    timer = threading.Timer(0.2, self._mark, args=(".krea2_fetch_complete",))
    timer.start()
    self.addCleanup(timer.cancel)
    with mock.patch.object(generate_krea2.max_logging, "log"):
      waited = generate_krea2.wait_for_model_dir(self.path, 10, poll_s=0.05)
    self.assertIsNotNone(waited)
    self.assertGreater(waited, 0.1)
    self.assertLess(waited, 5.0)


def _set_model_specific_special_tokens(message):
  raise AttributeError(message)


class Krea2TokenizerSingleLoadTest(unittest.TestCase):

  def _tokenizer_dir(self, extra_special_tokens):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    with open(os.path.join(tmp.name, "tokenizer_config.json"), "w", encoding="utf-8") as f:
      json.dump({"extra_special_tokens": extra_special_tokens, "model_max_length": 10}, f)
    return tmp.name

  def _load(self, tokenizer_dir, transformers_version="4.57.6"):
    from maxdiffusion.models.krea2.util import load_krea2_tokenizer  # pylint: disable=import-outside-toplevel

    sentinel = object()
    calls = []

    def fake_from_pretrained(path, **kwargs):
      calls.append((path, kwargs))
      if isinstance(extra, list) and kwargs.get("extra_special_tokens") != {} and transformers_version < "5":
        _set_model_specific_special_tokens("'list' object has no attribute 'keys'")
      return sentinel

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), encoding="utf-8") as f:
      extra = json.load(f)["extra_special_tokens"]
    with (
        mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained),
        mock.patch("transformers.__version__", transformers_version),
        mock.patch("maxdiffusion.max_logging.log") as log,
    ):
      tokenizer = load_krea2_tokenizer(tokenizer_dir, "/snap")
    self.assertIs(tokenizer, sentinel)
    return calls, log

  def test_list_in_config_builds_once(self):
    tokenizer_dir = self._tokenizer_dir(["<|a|>", "<|b|>"])
    calls, log = self._load(tokenizer_dir)
    self.assertEqual(calls, [(tokenizer_dir, {"local_files_only": True, "extra_special_tokens": {}})])
    log.assert_not_called()

  def test_dict_in_config_keeps_the_plain_load(self):
    tokenizer_dir = self._tokenizer_dir({"a": "<|a|>"})
    calls, _ = self._load(tokenizer_dir)
    self.assertEqual(calls, [(tokenizer_dir, {"local_files_only": True})])

  def test_transformers_5_keeps_the_plain_load(self):
    tokenizer_dir = self._tokenizer_dir(["<|a|>"])
    calls, _ = self._load(tokenizer_dir, transformers_version="5.0.0")
    self.assertEqual(calls, [(tokenizer_dir, {"local_files_only": True})])


class Krea2TorchFreeImportTest(unittest.TestCase):

  def test_is_flax_clip_text_model_without_transformers(self):
    with mock.patch.dict(sys.modules):
      sys.modules.pop("transformers", None)
      with mock.patch("builtins.__import__", side_effect=AssertionError("imported")):
        self.assertFalse(max_utils._is_flax_clip_text_model(object()))  # pylint: disable=protected-access

  def test_is_flax_clip_text_model_with_transformers(self):
    class FakeClip:
      pass

    class FakeClipPreTrained:
      pass

    fake = types.ModuleType("transformers")
    fake.FlaxCLIPTextModel = FakeClip
    fake.FlaxCLIPTextPreTrainedModel = FakeClipPreTrained
    check = max_utils._is_flax_clip_text_model  # pylint: disable=protected-access
    with mock.patch.dict(sys.modules, {"transformers": fake}):
      self.assertTrue(check(FakeClip()))
      self.assertTrue(check(FakeClipPreTrained()))
      self.assertFalse(check(object()))
      fake.FlaxCLIPTextModel = None
      self.assertFalse(check(FakeClip()))
    # A transformers without the classes skips the check.
    with mock.patch.dict(sys.modules, {"transformers": types.ModuleType("transformers")}):
      self.assertFalse(check(FakeClip()))

  def test_generate_process_imports_neither_torch_nor_transformers(self):
    out = _run_python(
        "import sys\n"
        "import maxdiffusion.max_utils\n"
        "print('max_utils', 'torch' in sys.modules, 'transformers' in sys.modules)\n"
        "import maxdiffusion.generate_krea2\n"
        "print('generate_krea2', 'torch' in sys.modules, 'transformers' in sys.modules)\n"
        "from maxdiffusion.models.krea2.util import load_krea2_tokenizer\n"
        "from maxdiffusion.models.qwen3_flax import FlaxQwen3Model\n"
        "from maxdiffusion.models.krea2.text_encoder_quant import quantize_text_encoder_model\n"
        "from maxdiffusion.models.krea2.transformer_quant import quantize_transformer_params\n"
        "from maxdiffusion.models.krea2.weight_cache import WeightCacheSpec\n"
        "from maxdiffusion.models.flux.util import cast_dict_to_bfloat16_inplace\n"
        "from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler\n"
        "from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline\n"
        "from maxdiffusion.loaders.krea2_lora_pipeline import maybe_load_krea2_lora\n"
        "from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan\n"
        "from maxdiffusion.models.wan.wan_utils import load_wan_vae\n"
        "print('late', 'torch' in sys.modules, 'transformers' in sys.modules)\n"
    )
    self.assertEqual(
        out.strip().splitlines()[-3:],
        ["max_utils False False", "generate_krea2 False False", "late False False"],
    )

  def test_loaders_package_exports_stay_available(self):
    out = _run_python(
        "import sys\n"
        "import maxdiffusion.loaders as loaders\n"
        "print('torch' in sys.modules)\n"
        "from maxdiffusion.loaders import FluxLoraLoaderMixin, StableDiffusionLoraLoaderMixin\n"
        "print(FluxLoraLoaderMixin.__name__, StableDiffusionLoraLoaderMixin.__name__)\n"
        "print(sorted(set(loaders.__all__) <= set(dir(loaders)) for _ in [0]))\n"
    )
    self.assertEqual(
        out.strip().splitlines()[-3:], ["False", "FluxLoraLoaderMixin StableDiffusionLoraLoaderMixin", "[True]"]
    )


class Krea2VaeNumpyReadTest(unittest.TestCase):

  def _write(self, tensors):
    from safetensors.numpy import save_file  # pylint: disable=import-outside-toplevel

    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    os.makedirs(os.path.join(tmp.name, "vae"))
    save_file(tensors, os.path.join(tmp.name, "vae", "diffusion_pytorch_model.safetensors"))
    return tmp.name

  def _tensors(self):
    rng = np.random.default_rng(0)
    return {
        "decoder.conv_in.bias": rng.standard_normal(4).astype(np.float32),
        "decoder.conv_in.weight": rng.standard_normal((4, 3, 1, 3, 3)).astype(np.float32),
        "decoder.norm_out.gamma": rng.standard_normal((4, 1, 1, 1)).astype(np.float32),
        "encoder.down_blocks.0.resnets.0.conv1.weight": rng.standard_normal((4, 4, 3, 3, 3)).astype(np.float32),
    }

  def test_numpy_read_matches_the_torch_read(self):
    from maxdiffusion.models.wan import wan_utils  # pylint: disable=import-outside-toplevel

    snapshot = self._write(self._tensors())
    path = os.path.join(snapshot, "vae", "diffusion_pytorch_model.safetensors")
    with_np = wan_utils._read_safetensors_as_jax(path, "np")  # pylint: disable=protected-access
    with_pt = wan_utils._read_safetensors_as_jax(path, "pt")  # pylint: disable=protected-access
    self.assertEqual(list(with_np), list(with_pt))
    for key, value in with_np.items():
      self.assertIsInstance(value, jax.Array)
      self.assertEqual(value.dtype, with_pt[key].dtype)
      np.testing.assert_array_equal(np.asarray(value), np.asarray(with_pt[key]))
    with self.assertRaises(ValueError):
      wan_utils._read_safetensors_as_jax(path, "tf")  # pylint: disable=protected-access

    eval_shapes = {}
    a = wan_utils.load_wan_vae(snapshot, eval_shapes, "cpu", framework="np")
    b = wan_utils.load_wan_vae(snapshot, eval_shapes, "cpu")
    flat_a = jax.tree_util.tree_leaves_with_path(a)
    flat_b = jax.tree_util.tree_leaves_with_path(b)
    self.assertEqual([p for p, _ in flat_a], [p for p, _ in flat_b])
    for (_, x), (_, y) in zip(flat_a, flat_b):
      self.assertEqual(x.dtype, y.dtype)
      np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

  def test_bfloat16_file_falls_back_to_torch(self):
    from maxdiffusion.models.wan import wan_utils  # pylint: disable=import-outside-toplevel

    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    path = os.path.join(tmp.name, "bf16.safetensors")
    # A hand-written safetensors file with one bf16 tensor (numpy cannot write one).
    payload = np.array([0x3F80, 0x4000], dtype="<u2").tobytes()  # 1.0, 2.0
    header = json.dumps({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, len(payload)]}}).encode()
    with open(path, "wb") as f:
      f.write(len(header).to_bytes(8, "little") + header + payload)
    with mock.patch.object(wan_utils.max_logging, "log") as log:
      tensors = wan_utils._read_safetensors_as_jax(path, "np")  # pylint: disable=protected-access
    self.assertIn("BF16", log.call_args.args[0])
    self.assertEqual(str(tensors["w"].dtype), "bfloat16")
    np.testing.assert_array_equal(np.asarray(tensors["w"], dtype=np.float32), [1.0, 2.0])

  def test_numpy_vae_load_imports_no_torch(self):
    snapshot = self._write(self._tensors())
    out = _run_python(
        "import sys\n"
        "from maxdiffusion.models.wan.wan_utils import load_wan_vae\n"
        f"params = load_wan_vae({snapshot!r}, {{}}, 'cpu', framework='np')\n"
        "print(len(params) > 0, 'torch' in sys.modules)\n"
    )
    self.assertEqual(out.strip().splitlines()[-1], "True False")

  def test_load_qwen_image_vae_reads_with_numpy(self):
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from maxdiffusion.models.wan import autoencoder_kl_wan, wan_utils  # pylint: disable=import-outside-toplevel

    class _Stop(Exception):
      pass

    recorded = {}

    def fake_load_wan_vae(snapshot_dir, params, device, **kwargs):
      recorded.update(snapshot_dir=snapshot_dir, device=device, **kwargs)
      raise _Stop()

    def tiny_vae(*args, rngs, **kwargs):
      del args, kwargs
      return nnx.Linear(2, 2, rngs=rngs)

    config = types.SimpleNamespace(activations_dtype="float32", weights_dtype="float32")
    with (
        mock.patch.object(autoencoder_kl_wan.AutoencoderKLWan, "from_config", side_effect=tiny_vae),
        mock.patch.object(wan_utils, "load_wan_vae", side_effect=fake_load_wan_vae),
        self.assertRaises(_Stop),
    ):
      generate_krea2.load_qwen_image_vae("/snap", config, None, nnx.Rngs(0))
    self.assertEqual(recorded, {"snapshot_dir": "/snap", "device": "cpu", "framework": "np"})


if __name__ == "__main__":
  unittest.main()
