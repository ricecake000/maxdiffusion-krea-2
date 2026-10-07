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

# CPU tests for the Krea 2 pipeline's per-call seed and argument checks. No
# model: the pipeline object is created without __init__ and only the code
# paths that run before any model work are exercised.

import types
import unittest
from unittest import mock

import numpy as np

from maxdiffusion.models.flux.util import pack_latents
from maxdiffusion.pipelines.krea2 import krea2_pipeline
from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline, resolve_latent_seed


def _bare_pipeline(**config):
  pipeline = object.__new__(FlaxKrea2Pipeline)
  pipeline._config = types.SimpleNamespace(**config)  # pylint: disable=protected-access
  return pipeline


def _states_equal(a, b):
  return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


class ResolveLatentSeedTest(unittest.TestCase):

  def test_call_seed_wins(self):
    self.assertEqual(resolve_latent_seed(5, types.SimpleNamespace(seed=7)), 5)
    self.assertEqual(resolve_latent_seed(0, types.SimpleNamespace(seed=7)), 0)

  def test_config_seed_next(self):
    self.assertEqual(resolve_latent_seed(None, types.SimpleNamespace(seed=7)), 7)
    self.assertEqual(resolve_latent_seed(None, types.SimpleNamespace(seed=0)), 0)

  def test_time_last(self):
    with mock.patch.object(krea2_pipeline.time, "time", return_value=float(2**31 + 123)):
      self.assertEqual(resolve_latent_seed(None, types.SimpleNamespace(seed=None)), 123)
      self.assertEqual(resolve_latent_seed(None, types.SimpleNamespace()), 123)


class PrepareLatentsTest(unittest.TestCase):

  def test_matches_the_legacy_global_draw_and_keeps_the_global_rng(self):
    pipeline = _bare_pipeline(seed=11)
    shape = (2, 16, 64 // 8, 96 // 8)

    np.random.seed(1234)
    before = np.random.get_state()
    latents = pipeline._prepare_latents(2, 64, 96)  # pylint: disable=protected-access
    self.assertTrue(_states_equal(before, np.random.get_state()))

    np.random.seed(11)
    legacy = pack_latents(np.random.randn(*shape).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(latents), np.asarray(legacy))
    self.assertEqual(np.asarray(latents).shape, (2, (64 // 16) * (96 // 16), 64))

  def test_call_seed_overrides_config_and_is_deterministic(self):
    pipeline = _bare_pipeline(seed=11)
    a = np.asarray(pipeline._prepare_latents(1, 32, 32, seed=3))  # pylint: disable=protected-access
    b = np.asarray(pipeline._prepare_latents(1, 32, 32, seed=3))  # pylint: disable=protected-access
    c = np.asarray(pipeline._prepare_latents(1, 32, 32))  # pylint: disable=protected-access
    np.testing.assert_array_equal(a, b)
    self.assertFalse(np.array_equal(a, c))
    expected = pack_latents(np.random.RandomState(3).randn(1, 16, 4, 4).astype(np.float32))
    np.testing.assert_array_equal(a, np.asarray(expected))


class CallArgumentChecksTest(unittest.TestCase):

  def _pipeline(self):
    pipeline = _bare_pipeline(seed=1)
    pipeline._setup_jit_functions = mock.Mock()  # pylint: disable=protected-access
    pipeline.encode_prompt = mock.Mock()
    return pipeline

  def _assert_untouched(self, pipeline):
    pipeline._setup_jit_functions.assert_not_called()  # pylint: disable=protected-access
    pipeline.encode_prompt.assert_not_called()

  def test_bad_output_type(self):
    pipeline = self._pipeline()
    for output_type in ("png", "PIL", None, "files"):
      with self.subTest(output_type=output_type):
        with self.assertRaisesRegex(ValueError, "output_type"):
          pipeline("a fox", None, None, output_type=output_type)
    self._assert_untouched(pipeline)

  def test_wrong_prompt_count(self):
    pipeline = self._pipeline()
    with self.assertRaisesRegex(ValueError, "2 prompt"):
      pipeline(["a", "b"], None, None, batch_size=1)
    with self.assertRaisesRegex(ValueError, "1 prompt"):
      pipeline(["a"], None, None, batch_size=2)
    self._assert_untouched(pipeline)

  def test_wrong_negative_prompt_count(self):
    pipeline = self._pipeline()
    with self.assertRaisesRegex(ValueError, "negative prompt"):
      pipeline("a", None, None, batch_size=2, negative_prompt=["x"])
    self._assert_untouched(pipeline)


if __name__ == "__main__":
  unittest.main()
