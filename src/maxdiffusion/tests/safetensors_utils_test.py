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

import os
import tempfile
import unittest

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

from maxdiffusion.models.flux.util import cast_dict_to_bfloat16_inplace
from maxdiffusion.safetensors_utils import SafetensorsShardReader


class SafetensorsShardReaderTest(unittest.TestCase):

  def test_reads_requested_keys_across_shards(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      save_file({"a": np.arange(4, dtype=np.float32)}, os.path.join(tmpdir, "model-00001.safetensors"))
      save_file({"b": np.arange(3, dtype=np.int32)}, os.path.join(tmpdir, "model-00002.safetensors"))

      reader = SafetensorsShardReader(tmpdir)
      self.assertEqual(len(reader.files), 2)
      with reader:
        self.assertEqual(len(reader), 2)
        np.testing.assert_array_equal(reader.get_tensor("b"), np.arange(3, dtype=np.int32))
        self.assertEqual(len(reader), 1)
        self.assertIn("a", reader)
        self.assertNotIn("b", reader.keys())

  def test_duplicate_key_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      save_file({"duplicate": np.ones((1,), dtype=np.float32)}, os.path.join(tmpdir, "model-00001.safetensors"))
      save_file({"duplicate": np.zeros((1,), dtype=np.float32)}, os.path.join(tmpdir, "model-00002.safetensors"))

      with self.assertRaisesRegex(ValueError, "Duplicate safetensors key"):
        with SafetensorsShardReader(tmpdir):
          pass


class DtypeNormalizationTest(unittest.TestCase):

  def test_matching_dtypes_are_reused_without_copy(self):
    bf16 = np.ones((4,), dtype=ml_dtypes.bfloat16)
    norm = np.ones((4,), dtype=np.float32)
    params = {"kernel": bf16, "input_norm": {"weight": norm}}

    cast_dict_to_bfloat16_inplace(params, exclude_keywords=("norm",))

    self.assertIs(params["kernel"], bf16)
    self.assertIs(params["input_norm"]["weight"], norm)

  def test_only_mismatched_dtypes_are_cast(self):
    kernel = np.ones((4,), dtype=np.float32)
    params = {"kernel": kernel}

    cast_dict_to_bfloat16_inplace(params)

    self.assertIsNot(params["kernel"], kernel)
    self.assertEqual(params["kernel"].dtype, np.dtype(ml_dtypes.bfloat16))


if __name__ == "__main__":
  unittest.main()
