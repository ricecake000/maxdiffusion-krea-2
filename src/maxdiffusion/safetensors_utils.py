# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy, shard-aware safetensors reads for host-side weight conversion."""

from contextlib import ExitStack
import glob
import os


class SafetensorsShardReader:
  """Dictionary-like lazy reader over one safetensors file or a shard directory.

  `safetensors.numpy.load_file` materializes every tensor in every shard before
  conversion starts. Large inference checkpoints often use only a subset of a
  component (Krea 2 uses the Qwen3-VL language tower but not its vision tower),
  so opening the files once and fetching keys on demand avoids unnecessary
  reads and keeps peak host memory bounded.
  """

  def __init__(self, path: str):
    if os.path.isdir(path):
      self.files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    else:
      self.files = [path]
    if not self.files:
      raise FileNotFoundError(f"No safetensors files found at {path}")

    self._stack = None
    self._key_to_handle = {}
    self._remaining_keys = set()

  def __enter__(self):
    from safetensors import safe_open

    self._stack = ExitStack()
    try:
      for filename in self.files:
        handle = self._stack.enter_context(safe_open(filename, framework="np"))
        for key in handle.keys():
          if key in self._key_to_handle:
            raise ValueError(f"Duplicate safetensors key '{key}' in {filename}")
          self._key_to_handle[key] = handle
      self._remaining_keys = set(self._key_to_handle)
      return self
    except Exception:
      self._stack.close()
      self._stack = None
      self._key_to_handle.clear()
      self._remaining_keys.clear()
      raise

  def __exit__(self, exc_type, exc_value, traceback):
    if self._stack is not None:
      self._stack.close()
    self._stack = None
    self._key_to_handle.clear()
    self._remaining_keys.clear()

  def __contains__(self, key):
    return key in self._remaining_keys

  def __len__(self):
    return len(self._remaining_keys)

  def keys(self):
    return self._remaining_keys

  def get_tensor(self, key):
    if key not in self._remaining_keys:
      raise KeyError(f"Weight '{key}' not found or already consumed from {self.files}")
    try:
      handle = self._key_to_handle[key]
    except KeyError as e:
      raise KeyError(f"Weight '{key}' not found in {self.files}") from e
    self._remaining_keys.remove(key)
    return handle.get_tensor(key)

  def pop(self, key):
    return self.get_tensor(key)
