# Copyright 2023 Google LLC
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

import importlib

# Loaded on first access (PEP 562), not with the package: these loaders import torch (through
# models.modeling_utils), seconds on every start of a process that only needs a torch-free loader such as
# loaders.krea2_lora_pipeline. `from maxdiffusion.loaders import X` works as before.
_LAZY_EXPORTS = {
    "StableDiffusionLoraLoaderMixin": ".lora_pipeline",
    "FluxLoraLoaderMixin": ".flux_lora_pipeline",
    "Wan2_1NNXLoraLoader": ".wan_lora_nnx_loader",
    "Wan2_2NNXLoraLoader": ".wan_lora_nnx_loader",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
  module = _LAZY_EXPORTS.get(name)
  if module is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  value = getattr(importlib.import_module(module, __name__), name)
  globals()[name] = value
  return value


def __dir__():
  return sorted(set(globals()) | set(_LAZY_EXPORTS))
