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
import pytest

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from maxdiffusion import pyconfig
from maxdiffusion import generate_krea2

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT = "a fox in the snow"


class GenerateKrea2SmokeTest(unittest.TestCase):
  """End-to-end smoke tests for Krea 2 Raw and Turbo."""

  def _run(self, config_name, run_name, output_name, ref_name, extra_args=()):
    ref_path = os.path.join(THIS_DIR, "images", ref_name)
    if not os.path.exists(ref_path):
      # Reference images are produced once from a verified TPU run and committed
      # to tests/images/. They must be generated with the config defaults, which
      # pin the model-card sampler settings (Raw: 52 steps / guidance 3.5,
      # Turbo: 8 steps / guidance 0.0). Until then the smoke test cannot score SSIM.
      self.skipTest(f"Reference image not found: {ref_path}. Generate it from a verified TPU run first.")
    base_image = np.array(Image.open(ref_path)).astype(np.uint8)

    output_dir = f"/mnt/data/{run_name}" if os.path.exists("/mnt/data") else f"/tmp/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    if os.path.exists(out_path):
      os.remove(out_path)

    pyconfig._config = None
    pyconfig.config = None
    args = [
        None,
        os.path.join(THIS_DIR, "..", "configs", config_name),
        f"run_name={run_name}",
        f"output_dir={output_dir}",
        "jax_cache_dir=/tmp/cache_dir",
        "skip_jax_distributed_system=True",
        f"prompt={PROMPT}",
        "height=512",
        "width=512",
        "batch_size=1",
        "seed=42",
        "ici_fsdp_parallelism=-1",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
    ] + list(extra_args)

    generate_krea2.main(args)

    self.assertTrue(os.path.exists(out_path), f"Smoke test {run_name} failed to produce output image!")
    test_image = np.array(Image.open(out_path)).astype(np.uint8)

    self.assertEqual(base_image.shape, test_image.shape)
    ssim_compare = ssim(base_image, test_image, channel_axis=-1, data_range=255)
    print(f"\n[SMOKE TEST {run_name}] SSIM Score: {ssim_compare:.6f}")
    self.assertGreaterEqual(ssim_compare, 0.75)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_krea2_raw_smoke(self):
    """End-to-end smoke test for Krea-2-Raw (52 steps, CFG 3.5) at 512x512."""
    self._run(
        config_name="base_krea2.yml",
        run_name="krea2_raw_smoke",
        output_name="krea2_generated_image.png",
        ref_name="ref_krea2_raw.png",
    )

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_krea2_turbo_smoke(self):
    """End-to-end smoke test for Krea-2-Turbo (8 steps, no guidance) at 512x512."""
    self._run(
        config_name="base_krea2_turbo.yml",
        run_name="krea2_turbo_smoke",
        output_name="krea2_turbo_generated_image.png",
        ref_name="ref_krea2_turbo.png",
    )

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions (requires TPU HBM)")
  def test_krea2_turbo_lora_zero_scale_smoke(self):
    """Runs Turbo with a LoRA adapter at scale=0 and scores against the base
    reference image: the full LoRA path (key conversion, param-tree insertion,
    interceptors around shape eval and the jitted denoise loop) must be
    numerically inert at zero scale. Set KREA2_LORA_PATH to any Krea 2 LoRA
    .safetensors file to enable."""
    lora_path = os.environ.get("KREA2_LORA_PATH", "")
    if not lora_path or not os.path.exists(lora_path):
      self.skipTest("Set KREA2_LORA_PATH to a Krea 2 LoRA .safetensors file to run this test.")
    lora_config = (
        'lora_config={"lora_model_name_or_path": ["%s"], "weight_name": ["%s"], '
        '"adapter_name": ["smoke"], "scale": [0.0], "from_pt": ["true"]}' % (lora_path, os.path.basename(lora_path))
    )
    self._run(
        config_name="base_krea2_turbo.yml",
        run_name="krea2_turbo_lora_smoke",
        output_name="krea2_turbo_lora_generated_image.png",
        ref_name="ref_krea2_turbo.png",
        extra_args=("output_name=krea2_turbo_lora_generated_image.png", lora_config),
    )


class PromptResolutionTest(unittest.TestCase):

  def test_prompt_file_preserves_one_prompt_per_nonempty_line(self):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as prompt_file:
      prompt_file.write("first prompt\n\n second prompt \n")
      prompt_file.flush()

      prompts = generate_krea2.resolve_prompts("ignored", 2, prompt_file.name)

    self.assertEqual(prompts, ["first prompt", "second prompt"])

  def test_prompt_file_requires_exact_batch_size(self):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as prompt_file:
      prompt_file.write("only one\n")
      prompt_file.flush()

      with self.assertRaisesRegex(ValueError, "exactly batch_size=2"):
        generate_krea2.resolve_prompts("ignored", 2, prompt_file.name)


if __name__ == "__main__":
  unittest.main()
