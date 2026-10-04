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

"""CPU (Pallas interpret mode) tests for the Krea 2 prefix-masked GQA kernel."""

import dataclasses
import math
import os
import tempfile
import types
import unittest
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from maxdiffusion import max_utils
from maxdiffusion.kernels import krea2_attention
from maxdiffusion.models import attention_flax

_LOG2E = math.log2(math.e)
_D = 128


def _random_qkv(batch, hq, hkv, seq_len, seed=0):
  keys = jax.random.split(jax.random.PRNGKey(seed), 3)
  q = jax.random.normal(keys[0], (batch, hq, seq_len, _D), jnp.float32).astype(jnp.bfloat16)
  k = jax.random.normal(keys[1], (batch, hkv, seq_len, _D), jnp.float32).astype(jnp.bfloat16)
  v = jax.random.normal(keys[2], (batch, hkv, seq_len, _D), jnp.float32).astype(jnp.bfloat16)
  return q, k, v


def _reference(q, k, v, valid_len, scale):
  """f32 softmax attention with a per-batch prefix key mask; (B, Hq, L, D)."""
  group = q.shape[1] // k.shape[1]
  qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
  kf = jnp.repeat(kf, group, axis=1)
  vf = jnp.repeat(vf, group, axis=1)
  logits = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
  kv_pos = jnp.arange(k.shape[2])
  valid = kv_pos[None, :] < jnp.asarray(valid_len)[:, None]  # (B, L_kv)
  logits = jnp.where(valid[:, None, None, :], logits, -1e30)
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum("bhqk,bhkd->bhqd", probs, vf)


def _run_kernel(q, k, v, valid_len, block_sizes, scale):
  """Pads like the registry wrapper and runs the kernel factory in interpret mode."""
  seq_len = q.shape[2]
  q_pad = krea2_attention.padded_len(seq_len, block_sizes.block_q) - seq_len
  kv_pad = krea2_attention.padded_len(seq_len, block_sizes.block_kv) - seq_len
  q_scaled = (q.astype(jnp.float32) * scale * _LOG2E).astype(q.dtype)
  q_scaled = jnp.pad(q_scaled, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
  k = jnp.pad(k, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
  v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
  fn = krea2_attention.make_krea2_attention(
      block_sizes, q_seq_len=seq_len, kv_seq_len=seq_len, use_base2_exp=True, interpret=True
  )
  out = fn(q_scaled, k, v, jnp.asarray(valid_len, jnp.int32))
  return jnp.swapaxes(out, 2, 3)


class Krea2AttentionKernelTest(unittest.TestCase):

  scale = 1.0 / math.sqrt(_D)

  def _assert_matches(self, out, ref):
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(ref, np.float32), atol=2e-2, rtol=2e-2)

  def test_exact_fit_gqa_full_validity(self):
    q, k, v = _random_qkv(1, 4, 2, 256)
    bs = krea2_attention.Krea2BlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    out = _run_kernel(q, k, v, [256], bs, self.scale)
    self.assertEqual(out.shape, (1, 4, 256, _D))
    self.assertEqual(out.dtype, jnp.bfloat16)
    self._assert_matches(out, _reference(q, k, v, [256], self.scale))

  def test_ragged_static_tail(self):
    # 320 = 2 * 128 + 64: the last kv block is ragged and q is padded to 384.
    q, k, v = _random_qkv(1, 4, 1, 320, seed=1)
    bs = krea2_attention.Krea2BlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    out = _run_kernel(q, k, v, [320], bs, self.scale)
    self.assertEqual(out.shape, (1, 4, 320, _D))
    self._assert_matches(out, _reference(q, k, v, [320], self.scale))

  def test_ragged_tail_with_register_tiling(self):
    # bkv=256 / compute=256 / compute_in=128 with 320 = 256 + 64: exercises the
    # multi-chunk inner loop and a sub-compute-block ragged tail.
    q, k, v = _random_qkv(1, 2, 1, 320, seed=2)
    bs = krea2_attention.Krea2BlockSizes(block_q=256, block_kv=256, block_kv_compute=256, block_kv_compute_in=128)
    for valid in ([320], [300], [140]):
      with self.subTest(valid=valid):
        out = _run_kernel(q, k, v, valid, bs, self.scale)
        self._assert_matches(out, _reference(q, k, v, valid, self.scale))

  def test_dynamic_prefix_mask(self):
    q, k, v = _random_qkv(2, 4, 2, 320, seed=3)
    bs = krea2_attention.Krea2BlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    valid = [200, 320]
    out = _run_kernel(q, k, v, valid, bs, self.scale)
    ref = _reference(q, k, v, valid, self.scale)
    for b in range(2):
      self._assert_matches(out[b], ref[b])
    # The mask must actually matter for batch 0.
    unmasked = _reference(q, k, v, [320, 320], self.scale)
    self.assertGreater(float(jnp.max(jnp.abs(unmasked[0] - ref[0]))), 0.05)

  def test_prefix_mask_spanning_blocks(self):
    # valid=130 with bkv=128: block 1 is partially valid, block 2 fully masked.
    q, k, v = _random_qkv(2, 4, 2, 320, seed=4)
    bs = krea2_attention.Krea2BlockSizes(block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128)
    valid = [130, 257]
    out = _run_kernel(q, k, v, valid, bs, self.scale)
    ref = _reference(q, k, v, valid, self.scale)
    for b in range(2):
      self._assert_matches(out[b], ref[b])

  def test_select_block_sizes(self):
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224).block_q, 1408)
    self.assertEqual(krea2_attention.select_krea2_block_sizes(16512).block_q, 1664)
    default = krea2_attention.select_krea2_block_sizes(4224)
    self.assertEqual((default.block_kv, default.block_kv_compute, default.block_kv_compute_in), (1024, 512, 256))
    # Tiny inputs cap block_q at the sequence rounded up to 128.
    self.assertEqual(krea2_attention.select_krea2_block_sizes(200).block_q, 256)
    user = max_utils.CustomFlashBlockSizes(
        block_q=2048,
        block_kv=2048,
        block_kv_compute=None,
        block_kv_compute_in=None,
        heads_per_tile=None,
        vmem_limit_bytes=None,
    )
    sizes = krea2_attention.select_krea2_block_sizes(4224, user=user)
    self.assertEqual(
        (sizes.block_q, sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in), (2048, 2048, 512, 256)
    )
    sizes = krea2_attention.select_krea2_block_sizes(4224, user={"block_kv_compute": 1024, "block_kv_compute_in": 512})
    self.assertEqual(
        (sizes.block_q, sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in), (1408, 1024, 1024, 512)
    )
    with self.assertRaises(ValueError):
      krea2_attention.select_krea2_block_sizes(4224, user={"block_q": 1000})
    with self.assertRaises(ValueError):
      krea2_attention.Krea2BlockSizes(block_q=512, block_kv=1024, block_kv_compute=384, block_kv_compute_in=128)
    with self.assertRaises(ValueError):
      krea2_attention.Krea2BlockSizes(block_q=512, block_kv=1024, block_kv_compute=512, block_kv_compute_in=384)
    self.assertEqual(krea2_attention.padded_len(4224, 1024), 5120)
    self.assertEqual(krea2_attention.padded_len(4096, 1024), 4096)


_V6E = "TPU v6 lite"
_V5E = "TPU v5 lite"

# Calibration record of the VMEM-budget extension (off by default since block
# selection revision 3), checked with the extension switched on:
# (device_kind, user kv sizes or None, max_auto_block_q, {seq_len: block_q}).
_AUTO_BLOCK_Q_CASES = (
    (
        _V6E,
        (2048, 1024, 256),
        4992,
        {
            200: 256,
            1152: 1152,
            3728: 3840,
            4224: 4224,
            4352: 4352,
            4480: 4480,
            4608: 4608,
            5000: 2560,
            9344: 4736,
            16512: 3328,
            16896: 4224,
        },
    ),
    (_V6E, (1024, 512, 256), 7808, {4224: 4224, 5000: 5120, 16512: 5504, 16640: 5632, 16896: 5632}),
    (_V6E, (4096, 1024, 256), 4480, {4480: 4480, 4608: 2304, 16512: 3328}),
    (_V6E, (2048, 2048, 256), 3072, {4224: 2176, 4352: 2176, 16512: 2816, 16896: 2816}),
    (_V6E, (2048, 1024, 128), 2048, {4224: 1408, 4352: 1536, 16512: 1664}),
    (_V6E, (1024, 256, 256), 8192, {16512: 5504}),
    (_V5E, (2048, 1024, 256), 2304, {4224: 2176, 4352: 2176, 4608: 2304, 16512: 1664}),
    (_V5E, None, 3840, {3728: 3840, 4224: 2176, 16512: 3328}),
)

# Compile-only calibration FAIL points:
# (device_kind, block_q, block_kv, block_kv_compute, block_kv_compute_in).
_VMEM_FAIL_POINTS = (
    (_V6E, 9088, 1024, 512, 256),
    (_V6E, 8832, 2048, 512, 256),
    (_V6E, 5888, 1024, 1024, 256),
    (_V6E, 5632, 2048, 1024, 256),
    (_V6E, 5120, 4096, 1024, 256),
    (_V6E, 3712, 2048, 2048, 256),
    (_V6E, 3328, 4096, 2048, 256),
    (_V6E, 5504, 2048, 1024, 128),
    (_V5E, 4608, 1024, 512, 256),
    (_V5E, 3072, 2048, 1024, 256),
    (_V5E, 3072, 4096, 1024, 256),
)


def _kv_user(kv_sizes, **extra):
  """User sizes for the flash variant: kernel "flash" is forced, since "auto" is hybrid on v6e (revision 4)."""
  user = {"kernel": "flash", **extra}
  if kv_sizes is not None:
    user.update(zip(("block_kv", "block_kv_compute", "block_kv_compute_in"), kv_sizes))
  return user


def _budget_extension(enabled):
  """Switches the module default of the VMEM-budget extension for a `with` block."""
  return mock.patch.object(krea2_attention, "AUTO_BLOCK_Q_BUDGET_EXTENSION", enabled)


# Automatic block_q with the default (extension off) for the v6e-1 preset's kv
# sizes at the Krea 2 preset sequence lengths (image tokens + 128 text tokens).
_PRESET_DEFAULT_BLOCK_Q = {
    4224: 1408,
    4016: 2048,
    4160: 1408,
    4184: 1408,
    16512: 1664,
    16256: 2048,
    16352: 2048,
    15680: 1792,
}

# Automatic block_q with the default maximum 2048 at every sequence length of
# `_AUTO_BLOCK_Q_CASES` (any kv sizes, extension off).
_BASE_MAX_BLOCK_Q = {
    200: 256,
    1152: 1152,
    3728: 1920,
    4224: 1408,
    4352: 1536,
    4480: 1536,
    4608: 1536,
    5000: 1280,
    9344: 1920,
    16512: 1664,
    16640: 1664,
    16896: 1920,
}


class Krea2AutoBlockQTest(unittest.TestCase):

  def test_max_auto_block_q(self):
    for device_kind, kv_sizes, expected_max, _ in _AUTO_BLOCK_Q_CASES:
      with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes):
        sizes = krea2_attention.select_krea2_block_sizes(4224, user=_kv_user(kv_sizes))
        kv = (sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in)
        self.assertEqual(krea2_attention.max_auto_block_q(device_kind, *kv, budget_extension=True), expected_max)
    # Uncalibrated chips and a user VMEM limit keep the pre-budget maximum.
    for device_kind in (None, "cpu", "TPU v4", "TPU v5p"):
      self.assertEqual(krea2_attention.max_auto_block_q(device_kind, 1024, 512, 256, budget_extension=True), 2048)
    self.assertEqual(
        krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, vmem_limit_bytes=67108864, budget_extension=True), 2048
    )
    # Never below the base, even when the base itself exceeds the estimate budget.
    self.assertEqual(krea2_attention.max_auto_block_q(_V5E, 4096, 2048, 256, budget_extension=True), 2048)

  def test_budget_extension_is_off_by_default(self):
    self.assertFalse(krea2_attention.AUTO_BLOCK_Q_BUDGET_EXTENSION)
    self.assertEqual(krea2_attention.KREA2_BLOCK_SELECTION_REVISION, 4)
    for device_kind in (_V6E, _V5E, None):
      with self.subTest(device_kind=device_kind):
        self.assertEqual(krea2_attention.max_auto_block_q(device_kind, 2048, 1024, 256), 2048)
        self.assertEqual(krea2_attention.max_auto_block_q(device_kind, 2048, 1024, 256, budget_extension=False), 2048)
    self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, budget_extension=True), 4992)
    # The keyword wins over the module default in both directions.
    with _budget_extension(True):
      self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256), 4992)
      self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, budget_extension=False), 2048)

  def test_default_block_q_at_preset_sequence_lengths(self):
    user = _kv_user((2048, 1024, 256))
    for seq_len, block_q in _PRESET_DEFAULT_BLOCK_Q.items():
      with self.subTest(seq_len=seq_len):
        self.assertEqual(krea2_attention._default_block_q(seq_len, 2048), block_q)  # pylint: disable=protected-access
        for device_kind in (_V6E, _V5E, None):
          sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=user, device_kind=device_kind)
          self.assertEqual(sizes.block_q, block_q)
          self.assertEqual((sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in), (2048, 1024, 256))

  def test_overhead_term_avoids_tiny_blocks(self):
    default_block_q = krea2_attention._default_block_q  # pylint: disable=protected-access
    # 2k@4:3 / 3:4 (seq 15680): 9 blocks of 1792 (waste 448) instead of 31 blocks of 512 (waste 192).
    self.assertEqual(default_block_q(15680, 2048), 1792)
    self.assertNotEqual(default_block_q(15680, 2048), 512)
    sizes = krea2_attention.select_krea2_block_sizes(15680, user=_kv_user((2048, 1024, 256)), device_kind=_V6E)
    self.assertEqual(sizes.block_q, 1792)
    # Without the per-block term the waste-only rule picks the tiny blocks.
    with mock.patch.object(krea2_attention, "_BLOCK_Q_OVERHEAD_ROWS", 0):
      self.assertEqual(default_block_q(15680, 2048), 512)
      self.assertEqual(default_block_q(4352, 2048), 896)
      self.assertEqual(default_block_q(16896, 2048), 1536)

  def test_auto_block_q_per_device(self):
    default_block_q = krea2_attention._default_block_q  # pylint: disable=protected-access
    for device_kind, kv_sizes, _, expected in _AUTO_BLOCK_Q_CASES:
      for seq_len, block_q in expected.items():
        with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes, seq_len=seq_len):
          sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
          kv = (sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in)
          if kv_sizes is not None:
            self.assertEqual(kv, kv_sizes)
          # The calibration table holds with the extension on ...
          extended_max = krea2_attention.max_auto_block_q(device_kind, *kv, budget_extension=True)
          self.assertEqual(default_block_q(seq_len, extended_max), block_q)
          with _budget_extension(True):
            extended = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
          self.assertEqual(extended.block_q, block_q)
          # ... while the default keeps the choice at most 2048.
          self.assertEqual(sizes.block_q, _BASE_MAX_BLOCK_Q[seq_len])
          self.assertEqual(sizes.block_q, default_block_q(seq_len, 2048))

  def test_uncalibrated_device_follows_default_rule(self):
    # Uncalibrated chips (None, "cpu") get the same default-rule choice with the
    # budget extension off and on: the extension never applies to them.
    for extension in (False, True):
      for device_kind in (None, "cpu"):
        for kv_sizes in (None, (2048, 1024, 256), (1024, 256, 256)):
          for seq_len, block_q in {4224: 1408, 4352: 1536, 16512: 1664}.items():
            with self.subTest(extension=extension, device_kind=device_kind, kv_sizes=kv_sizes, seq_len=seq_len):
              with _budget_extension(extension):
                sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
              self.assertEqual(sizes.block_q, block_q)

  def test_user_vmem_limit_and_block_q(self):
    for extension in (False, True):
      with self.subTest(extension=extension), _budget_extension(extension):
        self._check_user_vmem_limit_and_block_q()

  def _check_user_vmem_limit_and_block_q(self):
    user = _kv_user((2048, 1024, 256), vmem_limit_bytes=67108864)
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E).block_q, 1408)
    carrier = max_utils.CustomFlashBlockSizes(
        block_q=None,
        block_kv=2048,
        block_kv_compute=1024,
        block_kv_compute_in=256,
        heads_per_tile=None,
        vmem_limit_bytes=67108864,
        kernel="flash",
    )
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224, user=carrier, device_kind=_V6E).block_q, 1408)
    # A user block_q always wins.
    user = _kv_user((2048, 1024, 256), block_q=1408)
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E).block_q, 1408)

  def test_budget_only_for_bfloat16_operands(self):
    user = _kv_user((2048, 1024, 256))
    # The VMEM budget is calibrated with bf16 q/k/v; wider operands keep the previous choice.
    # Checked with the budget extension on (it is off by default).
    for dtype in (jnp.float32, jnp.float16, np.float32, "float32"):
      with self.subTest(dtype=dtype):
        self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, dtype=dtype, budget_extension=True), 2048)
        for seq_len, block_q in {4224: 1408, 16512: 1664}.items():
          with _budget_extension(True):
            sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=user, device_kind=_V6E, dtype=dtype)
          self.assertEqual(sizes.block_q, block_q)
    # None means bf16; bf16 is accepted as a jnp scalar type, a numpy dtype object and a string.
    for dtype in (None, jnp.bfloat16, np.dtype(jnp.bfloat16), "bfloat16", jnp.zeros((1,), jnp.bfloat16).dtype):
      with self.subTest(dtype=dtype):
        self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, dtype=dtype, budget_extension=True), 4992)
        with _budget_extension(True):
          sizes = krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E, dtype=dtype)
        self.assertEqual(sizes.block_q, 4224)

  def test_vmem_budget_covers_choices_and_excludes_failures(self):
    budgets = krea2_attention._VMEM_BUDGET_BYTES  # pylint: disable=protected-access
    estimate = krea2_attention._estimated_vmem_bytes  # pylint: disable=protected-access
    for device_kind, kv_sizes, _, expected in _AUTO_BLOCK_Q_CASES:
      for seq_len in expected:
        with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes, seq_len=seq_len):
          with _budget_extension(True):
            sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
          self.assertLessEqual(estimate(sizes.block_q, sizes.block_kv, sizes.block_kv_compute), budgets[device_kind])
    # Nobody may raise a budget past a block_q that is known to fail the compile.
    for device_kind, block_q, block_kv, block_kv_compute, block_kv_compute_in in _VMEM_FAIL_POINTS:
      with self.subTest(device_kind=device_kind, block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv_compute):
        self.assertGreater(estimate(block_q, block_kv, block_kv_compute), budgets[device_kind])
        max_block_q = krea2_attention.max_auto_block_q(
            device_kind, block_kv, block_kv_compute, block_kv_compute_in, budget_extension=True
        )
        self.assertLess(max_block_q, block_q)


def _rel_l2(out, ref):
  out, ref = np.asarray(out, np.float64), np.asarray(ref, np.float64)
  return float(np.linalg.norm(out - ref) / np.linalg.norm(ref))


def _hybrid(block_q, block_kv, block_kv_compute, block_kv_compute_in, block_kv_pv, block_q_strip):
  return krea2_attention.Krea2BlockSizes(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_kv_compute_in=block_kv_compute_in,
      variant="hybrid",
      block_kv_pv=block_kv_pv,
      block_q_strip=block_q_strip,
  )


# Hybrid numerics suite (the probe's small suite): seq 1100 with block_kv 512 is
# 3 kv blocks, the last one ragged (76 rows, shorter than block_kv_compute_in);
# block_q 384 is 3 q blocks of one 256-lane strip plus a 128-lane remainder
# strip. (name, batch, hq, hkv, valid, (bq, bkv, compute, compute_in, pv, strip)).
_HYBRID_CASES = (
    ("masked_last_ragged_tail_pv_split", 1, 4, 2, [1049], (384, 512, 512, 256, 128, 256)),
    ("masked_body_block", 1, 4, 2, [700], (384, 512, 512, 512, 256, 256)),
    ("two_batches_pv_eq_in", 2, 4, 2, [1100, 300], (384, 512, 512, 256, 256, 256)),
    ("two_qk_chunks_per_block", 1, 4, 2, [1049], (384, 512, 256, 256, 128, 256)),
    ("strip_128", 1, 4, 2, [1049], (384, 512, 512, 256, 128, 128)),
    ("gqa_4_unmasked", 1, 4, 1, [1100], (384, 512, 512, 512, 128, 256)),
)


class Krea2HybridKernelTest(unittest.TestCase):

  scale = 1.0 / math.sqrt(_D)

  def test_hybrid_matches_reference_and_flash(self):
    for name, batch, hq, hkv, valid, sizes in _HYBRID_CASES:
      with self.subTest(case=name):
        q, k, v = _random_qkv(batch, hq, hkv, 1100, seed=len(name))
        out = _run_kernel(q, k, v, valid, _hybrid(*sizes), self.scale)
        self.assertEqual(out.shape, (batch, hq, 1100, _D))
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out.astype(jnp.float32)))))
        ref = _reference(q, k, v, valid, self.scale)
        np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(ref, np.float32), atol=2e-2, rtol=2e-2)
        flash = _run_kernel(q, k, v, valid, krea2_attention.Krea2BlockSizes(*sizes[:4]), self.scale)
        np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(flash, np.float32), atol=2e-2, rtol=2e-2)
        # The outputs are small (|o| ~ 0.1 for 1000 random keys), so also bound the relative L2 error: bf16
        # rounding of the output and of P gives ~3e-3; a wrong l, mask or rescale gives > 1e-1.
        self.assertLess(_rel_l2(out, ref), 1e-2)
        self.assertLess(_rel_l2(out, flash), 1e-2)

  def test_hybrid_default_sizes_small_sequence(self):
    # The hybrid defaults (2048/2048/1024/256/256) on a sequence shorter than one kv block.
    q, k, v = _random_qkv(1, 2, 1, 320, seed=11)
    sizes = krea2_attention.select_krea2_block_sizes(320, user={"kernel": "hybrid"})
    self.assertEqual(sizes, _hybrid(384, 2048, 2048, 1024, 256, 256))
    out = _run_kernel(q, k, v, [250], sizes, self.scale)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(_reference(q, k, v, [250], self.scale), np.float32), atol=2e-2, rtol=2e-2
    )


# Unique transformer sequence lengths of the 22 aspect presets with the 128-token text bucket.
_PRESET_SEQ_LENS = (4224, 4160, 4016, 4184, 16512, 16256, 15680, 16352)


class Krea2KernelVariantSelectionTest(unittest.TestCase):

  def test_parse_kernel_choice(self):
    parse = krea2_attention.parse_kernel_choice
    for value in (None, "", "   ", "''", '""', " '' ", "' '", "auto", "AUTO", " Auto ", "'auto'"):
      with self.subTest(value=value):
        self.assertEqual(parse(value), "auto")
    for value, expected in (("flash", "flash"), (" FLASH", "flash"), ('"hybrid"', "hybrid"), ("Hybrid\n", "hybrid")):
      with self.subTest(value=value):
        self.assertEqual(parse(value), expected)
    for value in ("flash_custom", "splash", "hybrid2", "auto,flash", 0, 1, True, ["flash"]):
      with self.subTest(value=value), self.assertRaisesRegex(ValueError, "auto.*flash.*hybrid"):
        parse(value)
    self.assertEqual(krea2_attention.KERNEL_VARIANTS, ("flash", "hybrid"))
    self.assertEqual(krea2_attention.KERNEL_CHOICES, ("auto", "flash", "hybrid"))

  def test_resolve_kernel_variant(self):
    resolve = krea2_attention.resolve_kernel_variant
    self.assertEqual(resolve("auto", _V6E), "hybrid")
    self.assertEqual(resolve(None, _V6E), "hybrid")
    self.assertEqual(resolve("", _V6E), "hybrid")
    for device_kind in (_V5E, None, "cpu", "TPU v4", "TPU v5p", "TPU v6e"):
      with self.subTest(device_kind=device_kind):
        self.assertEqual(resolve("auto", device_kind), "flash")
    for device_kind in (_V6E, _V5E, None, "cpu"):
      with self.subTest(forced=device_kind):
        self.assertEqual(resolve("flash", device_kind), "flash")
        self.assertEqual(resolve("hybrid", device_kind), "hybrid")
    with self.assertRaises(ValueError):
      resolve("fast", _V6E)

  def test_block_sizes_validation(self):
    bs = krea2_attention.Krea2BlockSizes
    self.assertEqual(bs(block_q=512).variant, "flash")
    _hybrid(512, 2048, 2048, 1024, 256, 256)
    _hybrid(512, 2048, 2048, 1024, 1024, 384)  # pv == compute_in; a strip need not divide block_q
    invalid = (
        dict(variant="splash"),
        dict(block_kv_pv=256),  # flash with hybrid-only fields
        dict(block_q_strip=256),
        dict(variant="hybrid", block_q_strip=256),  # hybrid without block_kv_pv
        dict(variant="hybrid", block_kv_pv=256),
        dict(variant="hybrid", block_kv_pv=192, block_q_strip=256),
        dict(variant="hybrid", block_kv_pv=256, block_q_strip=0),
        dict(variant="hybrid", block_kv_pv=512, block_q_strip=256),  # compute_in 256 % pv 512
        dict(variant="hybrid", block_kv_pv=384, block_q_strip=256),
        dict(variant="hybrid", block_kv_pv=True, block_q_strip=256),
    )
    for kwargs in invalid:
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        bs(block_q=512, block_kv=1024, block_kv_compute=512, block_kv_compute_in=256, **kwargs)

  def test_defaults_per_variant_and_chip(self):
    select = krea2_attention.select_krea2_block_sizes
    kv = lambda s: (s.variant, s.block_kv, s.block_kv_compute, s.block_kv_compute_in, s.block_kv_pv, s.block_q_strip)
    hybrid = ("hybrid", 2048, 2048, 1024, 256, 256)
    cases = (
        (None, None, ("flash", 1024, 512, 256, None, None)),
        (None, "cpu", ("flash", 1024, 512, 256, None, None)),
        (None, _V5E, ("flash", 1024, 512, 256, None, None)),
        (None, _V6E, hybrid),
        ("auto", _V6E, hybrid),
        ("", _V6E, hybrid),
        ("flash", _V6E, ("flash", 2048, 1024, 256, None, None)),
        ("flash", None, ("flash", 1024, 512, 256, None, None)),
        ("hybrid", None, hybrid),
        ("hybrid", "cpu", hybrid),
        ("hybrid", _V5E, hybrid),
    )
    for kernel, device_kind, expected in cases:
      for user in ({"kernel": kernel}, max_utils.CustomFlashBlockSizes(kernel=kernel)):
        with self.subTest(kernel=kernel, device_kind=device_kind, user=type(user).__name__):
          self.assertEqual(kv(select(4224, user=user, device_kind=device_kind)), expected)
    self.assertEqual(kv(select(4224, device_kind=_V6E)), hybrid)
    self.assertEqual(
        krea2_attention.default_kv_block_sizes("flash", "TPU v4"),
        {"block_kv": 1024, "block_kv_compute": 512, "block_kv_compute_in": 256, "block_kv_pv": None, "block_q_strip": None},
    )

  def test_user_overrides(self):
    select = krea2_attention.select_krea2_block_sizes
    sizes = select(4224, user={"kernel": "hybrid", "block_kv_pv": 512, "block_q_strip": 128}, device_kind=_V6E)
    self.assertEqual(sizes, _hybrid(1408, 2048, 2048, 1024, 512, 128))
    # A smaller block_kv clamps the defaults below it, like flash.
    sizes = select(4224, user={"block_kv": 1024}, device_kind=_V6E)
    self.assertEqual(sizes, _hybrid(1408, 1024, 1024, 1024, 256, 256))
    sizes = select(4224, user={"block_kv_compute": 512}, device_kind=_V6E)
    self.assertEqual(sizes, _hybrid(1408, 2048, 512, 512, 256, 256))
    sizes = select(4224, user={"block_kv_compute_in": 128}, device_kind=_V6E)
    self.assertEqual(sizes, _hybrid(1408, 2048, 2048, 128, 128, 256))
    sizes = select(4224, user={"block_q": 1024, "kernel": "hybrid"})
    self.assertEqual(sizes.block_q, 1024)
    # Flash with hybrid-only fields is a user error, with the kernel forced or chosen by "auto".
    for user, device_kind in (
        ({"kernel": "flash", "block_kv_pv": 256}, _V6E),
        ({"block_q_strip": 256}, _V5E),
        (max_utils.CustomFlashBlockSizes(block_kv_pv=256), None),
    ):
      with self.subTest(user=user, device_kind=device_kind), self.assertRaisesRegex(ValueError, "hybrid"):
        select(4224, user=user, device_kind=device_kind)
    # Invalid combinations surface from Krea2BlockSizes.
    with self.assertRaises(ValueError):
      select(4224, user={"kernel": "hybrid", "block_kv_compute_in": 256, "block_kv_pv": 512})
    with self.assertRaises(ValueError):
      select(4224, user={"kernel": "nope"})

  def test_hybrid_auto_block_q_matches_flash_on_v6e(self):
    # The hybrid VMEM cap does not bind on v6e with its default sizes: the
    # automatic block_q equals the flash one at every preset length.
    select = krea2_attention.select_krea2_block_sizes
    self.assertEqual(krea2_attention.max_hybrid_block_q(_V6E, 2048, 2048), 2048)
    for seq_len in _PRESET_SEQ_LENS:
      with self.subTest(seq_len=seq_len):
        hybrid = select(seq_len, device_kind=_V6E)
        flash = select(seq_len, user={"kernel": "flash"}, device_kind=_V6E)
        self.assertEqual(hybrid.variant, "hybrid")
        self.assertEqual(hybrid.block_q, flash.block_q)
        self.assertEqual(hybrid.block_q, _PRESET_DEFAULT_BLOCK_Q[seq_len])
    self.assertEqual(select(4224, device_kind=_V6E).block_q, 1408)
    self.assertEqual(select(16512, device_kind=_V6E).block_q, 1664)

  def test_forced_hybrid_on_v5e_is_capped(self):
    select = krea2_attention.select_krea2_block_sizes
    # Calibrated: v5e compiles block_q 1024 / fails 1152 with block_kv_compute 2048 (cap 896), 1536 / 1664 with 1024.
    self.assertEqual(krea2_attention.max_hybrid_block_q(_V5E, 2048, 2048), 896)
    self.assertEqual(krea2_attention.max_hybrid_block_q(_V5E, 2048, 1024), 1408)
    for seq_len in (4224, 16512):
      with self.subTest(seq_len=seq_len):
        sizes = select(seq_len, user={"kernel": "hybrid"}, device_kind=_V5E)
        self.assertLessEqual(sizes.block_q, 896)
        self.assertEqual(sizes.block_q, krea2_attention._default_block_q(seq_len, 896))  # pylint: disable=protected-access
        sizes = select(seq_len, user={"kernel": "hybrid", "block_kv_compute": 1024}, device_kind=_V5E)
        self.assertLessEqual(sizes.block_q, 1408)
    # A user block_q is not capped (the compile decides).
    self.assertEqual(select(4224, user={"kernel": "hybrid", "block_q": 1408}, device_kind=_V5E).block_q, 1408)
    # A user VMEM limit sets the budget (same safety factor): 32 MiB on v5e behaves like v6e.
    user = {"kernel": "hybrid", "vmem_limit_bytes": 32 * 1024 * 1024}
    self.assertEqual(select(16512, user=user, device_kind=_V5E).block_q, 1664)
    self.assertEqual(krea2_attention.max_hybrid_block_q(None, 2048, 2048, vmem_limit_bytes=16 * 1024 * 1024), 896)
    # Unknown chips without a limit are not capped.
    self.assertEqual(krea2_attention.max_hybrid_block_q(None, 2048, 2048), 2048)
    self.assertEqual(krea2_attention.max_hybrid_block_q("TPU v4", 8192, 8192), 2048)

  def test_hybrid_too_big_for_vmem_raises(self):
    select = krea2_attention.select_krea2_block_sizes
    user = {"kernel": "hybrid", "block_kv": 8192, "block_kv_compute": 8192}
    with self.assertRaisesRegex(ValueError, "block_kv_compute"):
      select(4224, user=user, device_kind=_V5E)
    with self.assertRaisesRegex(ValueError, "VMEM"):
      select(4224, user={"kernel": "hybrid", "vmem_limit_bytes": 8 * 1024 * 1024}, device_kind=_V6E)
    # With block_q pinned there is no automatic choice to fail.
    self.assertEqual(select(4224, user=dict(user, block_q=512), device_kind=_V5E).block_q, 512)

  def test_hybrid_vmem_calibration_points(self):
    estimate = krea2_attention._estimated_hybrid_vmem_bytes  # pylint: disable=protected-access
    budget = krea2_attention._hybrid_vmem_budget  # pylint: disable=protected-access
    mib = 1024 * 1024
    # (device_kind, vmem_limit_bytes, block_kv_compute, smallest failing block_q), block_kv 2048.
    fail_points = (
        (_V6E, None, 2048, 2304),
        (_V6E, None, 1024, 3456),
        (_V5E, None, 2048, 1152),
        (_V5E, None, 1024, 1664),
        (_V5E, 32 * mib, 2048, 2304),
    )
    for device_kind, limit, compute, block_q in fail_points:
      with self.subTest(device_kind=device_kind, limit=limit, compute=compute, block_q=block_q):
        self.assertGreater(estimate(block_q, 2048, compute), budget(device_kind, limit))
        self.assertLess(krea2_attention.max_hybrid_block_q(device_kind, 2048, compute, limit), block_q)
    # Compiled OK and admitted by the budget.
    ok_points = ((_V6E, None, 2048, 2048), (_V5E, None, 2048, 896), (_V5E, None, 1024, 1408), (_V6E, 64 * mib, 2048, 2048))
    for device_kind, limit, compute, block_q in ok_points:
      with self.subTest(ok=(device_kind, limit, compute, block_q)):
        self.assertLessEqual(estimate(block_q, 2048, compute), budget(device_kind, limit))
    # Wider operands need more VMEM.
    self.assertGreater(estimate(1024, 2048, 2048, itemsize=4), estimate(1024, 2048, 2048))
    self.assertLess(krea2_attention.max_hybrid_block_q(_V5E, 2048, 2048, dtype=jnp.float32), 896)


class Krea2PresetConfigTest(unittest.TestCase):

  _CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

  def test_configs_define_the_kernel_choice(self):
    for name in ("base_krea2.yml", "base_krea2_turbo.yml", "base_krea2_turbo_v6e1.yml"):
      with self.subTest(name=name):
        with open(os.path.join(self._CONFIG_DIR, name), encoding="utf-8") as f:
          config = yaml.safe_load(f)
        self.assertEqual(config["krea2_attention_kernel"], "auto")
        self.assertEqual(krea2_attention.parse_kernel_choice(config["krea2_attention_kernel"]), "auto")

  def test_v6e1_preset_takes_the_kernel_default_sizes(self):
    with open(os.path.join(self._CONFIG_DIR, "base_krea2_turbo_v6e1.yml"), encoding="utf-8") as f:
      config = yaml.safe_load(f)
    self.assertEqual(config["attention"], "flash_custom")
    self.assertEqual(config["flash_block_sizes"], {})
    carrier = max_utils.get_flash_block_sizes(types.SimpleNamespace(**config))
    self.assertIsNone(carrier)
    # What generate_krea2 hands the kernel on v6e: hybrid, or the tuned flash sizes when forced.
    hybrid = krea2_attention.select_krea2_block_sizes(
        4224, user=max_utils.CustomFlashBlockSizes(kernel="auto"), device_kind=_V6E
    )
    self.assertEqual(hybrid, _hybrid(1408, 2048, 2048, 1024, 256, 256))
    flash = krea2_attention.select_krea2_block_sizes(
        4224, user=max_utils.CustomFlashBlockSizes(kernel="flash"), device_kind=_V6E
    )
    self.assertEqual(flash, krea2_attention.Krea2BlockSizes(1408, 2048, 1024, 256))

  def _load_preset(self, *overrides):
    from maxdiffusion import pyconfig  # pylint: disable=import-outside-toplevel

    prev = (pyconfig._config, pyconfig.config)  # pylint: disable=protected-access
    self.addCleanup(lambda: setattr(pyconfig, "_config", prev[0]) or setattr(pyconfig, "config", prev[1]))
    with tempfile.TemporaryDirectory() as out:
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
    return pyconfig.config

  def test_preset_loads_through_pyconfig_with_overrides(self):
    # pylint: disable-next=import-outside-toplevel
    from maxdiffusion.generate_krea2 import krea2_flash_block_sizes, resolve_krea2_attention_kernel

    for overrides, expected in (
        ((), "auto"),
        (("krea2_attention_kernel=flash",), "flash"),
        (("krea2_attention_kernel=''",), "auto"),
        (("krea2_attention_kernel=Hybrid",), "hybrid"),
    ):
      with self.subTest(overrides=overrides):
        config = self._load_preset(*overrides)
        self.assertEqual(dict(config.flash_block_sizes), {})
        self.assertEqual(resolve_krea2_attention_kernel(config), expected)
        self.assertEqual(krea2_flash_block_sizes(config), max_utils.CustomFlashBlockSizes(kernel=expected))

  def test_invalid_kernel_fails_before_model_load(self):
    from maxdiffusion import generate_krea2, pyconfig  # pylint: disable=import-outside-toplevel

    prev = (pyconfig._config, pyconfig.config)  # pylint: disable=protected-access
    self.addCleanup(lambda: setattr(pyconfig, "_config", prev[0]) or setattr(pyconfig, "config", prev[1]))
    with tempfile.TemporaryDirectory() as out:
      argv = [
          None,
          os.path.join(self._CONFIG_DIR, "base_krea2_turbo_v6e1.yml"),
          f"output_dir={out}/",
          "skip_jax_distributed_system=True",
          "krea2_attention_kernel=fastest",
      ]
      with (
          mock.patch.object(generate_krea2, "build_krea2_transformer") as build,
          mock.patch.object(generate_krea2, "create_device_mesh") as mesh,
          self.assertRaisesRegex(ValueError, "krea2_attention_kernel.*fastest"),
      ):
        generate_krea2.main(argv)
      build.assert_not_called()
      mesh.assert_not_called()

  def test_get_flash_block_sizes_copies_hybrid_fields(self):
    config = types.SimpleNamespace(
        attention="flash_custom",
        flash_block_sizes={"block_kv": 2048, "block_kv_pv": 512, "block_q_strip": 128, "vmem_limit_bytes": 1 << 26},
    )
    carrier = max_utils.get_flash_block_sizes(config)
    self.assertEqual(
        carrier,
        max_utils.CustomFlashBlockSizes(block_kv=2048, block_kv_pv=512, block_q_strip=128, vmem_limit_bytes=1 << 26),
    )
    self.assertIsNone(carrier.kernel)
    self.assertIsInstance(hash(carrier), int)
    self.assertIsInstance(hash(dataclasses.replace(carrier, kernel="hybrid")), int)


class Krea2FlashCustomRegistryTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._prev_interpret = krea2_attention.INTERPRET
    krea2_attention.INTERPRET = True

  def tearDown(self):
    krea2_attention.INTERPRET = self._prev_interpret
    super().tearDown()

  def _apply(self, kernel, q, k, v, heads, mask, flash_block_sizes=None):
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with mesh, nn.partitioning.axis_rules(()):
      return attention_flax._apply_attention(
          query=q,
          key=k,
          value=v,
          heads=heads,
          dim_head=_D,
          split_head_dim=True,
          float32_qk_product=True,
          attention_kernel=kernel,
          flash_min_seq_length=0,
          use_memory_efficient_attention=False,
          scale=1.0 / math.sqrt(_D),
          dtype=jnp.float32,
          mesh=mesh,
          axis_names_q=(attention_flax.BATCH, attention_flax.HEAD, attention_flax.LENGTH, attention_flax.D_KV),
          axis_names_kv=(attention_flax.BATCH, attention_flax.HEAD, attention_flax.KV_LENGTH, attention_flax.D_KV),
          flash_block_sizes=flash_block_sizes,
          dpa_layer=None,
          attention_mask=mask,
      )

  def test_flash_custom_matches_dot_product(self):
    batch, hq, hkv, seq_len = 2, 4, 2, 320
    q, k, v = _random_qkv(batch, hq, hkv, seq_len, seed=5)
    valid = [200, 320]
    mask = (jnp.arange(seq_len)[None, :] < jnp.asarray(valid)[:, None]).astype(jnp.int32)
    block_sizes = max_utils.CustomFlashBlockSizes(
        block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128, heads_per_tile=None, vmem_limit_bytes=None
    )
    out = self._apply("flash_custom", q, k, v, hq, mask, block_sizes)
    self.assertEqual(out.shape, (batch, seq_len, hq * _D))

    # dot_product ignores attention_mask, so emulate the prefix mask by cropping
    # keys/values per batch element; it also needs repeated (B, L, H*D) k/v.
    group = hq // hkv

    def flat(t):
      return jnp.swapaxes(t, 1, 2).reshape(t.shape[0], t.shape[2], -1)

    for b in range(batch):
      n = valid[b]
      qb = flat(q[b : b + 1])
      kb = flat(jnp.repeat(k[b : b + 1, :, :n], group, axis=1))
      vb = flat(jnp.repeat(v[b : b + 1, :, :n], group, axis=1))
      ref = self._apply("dot_product", qb, kb, vb, hq, None)
      np.testing.assert_allclose(np.asarray(out[b : b + 1], np.float32), np.asarray(ref, np.float32), atol=2e-2, rtol=2e-2)

  def _check_against_dot_product(self, out, q, k, v, valid):
    group = q.shape[1] // k.shape[1]

    def flat(t):
      return jnp.swapaxes(t, 1, 2).reshape(t.shape[0], t.shape[2], -1)

    for b in range(q.shape[0]):
      n = valid[b]
      ref = self._apply(
          "dot_product",
          flat(q[b : b + 1]),
          flat(jnp.repeat(k[b : b + 1, :, :n], group, axis=1)),
          flat(jnp.repeat(v[b : b + 1, :, :n], group, axis=1)),
          q.shape[1],
          None,
      )
      np.testing.assert_allclose(np.asarray(out[b : b + 1], np.float32), np.asarray(ref, np.float32), atol=2e-2, rtol=2e-2)

  def test_flash_custom_hybrid_matches_dot_product(self):
    batch, hq, hkv, seq_len = 2, 4, 2, 320
    q, k, v = _random_qkv(batch, hq, hkv, seq_len, seed=12)
    valid = [200, 320]
    mask = (jnp.arange(seq_len)[None, :] < jnp.asarray(valid)[:, None]).astype(jnp.int32)
    carriers = (
        max_utils.CustomFlashBlockSizes(
            block_q=256,
            block_kv=256,
            block_kv_compute=256,
            block_kv_compute_in=256,
            block_kv_pv=128,
            block_q_strip=128,
            kernel="hybrid",
        ),
        {"kernel": " 'HYBRID' "},  # dict carrier, hybrid defaults
    )
    spy = mock.patch.object(krea2_attention, "_krea2_hybrid_forward", wraps=krea2_attention._krea2_hybrid_forward)
    for carrier in carriers:
      with self.subTest(carrier=carrier), spy as hybrid_forward:
        out = self._apply("flash_custom", q, k, v, hq, mask, carrier)
        self.assertEqual(hybrid_forward.call_count, 1)
        self.assertEqual(out.shape, (batch, seq_len, hq * _D))
        self._check_against_dot_product(out, q, k, v, valid)

  def test_flash_custom_auto_picks_hybrid_on_v6e_only(self):
    batch, hq, hkv, seq_len = 1, 2, 1, 256
    q, k, v = _random_qkv(batch, hq, hkv, seq_len, seed=13)
    mask = (jnp.arange(seq_len) < 190)[None, :]
    for device_kind, kernel, expected in (
        (_V6E, None, "hybrid"),
        (_V6E, "auto", "hybrid"),
        (_V6E, "", "hybrid"),
        (_V6E, "flash", "flash"),
        (_V5E, "auto", "flash"),
        ("cpu", None, "flash"),
        ("cpu", "hybrid", "hybrid"),
    ):
      with self.subTest(device_kind=device_kind, kernel=kernel):
        carrier = max_utils.CustomFlashBlockSizes(kernel=kernel)
        hybrid_spy = mock.patch.object(krea2_attention, "_krea2_hybrid_forward", wraps=krea2_attention._krea2_hybrid_forward)
        flash_spy = mock.patch.object(
            krea2_attention, "_krea2_attention_forward", wraps=krea2_attention._krea2_attention_forward
        )
        kind = mock.patch.object(attention_flax, "_mesh_device_kind", return_value=device_kind)
        with hybrid_spy as hybrid_forward, flash_spy as flash_forward, kind:
          out = self._apply("flash_custom", q, k, v, hq, mask, carrier)
        self.assertEqual((hybrid_forward.call_count, flash_forward.call_count), (expected == "hybrid", expected == "flash"))
        used = (hybrid_forward if expected == "hybrid" else flash_forward).call_args.args[4]
        self.assertEqual(used.variant, expected)
        if device_kind == _V6E and expected == "flash":
          self.assertEqual((used.block_kv, used.block_kv_compute, used.block_kv_compute_in), (2048, 1024, 256))
        self._check_against_dot_product(out, q, k, v, [190])

  def test_flash_custom_rejects_pv_with_flash(self):
    q, k, v = _random_qkv(1, 2, 1, 256, seed=14)
    carrier = max_utils.CustomFlashBlockSizes(kernel="flash", block_kv_pv=256)
    with self.assertRaisesRegex(ValueError, "hybrid"):
      self._apply("flash_custom", q, k, v, 2, None, carrier)

  def test_carrier_is_hashable_static_config(self):
    carrier = max_utils.CustomFlashBlockSizes(kernel="hybrid", block_kv_pv=256, block_q_strip=256)
    self.assertEqual(
        hash(carrier), hash(max_utils.CustomFlashBlockSizes(kernel="hybrid", block_kv_pv=256, block_q_strip=256))
    )
    self.assertNotEqual(carrier, dataclasses.replace(carrier, kernel="flash"))

    @jax.jit
    def run(x):
      return x + 1

    # A carrier as a jit static argument (it lives in the module's static config).
    f = jax.jit(lambda x, c: x * (2 if c.kernel == "hybrid" else 3), static_argnums=1)
    self.assertEqual(int(f(jnp.int32(1), carrier)), 2)
    self.assertEqual(int(run(jnp.int32(1))), 2)

  def test_read_custom_block_sizes(self):
    read = attention_flax._read_custom_block_sizes  # pylint: disable=protected-access
    self.assertEqual(read(None)["kernel"], None)
    self.assertEqual(read({"kernel": ""})["kernel"], "")  # '' means auto downstream, not dropped as an int would be
    sizes = read(max_utils.CustomFlashBlockSizes(kernel="hybrid", block_kv_pv=256, block_q_strip=128))
    self.assertEqual((sizes["kernel"], sizes["block_kv_pv"], sizes["block_q_strip"]), ("hybrid", 256, 128))
    sizes = read({"block_kv_pv": 0, "block_q_strip": None, "block_q": 512})
    self.assertEqual((sizes["block_kv_pv"], sizes["block_q_strip"], sizes["block_q"]), (None, None, 512))

  def test_flash_custom_flat_inputs_and_broadcast_mask(self):
    # 3-D (B, L, H*D) GQA inputs, a (1, L) mask broadcast over batch, default blocks.
    batch, hq, hkv, seq_len = 2, 2, 1, 256
    q, k, v = _random_qkv(batch, hq, hkv, seq_len, seed=6)

    def flat(t):
      return jnp.swapaxes(t, 1, 2).reshape(t.shape[0], t.shape[2], -1)

    mask = (jnp.arange(seq_len) < 190)[None, :]
    # _apply_attention's input check requires equal q/k depths, so 3-D GQA
    # inputs can only reach the kernel through the registry entry directly.
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    context = {
        "heads": hq,
        "dim_head": _D,
        "scale": 1.0 / math.sqrt(_D),
        "mesh": mesh,
        "attention_mask": mask,
        "flash_block_sizes": None,
        "axis_names_q": (attention_flax.BATCH, attention_flax.HEAD, attention_flax.LENGTH, attention_flax.D_KV),
        "axis_names_kv": (attention_flax.BATCH, attention_flax.HEAD, attention_flax.KV_LENGTH, attention_flax.D_KV),
    }
    with mesh, nn.partitioning.axis_rules(()):
      out = attention_flax.KERNEL_REGISTRY["flash_custom"](flat(q), flat(k), flat(v), context)
    ref = _reference(q, k, v, [190, 190], 1.0 / math.sqrt(_D))
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(flat(ref), np.float32), atol=2e-2, rtol=2e-2)


class Krea2RejectShardedSequenceTest(unittest.TestCase):

  def test_size_one_sequence_axis_is_allowed(self):
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    spec = P("data", None, "data", None)
    attention_flax._krea2_reject_sharded_sequence(mesh, spec, spec)
    attention_flax._krea2_reject_sharded_sequence(mesh, P("data", None, None, None), P(None, None, ("data",), None))

  def test_sharded_sequence_axis_raises(self):
    mesh = types.SimpleNamespace(shape={"data": 1, "context": 2})
    unsharded = P("data", None, None, None)
    with self.assertRaises(NotImplementedError):
      attention_flax._krea2_reject_sharded_sequence(mesh, P("data", None, "context", None), unsharded)
    with self.assertRaises(NotImplementedError):
      attention_flax._krea2_reject_sharded_sequence(mesh, unsharded, P("data", None, "context", None))
    with self.assertRaises(NotImplementedError):
      attention_flax._krea2_reject_sharded_sequence(mesh, P(None, None, ("data", "context"), None), unsharded)
    attention_flax._krea2_reject_sharded_sequence(mesh, P(None, "context", "data", None), unsharded)


def _abstract_mesh(test, device_kind):
  """A 1-device AbstractMesh whose abstract device is `device_kind` (a TPU chip)."""
  try:
    from jax._src.mesh import AbstractDevice  # pylint: disable=import-outside-toplevel
  except ImportError:
    test.skipTest("this jax has no AbstractDevice")
  device = AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.AbstractMesh((1,), ("data",), abstract_device=device)


class Krea2MeshDeviceKindTest(unittest.TestCase):

  def test_real_mesh(self):
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    self.assertEqual(attention_flax._mesh_device_kind(mesh), "cpu")
    fake = types.SimpleNamespace(devices=np.array([types.SimpleNamespace(device_kind=_V6E)], dtype=object))
    self.assertEqual(attention_flax._mesh_device_kind(fake), _V6E)

  def test_abstract_mesh_reports_its_abstract_device(self):
    real_mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    self.assertEqual(attention_flax._mesh_device_kind(real_mesh.abstract_mesh), "cpu")
    for kind in (_V6E, _V5E):
      with self.subTest(kind=kind):
        self.assertEqual(attention_flax._mesh_device_kind(_abstract_mesh(self, kind)), kind)

  def _trace_flash_custom(self, mesh, kernel, seq_len=4224):
    """Traces the flash_custom wrapper under `mesh` (no compile) and returns the block sizes it built the kernel with."""
    context = {
        "heads": 4,
        "dim_head": _D,
        "scale": 1.0 / math.sqrt(_D),
        "mesh": mesh,
        "attention_mask": None,
        "flash_block_sizes": max_utils.CustomFlashBlockSizes(kernel=kernel),
        "axis_names_q": (attention_flax.BATCH, attention_flax.HEAD, attention_flax.LENGTH, attention_flax.D_KV),
        "axis_names_kv": (attention_flax.BATCH, attention_flax.HEAD, attention_flax.KV_LENGTH, attention_flax.D_KV),
    }
    make = mock.patch.object(krea2_attention, "make_krea2_attention", wraps=krea2_attention.make_krea2_attention)
    q = jax.ShapeDtypeStruct((1, seq_len, 4 * _D), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((1, seq_len, 2 * _D), jnp.bfloat16)
    with make as made, nn.partitioning.axis_rules(()), jax.sharding.use_abstract_mesh(mesh):
      out = jax.eval_shape(lambda q, k, v: attention_flax.KERNEL_REGISTRY["flash_custom"](q, k, v, context), q, kv, kv)
    self.assertEqual(out.shape, (1, seq_len, 4 * _D))
    return made.call_args.args[0]

  def test_abstract_mesh_drives_the_kernel_choice(self):
    # An AbstractMesh (no concrete devices) keeps the chip through its abstract device.
    sizes = self._trace_flash_custom(_abstract_mesh(self, _V6E), "auto")
    self.assertEqual(sizes, _hybrid(1408, 2048, 2048, 1024, 256, 256))
    sizes = self._trace_flash_custom(_abstract_mesh(self, _V5E), "hybrid")
    self.assertEqual(sizes, _hybrid(896, 2048, 2048, 1024, 256, 256))
    sizes = self._trace_flash_custom(_abstract_mesh(self, _V5E), "auto")
    self.assertEqual(sizes, krea2_attention.Krea2BlockSizes(1408, 1024, 512, 256))

  def test_undeterminable_device_kind_is_none(self):
    meshes = {
        "none": None,
        "abstract_mesh_without_device": jax.sharding.AbstractMesh((1,), ("data",)),
        "no_devices": types.SimpleNamespace(shape={"data": 1}),
        "empty_devices": types.SimpleNamespace(devices=np.empty((0,), dtype=object)),
        "no_device_kind": types.SimpleNamespace(devices=np.array([types.SimpleNamespace(id=0)], dtype=object)),
    }
    for name, mesh in meshes.items():
      with self.subTest(mesh=name):
        self.assertIsNone(attention_flax._mesh_device_kind(mesh))


class Krea2OperandDtypeTest(unittest.TestCase):

  def test_one_other_operand_keeps_the_previous_block_q(self):
    bf16, f32 = jnp.zeros((1,), jnp.bfloat16), jnp.zeros((1,), jnp.float32)
    f16, i32, b = jnp.zeros((1,), jnp.float16), jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.bool_)
    user = {"block_kv": 2048, "block_kv_compute": 1024, "block_kv_compute_in": 256, "kernel": "flash"}
    # Checked with the budget extension on, where the operand dtype matters.
    cases = [((bf16, bf16, bf16), 4224), ((f32, f32, f32), 1408)]
    # Integer and bool operands promote to bfloat16 with bfloat16: they must not pass either.
    for other in (f32, f16, i32, b):
      cases += [((other, bf16, bf16), 1408), ((bf16, other, bf16), 1408), ((bf16, bf16, other), 1408)]
    for operands, expected in cases:
      with self.subTest(dtypes=[str(x.dtype) for x in operands]):
        dtype = attention_flax._krea2_operand_dtype(*operands)
        with _budget_extension(True):
          sizes = krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E, dtype=dtype)
        self.assertEqual(sizes.block_q, expected)
        # The default (extension off) ignores the dtype.
        sizes = krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E, dtype=dtype)
        self.assertEqual(sizes.block_q, 1408)


if __name__ == "__main__":
  unittest.main()
