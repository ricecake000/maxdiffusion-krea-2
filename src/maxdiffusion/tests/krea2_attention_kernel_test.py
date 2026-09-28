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

import math
import types
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
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
    (_V6E, None, 7808, {4224: 4224, 5000: 5120, 16512: 5504, 16640: 3328, 16896: 5632}),
    (_V6E, (4096, 1024, 256), 4480, {4480: 4480, 4608: 2304, 16512: 3328}),
    (_V6E, (2048, 2048, 256), 3072, {4224: 1408, 4352: 2176, 16512: 1664, 16896: 2816}),
    (_V6E, (2048, 1024, 128), 2048, {4224: 1408, 4352: 896, 16512: 1664}),
    (_V6E, (1024, 256, 256), 8192, {16512: 5504}),
    (_V5E, (2048, 1024, 256), 2304, {4224: 1408, 4352: 2176, 4608: 2304, 16512: 1664}),
    (_V5E, None, 3840, {3728: 3840, 4224: 1408, 16512: 3328}),
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
  user = dict(extra)
  if kv_sizes is not None:
    user.update(zip(("block_kv", "block_kv_compute", "block_kv_compute_in"), kv_sizes))
  return user or None


class Krea2AutoBlockQTest(unittest.TestCase):

  def test_max_auto_block_q(self):
    for device_kind, kv_sizes, expected_max, _ in _AUTO_BLOCK_Q_CASES:
      with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes):
        sizes = krea2_attention.select_krea2_block_sizes(4224, user=_kv_user(kv_sizes))
        kv = (sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in)
        self.assertEqual(krea2_attention.max_auto_block_q(device_kind, *kv), expected_max)
    # Uncalibrated chips and a user VMEM limit keep the pre-budget maximum.
    for device_kind in (None, "cpu", "TPU v4", "TPU v5p"):
      self.assertEqual(krea2_attention.max_auto_block_q(device_kind, 1024, 512, 256), 2048)
    self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, vmem_limit_bytes=67108864), 2048)
    # Never below the base, even when the base itself exceeds the estimate budget.
    self.assertEqual(krea2_attention.max_auto_block_q(_V5E, 4096, 2048, 256), 2048)

  def test_auto_block_q_per_device(self):
    for device_kind, kv_sizes, _, expected in _AUTO_BLOCK_Q_CASES:
      for seq_len, block_q in expected.items():
        with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes, seq_len=seq_len):
          sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
          self.assertEqual(sizes.block_q, block_q)
          if kv_sizes is not None:
            self.assertEqual((sizes.block_kv, sizes.block_kv_compute, sizes.block_kv_compute_in), kv_sizes)

  def test_uncalibrated_device_keeps_previous_choice(self):
    for device_kind in (None, "cpu"):
      for kv_sizes in (None, (2048, 1024, 256), (1024, 256, 256)):
        for seq_len, block_q in {4224: 1408, 4352: 896, 16512: 1664}.items():
          with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes, seq_len=seq_len):
            sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
            self.assertEqual(sizes.block_q, block_q)

  def test_user_vmem_limit_and_block_q(self):
    user = _kv_user((2048, 1024, 256), vmem_limit_bytes=67108864)
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E).block_q, 1408)
    carrier = max_utils.CustomFlashBlockSizes(
        block_q=None,
        block_kv=2048,
        block_kv_compute=1024,
        block_kv_compute_in=256,
        heads_per_tile=None,
        vmem_limit_bytes=67108864,
    )
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224, user=carrier, device_kind=_V6E).block_q, 1408)
    # A user block_q always wins.
    user = _kv_user((2048, 1024, 256), block_q=1408)
    self.assertEqual(krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E).block_q, 1408)

  def test_budget_only_for_bfloat16_operands(self):
    user = _kv_user((2048, 1024, 256))
    # The VMEM budget is calibrated with bf16 q/k/v; wider operands keep the previous choice.
    for dtype in (jnp.float32, jnp.float16, np.float32, "float32"):
      with self.subTest(dtype=dtype):
        self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, dtype=dtype), 2048)
        for seq_len, block_q in {4224: 1408, 16512: 1664}.items():
          sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=user, device_kind=_V6E, dtype=dtype)
          self.assertEqual(sizes.block_q, block_q)
    # None means bf16; bf16 is accepted as a jnp scalar type, a numpy dtype object and a string.
    for dtype in (None, jnp.bfloat16, np.dtype(jnp.bfloat16), "bfloat16", jnp.zeros((1,), jnp.bfloat16).dtype):
      with self.subTest(dtype=dtype):
        self.assertEqual(krea2_attention.max_auto_block_q(_V6E, 2048, 1024, 256, dtype=dtype), 4992)
        sizes = krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E, dtype=dtype)
        self.assertEqual(sizes.block_q, 4224)

  def test_vmem_budget_covers_choices_and_excludes_failures(self):
    budgets = krea2_attention._VMEM_BUDGET_BYTES  # pylint: disable=protected-access
    estimate = krea2_attention._estimated_vmem_bytes  # pylint: disable=protected-access
    for device_kind, kv_sizes, _, expected in _AUTO_BLOCK_Q_CASES:
      for seq_len in expected:
        with self.subTest(device_kind=device_kind, kv_sizes=kv_sizes, seq_len=seq_len):
          sizes = krea2_attention.select_krea2_block_sizes(seq_len, user=_kv_user(kv_sizes), device_kind=device_kind)
          self.assertLessEqual(estimate(sizes.block_q, sizes.block_kv, sizes.block_kv_compute), budgets[device_kind])
    # Nobody may raise a budget past a block_q that is known to fail the compile.
    for device_kind, block_q, block_kv, block_kv_compute, block_kv_compute_in in _VMEM_FAIL_POINTS:
      with self.subTest(device_kind=device_kind, block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv_compute):
        self.assertGreater(estimate(block_q, block_kv, block_kv_compute), budgets[device_kind])
        max_block_q = krea2_attention.max_auto_block_q(device_kind, block_kv, block_kv_compute, block_kv_compute_in)
        self.assertLess(max_block_q, block_q)


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


class Krea2MeshDeviceKindTest(unittest.TestCase):

  def test_real_mesh(self):
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    self.assertEqual(attention_flax._mesh_device_kind(mesh), "cpu")
    fake = types.SimpleNamespace(devices=np.array([types.SimpleNamespace(device_kind=_V6E)], dtype=object))
    self.assertEqual(attention_flax._mesh_device_kind(fake), _V6E)

  def test_undeterminable_device_kind_is_none(self):
    real_mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    meshes = {
        "none": None,
        "abstract_mesh": real_mesh.abstract_mesh,
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
    user = {"block_kv": 2048, "block_kv_compute": 1024, "block_kv_compute_in": 256}
    cases = [((bf16, bf16, bf16), 4224), ((f32, f32, f32), 1408)]
    # Integer and bool operands promote to bfloat16 with bfloat16: they must not pass either.
    for other in (f32, f16, i32, b):
      cases += [((other, bf16, bf16), 1408), ((bf16, other, bf16), 1408), ((bf16, bf16, other), 1408)]
    for operands, expected in cases:
      with self.subTest(dtypes=[str(x.dtype) for x in operands]):
        dtype = attention_flax._krea2_operand_dtype(*operands)
        sizes = krea2_attention.select_krea2_block_sizes(4224, user=user, device_kind=_V6E, dtype=dtype)
        self.assertEqual(sizes.block_q, expected)


if __name__ == "__main__":
  unittest.main()
