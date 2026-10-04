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

"""CPU (Pallas interpret mode) tests for the fused Krea 2 q/k post-processing (kernels/krea2_qk_prep.py)."""

import math
import unittest
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion import max_utils
from maxdiffusion.kernels import krea2_attention
from maxdiffusion.kernels import krea2_qk_prep
from maxdiffusion.models import attention_flax
from maxdiffusion.models.krea2.transformer_krea2_flax import (
    Krea2Attention,
    Krea2RMSNorm,
    apply_rope_rotate_half,
    krea2_rotary_tables,
)
from maxdiffusion.models.krea2.util import prepare_krea2_image_ids

_LOG2E = math.log2(math.e)
_EPS = 1e-5


def _random(shape, seed, scale=1.0):
  return jnp.asarray(scale * np.random.RandomState(seed).randn(*shape), jnp.float32)


def _tables(seq_len, head_dim, seed=1):
  # Random angles exercise every lane (the real tables are cos/sin of position ids).
  angles = _random((seq_len, head_dim // 2), seed, scale=3.0)
  return jnp.cos(angles), jnp.sin(angles)


def _old_path(x, weight, cos, sin):
  """The unfused reference: Krea2RMSNorm (cast back to x.dtype) -> (B, H, L, D) -> apply_rope_rotate_half."""
  normed = Krea2RMSNorm(x.shape[-1], eps=_EPS).apply({"params": {"weight": weight}}, x)
  return apply_rope_rotate_half(jnp.transpose(normed, (0, 2, 1, 3)), cos, sin)


def _kernel(x, weight, cos, sin, **kwargs):
  """The prep kernel in interpret mode on a (B, L, H, D) projection."""
  cos2, sin2 = krea2_qk_prep.rotate_half_full_tables(cos, sin)
  kwargs.setdefault("padded_len", x.shape[1])
  return krea2_qk_prep.krea2_qk_prep(
      jnp.transpose(x, (0, 2, 1, 3)), weight, cos2, sin2, eps=_EPS, interpret=True, **kwargs
  )


class FullTablesTest(unittest.TestCase):

  def test_full_tables(self):
    cos, sin = _tables(5, 8)
    cos2, sin2 = krea2_qk_prep.rotate_half_full_tables(cos, sin)
    np.testing.assert_array_equal(np.asarray(cos2), np.concatenate([cos, cos], -1))
    np.testing.assert_array_equal(np.asarray(sin2), np.concatenate([-sin, sin], -1))
    self.assertEqual(cos2.dtype, jnp.float32)

  def test_block_choices(self):
    self.assertEqual(krea2_qk_prep.pick_block_rows(4608), 768)
    self.assertEqual(krea2_qk_prep.pick_block_rows(6144), 1024)
    self.assertEqual(krea2_qk_prep.pick_block_rows(48), 48)
    self.assertEqual(krea2_qk_prep.pick_block_rows(40), 40)  # no multiple of 16 divides it
    # No multiple of 16 divides 2040 = 8 * 3 * 5 * 17; the largest multiple of 8 <= 1024 dividing it is 680.
    self.assertEqual(krea2_qk_prep.pick_block_rows(2040), 680)
    # Small lengths with no multiple-of-8 divisor at all still run as one block.
    self.assertEqual(krea2_qk_prep.pick_block_rows(36), 36)
    self.assertEqual(krea2_qk_prep.pick_block_rows(12, max_rows=16), 12)
    # Large lengths stay under max_rows: 32792 = 8 * 4099 (4099 prime) -> 8-row blocks, not one 32792-row block.
    self.assertEqual(krea2_qk_prep.pick_block_rows(32792, max_rows=1024), 8)
    # 8198 = 2 * 4099 has no multiple-of-8 divisor and exceeds max_rows: refuse instead of one huge block.
    with self.assertRaisesRegex(ValueError, r"padded_len=8198.*block_rows"):
      krea2_qk_prep.pick_block_rows(8198, max_rows=1024)
    self.assertEqual(krea2_qk_prep.pick_heads_per_block(48), 4)
    self.assertEqual(krea2_qk_prep.pick_heads_per_block(6), 3)
    self.assertEqual(krea2_qk_prep.pick_heads_per_block(1), 1)


class QkPrepReferenceTest(unittest.TestCase):

  def test_reference_is_bit_exact_to_the_old_path_in_float32(self):
    # Without the intermediate bf16 cast (float32 input) the fused arithmetic is the old one exactly.
    x = _random((2, 24, 3, 128), 0)
    weight = _random((128,), 2, scale=0.3)
    cos, sin = _tables(24, 128)
    expected = _old_path(x, weight, cos, sin)
    actual = krea2_qk_prep.qk_prep_reference(x, weight, _EPS, cos, sin)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

  def test_reference_scale_and_padding(self):
    x = _random((1, 10, 2, 8), 3)
    weight = _random((8,), 4, scale=0.3)
    cos, sin = _tables(10, 8)
    plain = krea2_qk_prep.qk_prep_reference(x, weight, _EPS, cos, sin)
    scaled = krea2_qk_prep.qk_prep_reference(x, weight, _EPS, cos, sin, scale=0.25, padded_len=16)
    self.assertEqual(scaled.shape, (1, 2, 16, 8))
    np.testing.assert_allclose(np.asarray(scaled[:, :, :10]), 0.25 * np.asarray(plain), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(scaled[:, :, 10:]), 0.0)


class QkPrepKernelTest(unittest.TestCase):

  def test_kernel_matches_reference_float32(self):
    # (seq, padded_len, block_rows, heads_per_block): exact fit, a partial last input block,
    # output blocks past the end of the input (clamped), several heads per step.
    cases = ((32, 32, 16, None), (37, 48, 16, None), (20, 64, 16, 1), (40, 40, None, 2))
    weight = _random((128,), 2, scale=0.3)
    for i, (seq_len, padded_len, block_rows, heads_per_block) in enumerate(cases):
      with self.subTest(seq_len=seq_len, padded_len=padded_len, block_rows=block_rows):
        x = _random((2, seq_len, 4, 128), 10 + i)
        cos, sin = _tables(seq_len, 128, seed=20 + i)
        expected = krea2_qk_prep.qk_prep_reference(x, weight, _EPS, cos, sin, scale=0.3, padded_len=padded_len)
        actual = _kernel(
            x, weight, cos, sin, scale=0.3, padded_len=padded_len, block_rows=block_rows, heads_per_block=heads_per_block
        )
        self.assertEqual(actual.shape, (2, 4, padded_len, 128))
        # Same float32 operations; the interpreted lane reduction may round in another order (~1 ulp).
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(actual[:, :, seq_len:]), 0.0)

  def test_bf16_kernel_matches_the_old_path_within_bf16_rounding(self):
    x = _random((1, 40, 4, 128), 5).astype(jnp.bfloat16)
    weight = _random((128,), 6, scale=0.3)
    cos, sin = _tables(40, 128)
    old = np.asarray(_old_path(x, weight, cos, sin), np.float32)
    exact = np.asarray(_old_path(x.astype(jnp.float32), weight, cos, sin))  # no bf16 rounding at all
    new = np.asarray(_kernel(x, weight, cos, sin, block_rows=8), np.float32)
    # The kernel rounds once (at the end), the old path twice (after the norm and after RoPE).
    bound = 2.0**-8 * np.abs(exact) + 1e-6
    self.assertTrue(np.all(np.abs(new - exact) <= bound))
    tolerance = 3 * 2.0**-8 * np.max(np.abs(exact))
    self.assertLessEqual(float(np.max(np.abs(new - old))), tolerance)
    self.assertGreater(float(np.max(np.abs(new - old))), 0.0)  # the dropped intermediate cast is visible

  def test_rejects_bad_shapes(self):
    x = jnp.zeros((1, 2, 16, 8))
    cos2 = sin2 = jnp.zeros((16, 8))
    with self.assertRaises(ValueError):
      krea2_qk_prep.krea2_qk_prep(x, jnp.zeros(8), cos2, sin2, eps=_EPS, padded_len=8, interpret=True)
    with self.assertRaises(ValueError):
      krea2_qk_prep.krea2_qk_prep(x, jnp.zeros(8), cos2, sin2, eps=_EPS, padded_len=24, block_rows=16, interpret=True)
    with self.assertRaises(ValueError):
      krea2_qk_prep.krea2_qk_prep(x, jnp.zeros(8), cos2[:8], sin2[:8], eps=_EPS, padded_len=16, interpret=True)


class FlashCustomQkPrepTest(unittest.TestCase):
  """Krea2Attention -> flash_custom with rotate_half RoPE: the raw q/k go through the prep kernel."""

  def setUp(self):
    super().setUp()
    self._prev_interpret = krea2_attention.INTERPRET
    krea2_attention.INTERPRET = True

  def tearDown(self):
    krea2_attention.INTERPRET = self._prev_interpret
    super().tearDown()

  def test_matches_masked_dot_product(self):
    seq_len, dim, heads, kv_heads = 200, 256, 2, 1
    x = _random((2, seq_len, dim), 7)
    ids = prepare_krea2_image_ids(1, 10, 20)[0]
    rotary = krea2_rotary_tables(ids, (32, 48, 48), 1000.0, "rotate_half")
    mask = jnp.arange(seq_len)[None, :] < jnp.asarray([150, 200])[:, None]
    common = dict(dim=dim, num_heads=heads, num_kv_heads=kv_heads, head_dim=128, rope_layout="rotate_half")
    reference = Krea2Attention(**common)
    params = jax.tree_util.tree_map(
        lambda p: p.unbox() if isinstance(p, nn.LogicallyPartitioned) else p,
        reference.init(jax.random.PRNGKey(0), x, mask, rotary)["params"],
        is_leaf=lambda p: isinstance(p, nn.LogicallyPartitioned),
    )
    params = jax.tree_util.tree_map(lambda p: p + 0.1 * jnp.ones_like(p), params)  # non-zero norm weights
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    custom = Krea2Attention(
        **common,
        attention_kernel="flash_custom",
        flash_min_seq_length=0,
        mesh=mesh,
        flash_block_sizes=max_utils.CustomFlashBlockSizes(
            block_q=128, block_kv=128, block_kv_compute=128, block_kv_compute_in=128, heads_per_tile=None
        ),
    )
    expected = reference.apply({"params": params}, x, mask, rotary)
    seen = []
    original = krea2_qk_prep.krea2_qk_prep

    def spy(x_local, *args, **kwargs):
      seen.append((x_local.shape, kwargs["padded_len"], kwargs["scale"]))
      return original(x_local, *args, **kwargs)

    with mesh, nn.partitioning.axis_rules(()), mock.patch.object(krea2_qk_prep, "krea2_qk_prep", spy):
      actual = custom.apply({"params": params}, x, mask, rotary)
    # q padded to block_q with the scale folded in, k padded to block_kv unscaled.
    self.assertEqual(
        seen,
        [((2, heads, seq_len, 128), 256, 1.0 / math.sqrt(128) * _LOG2E), ((2, kv_heads, seq_len, 128), 256, None)],
    )
    # bf16-free float32 inputs, but the kernel's P.V runs in bf16.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-2, atol=2e-2)

  def test_extra_context_never_reaches_dot_product(self):
    q = jnp.zeros((1, 2, 8, 128))
    kv = jnp.zeros((1, 1, 8, 128))
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with self.assertRaisesRegex(ValueError, "falls back to dot_product"):
      attention_flax._apply_attention(  # pylint: disable=protected-access
          query=q,
          key=kv,
          value=kv,
          heads=2,
          dim_head=128,
          split_head_dim=True,
          float32_qk_product=True,
          attention_kernel="flash_custom",
          flash_min_seq_length=512,  # 8 < 512: the dispatcher falls back to dot_product
          use_memory_efficient_attention=False,
          scale=1.0,
          dtype=jnp.float32,
          mesh=mesh,
          axis_names_q=(attention_flax.BATCH, attention_flax.HEAD, attention_flax.LENGTH, attention_flax.D_KV),
          axis_names_kv=(attention_flax.BATCH, attention_flax.HEAD, attention_flax.KV_LENGTH, attention_flax.D_KV),
          flash_block_sizes=None,
          dpa_layer=None,
          extra_context={"krea2_qk_prep": {}},
      )

  def test_attention_ops_forward_extra_context(self):
    q = jnp.zeros((1, 2, 8, 128))
    kv = jnp.zeros((1, 1, 8, 128))
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    common = dict(mesh=mesh, attention_kernel="flash_custom", scale=1.0, heads=2, dim_head=128)
    ops = {
        "AttentionOp": attention_flax.AttentionOp(**common).bind({}),
        "NNXAttentionOp": attention_flax.NNXAttentionOp(**common),
    }
    for name, op in ops.items():
      with self.subTest(op=name):
        extra_context = {"krea2_qk_prep": {}}
        sentinel = object()
        with mock.patch.object(attention_flax, "_apply_attention", return_value=sentinel) as dispatcher:
          self.assertIs(op.apply_attention(q, kv, kv, extra_context=extra_context), sentinel)
        dispatcher.assert_called_once()
        self.assertIs(dispatcher.call_args.kwargs["extra_context"], extra_context)


if __name__ == "__main__":
  unittest.main()
