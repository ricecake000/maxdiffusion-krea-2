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

# MXFP6 matmuls for the Krea 2 DiT block projections on AMD gfx950 GPUs (MI350X; transformer quantization 'mxfp6').
#
# Both operands of `x @ w.T` are MXFP6: 6-bit E2M3 values with one e8m0 (power of two) scale per 32 values along
# the contraction K, after an orthonormal 32-point Hadamard rotation of every 32-group (applied to both operands,
# so it cancels in the product). The product runs in AMD's hand-written f6gemm kernel with float32 accumulation
# and returns bfloat16.
#
#   weights      quantized and packed once on the host by `convert_kernel` (pure JAX: any backend) and stored
#                packed in the weight cache.
#   activations  quantized and packed inside the jitted step: AMD's HIP quantizer on the GPU (`impl` 'kernel'),
#                the pure-JAX one elsewhere.
#   impl         'kernel' = the two XLA FFI targets of krea2_mxfp6_shims/ (GemmA6W6JA, QuantMxfp6JA), registered
#                for the ROCm platform only. 'ref' = the same packed operands unpacked, dequantized and multiplied
#                by XLA in float32: for CPU checks of the model and the weight path, never for serving.
#                `default_impl()` picks by backend; nothing reads the environment for it.
#
# Packed layout (aiter's "mxfp6_c0c1_256_padk2"): rows in tiles of 256, K in steps of 128 plus two guard steps
# per row tile; 24576 bytes of codes and 1024 bytes of scales per (row tile, K step). So K must be a multiple of
# 128, rows are padded to 256 (and the product sliced), and no buffer may exceed 2 GiB.
#
# The quantizer, the packer and their inverses below (`quant_fp6` ... `dequant_fp6`) are a JAX port of AITER's
# aiter/ops/gemm_op_a6w6.py (quant_mxfp6, pack_big, pack_scale), which is
#   SPDX-License-Identifier: MIT
#   Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# (full text: krea2_mxfp6_shims/LICENSE.aiter).
#
# The kernel implementation needs a JAX-AITER tree (https://github.com/ROCm/jax-aiter) on PYTHONPATH with
# JA_ROOT_DIR and GPU_ARCHS=gfx950 set, and the two shims built by krea2_mxfp6_shims/build.sh into
# `$KREA2_MXFP6_LIB_DIR` (default `$JA_ROOT_DIR/build/krea2_mxfp6`). Where the libraries are is deployment, not
# model configuration: it changes no result.

import ctypes
import math
import os
import threading

import jax
import jax.numpy as jnp
import numpy as np

IMPLS = ("kernel", "ref")
GEMM_TARGET = "GemmA6W6JA"
QUANT_TARGET = "QuantMxfp6JA"
_SHIMS = (("gemm_a6w6_ja.so", GEMM_TARGET), ("quant_mxfp6_ja.so", QUANT_TARGET))

_TILE, _K_TILE, _GROUP, _PAD_K = 256, 128, 32, 2
_PACKED_TILE_BYTES, _SCALE_TILE_BYTES = 24576, 1024
_MAX_BUFFER_BYTES = 1 << 31

_register_lock = threading.Lock()
_registered = False


def _ceil(x, m):
  return (x + m - 1) // m * m


def default_impl() -> str:
  """'kernel' on an AMD GPU (the ROCm backend), 'ref' on CPU; raises ValueError on any other backend.

  Read at trace time, like the cuDNN choice of attention 'cudnn'. A backend that is neither must not fall back to
  the reference silently: it is many times slower than any other transformer mode.
  """
  backend = jax.default_backend()
  if backend == "cpu":
    return "ref"
  if backend == "gpu" and str(jax.devices()[0]).startswith("rocm"):
    return "kernel"
  raise ValueError(
      f"Transformer quantization 'mxfp6' runs on AMD gfx950 GPUs (JAX's ROCm backend), or on CPU as a slow "
      f"reference; this process uses backend {backend!r} ({jax.devices()[0]})."
  )


def lib_dir() -> str:
  """The directory of the two shim libraries: `$KREA2_MXFP6_LIB_DIR`, else `$JA_ROOT_DIR/build/krea2_mxfp6`."""
  explicit = os.environ.get("KREA2_MXFP6_LIB_DIR", "").strip()
  if explicit:
    return explicit
  root = os.environ.get("JA_ROOT_DIR", "").strip()
  if not root:
    raise RuntimeError(
        "MXFP6 kernels: set KREA2_MXFP6_LIB_DIR to the directory of gemm_a6w6_ja.so / quant_mxfp6_ja.so, or "
        "JA_ROOT_DIR to the JAX-AITER tree they were built into (kernels/krea2_mxfp6_shims/build.sh)."
    )
  return os.path.join(root, "build", "krea2_mxfp6")


def register() -> None:
  """Loads the two shims and registers their XLA FFI targets for the ROCm platform. Idempotent, thread-safe.

  Nothing below calls this: tracing `quantize_activation` / `linear` needs no library (so shape evaluation, e.g.
  a weight cache build, works without them), compiling or running the traced program does. The runtime calls it
  before it compiles anything or loads an executable of the AOT cache; another caller of the kernel
  implementation must do the same, else XLA reports "No FFI handler registered". Raises when the JAX-AITER tree
  or a shim is missing, instead of at the first image.
  """
  global _registered
  with _register_lock:
    if _registered:
      return
    import jax_aiter  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import,import-error  (sets AITER_ASM_DIR)
    from jax_aiter.ffi import registry  # pylint: disable=import-outside-toplevel,import-error

    # JAX's ROCm plugin loads libamdhip64 first (the shims have no RUNPATH); the umbrella library holds the
    # AITER symbols they leave undefined (the code-object loader).
    registry.ensure_jax_backend_ready()
    registry.load_umbrella_library()
    directory = lib_dir()
    for lib_name, target in _SHIMS:
      path = os.path.join(directory, lib_name)
      if not os.path.exists(path):
        raise RuntimeError(f"MXFP6 kernels: {path} is missing (build it with kernels/krea2_mxfp6_shims/build.sh).")
      lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
      pointer = ctypes.cast(getattr(lib, target), ctypes.c_void_p).value
      if not pointer:
        raise RuntimeError(f"MXFP6 kernels: {target} not found in {path}.")
      jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(pointer), platform="ROCM")
    _registered = True


def _e2m3_table():
  """The 64 E2M3 values by code: bit 5 sign, bits 4..3 exponent, bits 2..0 mantissa (gemm_op_a6w6.py _e2m3_table)."""
  values = np.empty(64, np.float32)
  for code in range(64):
    sign, exponent, mantissa = (code >> 5) & 1, (code >> 3) & 3, code & 7
    value = (mantissa / 8.0) if exponent == 0 else (2.0 ** (exponent - 1)) * (1.0 + mantissa / 8.0)
    values[code] = -value if sign else value
  return values


def _hadamard32():
  """Sylvester 32 x 32 divided by sqrt(32) (gemm_op_a6w6.py _hadamard32_np)."""
  h = np.ones((1, 1), np.float32)
  while h.shape[0] < _GROUP:
    h = np.block([[h, h], [h, -h]])
  return (h / np.sqrt(float(_GROUP))).astype(np.float32)


_E2M3 = _e2m3_table()
_HADAMARD32 = _hadamard32()


def pack_sizes(rows, k):
  """`(packed bytes, scale bytes)` of an operand with `rows` (a multiple of 256) rows and contraction `k` (a
  multiple of 128). Raises ValueError otherwise, or when a buffer would pass the kernel's 2 GiB range."""
  if rows <= 0 or k <= 0 or rows % _TILE or k % _K_TILE:
    raise ValueError(f"MXFP6 packing needs rows % 256 == 0 and K % 128 == 0: got rows {rows}, K {k}.")
  steps = k // _K_TILE + _PAD_K
  sizes = (rows // _TILE * steps * _PACKED_TILE_BYTES, rows // _TILE * steps * _SCALE_TILE_BYTES)
  if max(sizes) > _MAX_BUFFER_BYTES:
    raise ValueError(f"MXFP6 packing of rows {rows} x K {k} exceeds the kernel's 2 GiB range.")
  return sizes


def quant_fp6(x):
  """`(R, K)` float -> codes `(R, K)` uint8 (6-bit E2M3) and e8m0 scales `(R, K/32)` uint8.

  gemm_op_a6w6.py quant_mxfp6 in JAX. The rotation is float32 as in the numpy original; the nearest level is
  computed as AITER's Triton kernel computes it, which rounds exact midpoints up where the numpy original rounds
  them down.
  """
  r, k = x.shape
  groups = k // _GROUP
  rotated = jnp.einsum(
      "rbi,ij->rbj",
      x.astype(jnp.float32).reshape(r, groups, _GROUP),
      jnp.asarray(_HADAMARD32),
      precision=jax.lax.Precision.HIGHEST,
  )
  absmax = jnp.max(jnp.abs(rotated), axis=-1)
  exponent = jnp.floor(jnp.log2(jnp.where(absmax > 0, absmax, 1.0))).astype(jnp.int32)
  scale_exponent = jnp.where(absmax > 0, jnp.clip(exponent - 2, -127, 127), 0)  # E2M3's largest exponent is 2
  scales = (scale_exponent + 127).astype(jnp.uint8)
  scaled = rotated * jnp.exp2(-scale_exponent.astype(jnp.float32))[..., None]
  magnitude = jnp.minimum(jnp.abs(scaled), 7.5)
  step = jnp.where(magnitude < 2.0, 8.0, jnp.where(magnitude < 4.0, 4.0, 2.0))
  bias = jnp.where(magnitude < 2.0, 0.0, jnp.where(magnitude < 4.0, 8.0, 16.0))
  code = (jnp.floor(magnitude * step + 0.5) + bias).astype(jnp.int32) + (scaled < 0).astype(jnp.int32) * 32
  return code.astype(jnp.uint8).reshape(r, k), scales


def pack_codes(codes):
  """`(R, K)` 6-bit codes -> the kernel's packed blob, 1-D uint8 (gemm_op_a6w6.py pack_big as a permutation).

  Row tile t, K step s: 1024 blocks of 32 codes, block `rb * 64 + kg * 16 + r16` for row `t*256 + rb*16 + r16`
  and columns `s*128 + kg*32 ..+32`. A block's 24 bytes split 16 into the step's first 16384 bytes and 8 into
  its last 8192. The two guard steps of every row tile are zeros.
  """
  r, k = codes.shape
  tiles, steps = r // _TILE, k // _K_TILE
  blocks = codes.reshape(tiles, 16, 16, steps, 4, _GROUP).transpose(0, 3, 1, 4, 2, 5)  # (t, s, rb, kg, r16, 32)
  quads = blocks.reshape(tiles, steps, 1024, 8, 4).astype(jnp.uint32)  # 4 codes -> 3 bytes
  b0 = (quads[..., 0] | (quads[..., 1] << 6)) & 0xFF
  b1 = ((quads[..., 1] >> 2) | (quads[..., 2] << 4)) & 0xFF
  b2 = ((quads[..., 2] >> 4) | (quads[..., 3] << 2)) & 0xFF
  packed = jnp.stack([b0, b1, b2], axis=-1).astype(jnp.uint8).reshape(tiles, steps, 1024, 24)
  step = jnp.concatenate(
      [packed[..., :16].reshape(tiles, steps, 1024 * 16), packed[..., 16:].reshape(tiles, steps, 1024 * 8)], -1
  )
  return jnp.pad(step, ((0, 0), (0, _PAD_K), (0, 0))).reshape(-1)


def pack_scales(scales):
  """`(R, K/32)` e8m0 scales -> the kernel's scale blob, 1-D uint8 (gemm_op_a6w6.py pack_scale as a permutation).

  Byte of a step = `su * 512 + kg * 128 + r16 * 8 + sub` for row `su * 128 + sub * 16 + r16` and group
  `s*4 + kg`. The guard steps hold 127 (2^0).
  """
  r, groups = scales.shape
  tiles, steps = r // _TILE, groups // 4
  s = scales.reshape(tiles, 2, 8, 16, steps, 4).transpose(0, 4, 1, 5, 3, 2)  # (t, s, su, kg, r16, sub)
  s = s.reshape(tiles, steps, _SCALE_TILE_BYTES)
  return jnp.pad(s, ((0, 0), (0, _PAD_K), (0, 0)), constant_values=127).reshape(-1)


def quant_pack(x):
  """`(R, K)` float (R % 256 == 0, K % 128 == 0) -> `(packed, packed_scale)`: AITER's quant_mxfp6_gemm in JAX."""
  pack_sizes(*x.shape)
  codes, scales = quant_fp6(x)
  return pack_codes(codes), pack_scales(scales)


def unpack_codes(blob, rows, k):
  """Inverse of `pack_codes`: the packed blob of `rows` x `k` -> codes `(rows, k)` uint8 (guard steps skipped)."""
  pack_sizes(rows, k)
  tiles, steps = rows // _TILE, k // _K_TILE
  step = blob.reshape(tiles, steps + _PAD_K, _PACKED_TILE_BYTES)[:, :steps]
  first = step[..., : 1024 * 16].reshape(tiles, steps, 1024, 16)
  last = step[..., 1024 * 16 :].reshape(tiles, steps, 1024, 8)
  triples = jnp.concatenate([first, last], axis=-1).reshape(tiles, steps, 1024, 8, 3).astype(jnp.uint32)
  b0, b1, b2 = triples[..., 0], triples[..., 1], triples[..., 2]
  codes = jnp.stack([b0 & 63, ((b0 >> 6) | (b1 << 2)) & 63, ((b1 >> 4) | (b2 << 4)) & 63, (b2 >> 2) & 63], -1)
  blocks = codes.reshape(tiles, steps, 16, 4, 16, _GROUP)  # (t, s, rb, kg, r16, 32)
  return jnp.transpose(blocks, (0, 2, 4, 1, 3, 5)).reshape(rows, k).astype(jnp.uint8)


def unpack_scales(blob, rows, k):
  """Inverse of `pack_scales`: the scale blob of `rows` x `k` -> e8m0 scales `(rows, k/32)` uint8."""
  pack_sizes(rows, k)
  tiles, steps = rows // _TILE, k // _K_TILE
  s = blob.reshape(tiles, steps + _PAD_K, _SCALE_TILE_BYTES)[:, :steps].reshape(tiles, steps, 2, 4, 16, 8)
  return jnp.transpose(s, (0, 2, 5, 4, 1, 3)).reshape(rows, k // _GROUP)  # from (t, s, su, kg, r16, sub)


def dequant_fp6(codes, scales):
  """Codes and e8m0 scales -> float32 `(R, K)` in the rotated domain (`a_rot @ b_rot.T == a @ b.T`)."""
  r, k = codes.shape
  values = jnp.asarray(_E2M3)[codes.astype(jnp.int32)].reshape(r, k // _GROUP, _GROUP)
  return (values * jnp.exp2(scales.astype(jnp.float32) - 127.0)[..., None]).reshape(r, k)


def quant_pack_hip(x, pad_mode=1):
  """AMD's HIP quantizer (krea2_mxfp6_shims/quant_mxfp6_ja.cu): `(rows, K)` bf16 / fp16, any `rows` -> the
  `(packed, packed_scale)` of `quant_pack` for `rows` padded to 256.

  The kernel leaves the guard steps and the rows >= `rows` unwritten; `pad_mode` 1 sets them first to what
  `quant_pack` writes for zero rows (codes 0, scales 127), 0 leaves what the buffer held, 2 writes poison (a
  test: the GEMM ignores those bytes, measured on an MI350X). Against `quant_fp6` the codes are not always equal:
  AMD's kernel rotates with the bf16-rounded 1/sqrt(32) and the hardware conversion may round a midpoint the
  other way (measured: 99.85 % equal bytes). Needs `register()` before the traced call is compiled.
  """
  rows, k = x.shape
  if x.dtype not in (jnp.bfloat16, jnp.float16):
    raise ValueError(f"The HIP MXFP6 quantizer takes a bf16 or fp16 (rows, K) input, got {x.dtype}.")
  packed, scale = pack_sizes(_ceil(rows, _TILE), k)
  call = jax.ffi.ffi_call(
      QUANT_TARGET,
      (jax.ShapeDtypeStruct((packed,), jnp.uint8), jax.ShapeDtypeStruct((scale,), jnp.uint8)),
      vmap_method="broadcast_all",
  )
  return call(x, pad_mode=np.int32(pad_mode))


def gemm_packed(a, b, a_scale, b_scale, m, n, k):
  """AMD's MXFP6 GEMM (krea2_mxfp6_shims/gemm_a6w6_ja.cu): packed A (`m` rows), B (`n` rows) and their scales
  -> `(m, n)` bfloat16 = `a @ b.T`. The operand shapes are the caller's to check (`linear` does). Needs
  `register()` before the traced call is compiled."""
  call = jax.ffi.ffi_call(GEMM_TARGET, jax.ShapeDtypeStruct((m, n), jnp.bfloat16), vmap_method="broadcast_all")
  return call(a, b, a_scale, b_scale, k=np.int32(k))


def _check_impl(impl):
  impl = default_impl() if impl is None else impl
  if impl not in IMPLS:
    raise ValueError(f"MXFP6 impl must be one of {IMPLS}, got {impl!r}.")
  return impl


def kernel_param_shapes(k, n):
  """`((kernel shape, uint8), (scale shape, uint8))` of a projection with `k` inputs and `n` outputs.

  The packed weight has `n` padded to 256 rows (the real output widths 1536 / 6144 / 16384 need none). The same
  for every impl, so the parameter tree does not depend on it.
  """
  packed, scale = pack_sizes(_ceil(n, _TILE), k)
  return ((packed,), jnp.uint8), ((scale,), jnp.uint8)


@jax.jit
def _convert_kernel(w_nk):
  n = w_nk.shape[0]
  padded_n = _ceil(n, _TILE)
  w = w_nk.astype(jnp.float32)
  if padded_n != n:
    w = jnp.pad(w, ((0, padded_n - n), (0, 0)))
  return quant_pack(w)


def convert_kernel(w_nk):
  """Float `(n, k)` weight (the model's `x @ w_nk.T` orientation) -> `(kernel, scale)` of `kernel_param_shapes`.

  Pure JAX on the default device, the same values for every impl."""
  if len(w_nk.shape) != 2:
    raise ValueError(f"convert_kernel takes a 2-D (n, k) weight, got shape {tuple(w_nk.shape)}.")
  n, k = w_nk.shape
  (kernel_shape, _), (scale_shape, _) = kernel_param_shapes(k, n)
  kernel, scale = _convert_kernel(w_nk)
  if (kernel.shape, scale.shape) != (kernel_shape, scale_shape):
    raise AssertionError(f"MXFP6 kernel {(kernel.shape, scale.shape)} is not {(kernel_shape, scale_shape)}.")
  return kernel, scale


def quantize_activation(x, impl=None):
  """`(..., k)` float -> the packed operand pair `(a, a_scale)` of its `prod(leading dims)` rows, padded to 256.

  Traceable under jit. The rows are cast to bfloat16 first (the HIP quantizer's input; the JAX one gets the same
  values). `impl` None = `default_impl()`.
  """
  impl = _check_impl(impl)
  k = x.shape[-1]
  rows = math.prod(x.shape[:-1])
  padded_rows = _ceil(rows, _TILE)
  pack_sizes(padded_rows, k)
  x2 = x.reshape(rows, k).astype(jnp.bfloat16)
  if impl == "kernel":
    return tuple(quant_pack_hip(x2))  # pads the rows itself
  if padded_rows != rows:
    x2 = jnp.pad(x2, ((0, padded_rows - rows), (0, 0)))
  return quant_pack(x2)


def linear(quantized, kernel, scale, lead_shape, k, n, impl=None):
  """`quantize_activation`'s pair and `convert_kernel`'s -> `lead_shape + (n,)` bfloat16, about `x @ w_nk.T`.

  Traceable under jit; `lead_shape`, `k` and `n` are static. Operand shapes and the kernel's 2 GiB limits (packed
  operands and the `(rows, n)` bf16 product) are checked at trace time, for 'ref' too so that a CPU check sees
  what the GPU would refuse: ValueError.
  """
  impl = _check_impl(impl)
  lead_shape = tuple(lead_shape)
  rows = math.prod(lead_shape)
  padded_rows, padded_n = _ceil(rows, _TILE), _ceil(n, _TILE)
  a, a_scale = quantized
  want = tuple((size,) for size in pack_sizes(padded_rows, k) + pack_sizes(padded_n, k))
  got = (a.shape, a_scale.shape, kernel.shape, scale.shape)
  if got != want or any(t.dtype != jnp.uint8 for t in (a, a_scale, kernel, scale)):
    raise ValueError(f"MXFP6 linear rows {rows} k {k} n {n}: operand shapes {got} are not {want} uint8.")
  if padded_rows * padded_n * 2 > _MAX_BUFFER_BYTES:
    raise ValueError(
        f"MXFP6 linear: the ({padded_rows}, {padded_n}) bf16 product exceeds the kernel's 2 GiB range "
        f"(rows = batch x sequence = {rows}); use a smaller batch."
    )
  if impl == "kernel":
    out = gemm_packed(a, kernel, a_scale, scale, padded_rows, padded_n, k)
  else:
    xa = dequant_fp6(unpack_codes(a, padded_rows, k), unpack_scales(a_scale, padded_rows, k))[:rows]
    wa = dequant_fp6(unpack_codes(kernel, padded_n, k), unpack_scales(scale, padded_n, k))
    out = jax.lax.dot_general(
        xa, wa, (((1,), (1,)), ((), ())), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )
  if out.shape != (rows, n):
    out = out[:rows, :n]
  return out.astype(jnp.bfloat16).reshape(lead_shape + (n,))
