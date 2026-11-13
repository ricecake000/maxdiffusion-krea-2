// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// (the host entry below follows AITER's csrc/kernels/quant_mxfp6_gemm.cu `quant_mxfp6_gemm_hip`; full text:
// LICENSE.aiter)
//
// XLA FFI handler for AITER's HIP MXFP6 activation quantizer on gfx950 (Krea 2 transformer quantization 'mxfp6').
//   input [rows, K] bf16 / fp16 -> packed u8 1-D, packed_scale u8 1-D in aiter's layout "mxfp6_c0c1_256_padk2"
//   (pad256(rows) / 256 row tiles x (pad128(K) / 128 + 2) K steps of 24576 bytes, scales 1024 bytes per step):
//   the A / B operand format of gemm_a6w6_ja.cu, the same bytes as kernels/krea2_mxfp6.py quant_pack up to the
//   rounding differences listed in its quant_pack_hip.
// The kernel is AMD's own `quant_mxfp6_gemm_kernel` from aiter's csrc/kernels/quant_mxfp6_gemm.cu (MIT, Advanced
// Micro Devices), #included below from the aiter tree at build time: the 32-point Hadamard rotation along K, E2M3
// codes through the gfx950 instruction v_cvt_scalef32_2xpk16_fp6_f32, e8m0 scales and the packing, in one pass.
// This file replaces only AMD's host entry `quant_mxfp6_gemm_hip` (same file): the same checks and the same grid,
// but the launch goes to the stream XLA hands the handler, not to aiter::getCurrentHIPStream() (a thread-local
// that nothing sets under XLA: it would be the null stream), and a failed check is an FFI error, not an abort.
// One addition: AMD's kernel never writes the two guard K steps of each row tile nor the rows >= `rows` of the
// last row tile (gemm_op_a6w6.py quant_mxfp6_gemm: "Leave guard and row-padding slots unspecified"; the GEMM is
// said not to accumulate the guard steps, which cannot be checked in its .co binaries). Attribute pad_mode:
//   1 (the Python wrapper's default) those bytes are set first, on the same stream, by one small kernel of ours
//     (fill_pad below: one launch, 16-byte stores, a few MB at the 2k shapes) to what aiter's numpy packer and our
//     JAX port write there: codes 0, scales 127 (2^0). The output is then fully defined.
//   0 left as the buffer held them (AMD's own behaviour), 2 set to poison (0xA5 / 0x8A = 2^11: a test that the GEMM
//     really ignores them; it does, measured on an MI350X).
// Not marked command-buffer compatible (nothing here needs it; kept out of graphs as gemm_a6w6_ja.cu is).
#include <cstdint>
#include <string>

#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// AMD's #else branches of quant_mxfp6_gemm.cu write all-zero codes when __gfx950__ is not defined. hipcc compiles
// this file once for the host (no arch macro, no device code emitted) and once per --offload-arch with
// __HIP_DEVICE_COMPILE__ and that arch's macro (__gfx950__ for gfx950) defined: any device pass for another arch
// stops here, so the .so can only carry the real conversion.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx950__)
#error "quant_mxfp6_ja.cu needs --offload-arch=gfx950 only: AMD's kernel writes zeros on other architectures"
#endif

#include "quant_mxfp6_gemm.cu"   // aiter/csrc/kernels (build.sh puts it on the include path)

namespace ffi = xla::ffi;

namespace krea2_qfp6 {

// AMD's constants (anonymous namespace of quant_mxfp6_gemm.cu, visible in this translation unit).
using aiter::kBlockThreads;
using aiter::kGroupSize;
using aiter::kKGuardTiles;
using aiter::kKTile;
using aiter::kLargeKStepsPerBlock;
using aiter::kLargeKThreshold;
using aiter::kMaxBufferBytes;
using aiter::kOutputAlignment;
using aiter::kPackedTileBytes;
using aiter::kScaleTileBytes;
using aiter::kSmallKStepsPerBlock;
using aiter::kTileRows;
using aiter::kGroupsPerKTile;
constexpr int kScalePad = 127;   // e8m0 2^0: aiter's pack_scale pads with 127
constexpr int kPoisonCode = 0xA5, kPoisonScale = 0x8A;   // pad_mode 2 (0x8A = 2^11: a large term if accumulated)

static ffi::Error invalid(const std::string& why) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, "QuantMxfp6JA: " + why);
}

// quant_mxfp6_gemm_hip's `get_gpu_arch() == "gfx950"`, asked once (it reads the device properties).
static bool device_is_gfx950() {
  static const bool is_gfx950 = get_gpu_arch() == "gfx950";
  return is_gfx950;
}

// pad_mode 1 / 2: one 16-byte store per thread over (a) the two guard steps of every row tile that is not the
// last padded one, packed then scales, and (b) when rows % 256 != 0 the whole last row tile, packed then scales.
// The regions are disjoint, start at multiples of 1024 bytes and the buffers are 16-byte aligned (checked).
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
constexpr int64_t kPackedUnits = kPackedTileBytes / 16, kScaleUnits = kScaleTileBytes / 16;   // per step
constexpr int kFillThreads = 256;

__global__ __launch_bounds__(kFillThreads) void fill_pad(uint8_t* __restrict__ packed, uint8_t* __restrict__ scale,
                                                         int64_t guard_tiles, int64_t last_tile, int32_t steps,
                                                         int32_t nk_pad, uint32_t code4, uint32_t scale4,
                                                         int64_t units) {
  const int64_t i0 = static_cast<int64_t>(blockIdx.x) * kFillThreads + threadIdx.x;
  if (i0 >= units) return;
  const int64_t guard_p = guard_tiles * kKGuardTiles * kPackedUnits, guard_s = guard_tiles * kKGuardTiles * kScaleUnits;
  const int64_t last_p = nk_pad * kPackedUnits;   // the last tile's units (used only when last_tile >= 0)
  int64_t i = i0;
  uint8_t* dst;
  uint32_t v;
  if (i < guard_p) {
    const int64_t t = i / (kKGuardTiles * kPackedUnits), u = i % (kKGuardTiles * kPackedUnits);
    dst = packed + (t * nk_pad + steps) * kPackedTileBytes + u * 16;
    v = code4;
  } else if ((i -= guard_p) < guard_s) {
    const int64_t t = i / (kKGuardTiles * kScaleUnits), u = i % (kKGuardTiles * kScaleUnits);
    dst = scale + (t * nk_pad + steps) * kScaleTileBytes + u * 16;
    v = scale4;
  } else if ((i -= guard_s) < last_p) {
    dst = packed + last_tile * nk_pad * kPackedTileBytes + i * 16;
    v = code4;
  } else {
    i -= last_p;
    dst = scale + last_tile * nk_pad * kScaleTileBytes + i * 16;
    v = scale4;
  }
  *reinterpret_cast<u32x4*>(dst) = u32x4{v, v, v, v};
}

template <typename input_t>
static hipError_t launch(const void* input, uint8_t* packed, uint8_t* packed_scale, int64_t rows64, int32_t cols,
                         int32_t num_groups, int32_t nk_pad, int32_t k_steps_per_block, int64_t grid_size,
                         hipStream_t stream) {
  // quant_mxfp6_gemm_hip: the two instantiations and the launch shape, verbatim but for the stream.
  if (k_steps_per_block == kLargeKStepsPerBlock) {
    aiter::quant_mxfp6_gemm_kernel<input_t, kLargeKStepsPerBlock>
        <<<static_cast<int32_t>(grid_size), kBlockThreads, 0, stream>>>(
            reinterpret_cast<const input_t*>(input), packed, packed_scale, rows64, cols, num_groups, nk_pad);
  } else {
    aiter::quant_mxfp6_gemm_kernel<input_t, kSmallKStepsPerBlock>
        <<<static_cast<int32_t>(grid_size), kBlockThreads, 0, stream>>>(
            reinterpret_cast<const input_t*>(input), packed, packed_scale, rows64, cols, num_groups, nk_pad);
  }
  return hipGetLastError();
}

ffi::Error QuantMxfp6_Bridge(hipStream_t stream,
                             ffi::AnyBuffer input,                   // [rows, K] bf16 or fp16, row-major
                             int32_t pad_mode,                       // 0 leave, 1 codes 0 / scales 127, 2 poison
                             ffi::Result<ffi::AnyBuffer> packed,     // u8, pad256(rows)/256 * nk_pad * 24576
                             ffi::Result<ffi::AnyBuffer> packed_scale) {  // u8, pad256(rows)/256 * nk_pad * 1024
  // quant_mxfp6_gemm_hip's checks, in its order. is_gpu / same device / contiguous hold for XLA buffers (one
  // device per executable, dense row-major default layouts).
  if (!device_is_gfx950()) return invalid("requires gfx950 hardware FP6 conversion");
  auto dims = input.dimensions();
  if (dims.size() != 2) return invalid("expected a 2D [rows, K] input");
  if (packed->element_type() != ffi::DataType::U8) return invalid("expected uint8 packed output");
  if (packed_scale->element_type() != ffi::DataType::U8) return invalid("expected uint8 packed-scale output");
  uint8_t* packed_ptr = static_cast<uint8_t*>(packed->untyped_data());
  uint8_t* scale_ptr = static_cast<uint8_t*>(packed_scale->untyped_data());
  if (reinterpret_cast<uintptr_t>(packed_ptr) % kOutputAlignment != 0 ||
      reinterpret_cast<uintptr_t>(scale_ptr) % kOutputAlignment != 0)
    return invalid("expected 16-byte-aligned outputs");
  const ffi::DataType dtype = input.element_type();
  if (dtype != ffi::DataType::BF16 && dtype != ffi::DataType::F16) return invalid("expected bf16 or fp16 input");

  const int64_t rows64 = dims[0];
  const int64_t cols64 = dims[1];
  if (!(rows64 > 0 && cols64 > 0)) return invalid("expected non-empty input");
  if (!(rows64 <= INT64_MAX - (kTileRows - 1))) return invalid("rows exceed int64 range");
  if (!(cols64 <= INT32_MAX - (kKTile - 1))) return invalid("K exceeds int32 range");
  const int32_t cols = static_cast<int32_t>(cols64);

  const int32_t pad_cols = (cols + kKTile - 1) / kKTile * kKTile;
  const int64_t pad_rows = (rows64 + kTileRows - 1) / kTileRows * kTileRows;
  const int32_t num_groups = pad_cols / kGroupSize;
  const int32_t nk_pad = pad_cols / kKTile + kKGuardTiles;
  const int64_t tile_rows = pad_rows / kTileRows;
  if (!(tile_rows <= kMaxBufferBytes / (static_cast<int64_t>(nk_pad) * kPackedTileBytes)))
    return invalid("packed output exceeds the 2 GiB address range");
  const int64_t expected_packed = tile_rows * static_cast<int64_t>(nk_pad) * kPackedTileBytes;
  const int64_t expected_scale = tile_rows * static_cast<int64_t>(nk_pad) * kScaleTileBytes;
  // numel() == bytes for u8; the 1-D shape is the wrapper's contract, checked as well
  if (packed->dimensions().size() != 1 || static_cast<int64_t>(packed->size_bytes()) != expected_packed)
    return invalid("packed output has " + std::to_string(packed->size_bytes()) + " bytes, expected 1-D " +
                   std::to_string(expected_packed));
  if (packed_scale->dimensions().size() != 1 || static_cast<int64_t>(packed_scale->size_bytes()) != expected_scale)
    return invalid("packed-scale output has " + std::to_string(packed_scale->size_bytes()) + " bytes, expected 1-D " +
                   std::to_string(expected_scale));

  const int64_t row_blocks = (rows64 + 15) / 16;
  const int32_t num_steps = num_groups / kGroupsPerKTile;
  const int32_t k_steps_per_block = cols >= kLargeKThreshold ? kLargeKStepsPerBlock : kSmallKStepsPerBlock;
  const int32_t num_work_steps = (num_steps + k_steps_per_block - 1) / k_steps_per_block;
  const int64_t grid_size = row_blocks * num_work_steps;
  if (!(grid_size <= 2147483647LL)) return invalid("grid size exceeds maximum");

  if (pad_mode < 0 || pad_mode > 2) return invalid("pad_mode must be 0, 1 or 2");
  if (pad_mode != 0) {
    // The kernel writes every logical K step (num_steps of them; K beyond `cols` is read as zeros) of every row
    // < rows, nothing else. Before it, on the same stream: the two guard steps of every row tile, and when
    // rows % 256 != 0 the whole last row tile instead (the kernel then overwrites its rows < rows). A step is
    // contiguous: offset (tile * nk_pad + step) * tile bytes, as the kernel addresses it.
    const bool partial = rows64 % kTileRows != 0;
    const int64_t guard_tiles = partial ? tile_rows - 1 : tile_rows;
    const int64_t units = guard_tiles * kKGuardTiles * (kPackedUnits + kScaleUnits) +
                          (partial ? static_cast<int64_t>(nk_pad) * (kPackedUnits + kScaleUnits) : 0);
    const uint32_t code4 = (pad_mode == 1 ? 0u : static_cast<uint32_t>(kPoisonCode)) * 0x01010101u;
    const uint32_t scale4 = static_cast<uint32_t>(pad_mode == 1 ? kScalePad : kPoisonScale) * 0x01010101u;
    (void)hipGetLastError();   // the error state is per thread and sticky: an earlier, unrelated one is not ours
    fill_pad<<<static_cast<int32_t>((units + kFillThreads - 1) / kFillThreads), kFillThreads, 0, stream>>>(
        packed_ptr, scale_ptr, guard_tiles, partial ? tile_rows - 1 : -1, num_steps, nk_pad, code4, scale4, units);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
      return ffi::Error(ffi::ErrorCode::kInternal, std::string("QuantMxfp6JA: fill launch failed: ") + hipGetErrorString(err));
  }

  // AITER_DISPATCH_FLOATING16_TYPES_rmTorch's element types (aiter_dispatch.h cpp_type_from_aiter_dtype).
  using bf16_t = aiter::dispatch_detail::cpp_type_from_aiter_dtype<AITER_DTYPE_bf16>::type;
  using fp16_t = aiter::dispatch_detail::cpp_type_from_aiter_dtype<AITER_DTYPE_fp16>::type;
  (void)hipGetLastError();   // as above
  hipError_t err = dtype == ffi::DataType::BF16
      ? launch<bf16_t>(input.untyped_data(), packed_ptr, scale_ptr, rows64, cols, num_groups, nk_pad,
                       k_steps_per_block, grid_size, stream)
      : launch<fp16_t>(input.untyped_data(), packed_ptr, scale_ptr, rows64, cols, num_groups, nk_pad,
                       k_steps_per_block, grid_size, stream);
  if (err != hipSuccess)
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("QuantMxfp6JA: launch failed: ") + hipGetErrorString(err));
  return ffi::Error::Success();
}

}  // namespace krea2_qfp6

#pragma GCC visibility push(default)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    QuantMxfp6JA, krea2_qfp6::QuantMxfp6_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()      // input [rows, K] bf16 / fp16
        .Attr<int32_t>("pad_mode")  // guard steps + padding rows: 0 leave, 1 codes 0 / scales 127, 2 poison
        .Ret<ffi::AnyBuffer>()      // packed (1-D u8)
        .Ret<ffi::AnyBuffer>());    // packed_scale (1-D u8)
#pragma GCC visibility pop
