// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// (the launcher below follows AITER's csrc/py_itfs_cu/asm_gemm_a6w6.cu `gemm_a6w6_asm`; full text: LICENSE.aiter)
//
// XLA FFI handler for AITER's hand-written MXFP6 GEMM on gfx950 (Krea 2 transformer quantization 'mxfp6').
//   Out[M, N] bf16 = A @ B^T, A and B in MXFP6 (E2M3 values, one e8m0 scale per 32 along K), fp32 accumulation
// A, B, A_scale, B_scale are the packed blobs of aiter's quant_mxfp6_gemm (layout "mxfp6_c0c1_256_padk2":
// 256-row tiles x (K/128 + 2) K steps of 24576 bytes, scales 1024 bytes per step; aiter/ops/gemm_op_a6w6.py
// pack_big / pack_scale, ported to JAX in kernels/krea2_mxfp6.py). The kernels are the f6gemm code objects of
// aiter's hsa/gfx950/f6gemm. AITER's launcher is rewritten here for XLA FFI buffers: the same checks in the same
// order, the launch on the stream XLA hands the handler, a failed check an FFI error instead of an abort.
// Kernel choice: aiter/ops/gemm_op_a6w6.py `_default_gemm_a6w6_kernel` (the untuned path of `gemm_a6w6`). The
// exact-shape kernels of the table are never picked (none matches a Krea 2 shape).
// Not marked command-buffer compatible: the first call for a kernel loads its code object.
// Written against aiter a3158e4800b6080e3adb25eb58bc48686bc1859c (the commit jax-aiter ac59012c records).
#include <exception>
#include <hip/hip_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_f6gemm_configs.hpp"

namespace ffi = xla::ffi;

namespace krea2_a6w6 {

// asm_gemm_a6w6.cu, anonymous namespace constants.
constexpr int kTileSize = 256;
constexpr int kKTileSize = 128;
constexpr int kScaleGroupSize = 32;
constexpr int kKPaddingTiles = 2;
constexpr size_t kPackedTileBytes = 24576;
constexpr size_t kScaleTileBytes = 1024;
constexpr char kPackLayout[] = "mxfp6_c0c1_256_padk2";
// gemm_op_a6w6.py: _KERNEL_NAME, _SAFE_FALLBACK_KERNEL_NAME, _SHORT_K_SWIZZLE_LIMIT, _GROUPED_SWIZZLE_MAX_M / _N,
// _MAX_BUFFER_BYTES.
constexpr char kKernelName[] = "f6gemm_dmabig_kernel_func";
constexpr char kSafeFallbackKernelName[] = "f6gemm_dmabig_swz0_kernel_func";
constexpr long long kShortKSwizzleLimit = 48LL * kKTileSize;
constexpr long long kGroupedSwizzleMaxM = 16LL * 32 * kTileSize;
constexpr long long kGroupedSwizzleMaxN = 64LL * kTileSize;
constexpr size_t kMaxBufferBytes = size_t(1) << 31;

// asm_gemm_a6w6.cu KernelArgs, field for field: "identical to the a4w4 asm gemm ABI ... 0x180 bytes".
struct __attribute__((packed)) KernelArgs {
  void* ptr_D;    p2 _p0;
  void* ptr_C;    p2 _p1;
  void* ptr_A;    p2 _p2;
  void* ptr_B;    p2 _p3;
  float alpha;    p3 _p4;
  float beta;     p3 _p5;
  unsigned int stride_D0; p3 _p6;
  unsigned int stride_D1; p3 _p7;
  unsigned int stride_C0; p3 _p8;
  unsigned int stride_C1; p3 _p9;
  unsigned int stride_A0; p3 _p10;
  unsigned int stride_A1; p3 _p11;
  unsigned int stride_B0; p3 _p12;
  unsigned int stride_B1; p3 _p13;
  unsigned int M; p3 _p14;
  unsigned int N; p3 _p15;
  unsigned int K; p3 _p16;
  void* ptr_ScaleA; p2 _p17;
  void* ptr_ScaleB; p2 _p18;
  unsigned int stride_ScaleA0; p3 _p19;
  unsigned int stride_ScaleA1; p3 _p20;
  unsigned int stride_ScaleB0; p3 _p21;
  unsigned int stride_ScaleB1; p3 _p22;
  int log2_k_split; p3 _p23;
};
static_assert(sizeof(KernelArgs) == 0x180, "a6w6 KernelArgs must be 0x180 bytes");

// gemm_op_a6w6.py _default_gemm_a6w6_kernel(M, N, K), with M, N, K already multiples of the tiles.
static std::string default_kernel(long long M, long long N, long long K) {
  bool short_k = K <= kShortKSwizzleLimit;
  bool grouped_grid_in_bounds = M <= kGroupedSwizzleMaxM && N <= kGroupedSwizzleMaxN;
  return (!short_k || grouped_grid_in_bounds) ? kKernelName : kSafeFallbackKernelName;
}

struct Kernel {
  std::unique_ptr<AiterAsmKernel> impl;
  int tile_m;
  int tile_n;
  int block_size;
};

// The checks of asm_gemm_a6w6.cu that depend on the kernel's table row, then the code object (loaded once per
// kernel under the mutex). Returns an empty string on success, else the reason; `out` is set on success.
static std::string pick_kernel(int M, int N, int K, const Kernel** out) {
  static std::mutex mu;
  static std::unordered_map<std::string, Kernel> loaded;
  static std::string arch;
  std::lock_guard<std::mutex> lock(mu);
  if (arch.empty())
    arch = get_gpu_arch();

  std::string name = default_kernel(M, N, K);
  std::string key = arch + name;   // asm_gemm_a6w6.cu: kname = arch_id + kernelName
  auto it = cfg_f6gemm_bf16_per1x32Fp6.find(key);
  if (it == cfg_f6gemm_bf16_per1x32Fp6.end())
    return "no f6gemm kernel " + key + " for this architecture";
  const auto& cfg = it->second;
  if (cfg.tile_M <= 0 || cfg.tile_N <= 0) return "invalid a6w6 tile dimensions";
  if (cfg.block_size <= 0) return "invalid a6w6 workgroup size";
  if (cfg.splitK != 0) return "a6w6 split-K is not supported";
  if (cfg.pack_layout != kPackLayout) return "unsupported a6w6 packed layout " + cfg.pack_layout;
  if (cfg.exact_K > 0 && (M != cfg.exact_M || N != cfg.exact_N || K != cfg.exact_K))
    return "kernel " + cfg.knl_name + " does not support the requested physical shape";
  if (cfg.swizzle_max_K > 0 && K <= cfg.swizzle_max_K) {
    if (M > cfg.swizzle_max_M) return "kernel " + cfg.knl_name + " exceeds its M swizzle bound";
    if (N > cfg.swizzle_max_N) return "kernel " + cfg.knl_name + " exceeds its N swizzle bound";
  }
  // Not in aiter (its tiles are always 256x256): the grid below assumes the packed tile of the layout.
  if (cfg.tile_M != kTileSize || cfg.tile_N != kTileSize) return "a6w6 kernel tile is not 256x256";
  auto lit = loaded.find(key);
  if (lit == loaded.end()) {
    Kernel k;
    k.impl = std::make_unique<AiterAsmKernel>(cfg.knl_name.c_str(), cfg.co_name.c_str());
    k.tile_m = cfg.tile_M;
    k.tile_n = cfg.tile_N;
    k.block_size = cfg.block_size;
    lit = loaded.emplace(key, std::move(k)).first;
  }
  *out = &lit->second;
  return "";
}

ffi::Error GemmA6W6_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer a,        // packed MXFP6 A, (M/256) * (K/128 + 2) * 24576 bytes, u8
    ffi::AnyBuffer b,        // packed MXFP6 B, (N/256) * (K/128 + 2) * 24576 bytes, u8
    ffi::AnyBuffer a_scale,  // packed e8m0 A scales, (M/256) * (K/128 + 2) * 1024 bytes, u8
    ffi::AnyBuffer b_scale,  // packed e8m0 B scales, (N/256) * (K/128 + 2) * 1024 bytes, u8
    int32_t k_attr,          // K: the contraction size of the packed layout (a multiple of 128)
    ffi::Result<ffi::AnyBuffer> out) {  // [M, N] bf16
  // asm_gemm_a6w6.cu gemm_a6w6_asm: M, N from out, K passed in; the same checks in the same order.
  auto od = out->dimensions();
  if (od.size() != 2)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: Out must be [M, N]");
  if (out->element_type() != ffi::DataType::BF16)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: only bf16 output");
  long long M = od[0], N = od[1], K = k_attr;
  if (M <= 0 || M % kTileSize != 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: M must be a positive multiple of 256");
  if (N <= 0 || N % kTileSize != 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: N must be a positive multiple of 256");
  if (K <= 0 || K % kKTileSize != 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: K must be a positive multiple of 128");
  if (a.element_type() != ffi::DataType::U8 || b.element_type() != ffi::DataType::U8 ||
      a_scale.element_type() != ffi::DataType::U8 || b_scale.element_type() != ffi::DataType::U8)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: packed operands and scales must be u8");
  const size_t nk_pad = static_cast<size_t>(K / kKTileSize + kKPaddingTiles);
  const size_t expected_a = static_cast<size_t>(M / kTileSize) * nk_pad * kPackedTileBytes;
  const size_t expected_b = static_cast<size_t>(N / kTileSize) * nk_pad * kPackedTileBytes;
  const size_t expected_sa = static_cast<size_t>(M / kTileSize) * nk_pad * kScaleTileBytes;
  const size_t expected_sb = static_cast<size_t>(N / kTileSize) * nk_pad * kScaleTileBytes;
  if (a.size_bytes() != expected_a || b.size_bytes() != expected_b)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "a6w6 GEMM: packed operand buffer sizes must exactly match the launch dimensions");
  if (a_scale.size_bytes() != expected_sa || b_scale.size_bytes() != expected_sb)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "a6w6 GEMM: packed scale buffer sizes must exactly match the launch dimensions");
  // gemm_op_a6w6.py _compute_mxfp6_gemm_pack_layout: "exceeds the kernel's 2 GiB range"; Out by the same rule.
  if (expected_a > kMaxBufferBytes || expected_b > kMaxBufferBytes ||
      static_cast<size_t>(M) * static_cast<size_t>(N) * 2 > kMaxBufferBytes)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: a buffer exceeds the kernel's 2 GiB range");

  // aiter's checks abort the process unless its thread-local throw flag is set: with the guard a code object that
  // cannot be loaded comes back as an FFI error. (HIP_CALL failures still abort; nothing here can catch those.)
  const Kernel* kernel = nullptr;
  std::string why;
  try {
    aiter_detail::AiterThrowGuard guard;
    why = pick_kernel(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), &kernel);
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("a6w6 GEMM: loading the kernel failed: ") + e.what());
  }
  if (!why.empty())
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a6w6 GEMM: " + why);
  if (kernel == nullptr || !kernel->impl)
    return ffi::Error(ffi::ErrorCode::kInternal, "a6w6 GEMM: no kernel");

  // asm_gemm_a6w6.cu: the fields it sets; the rest stay zero ("left zeroed and ignored by the kernel").
  KernelArgs args{};
  args.ptr_D = out->untyped_data();
  args.ptr_C = nullptr;
  args.ptr_A = const_cast<void*>(a.untyped_data());
  args.ptr_B = const_cast<void*>(b.untyped_data());
  args.alpha = 1.0f;                                    // gemm_a6w6 supports only alpha = 1.0
  args.beta = 0.0f;
  args.stride_D0 = static_cast<unsigned int>(N);       // out->stride(0): elements of a contiguous [M, N]
  args.M = static_cast<unsigned int>(M);
  args.N = static_cast<unsigned int>(N);
  args.K = static_cast<unsigned int>(K);
  args.ptr_ScaleA = const_cast<void*>(a_scale.untyped_data());
  args.ptr_ScaleB = const_cast<void*>(b_scale.untyped_data());
  args.stride_ScaleA0 = static_cast<unsigned int>(K / kScaleGroupSize);
  args.stride_ScaleB0 = static_cast<unsigned int>(K / kScaleGroupSize);
  args.log2_k_split = 0;
  size_t arg_size = sizeof(args);
  int gdx = static_cast<int>((N + kernel->tile_n - 1) / kernel->tile_n);
  int gdy = static_cast<int>((M + kernel->tile_m - 1) / kernel->tile_m);
  try {
    aiter_detail::AiterThrowGuard guard;
    kernel->impl->launch_kernel({&args, &arg_size, gdx, gdy, 1, kernel->block_size, 1, 1, stream});
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("a6w6 GEMM: the launch failed: ") + e.what());
  }
  return ffi::Error::Success();
}

}  // namespace krea2_a6w6

#pragma GCC visibility push(default)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmA6W6JA, krea2_a6w6::GemmA6W6_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()    // A (packed)
        .Arg<ffi::AnyBuffer>()    // B (packed)
        .Arg<ffi::AnyBuffer>()    // A_scale (packed)
        .Arg<ffi::AnyBuffer>()    // B_scale (packed)
        .Attr<int32_t>("k")       // K
        .Ret<ffi::AnyBuffer>());  // Out
#pragma GCC visibility pop
