#!/bin/bash
# Builds the two XLA FFI shims of kernels/krea2_mxfp6.py (Krea 2 transformer quantization 'mxfp6', AMD gfx950) against
# a JAX-AITER tree.
#   build.sh JA_ROOT VENV [OUT]     last line: MXFP6_BUILD_OK <dir> or MXFP6_BUILD_FAIL <why>
#     JA_ROOT  a jax-aiter checkout with its submodules and its umbrella library built
#              (build/jax_aiter_build/libjax_aiter.so), i.e. what serving sets JA_ROOT_DIR to
#     VENV     the serving venv: jax, the ROCm wheels with the `devel` extra (hipcc, rocm-sdk), pandas (AITER's codegen)
#     OUT      default JA_ROOT/build/krea2_mxfp6, where krea2_mxfp6.py looks without KREA2_MXFP6_LIB_DIR. Not
#              build/jax_aiter_build: jax_aiter loads every library it finds there.
# Output: gemm_a6w6_ja.so (GemmA6W6JA: AMD's f6gemm kernels, with the kernel table AITER's hsa/codegen.py writes) and
# quant_mxfp6_ja.so (QuantMxfp6JA: AITER's HIP quantizer csrc/kernels/quant_mxfp6_gemm.cu, #included from the tree).
# Both are needed and both are compiled for gfx950 only (the quantizer stops with #error for another --offload-arch).
# The sources were written against aiter a3158e4800b6080e3adb25eb58bc48686bc1859c.
set -uo pipefail
[ $# -ge 2 ] || { echo "MXFP6_BUILD_FAIL usage: build.sh JA_ROOT VENV [OUT]"; exit 1; }
fail() { echo "MXFP6_BUILD_FAIL $*"; exit 1; }
# Absolute paths: the code generator runs from inside the aiter tree.
HERE=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$1" 2> /dev/null && pwd) || fail "no directory $1"
VENV=$(cd "$2" 2> /dev/null && pwd) || fail "no directory $2"
O=${3:-$S/build/krea2_mxfp6}; mkdir -p "$O" || fail "cannot make $O"; O=$(cd "$O" && pwd)
[ -e "$S/build/jax_aiter_build/libjax_aiter.so" ] || fail "no built JAX-AITER tree in $S"
[ -x "$VENV/bin/hipcc" ] || fail "no hipcc in $VENV/bin"
A=$S/third_party/aiter; G=$O/gen
[ -s "$A/csrc/kernels/quant_mxfp6_gemm.cu" ] || fail "no csrc/kernels/quant_mxfp6_gemm.cu in the aiter tree"
mkdir -p "$G" || fail "cannot make $G"
LOG=$O/build.log; : > "$LOG"

( cd "$A" && AITER_GPU_ARCHS=gfx950 timeout -k 5 300 "$VENV/bin/python" hsa/codegen.py -m f6gemm -o "$G" ) >> "$LOG" 2>&1
[ -s "$G/asm_f6gemm_configs.hpp" ] || fail "codegen f6gemm (needs pandas): $(tail -n 2 "$LOG" | cut -c1-300 | tr '\n' '|')"
FFI_INC=$("$VENV/bin/python" -c 'from jax import ffi; print(ffi.include_dir())' 2> /dev/null | tail -n 1)
[ -d "$FFI_INC" ] || fail "no jax ffi include dir"

exports() {   # exports SO SYMBOL: SYMBOL is a defined dynamic symbol of SO (no check without nm)
  command -v nm > /dev/null || return 0   # not `nm | grep -q`: grep's early exit is a SIGPIPE failure under pipefail
  local syms; syms=$(nm -D --defined-only "$1" 2> /dev/null)
  [[ $syms == *" $2"* ]]
}
for pair in gemm_a6w6_ja:GemmA6W6JA quant_mxfp6_ja:QuantMxfp6JA; do
  src=${pair%%:*}; target=${pair##*:}
  rm -f "$O/$src.so"   # no stale library survives a failed compile
  PATH=$VENV/bin:$PATH timeout -k 5 600 "$VENV/bin/hipcc" -shared -fPIC -std=c++20 -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
    -DENABLE_CK=1 -fvisibility-inlines-hidden -fvisibility=hidden --offload-arch=gfx950 \
    -I"$A/3rdparty/composable_kernel/include" -I"$A/3rdparty/composable_kernel/library/include" \
    -I"$FFI_INC" -I"$S/csrc/common" -I"$A/csrc/include" -I"$A/csrc/kernels" -I"$G" \
    "$HERE/$src.cu" -o "$O/$src.so" >> "$LOG" 2>&1 ||
    { rm -f "$O/$src.so"; fail "hipcc $src: $(grep -m 3 -E ' error' "$LOG" | cut -c1-300 | tr '\n' '|')"; }
  [ -s "$O/$src.so" ] || fail "no $src.so after the compile"
  exports "$O/$src.so" "$target" || { rm -f "$O/$src.so"; fail "$target is not exported by $src.so"; }
done
echo "MXFP6_BUILD_OK $O"
