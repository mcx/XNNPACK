// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qb4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstdint>
#include <functional>

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"

namespace {

struct ConstantOrFunction {
  ConstantOrFunction(size_t x) : fn([x]() { return x; }) {}  //NOLINT
  ConstantOrFunction(int x) : fn([x]() { return x; }) {}  //NOLINT
  template <typename Fn>
  ConstantOrFunction(Fn fn) : fn(std::move(fn)) {}  //NOLINT

  std::function<size_t()> fn;

  operator size_t() const { return fn(); }  //NOLINT
};

}  // namespace



#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm)
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm)
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__avx2)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__avx2)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__avx2,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__avx2)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
