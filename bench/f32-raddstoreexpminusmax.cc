// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/raddstoreexpminusmax.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/buffer.h"
#include <benchmark/benchmark.h>

static void f32_raddstoreexpminusmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_fn rmax,
  xnn_f32_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax,
  xnn_init_f32_expminus_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_elements * sizeof(float));
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> x(elements);
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> y(packed_elements * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float x_max = -std::numeric_limits<float>::infinity();
    rmax(elements * sizeof(float), x.data(), &x_max, /*params=*/nullptr);
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    float y_sum;
    raddstoreexpminusmax(elements * sizeof(float), x.data(), &x_max, y.data() + buffer_index * packed_elements, &y_sum, nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_u4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_u8_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_u16_acc4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_u4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_u8_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_u16_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_u16_acc4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4,
                    nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_u4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_u8_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_u16_acc4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();


  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_u4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_u8_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_u16_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_u16_acc4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4,
                    nullptr,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, rvv_rr2_p6_u2v,
                    xnn_f32_rmax_ukernel__rvv_u8v,
                    xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v,
                    nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, rvv_rr2_p6_u4v,
                    xnn_f32_rmax_ukernel__rvv_u8v,
                    xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v,
                    nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx256skx_rr2_p5_u8,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u8,
                    nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx256skx_rr2_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx256skx_rr2_p5_u32_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx256skx_rr2_p5_u32_acc4,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc4,
                    nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_u16,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u16,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_u32_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u32_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_u64_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_u64_acc4,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr2_p5_u16,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u16,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr2_p5_u32_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u32_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr2_p5_u64_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr2_p5_u64_acc4,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc4,
                    nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_u8,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u8,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_u32_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_u32_acc4,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc4,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr2_p5_u8,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u8,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr2_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u16_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr2_p5_u32_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr2_p5_u32_acc4,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc4,
                    nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_u4,
                    xnn_f32_rmax_ukernel__sse_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_u8_acc2,
                    xnn_f32_rmax_ukernel__sse_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__sse_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_u16_acc4,
                    xnn_f32_rmax_ukernel__sse_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_u4,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_u8_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_u16_acc4,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_u4,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_u8_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_u16_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_u16_acc4,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4,
                    nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_u1,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_u2_acc2,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_u4_acc2,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_u4_acc4,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_u1,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u1,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_u2_acc2,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_u4_acc2,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_u4_acc4,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4,
                  nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
