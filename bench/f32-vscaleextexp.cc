#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/raddextexp.h"
#include "src/xnnpack/vscaleextexp.h"
#include <benchmark/benchmark.h>

static void f32_vscaleextexp(
    benchmark::State& state, xnn_f32_raddextexp_ukernel_fn raddextexp,
    xnn_f32_vscaleextexp_ukernel_fn vscaleextexp,
    uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n =
      benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(
      std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> x(elements);
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float scale[2];
    raddextexp(elements * sizeof(float), x.data(), scale);
    const float ext_mantissa = 1.0f / scale[0];
    const float ext_exponent = -scale[1];
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    vscaleextexp(elements * sizeof(float), x.data(),
                 y.data() + packed_n * buffer_index, ext_mantissa,
                 ext_exponent);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_u16,
                  xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u16,
                  xnn_arch_x86_avx512f)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_u32,
                  xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u32,
                  xnn_arch_x86_avx512f)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_u48,
                  xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u48,
                  xnn_arch_x86_avx512f)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_u64,
                  xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u64,
                  xnn_arch_x86_avx512f)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_u8,
                  xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx2_p5_u8,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_u16,
                  xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx2_p5_u16,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_u24,
                  xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx2_p5_u24,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_u32,
                  xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2,
                  xnn_f32_vscaleextexp_ukernel__avx2_p5_u32,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
