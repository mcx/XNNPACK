// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench/dconv.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/conv.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include <benchmark/benchmark.h>

static void f32_conv_hwc(
    benchmark::State& state, xnn_f32_conv_hwc_ukernel_fn conv,
    xnn_init_f32_minmax_params_fn init_params, uint32_t output_channels_tile,
    uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t output_channels = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f),
                          std::ref(rng));

  const size_t input_channels = 3;
  const size_t kernel_size = 3;
  const size_t padding = 1;
  const size_t subsampling = 2;

  const size_t output_height =
      (input_height + 2 * padding - kernel_size) / subsampling + 1;
  const size_t output_width =
      (input_width + 2 * padding - kernel_size) / subsampling + 1;

  xnnpack::Buffer<float> input(input_height * input_width * input_channels,
                               xnnpack::XnnExtraBytes);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  xnnpack::Buffer<float> kernel(output_channels * kernel_size * kernel_size *
                                input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  xnnpack::Buffer<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> zero(
      input_channels * input_width, xnnpack::XnnExtraBytes);

  const size_t weights_elements =
      (kernel_size * kernel_size * input_channels + 1) *
      benchmark::utils::RoundUp<size_t>(output_channels, output_channels_tile);
  const size_t output_elements = output_height * output_width * output_channels;
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * (weights_elements + output_elements));

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      weights_elements * num_buffers);
  xnn_pack_f32_dconv_oki_w(
      output_channels, input_channels, output_channels_tile,
      kernel_size /* kernel height */, kernel_size /* kernel width */,
      kernel.data(), bias.data(), packed_weights.data(), nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(packed_weights.cbegin(),
              packed_weights.cbegin() + weights_elements,
              packed_weights.begin() + n * weights_elements);
  }

  xnnpack::Buffer<float> output(output_elements * num_buffers);

  xnn_f32_minmax_params params;
  init_params(&params, -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    conv(input_height, input_width, 0 /* output_y_start */,
         output_height /* output_y_end */, input.data(), zero.data(),
         packed_weights.data() + buffer_index * weights_elements,
         output.data() + buffer_index * output_elements, padding,
         output_channels, output_channels * output_width * sizeof(float),
         output_channels * sizeof(float), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      uint64_t(state.iterations()) * 2 * output_height * output_width *
          input_channels * output_channels * kernel_size * kernel_size,
      benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
static void f32_conv_hwc_3x3s2p1c3x8__aarch64_neonfma_2x1(
    benchmark::State& state, const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__aarch64_neonfma_2x1,
               xnn_init_f32_minmax_scalar_params, 8 /* output channel tile */,
               xnn_arch_arm_neon_fma);
}
static void f32_conv_hwc_3x3s2p1c3x4__aarch64_neonfma_2x1(
    benchmark::State& state, const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x1,
               xnn_init_f32_minmax_scalar_params, 4 /* output channel tile */,
               xnn_arch_arm_neon_fma);
}
static void f32_conv_hwc_3x3s2p1c3x8__aarch64_neonfma_2x2(
    benchmark::State& state, const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__aarch64_neonfma_2x2,
               xnn_init_f32_minmax_scalar_params, 8 /* output channel tile */,
               xnn_arch_arm_neon_fma);
}
static void f32_conv_hwc_3x3s2p1c3x4__aarch64_neonfma_2x2(
    benchmark::State& state, const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2,
               xnn_init_f32_minmax_scalar_params, 4 /* output channel tile */,
               xnn_arch_arm_neon_fma);
}

BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x8__aarch64_neonfma_2x1);
BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x4__aarch64_neonfma_2x1);
BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x8__aarch64_neonfma_2x2);
BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x4__aarch64_neonfma_2x2);
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
static void f32_conv_hwc_3x3s2p1c3x8__neon_2x1(benchmark::State& state,
                                               const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neon_2x1,
               xnn_init_f32_minmax_scalar_params, 8 /* output channel tile */,
               xnn_arch_arm_neon);
}
static void f32_conv_hwc_3x3s2p1c3x4__neon_2x1(benchmark::State& state,
                                               const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neon_2x1,
               xnn_init_f32_minmax_scalar_params, 4 /* output channel tile */,
               xnn_arch_arm_neon);
}
static void f32_conv_hwc_3x3s2p1c3x8__neon_2x2(benchmark::State& state,
                                               const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neon_2x2,
               xnn_init_f32_minmax_scalar_params, 8 /* output channel tile */,
               xnn_arch_arm_neon);
}
static void f32_conv_hwc_3x3s2p1c3x4__neon_2x2(benchmark::State& state,
                                               const char* net) {
  f32_conv_hwc(state, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neon_2x2,
               xnn_init_f32_minmax_scalar_params, 4 /* output channel tile */,
               xnn_arch_arm_neon);
}

BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x8__neon_2x1);
BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x4__neon_2x1);
BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x8__neon_2x2);
BENCHMARK_DCONV(f32_conv_hwc_3x3s2p1c3x4__neon_2x2);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
