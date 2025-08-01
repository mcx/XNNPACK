// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "bench/conv.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include <benchmark/benchmark.h>

static void f16_igemm(benchmark::State& state,
                      xnn_f16_igemm_minmax_ukernel_fn igemm,
                      xnn_init_f16_minmax_params_fn init_params, uint32_t mr,
                      uint32_t nr, uint32_t kr, uint32_t sr,
                      uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t kernel_height = state.range(2);
  const size_t kernel_width = state.range(3);
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t padding_height = state.range(4);
  const size_t padding_width = state.range(5);
  const size_t subsampling = state.range(6);
  const size_t dilation = state.range(7);
  const size_t group_input_channels = state.range(8);
  const size_t group_output_channels = state.range(9);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  const size_t output_pixel_stride = group_output_channels;
  const size_t input_pixel_stride = group_input_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height =
      (input_height + padding_height - effective_kernel_height) / subsampling +
      1;
  const size_t output_width =
      (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;

  const size_t mc_stride = benchmark::utils::RoundUp<size_t>(output_size, mr);
  const size_t nc_stride =
      benchmark::utils::RoundUp<size_t>(group_output_channels, nr);
  const size_t kc_stride =
      benchmark::utils::RoundUp<size_t>(group_input_channels, kr * sr);

  xnnpack::Buffer<xnn_float16> a(
      input_height * input_width * input_pixel_stride, xnnpack::XnnExtraBytes);
  std::generate(a.begin(), a.end(), f32rng);
  xnnpack::Buffer<xnn_float16> k(group_output_channels * kernel_height *
                                 kernel_width * group_input_channels);
  std::generate(k.begin(), k.end(), f32rng);
  xnnpack::Buffer<xnn_float16> b(group_output_channels);
  std::generate(b.begin(), b.end(), f32rng);

  xnnpack::Buffer<xnn_float16> z(group_input_channels, xnnpack::XnnExtraBytes);

  const size_t w_elements = (kernel_size * kc_stride + 1) * nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(xnn_float16) * (w_elements + c_elements) +
                  sizeof(void*) * i_elements);

  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> w(w_elements *
                                                           num_buffers);
  xnn_pack_f16_conv_goki_w(
      /*groups=*/1, group_output_channels, kernel_size, group_input_channels,
      nr, kr, sr, reinterpret_cast<const uint16_t*>(k.data()),
      reinterpret_cast<const uint16_t*>(b.data()),
      /*scale=*/nullptr, reinterpret_cast<uint16_t*>(w.data()),
      /*extra_bytes=*/0, /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  xnnpack::Buffer<const xnn_float16*> i(i_elements * num_buffers);
  const size_t tiled_output_size = round_up(output_size, mr);
  xnn_indirection_init_conv2d(
      /*output_tile_size=*/mr,
      /*output_start=*/0,
      /*output_end=*/tiled_output_size,
      reinterpret_cast<const void**>(i.data()), a.data(), z.data(),
      input_pixel_stride << XNN_LOG2_SIZEOF_HALF, input_height, input_width,
      output_height, output_width, kernel_height, kernel_width, subsampling,
      subsampling, dilation, dilation, padding_top, padding_left);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  xnnpack::Buffer<xnn_float16> c(c_elements * num_buffers);

  // Prepare minmax parameters.
  xnn_f16_minmax_params params;
  init_params(&params, static_cast<xnn_float16>(-INFINITY),
              static_cast<xnn_float16>(INFINITY));

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(xnn_float16));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < output_size; m += mr) {
      const uint32_t mb = min(output_size - m, mr);
      for (uint32_t n = 0; n < group_output_channels; n += nr) {
        const uint32_t nb = min(group_output_channels - n, nr);
        igemm(mb, nb, group_input_channels * sizeof(xnn_float16),
              kernel_size * mr * sizeof(void*),
              reinterpret_cast<const xnn_float16**>(i.data()) +
                  buffer_index * i_elements + m,
              w.data() + buffer_index * w_elements +
                  n * (kc_stride * kernel_size + 1),
              c.data() + buffer_index * c_elements + m * group_output_channels +
                  n,
              group_output_channels * sizeof(xnn_float16),
              nr * sizeof(xnn_float16), 0, z.data(), &params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      uint64_t(state.iterations()) * 2 * output_height * output_width *
          group_input_channels * group_output_channels * kernel_height *
          kernel_width,
      benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
static void f16_igemm_6x16__asm_aarch64_neonfp16arith_cortex_a55(
    benchmark::State& state, const char* net) {
  f16_igemm(
      state,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0(
    benchmark::State& state, const char* net) {
  f16_igemm(
      state,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_6x16__asm_aarch64_neonfp16arith_cortex_a75(
    benchmark::State& state, const char* net) {
  f16_igemm(
      state,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_6x16__asm_aarch64_neonfp16arith_ld64(
    benchmark::State& state, const char* net) {
  f16_igemm(state,
            xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_4x16__asm_aarch64_neonfp16arith_ld32(
    benchmark::State& state, const char* net) {
  f16_igemm(state,
            xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_4x16__asm_aarch64_neonfp16arith_ld64(
    benchmark::State& state, const char* net) {
  f16_igemm(state,
            xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_1x16__asm_aarch64_neonfp16arith_ld32(
    benchmark::State& state, const char* net) {
  f16_igemm(state,
            xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_1x16__asm_aarch64_neonfp16arith_ld64(
    benchmark::State& state, const char* net) {
  f16_igemm(state,
            xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}

BENCHMARK_CONV(f16_igemm_6x16__asm_aarch64_neonfp16arith_cortex_a55)
BENCHMARK_CONV(f16_igemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0)
BENCHMARK_CONV(f16_igemm_6x16__asm_aarch64_neonfp16arith_cortex_a75)
BENCHMARK_CONV(f16_igemm_6x16__asm_aarch64_neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_4x16__asm_aarch64_neonfp16arith_ld32)
BENCHMARK_CONV(f16_igemm_4x16__asm_aarch64_neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_1x16__asm_aarch64_neonfp16arith_ld32)
BENCHMARK_CONV(f16_igemm_1x16__asm_aarch64_neonfp16arith_ld64)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
static void f16_igemm_1x8__neonfp16arith_ld64(benchmark::State& state,
                                              const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_4x8__neonfp16arith_ld64(benchmark::State& state,
                                              const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_6x8__neonfp16arith_ld64(benchmark::State& state,
                                              const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_8x8__neonfp16arith_ld64(benchmark::State& state,
                                              const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_1x16__neonfp16arith_ld64(benchmark::State& state,
                                               const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_4x16__neonfp16arith_ld64(benchmark::State& state,
                                               const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_6x16__neonfp16arith_ld64(benchmark::State& state,
                                               const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}
static void f16_igemm_8x16__neonfp16arith_ld64(benchmark::State& state,
                                               const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_arm_neon_fp16_arith);
}

BENCHMARK_CONV(f16_igemm_1x8__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_4x8__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_6x8__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_8x8__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_1x16__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_4x16__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_6x16__neonfp16arith_ld64)
BENCHMARK_CONV(f16_igemm_8x16__neonfp16arith_ld64)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
static void f16_igemm_1x8__avx2_broadcast(benchmark::State& state,
                                          const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_4x8__avx2_broadcast(benchmark::State& state,
                                          const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_5x8__avx2_broadcast(benchmark::State& state,
                                          const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_6x8__avx2_broadcast(benchmark::State& state,
                                          const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_7x8__avx2_broadcast(benchmark::State& state,
                                          const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_1x16__avx2_broadcast(benchmark::State& state,
                                           const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_3x16__avx2_broadcast(benchmark::State& state,
                                           const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_4x16__avx2_broadcast(benchmark::State& state,
                                           const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}
static void f16_igemm_5x16__avx2_broadcast(benchmark::State& state,
                                           const char* net) {
  f16_igemm(state, xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast,
            xnn_init_f16_minmax_scalar_params,
            /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
            xnn_arch_x86_avx2);
}

BENCHMARK_CONV(f16_igemm_1x8__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_4x8__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_5x8__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_6x8__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_7x8__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_1x16__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_3x16__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_4x16__avx2_broadcast)
BENCHMARK_CONV(f16_igemm_5x16__avx2_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
