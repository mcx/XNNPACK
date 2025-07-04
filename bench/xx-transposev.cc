// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/transpose.h"
#include <benchmark/benchmark.h>

void transpose(benchmark::State& state, xnn_transposev_ukernel_fn transpose,
               uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }
  const size_t height = state.range(0);
  const size_t width = state.range(1);
  const size_t element_size = state.range(2);
  const size_t tile_hbytes = height * 3;
  const size_t tile_wbytes = width * 3;

  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> x(
      height * width * element_size, xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> y(
      height * width * element_size, xnnpack::XnnExtraBytes);
  std::iota(x.begin(), x.end(), 0);

  for (auto _ : state) {
    transpose(x.data(), y.data(), tile_wbytes, tile_hbytes, element_size,
              element_size, element_size, width, height);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b) {
  b->ArgNames({"height", "width", "element_size"});
  // b->Args({32, 32, 3});
  // b->Args({64, 64, 3});
  // b->Args({117, 117, 3});
  // b->Args({1024, 1024, 3});
  b->Args({49153, 8, 128});
}

BENCHMARK_CAPTURE(transpose, 1x1_scalar_memcpy,
                  xnn_xx_transposev_ukernel__1x1_scalar_memcpy)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
