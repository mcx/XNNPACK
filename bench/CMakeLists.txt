# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---[ Build google benchmark
IF(NOT TARGET benchmark)
  IF(XNNPACK_USE_SYSTEM_LIBS)
    FIND_PACKAGE(benchmark REQUIRED)
  ELSE()
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark")
  ENDIF()
ENDIF()

ADD_LIBRARY(xnnpack-bench-utils STATIC utils.cc)
TARGET_LINK_LIBRARIES(xnnpack-bench-utils PRIVATE benchmark::benchmark cpuinfo pthreadpool)
TARGET_LINK_LIBRARIES(xnnpack-bench-utils PRIVATE xnnpack-base xnnpack-hardware-config)
IF(XNNPACK_BUILD_LIBRARY)
  TARGET_LINK_LIBRARIES(xnnpack-bench-utils PRIVATE xnnpack-logging xnnpack-memory)
ENDIF()

# Helper libraries
ADD_LIBRARY(packq-benchmark STATIC packq-benchmark.cc)
TARGET_LINK_LIBRARIES(packq-benchmark PRIVATE XNNPACK benchmark::benchmark xnnpack-bench-utils)
IF(XNNPACK_ENABLE_KLEIDIAI)
  TARGET_LINK_LIBRARIES(packq-benchmark PRIVATE kleidiai)
ENDIF()

ADD_LIBRARY(gemm-benchmark STATIC gemm-benchmark.cc)
TARGET_LINK_LIBRARIES(gemm-benchmark PRIVATE XNNPACK benchmark::benchmark xnnpack-bench-utils)
IF(XNNPACK_ENABLE_KLEIDIAI)
  TARGET_LINK_LIBRARIES(gemm-benchmark PUBLIC kleidiai)
ENDIF()

ADD_SUBDIRECTORY(subgraph)
ADD_SUBDIRECTORY(operators)

# ---[ Build microkernel-level microbenchmarks
SET(MICROKERNEL_BENCHMARKS
    bf16-gemm
    f16-conv-hwc2chw
    f16-dwconv
    f16-dwconv2d-chw
    f16-f32acc-gemm
    f16-f32acc-igemm
    f16-gemm
    f16-gemm-minmax
    f16-igemm
    f16-raddstoreexpminusmax
    f32-bgemm
    f32-conv-hwc
    f32-conv-hwc2chw
    f32-dwconv
    f32-dwconv2d-chw
    f32-gemm
    f32-gemm-minmax
    f32-igemm
    f32-qc4w-gemm
    f32-qc8w-gemm
    f32-raddexpminusmax
    f32-raddextexp
    f32-raddstoreexpminusmax
    f32-softmax
    f16-vcmul
    f32-vcmul
    f32-vscaleexpminusmax
    f32-vscaleextexp
    pf16-gemm-minmax
    pf32-gemm-minmax
    qb4-packw
    qd8-f16-qb4w-gemm
    qd8-f16-qc4w-gemm
    qd8-f16-qc8w-gemm
    qd8-f32-qb4w-gemm
    qd8-f32-qc4w-gemm
    qd8-f32-qc8w-gemm
    qp8-f32-qc4w-gemm
    qp8-f32-qc8w-gemm
    qp8-f32-qb4w-gemm
    qs8-dwconv
    qs8-gemm
    qs8-qc4w-gemm-fp32
    qs8-qc8w-gemm-fp32
    qu8-gemm
    qu8-gemm-fp32
    qu8-gemm-rndnu
    rdminmax
    rdsum
    rminmax
    rsum
    spmm
    x16-packw
    x32-packw
    x8-lut
    x8-packq
    x8-packw
    vunary
    vbinary
    xN-transposec
    xx-transposev)
FOREACH(BENCH ${MICROKERNEL_BENCHMARKS})
  ADD_EXECUTABLE(${BENCH}-bench ${BENCH}.cc)
  IF(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND NOT XNNPACK_TARGET_PROCESSOR MATCHES "^riscv")
    # Attempt to work around slow compilation/linking of benchmarks with a lot of functions
    TARGET_COMPILE_OPTIONS(${BENCH}-bench PRIVATE "-fno-function-sections")
    TARGET_COMPILE_OPTIONS(${BENCH}-bench PRIVATE "-Os")
  ENDIF()
  TARGET_LINK_LIBRARIES(${BENCH}-bench PRIVATE
    xnnpack-bench-utils
    benchmark::benchmark
    gemm-benchmark
    xnnpack-hardware-config
    xnnpack-indirection
    xnnpack-logging
    xnnpack-microkernels-all
    xnnpack-microparams-init
    xnnpack-packing
    packq-benchmark
    pthreadpool)
  SET_TARGET_PROPERTIES(${BENCH}-bench PROPERTIES CXX_EXTENSIONS YES)
ENDFOREACH()

# Special-case
IF(XNNPACK_ENABLE_KLEIDIAI)
  TARGET_LINK_LIBRARIES(x8-packq-bench PRIVATE kleidiai)
ENDIF()
