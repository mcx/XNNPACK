// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc8w-gemm-minmax-fp32.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_PRFM_CORTEX_A35, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_PRFM_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_PRFM_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C16__ASM_AARCH64_NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_eq_4) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(4)
      .sr(1)
      .m(1)
      .n(1)
      .k(4)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, strided_cn) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(4)
      .sr(1)
      .m(1)
      .n(1)
      .k(4)
      .cn_stride(3)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(4)
      .sr(1)
      .m(1)
      .n(1)
      .k(4)
      .a_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 1; n <= 1; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(4)
        .sr(1)
        .m(m)
        .n(1)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 1; n <= 1; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_lt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(4)
        .sr(1)
        .m(1)
        .n(1)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(4)
        .sr(1)
        .m(1)
        .n(1)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 1; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(1)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_gt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(4)
        .sr(1)
        .m(1)
        .n(1)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 1; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(1)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_div_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(4)
        .sr(1)
        .m(1)
        .n(1)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, k_div_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 1; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(1)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_gt_1) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n < 2; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_gt_1_strided_cn) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n < 2; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(3)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_gt_1_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n < 2; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_gt_1_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n < 2; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(1)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_div_1) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n <= 3; n += 1) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_div_1_strided_cn) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n <= 3; n += 1) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(3)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_div_1_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n <= 3; n += 1) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, n_div_1_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 2; n <= 3; n += 1) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(1)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 1; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(1)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(3)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X1C4__ARMSIMD32, strided_cm) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(4)
      .sr(1)
      .m(1)
      .n(1)
      .k(4)
      .cm_stride(3)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_eq_4) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(4)
      .sr(1)
      .m(2)
      .n(2)
      .k(4)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, strided_cn) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(4)
      .sr(1)
      .m(2)
      .n(2)
      .k(4)
      .cn_stride(5)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(4)
      .sr(1)
      .m(2)
      .n(2)
      .k(4)
      .a_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(4)
        .sr(1)
        .m(m)
        .n(2)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_lt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(4)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(4)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_gt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(4)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_div_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(4)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, k_div_4_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_gt_2) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_gt_2_strided_cn) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_gt_2_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_gt_2_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_div_2) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_div_2_strided_cn) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_div_2_strided_a) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, n_div_2_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2C4__ARMSIMD32, strided_cm) {
    TEST_REQUIRES_ARM_SIMD32;
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(4)
      .sr(1)
      .m(2)
      .n(2)
      .k(4)
      .cm_stride(5)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32, xnn_init_qc8_conv_minmax_fp32_armsimd32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16__NEONV8_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_6X8C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_8X8C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot, xnn_init_qc8_conv_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_eq_8) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, strided_cn) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD64, strided_cm) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__XOP_LD64, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE2_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__XOP_LD128, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__XOP_LD64, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_eq_8) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, strided_cn) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(4)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, strided_cm) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_eq_8) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, strided_cn) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD64, strided_cm) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__XOP_LD64, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_eq_8) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, strided_cn) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD64, strided_cm) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD64, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__SSE2_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__SSE2_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__SSE2_LD128, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, xnn_init_qc8_conv_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_eq_8) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, strided_cn) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__AVX_LD128, strided_cm) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_eq_8) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, strided_cn) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__AVX_LD128, strided_cm) {
    TEST_REQUIRES_X86_AVX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__XOP_LD128, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_eq_8) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, strided_cn) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, k_div_8_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_gt_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_gt_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_gt_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_div_4) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_div_4_strided_a) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, n_div_4_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, strided_cm_subtile) {
    TEST_REQUIRES_X86_XOP;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__XOP_LD128, strided_cm) {
    TEST_REQUIRES_X86_XOP;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, xnn_init_qc8_conv_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX2, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_1X8C8__AVX2, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .extended_weights(true)
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .extended_weights(true)
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .extended_weights(true)
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_XW_MINMAX_FP32_2X8C8__AVX2, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .extended_weights(true)
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_qc8w_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qc8_conv_minmax_fp32_avx2_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qc8_conv_minmax_fp32_avx512_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_eq_8) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_eq_8) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_eq_8) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, strided_cn) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, strided_cm) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_eq_8) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, strided_cn) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD128, strided_cm) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, strided_cn) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(2)
        .sr(4)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, strided_cm) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(2)
      .sr(4)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_eq_8) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, strided_cn) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_eq_8_strided_a) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile_m) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_eq_8_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_lt_8_strided_a) {
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_lt_8_subtile) {
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_gt_8) {
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_gt_8_subtile) {
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_div_8) {
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, k_div_8_subtile) {
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, strided_cm_subtile) {
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, strided_cm) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_eq_16) {
    TEST_REQUIRES_WASM_SDOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(16)
      .sr(1)
      .m(2)
      .n(4)
      .k(16)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, strided_cn) {
    TEST_REQUIRES_WASM_SDOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(16)
      .sr(1)
      .m(2)
      .n(4)
      .k(16)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_eq_16_strided_a) {
    TEST_REQUIRES_WASM_SDOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(16)
      .sr(1)
      .m(2)
      .n(4)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_eq_16_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_eq_16_subtile_m) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(16)
        .sr(1)
        .m(m)
        .n(4)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_eq_16_subtile_n) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(16)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_lt_16) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(16)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_lt_16_strided_a) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(16)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_lt_16_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_gt_16) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(16)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_gt_16_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_div_16) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(16)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, k_div_16_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_gt_4) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_gt_4_strided_cn) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_gt_4_strided_a) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_gt_4_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_div_4) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_div_4_strided_cn) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_div_4_strided_a) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, n_div_4_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, strided_cm_subtile) {
    TEST_REQUIRES_WASM_SDOT;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4C16__WASMSDOT, strided_cm) {
    TEST_REQUIRES_WASM_SDOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(16)
      .sr(1)
      .m(2)
      .n(4)
      .k(16)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot, xnn_init_qc8_conv_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_eq_1) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(1)
      .cn_stride(5)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(1)
      .a_stride(3)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_gt_2) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_gt_2_strided_cn) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_gt_2_strided_a) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_gt_2_subtile) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_div_2) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_div_2_strided_cn) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_div_2_strided_a) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, n_div_2_subtile) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__WASM_FMAGIC, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(1)
      .cm_stride(5)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_eq_1) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .cn_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .a_stride(3)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__WASM_FMAGIC, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .cm_stride(7)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_eq_1) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(1)
      .cn_stride(5)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(1)
      .a_stride(3)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_gt_2) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_gt_2_strided_cn) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_gt_2_strided_a) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_gt_2_subtile) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_div_2) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_div_2_strided_cn) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_div_2_strided_a) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, n_div_2_subtile) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X2__WASM_FMAGIC, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(1)
      .cm_stride(5)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_eq_1) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}


TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_eq_1) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}


TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf, xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}


TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_eq_1) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, strided_cn) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 3; m++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(3)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(3)
      .n(4)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_3X4__SCALAR_FMAGIC, strided_cm) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}


TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_FMAGIC, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}


TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic, xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}


TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
      }
    }
  }
}

TEST(QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic, xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}
