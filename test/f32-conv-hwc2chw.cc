// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-conv-hwc2chw.yaml
//   Generator: tools/generate-conv-hwc2chw-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/conv.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "test/conv-hwc2chw-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, input_width_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(4)
      .output_channels(4)
      .input_width(4)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, input_width_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t input_width = 8; input_width <= 32; input_width += 12) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, input_width_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t input_width = 1; input_width < 4; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, input_width_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t input_width = 5; input_width < 8; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, output_channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_channels = 1; output_channels < 4; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, output_channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_channels = 8; output_channels <= 16; output_channels += 4) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, output_channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_channels = 5; output_channels < 8; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__AARCH64_NEONFMA_2X2, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, input_width_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(4)
      .output_channels(4)
      .input_width(4)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, input_width_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t input_width = 8; input_width <= 32; input_width += 12) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, input_width_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t input_width = 1; input_width < 4; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, input_width_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t input_width = 5; input_width < 8; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, output_channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_channels = 1; output_channels < 4; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, output_channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_channels = 8; output_channels <= 16; output_channels += 4) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, output_channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_channels = 5; output_channels < 8; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__NEON_2X2, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, input_width_eq_1) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(4)
      .output_channels(4)
      .input_width(1)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, input_width_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 2; input_width < 33; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, output_channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 4; output_channels++) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, output_channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 8; output_channels <= 16; output_channels += 4) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, output_channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 5; output_channels < 8; output_channels++) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_1X1, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, input_width_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(4)
      .output_channels(4)
      .input_width(4)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, input_width_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 8; input_width <= 32; input_width += 12) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, input_width_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 1; input_width < 4; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, input_width_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 5; input_width < 8; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, output_channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 4; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, output_channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 8; output_channels <= 16; output_channels += 4) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, output_channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 5; output_channels < 8; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SSE_2X2, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, input_width_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(4)
      .output_channels(4)
      .input_width(4)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, input_width_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 8; input_width <= 32; input_width += 12) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, input_width_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 1; input_width < 4; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, input_width_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_width = 5; input_width < 8; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(4)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, output_channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 4; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, output_channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 8; output_channels <= 16; output_channels += 4) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, output_channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 5; output_channels < 8; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(4)
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__WASMSIMD_2X2, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, input_width_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  ConvHWC2CHWMicrokernelTester()
    .kernel_size(3)
    .subsampling(2)
    .padding_width(1)
    .input_channels(3)
    .output_channels_tile(4)
    .output_channels(4)
    .input_width(1)
    .input_height(3)
    .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, input_width_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t input_width = 2; input_width < 33; input_width++) {
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(4)
      .output_channels(4)
      .input_width(input_width)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, output_channels_lt_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_channels = 1; output_channels < 4; output_channels++) {
    for (size_t input_width = 1; input_width < 8; input_width += 1) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, output_channels_div_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_channels = 8; output_channels <= 16; output_channels += 4) {
    for (size_t input_width = 1; input_width < 8; input_width += 1) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, output_channels_gt_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_channels = 5; output_channels < 8; output_channels++) {
    for (size_t input_width = 1; input_width < 8; input_width += 1) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, input_height_lt_3) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t input_height = 1; input_height < 3; input_height++) {
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding(1)  // padded input height of at least 3 required
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(input_height)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, input_height_gt_3) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t input_height = 4; input_height <= 9; input_height++) {
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(input_height)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, padding_top) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
    for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .padding_top(padding_top)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, padding_bottom) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
    for (size_t output_channels = 1; output_channels < 16; output_channels += 7) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .padding_bottom(padding_bottom)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, output_y_start) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .output_y_start(output_y_start)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, output_y_end) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
    for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(4)
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .output_y_end(output_y_end)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, qmin) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
    for (size_t input_width = 1; input_width < 8; input_width += 1) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(6)
        .qmin(128)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_CONV_HWC2CHW_3X3S2P1C3X4__SCALAR_1X1, qmax) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t output_channels = 1; output_channels < 8; output_channels += 3) {
    for (size_t input_width = 1; input_width < 8; input_width += 1) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(4)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(6)
        .qmax(128)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, input_width_eq_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .input_width(1)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, input_width_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 2; input_width < 33; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, output_channels_lt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile; output_channels++) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, output_channels_div_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = output_channels_tile * 2; output_channels <= output_channels_tile * 4; output_channels += output_channels_tile) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, output_channels_gt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = output_channels_tile + 1; output_channels < output_channels_tile * 2; output_channels++) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 4; output_channels += output_channels_tile * 2 - 1) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 4; output_channels += output_channels_tile * 2 - 1) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 8; input_width += 1) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_1X1, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
      for (size_t input_width = 1; input_width < 8; input_width += 1) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, input_width_eq_2) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .input_width(2)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, input_width_div_2) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 4; input_width <= 16; input_width += 6) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, input_width_lt_2) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 1; input_width < 2; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, input_width_gt_2) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 3; input_width < 4; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, output_channels_lt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile; output_channels++) {
      for (size_t input_width = 1; input_width < 16; input_width += 3) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, output_channels_div_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = output_channels_tile * 2; output_channels <= output_channels_tile * 4; output_channels += output_channels_tile) {
      for (size_t input_width = 1; input_width < 16; input_width += 3) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, output_channels_gt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = output_channels_tile + 1; output_channels < output_channels_tile * 2; output_channels++) {
      for (size_t input_width = 1; input_width < 16; input_width += 3) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 16; input_width += 3) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 16; input_width += 3) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 4; output_channels += output_channels_tile * 2 - 1) {
        for (size_t input_width = 1; input_width < 16; input_width += 3) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 4; output_channels += output_channels_tile * 2 - 1) {
        for (size_t input_width = 1; input_width < 16; input_width += 3) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 16; input_width += 3) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 16; input_width += 3) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
      for (size_t input_width = 1; input_width < 16; input_width += 3) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X1, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
      for (size_t input_width = 1; input_width < 16; input_width += 3) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, input_width_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    ConvHWC2CHWMicrokernelTester()
      .kernel_size(3)
      .subsampling(2)
      .padding_width(1)
      .input_channels(3)
      .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .input_width(4)
      .input_height(3)
      .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, input_width_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 8; input_width <= 32; input_width += 12) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, input_width_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 1; input_width < 4; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, input_width_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    for (size_t input_width = 5; input_width < 8; input_width++) {
      ConvHWC2CHWMicrokernelTester()
        .kernel_size(3)
        .subsampling(2)
        .padding_width(1)
        .input_channels(3)
        .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .output_channels(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
        .input_width(input_width)
        .input_height(3)
        .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, output_channels_lt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, output_channels_div_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = output_channels_tile * 2; output_channels <= output_channels_tile * 4; output_channels += output_channels_tile) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, output_channels_gt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = output_channels_tile + 1; output_channels < output_channels_tile * 2; output_channels++) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(3)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, input_height_lt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t input_height = 1; input_height < 3; input_height++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding(1)  // padded input height of at least 3 required
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, input_height_gt_3) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t input_height = 4; input_height <= 9; input_height++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(input_height)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, padding_top) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 4; output_channels += output_channels_tile * 2 - 1) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_top(padding_top)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, padding_bottom) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 4; output_channels += output_channels_tile * 2 - 1) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .padding_bottom(padding_bottom)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, output_y_start) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_start(output_y_start)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, output_y_end) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
      for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
        for (size_t input_width = 1; input_width < 32; input_width += 7) {
          ConvHWC2CHWMicrokernelTester()
            .kernel_size(3)
            .subsampling(2)
            .padding_width(1)
            .input_channels(3)
            .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
            .output_channels(output_channels)
            .input_width(input_width)
            .input_height(9)
            .output_y_end(output_y_end)
            .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, qmin) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmin(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_CONV_HWC2CHW_3X3S2P1C3X2V__RVV_2X2, qmax) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    size_t output_channels_tile = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t output_channels = 1; output_channels < output_channels_tile * 2; output_channels += output_channels_tile - 1) {
      for (size_t input_width = 1; input_width < 32; input_width += 7) {
        ConvHWC2CHWMicrokernelTester()
          .kernel_size(3)
          .subsampling(2)
          .padding_width(1)
          .input_channels(3)
          .output_channels_tile(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(6)
          .qmax(128)
          .Test(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
