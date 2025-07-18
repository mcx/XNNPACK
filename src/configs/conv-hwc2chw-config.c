// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/conv.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"

static struct xnn_conv_hwc2chw_config f16_conv_hwc2chw_3x3c3s2_config = {0};
static struct xnn_conv_hwc2chw_config f32_conv_hwc2chw_3x3c3s2_config = {0};

XNN_INIT_ONCE_GUARD(f16_conv_hwc2chw_3x3c3s2);
XNN_INIT_ONCE_GUARD(f32_conv_hwc2chw_3x3c3s2);

// Macros to log the microkernel names if and when they are registered.
#define XNN_INIT_CONV_UKERNEL(ukernel) \
  (xnn_conv_hwc2chw_ukernel_fn) ukernel;    \
  xnn_log_info("Using conv_hwc2chw microkernel '%s'.", #ukernel);

static void init_f16_conv_hwc2chw_3x3c3s2_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith)) {
      f16_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        XNN_INIT_CONV_UKERNEL(xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2);
      f16_conv_hwc2chw_3x3c3s2_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f16_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith)) {
      f16_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        XNN_INIT_CONV_UKERNEL(xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2);
      f16_conv_hwc2chw_3x3c3s2_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f16_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
    }
  #endif
}

static void init_f32_conv_hwc2chw_3x3c3s2_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2);
      f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
    } else {
      f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1);
      f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2);
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2);
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2);
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2);
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 2 * hardware_config->vlenb / sizeof(float);
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
  #else
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      XNN_INIT_CONV_UKERNEL(xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1);
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 1;
  #endif
}

const struct xnn_conv_hwc2chw_config* xnn_init_f16_conv_hwc2chw_3x3c3s2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_conv_hwc2chw_3x3c3s2);
  return &f16_conv_hwc2chw_3x3c3s2_config;
}

const struct xnn_conv_hwc2chw_config* xnn_init_f32_conv_hwc2chw_3x3c3s2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_conv_hwc2chw_3x3c3s2);
  return &f32_conv_hwc2chw_3x3c3s2_config;
}
