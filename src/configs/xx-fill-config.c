// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/fill.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"

static struct xnn_xx_fill_config xx_fill_config = {0};

XNN_INIT_ONCE_GUARD(xx_fill);

static void init_xx_fill_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__neon_u64;
    } else {
      xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__scalar_u16;
    }
  #elif XNN_ARCH_ARM64
    xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__neon_u64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__sse2_u64;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__wasmsimd_u64;
  #elif XNN_ARCH_RISCV
    xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__scalar_u16;
  #else
    xx_fill_config.ukernel = (xnn_fill_ukernel_fn) xnn_xx_fill_ukernel__scalar_u16;
  #endif
}

const struct xnn_xx_fill_config* xnn_init_xx_fill_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(xx_fill);
  return &xx_fill_config;
}
