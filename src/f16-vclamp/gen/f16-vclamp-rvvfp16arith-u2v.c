// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vclamp/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vclamp_ukernel__rvvfp16arith_u2v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(_Float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const _Float16 vmin = *(const _Float16*) &params->scalar.min;
  const _Float16 vmax = *(const _Float16*) &params->scalar.max;

  batch >>= XNN_LOG2_SIZEOF_HALF;
  do {
    const size_t n = __riscv_vsetvl_e16m2(batch);
    vfloat16m2_t vacc = __riscv_vle16_v_f16m2((const void*) i, n);
    i += n;
    vacc = __riscv_vfmax_vf_f16m2(vacc, vmin, n);
    vacc = __riscv_vfmin_vf_f16m2(vacc, vmax, n);
    __riscv_vse16_v_f16m2((void*) o, vacc, n);
    o += n;

    batch -= n;
  } while (batch != 0);
}
