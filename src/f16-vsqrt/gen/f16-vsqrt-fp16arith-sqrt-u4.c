// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/fp16arith-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_fp16.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16_t* i = (const float16_t*) input;
  float16_t* o = (float16_t*) output;
  for (; batch >= 4 * sizeof(float16_t); batch -= 4 * sizeof(float16_t)) {
    float16_t vacc0 = *i++;
    float16_t vacc1 = *i++;
    float16_t vacc2 = *i++;
    float16_t vacc3 = *i++;

    vacc0 = vsqrth_f16(vacc0);
    vacc1 = vsqrth_f16(vacc1);
    vacc2 = vsqrth_f16(vacc2);
    vacc3 = vsqrth_f16(vacc3);

    *o++ = vacc0;
    *o++ = vacc1;
    *o++ = vacc2;
    *o++ = vacc3;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float16_t vacc = *i++;
      vacc = vsqrth_f16(vacc);
      *o++ = vacc;
      batch -= sizeof(float16_t);
    } while (batch != 0);
  }
}
