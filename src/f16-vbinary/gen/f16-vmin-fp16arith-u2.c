// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-fp16arith.c.in
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
#include "src/xnnpack/vbinary.h"


void xnn_f16_vmin_ukernel__fp16arith_u2(
    size_t batch,
    const xnn_float16* restrict input_a,
    const xnn_float16* restrict input_b,
    xnn_float16* restrict output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float16_t* a = (const float16_t*) input_a;
  const float16_t* b = (const float16_t*) input_b;
  float16_t* o = (float16_t*) output;

  for (; batch >= 2 * sizeof(float16_t); batch -= 2 * sizeof(float16_t)) {
    const float16_t va0 = *a++;
    const float16_t va1 = *a++;

    const float16_t vb0 = *b++;
    const float16_t vb1 = *b++;

    float16_t vacc0 = vminnmh_f16(va0, vb0);
    float16_t vacc1 = vminnmh_f16(va1, vb1);


    *o++ = vacc0;
    *o++ = vacc1;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16_t va = *a;
    const float16_t vb = *b;
    float16_t vacc = vminnmh_f16(va, vb);
    *o = vacc;
  }
}
