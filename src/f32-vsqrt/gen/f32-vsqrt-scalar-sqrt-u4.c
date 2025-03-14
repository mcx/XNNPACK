// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/scalar-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vsqrt_ukernel__scalar_sqrt_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vy0 = sqrtf(vx0);
    const float vy1 = sqrtf(vx1);
    const float vy2 = sqrtf(vx2);
    const float vy3 = sqrtf(vx3);

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = sqrtf(vx);
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
