// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vcmul/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"


void xnn_f32_vcmul_ukernel__scalar_u1(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float* ar = input_a;
  const float* ai = (const float*) ((uintptr_t) input_a + batch);
  const float* br = input_b;
  const float* bi = (const float*) ((uintptr_t) input_b + batch);
  float* or = output;
  float* oi = (float*) ((uintptr_t) output + batch);
  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    const float var = *ar++;
    const float vai = *ai++;
    const float vbr = *br++;
    const float vbi = *bi++;
    const float vaccr = var * vbr - vai * vbi;
    const float vacci = var * vbi + vai * vbr;
    *or++ = vaccr;
    *oi++ = vacci;
  }
}
