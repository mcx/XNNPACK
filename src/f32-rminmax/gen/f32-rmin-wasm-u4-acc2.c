// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

void xnn_f32_rmin_ukernel__wasm_u4_acc2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;

  float vmin0 = output[0];
  float vmin1 = vmin0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vt0 = i[0];
    const float vt1 = i[1];
    const float vt2 = i[2];
    const float vt3 = i[3];
    i += 4;

    vmin0 = __builtin_wasm_min_f32(vmin0, vt0);
    vmin1 = __builtin_wasm_min_f32(vmin1, vt1);
    vmin0 = __builtin_wasm_min_f32(vmin0, vt2);
    vmin1 = __builtin_wasm_min_f32(vmin1, vt3);
  }
  vmin0 = __builtin_wasm_min_f32(vmin0, vmin1);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vt = *i++;
      vmin0 = __builtin_wasm_min_f32(vmin0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  output[0] = vmin0;
}
