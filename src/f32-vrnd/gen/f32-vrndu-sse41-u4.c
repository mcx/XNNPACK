// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <smmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vrndu_ukernel__sse41_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    input += 4;

    const __m128 vy0123 = _mm_round_ps(vx0123, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy0123);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
