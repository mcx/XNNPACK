// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/scalar-bitcast.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1(
    size_t batch,
    const float* input,
    xnn_float16* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vnonsign_mask = UINT32_C(0x7FFFFFFF);
  const uint32_t vexp_bias = UINT32_C(0x07800000);
  const float vscale_to_inf = 0x1.0p+112f;
  const uint32_t vexpw_max = UINT32_C(0x7F800000);
  const float vscale_to_zero = 0x1.0p-110f;
  const uint32_t vbias_min = UINT32_C(0x40000000);
  const uint16_t vexph_mask = UINT16_C(0x7C00);
  const uint16_t vmanth_mask = UINT16_C(0x0FFF);
  const uint16_t vnanh = UINT16_C(0x7E00);

  const uint32_t* i = (const uint32_t*) input;
  uint16_t* o = (uint16_t*) output;
  do {
    const uint32_t vw = *i++;

    const uint32_t vnonsignw = vw & vnonsign_mask;

    float vf = uint32_as_float(vnonsignw);
    const uint32_t vsignw = vw ^ vnonsignw;
    uint32_t vbias = vnonsignw + vexp_bias;

    vf *= vscale_to_inf;
    vbias &= vexpw_max;

    vf *= vscale_to_zero;
    vbias = math_max_u32(vbias, vbias_min);

    vf += uint32_as_float(vbias);

    const uint32_t vbits = float_as_uint32(vf);

    const uint16_t vexph = (uint16_t) (vbits >> 13) & vexph_mask;
    const uint16_t vmanth = (uint16_t) vbits & vmanth_mask;
    const uint16_t vsignh = (uint16_t) (vsignw >> 16);

    uint16_t vh = vexph + vmanth;
    if XNN_UNPREDICTABLE(vnonsignw > vexpw_max) {
      vh = vnanh;
    }
    vh |= vsignh;

    *o++ = vh;

    batch -= sizeof(float);
  } while (batch != 0);
}
