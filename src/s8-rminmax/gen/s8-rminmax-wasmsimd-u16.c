// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-rminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// Architecture-specific headers.
#include "src/xnnpack/simd/s8-wasmsimd.h"

// XNNPACK headers.
#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

void xnn_s8_rminmax_ukernel__wasmsimd_u16(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s8 == 16);

  xnn_simd_s8_t vmin0 = xnn_set1_s8(output[0]);
  xnn_simd_s8_t vmax0 = xnn_set1_s8(output[1]);


  for (; batch >= xnn_simd_bytes_s8; batch -= xnn_simd_bytes_s8) {
    xnn_simd_s8_t vt = xnn_loadu_s8(input);
    input += xnn_simd_size_s8;

    vmin0 = xnn_min_s8(vmin0, vt);
    vmax0 = xnn_max_s8(vmax0, vt);
  }

  int8_t min0 = xnn_reduce_min_s8(vmin0);
  int8_t max0 = xnn_reduce_max_s8(vmax0);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const int8_t vt = *input++;

      min0 = vt > min0 ? min0 : vt;
      max0 = vt > max0 ? vt : max0;
    } while (--batch != 0);
  }

  *output = min0;
  output += sizeof(int8_t);
  *output = max0;
}
