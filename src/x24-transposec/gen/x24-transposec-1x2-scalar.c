// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x24-transposec/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/transpose.h"

void xnn_x24_transposec_ukernel__1x2_scalar(
    const void *input,
    void * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t input_reset = 6 - block_height * input_stride;
  const size_t output_reset = 2 * output_stride - block_height * 3;
  const size_t input_offset = 1 * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      o1[0] = i0[3];
      o1[1] = i0[4];
      o1[2] = i0[5];
      o1 += 3;
      o0[0] = i0[0];
      o0[1] = i0[1];
      o0[2] = i0[2];
      o0 += 3;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, 2);
  } while (block_width != 0);
}
