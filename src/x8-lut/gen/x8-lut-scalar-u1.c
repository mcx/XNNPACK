// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-lut/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/lut.h"


void xnn_x8_lut_ukernel__scalar_u1(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t* restrict table)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  do {
    const size_t vx = (size_t) *input++;
    const uint32_t vt = (uint32_t) table[vx];
    *output++ = (uint8_t) vt;
    batch -= sizeof(uint8_t);
  } while (batch != 0);
}
