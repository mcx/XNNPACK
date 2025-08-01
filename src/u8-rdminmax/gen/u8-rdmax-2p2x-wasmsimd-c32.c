// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-rdminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

#include "src/xnnpack/simd/u8-wasmsimd.h"


void xnn_u8_rdmax_ukernel_2p2x__wasmsimd_c32(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const uint8_t* i0 = input;
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);

    xnn_simd_u8_t vmax0 = xnn_loadu_u8(output);
    xnn_simd_u8_t vmax16 = xnn_loadu_u8(output + 16 * sizeof(uint8_t));;

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      xnn_simd_u8_t vin_0_0 = xnn_loadu_u8(&i0[0]);
      xnn_simd_u8_t vin_0_16 = xnn_loadu_u8(&i0[16]);
      xnn_simd_u8_t vin_1_0 = xnn_loadu_u8(&i1[0]);
      xnn_simd_u8_t vin_1_16 = xnn_loadu_u8(&i1[16]);
      vmax0 = xnn_max_u8(vmax0, vin_0_0);
      vmax16 = xnn_max_u8(vmax16, vin_0_16);
      vmax0 = xnn_max_u8(vmax0, vin_1_0);
      vmax16 = xnn_max_u8(vmax16, vin_1_16);

      i0 += input_increment;
      i1 += input_increment;
    }

    xnn_storeu_u8(output, vmax0);
    output += xnn_simd_bytes_u8;
    xnn_storeu_u8(output, vmax16);
    output += xnn_simd_bytes_u8;

    input += 32 * sizeof(uint8_t);
  }
  if (channels != 0) {
    input_increment = 2 * input_stride;
    do {
      const uint8_t* i0 = input;
      const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);

      xnn_simd_u8_t vmax;

      if (channels >= xnn_simd_size_u8) {
        vmax = xnn_loadu_u8(output);
      } else {
        vmax = xnn_load_tail_safe_u8(output, channels);
      }

      for (int r = rows; r > 0; r -= 2) {
        if XNN_UNPREDICTABLE(r < 2) {
          i1 = i0;
        }
        xnn_simd_u8_t vin0 = xnn_loadu_u8(&i0[0]);
        xnn_simd_u8_t vin1 = xnn_loadu_u8(&i1[0]);
        vmax = xnn_max_u8(vmax, vin0);
        vmax = xnn_max_u8(vmax, vin1);
        i0 += input_increment;
        i1 += input_increment;
      }

      if (channels >= xnn_simd_size_u8) {
        xnn_storeu_u8(output, vmax);
        output += xnn_simd_bytes_u8;
        input += xnn_simd_bytes_u8;
        channels -= xnn_simd_size_u8;
      } else {
        xnn_store_tail_u8(output, vmax, channels);

        channels = 0;
      }
    } while (channels != 0);
  }
}
