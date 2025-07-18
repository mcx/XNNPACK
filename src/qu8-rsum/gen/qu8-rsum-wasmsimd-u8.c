// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

void xnn_qu8_rsum_ukernel__wasmsimd_u8(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const int16_t mask_table[16] = {
    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0, 0, 0, 0, 0, 0, 0, 0
  };

  v128_t vacc0 = wasm_i32x4_const_splat(0);

  // 256 int8s may be summed into an int16 before overflowing.
  // Each register has 8 lanes and there are 1 accumulators so batch size is 2048
  for (; batch >= 2048; batch -= 2048) {
    v128_t vacc16_0 = wasm_i16x8_const_splat(0);
    for (size_t current_batch = 2048; current_batch > 0; current_batch -= 8) {
      const v128_t vt0 = wasm_u16x8_load8x8(input); input += 8;
      vacc16_0 = wasm_i16x8_add(vacc16_0, vt0);
    }
    vacc0 = wasm_i32x4_add(vacc0, wasm_u32x4_extend_low_u16x8(vacc16_0));
    vacc0 = wasm_i32x4_add(vacc0, wasm_u32x4_extend_high_u16x8(vacc16_0));
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 2048);
    v128_t vacc16 = wasm_i16x8_const_splat(0);
    for (; batch >= 8; batch -= 8) {
      const v128_t vt = wasm_u16x8_load8x8(input); input += 8;
      vacc16 = wasm_i16x8_add(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 7);
      const v128_t mask = wasm_v128_load(&mask_table[8 - batch]);
      const v128_t vt = wasm_v128_bitselect(wasm_u16x8_load8x8(input), wasm_i16x8_const_splat(0), mask);
      vacc16 = wasm_i16x8_add(vacc16, vt);
    }
    vacc0 = wasm_i32x4_add(vacc0, wasm_u32x4_extend_low_u16x8(vacc16));
    vacc0 = wasm_i32x4_add(vacc0, wasm_u32x4_extend_high_u16x8(vacc16));
  }

  v128_t vacc_shifted = wasm_i32x4_shuffle(vacc0, vacc0, 1, 0, 3, 2);
  v128_t vacc_lo = wasm_i32x4_add(vacc0, vacc_shifted);

  v128_t vacc_final_shifted = wasm_i32x4_shuffle(vacc_lo, vacc_lo, 2, 3, 0, 1);
  v128_t vacc_final = wasm_i32x4_add(vacc_lo, vacc_final_shifted);
  const int32_t vacc = wasm_u32x4_extract_lane(vacc_final, 0);

  *output += vacc;
}
