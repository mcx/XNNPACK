// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));

  const v128_t vmagic_bias = wasm_f32x4_const_splat(12582912.0f);
  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const v128_t vmagic_min = wasm_i32x4_splat((int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point));
  const v128_t vmagic_bias_less_output_zero_point = wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point);
  const v128_t voutput_max = wasm_i8x16_splat(params->fp32_scalar.output_max);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vmagic_min);
  XNN_FORCE_REALIZATION(vmagic_bias_less_output_zero_point);
  XNN_FORCE_REALIZATION(voutput_max);

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        v128_t vxa0 = wasm_i16x8_load8x8(a0);
        a0 += 8;

        const v128_t vb01 = wasm_v128_load(w);
        const v128_t vxb0 = wasm_i16x8_extend_low_i8x16(vb01);
        const v128_t vxb1 = wasm_i16x8_extend_high_i8x16(vb01);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb0));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb1));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vb23 = wasm_v128_load((const int8_t*) w + 16);
        const v128_t vxb2 = wasm_i16x8_extend_low_i8x16(vb23);
        const v128_t vxb3 = wasm_i16x8_extend_high_i8x16(vb23);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb2));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);

    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c0, vout, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vout, 0);
        c0 += 2;

        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
