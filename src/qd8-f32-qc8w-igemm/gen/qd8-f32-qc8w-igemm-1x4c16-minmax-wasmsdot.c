// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c16-wasmdot.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c16__wasmsdot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
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

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    v128_t vksum0 = wasm_v128_load32_zero(w);
    v128_t vksum1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vksum2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vksum3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    const v128_t vinput_zero_point = wasm_v128_load32_splat(&quantization_params->zero_point);
    v128_t vacc0x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc0x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc0x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc0x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c0, vacc0x0123);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
