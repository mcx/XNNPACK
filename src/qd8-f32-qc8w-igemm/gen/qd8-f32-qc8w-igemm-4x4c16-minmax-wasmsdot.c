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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c16__wasmsdot(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

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
    v128_t vacc1x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc1x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc1x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc1x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    v128_t vacc2x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc2x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc2x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc2x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    v128_t vacc3x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc3x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc3x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc3x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 16;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 16;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
        vacc2x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va2, vacc2x0);
        vacc3x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va3, vacc3x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
        vacc2x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va2, vacc2x1);
        vacc3x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va3, vacc3x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
        vacc2x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va2, vacc2x2);
        vacc3x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va3, vacc3x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
        vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);
        vacc2x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va2, vacc2x3);
        vacc3x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va3, vacc3x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc2x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x0, vacc2x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x0, vacc2x2, 2, 6, 3, 7));
    const v128_t vacc2x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x1, vacc2x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x1, vacc2x3, 2, 6, 3, 7));
    const v128_t vacc3x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x0, vacc3x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x0, vacc3x2, 2, 6, 3, 7));
    const v128_t vacc3x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x1, vacc3x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x1, vacc3x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x02, vacc2x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x02, vacc2x13, 2, 6, 3, 7));
    v128_t vacc3x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x02, vacc3x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x02, vacc3x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vinput_scale);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vfilter_output_scale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vbias0123);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);
    vacc2x0123 = wasm_f32x4_pmax(vacc2x0123, vmin);
    vacc3x0123 = wasm_f32x4_pmax(vacc3x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);
    vacc2x0123 = wasm_f32x4_pmin(vacc2x0123, vmax);
    vacc3x0123 = wasm_f32x4_pmin(vacc3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c0, vacc0x0123);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        c3 += 2;
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        c2 += 2;
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        c1 += 2;
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
