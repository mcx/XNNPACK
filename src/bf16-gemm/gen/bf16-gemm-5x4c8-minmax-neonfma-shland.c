// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-gemm/c8-neon-shland.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_bfloat16* restrict a,
    size_t a_stride,
    const xnn_bfloat16* restrict w_ptr,
    xnn_bfloat16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_bf16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }

  const uint16_t* w = (const uint16_t*) w_ptr;
  const uint16x8_t vmask = vreinterpretq_u16_u32(vmovq_n_u32(UINT32_C(0xFFFF0000)));
  do {
    float32x4_t vacc0x0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc1x2 = vacc0x2;
    float32x4_t vacc1x3 = vacc0x3;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc2x2 = vacc0x2;
    float32x4_t vacc2x3 = vacc0x3;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc3x2 = vacc0x2;
    float32x4_t vacc3x3 = vacc0x3;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc4x2 = vacc0x2;
    float32x4_t vacc4x3 = vacc0x3;

    size_t k = kc;
    for (; k >= 8 * sizeof(uint16_t); k -= 8 * sizeof(uint16_t)) {
      const uint16x8_t va0 = vld1q_u16(a0); a0 += 8;
      const uint16x8_t va1 = vld1q_u16(a1); a1 += 8;
      const uint16x8_t va2 = vld1q_u16(a2); a2 += 8;
      const uint16x8_t va3 = vld1q_u16(a3); a3 += 8;
      const uint16x8_t va4 = vld1q_u16(a4); a4 += 8;

      const uint16x8_t vb0 = vld1q_u16(w); w += 8;
      const uint16x8_t vb1 = vld1q_u16(w); w += 8;
      const uint16x8_t vb2 = vld1q_u16(w); w += 8;
      const uint16x8_t vb3 = vld1q_u16(w); w += 8;

      const float32x4_t va0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0), 16));
      const float32x4_t va1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va1), 16));
      const float32x4_t va2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va2), 16));
      const float32x4_t va3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va3), 16));
      const float32x4_t va4e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va4), 16));

      const float32x4_t vb0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb0), 16));
      const float32x4_t vb1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb1), 16));
      const float32x4_t vb2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb2), 16));
      const float32x4_t vb3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb3), 16));

      vacc0x0 = vfmaq_f32(vacc0x0, va0e, vb0e);
      vacc1x0 = vfmaq_f32(vacc1x0, va1e, vb0e);
      vacc2x0 = vfmaq_f32(vacc2x0, va2e, vb0e);
      vacc3x0 = vfmaq_f32(vacc3x0, va3e, vb0e);
      vacc4x0 = vfmaq_f32(vacc4x0, va4e, vb0e);
      vacc0x1 = vfmaq_f32(vacc0x1, va0e, vb1e);
      vacc1x1 = vfmaq_f32(vacc1x1, va1e, vb1e);
      vacc2x1 = vfmaq_f32(vacc2x1, va2e, vb1e);
      vacc3x1 = vfmaq_f32(vacc3x1, va3e, vb1e);
      vacc4x1 = vfmaq_f32(vacc4x1, va4e, vb1e);
      vacc0x2 = vfmaq_f32(vacc0x2, va0e, vb2e);
      vacc1x2 = vfmaq_f32(vacc1x2, va1e, vb2e);
      vacc2x2 = vfmaq_f32(vacc2x2, va2e, vb2e);
      vacc3x2 = vfmaq_f32(vacc3x2, va3e, vb2e);
      vacc4x2 = vfmaq_f32(vacc4x2, va4e, vb2e);
      vacc0x3 = vfmaq_f32(vacc0x3, va0e, vb3e);
      vacc1x3 = vfmaq_f32(vacc1x3, va1e, vb3e);
      vacc2x3 = vfmaq_f32(vacc2x3, va2e, vb3e);
      vacc3x3 = vfmaq_f32(vacc3x3, va3e, vb3e);
      vacc4x3 = vfmaq_f32(vacc4x3, va4e, vb3e);

      const float32x4_t va0o = vreinterpretq_f32_u16(vandq_u16(va0, vmask));
      const float32x4_t va1o = vreinterpretq_f32_u16(vandq_u16(va1, vmask));
      const float32x4_t va2o = vreinterpretq_f32_u16(vandq_u16(va2, vmask));
      const float32x4_t va3o = vreinterpretq_f32_u16(vandq_u16(va3, vmask));
      const float32x4_t va4o = vreinterpretq_f32_u16(vandq_u16(va4, vmask));

      const float32x4_t vb0o = vreinterpretq_f32_u16(vandq_u16(vb0, vmask));
      const float32x4_t vb1o = vreinterpretq_f32_u16(vandq_u16(vb1, vmask));
      const float32x4_t vb2o = vreinterpretq_f32_u16(vandq_u16(vb2, vmask));
      const float32x4_t vb3o = vreinterpretq_f32_u16(vandq_u16(vb3, vmask));

      vacc0x0 = vfmaq_f32(vacc0x0, va0o, vb0o);
      vacc1x0 = vfmaq_f32(vacc1x0, va1o, vb0o);
      vacc2x0 = vfmaq_f32(vacc2x0, va2o, vb0o);
      vacc3x0 = vfmaq_f32(vacc3x0, va3o, vb0o);
      vacc4x0 = vfmaq_f32(vacc4x0, va4o, vb0o);
      vacc0x1 = vfmaq_f32(vacc0x1, va0o, vb1o);
      vacc1x1 = vfmaq_f32(vacc1x1, va1o, vb1o);
      vacc2x1 = vfmaq_f32(vacc2x1, va2o, vb1o);
      vacc3x1 = vfmaq_f32(vacc3x1, va3o, vb1o);
      vacc4x1 = vfmaq_f32(vacc4x1, va4o, vb1o);
      vacc0x2 = vfmaq_f32(vacc0x2, va0o, vb2o);
      vacc1x2 = vfmaq_f32(vacc1x2, va1o, vb2o);
      vacc2x2 = vfmaq_f32(vacc2x2, va2o, vb2o);
      vacc3x2 = vfmaq_f32(vacc3x2, va3o, vb2o);
      vacc4x2 = vfmaq_f32(vacc4x2, va4o, vb2o);
      vacc0x3 = vfmaq_f32(vacc0x3, va0o, vb3o);
      vacc1x3 = vfmaq_f32(vacc1x3, va1o, vb3o);
      vacc2x3 = vfmaq_f32(vacc2x3, va2o, vb3o);
      vacc3x3 = vfmaq_f32(vacc3x3, va3o, vb3o);
      vacc4x3 = vfmaq_f32(vacc4x3, va4o, vb3o);
    }
    if XNN_UNLIKELY(k != 0) {
      const uint16x8_t va0 = vld1q_u16(a0); a0 = (const uint16_t*) ((uintptr_t) a0 + k);
      const uint16x8_t va1 = vld1q_u16(a1); a1 = (const uint16_t*) ((uintptr_t) a1 + k);
      const uint16x8_t va2 = vld1q_u16(a2); a2 = (const uint16_t*) ((uintptr_t) a2 + k);
      const uint16x8_t va3 = vld1q_u16(a3); a3 = (const uint16_t*) ((uintptr_t) a3 + k);
      const uint16x8_t va4 = vld1q_u16(a4); a4 = (const uint16_t*) ((uintptr_t) a4 + k);

      const uint16x8_t vb0 = vld1q_u16(w); w += 8;
      const uint16x8_t vb1 = vld1q_u16(w); w += 8;
      const uint16x8_t vb2 = vld1q_u16(w); w += 8;
      const uint16x8_t vb3 = vld1q_u16(w); w += 8;

      const uint16x8_t vm0 = vceqq_u16(vb0, vmovq_n_u16(0));
      const uint16x8_t vm1 = vceqq_u16(vb1, vmovq_n_u16(0));
      const uint16x8_t vm2 = vceqq_u16(vb2, vmovq_n_u16(0));
      const uint16x8_t vm3 = vceqq_u16(vb3, vmovq_n_u16(0));

      const float32x4_t vb0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb0), 16));
      const float32x4_t vb1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb1), 16));
      const float32x4_t vb2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb2), 16));
      const float32x4_t vb3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb3), 16));

      const uint16x8_t va0x0 = vbicq_u16(va0, vm0);
      const uint16x8_t va1x0 = vbicq_u16(va1, vm0);
      const uint16x8_t va2x0 = vbicq_u16(va2, vm0);
      const uint16x8_t va3x0 = vbicq_u16(va3, vm0);
      const uint16x8_t va4x0 = vbicq_u16(va4, vm0);
      const uint16x8_t va0x1 = vbicq_u16(va0, vm1);
      const uint16x8_t va1x1 = vbicq_u16(va1, vm1);
      const uint16x8_t va2x1 = vbicq_u16(va2, vm1);
      const uint16x8_t va3x1 = vbicq_u16(va3, vm1);
      const uint16x8_t va4x1 = vbicq_u16(va4, vm1);
      const uint16x8_t va0x2 = vbicq_u16(va0, vm2);
      const uint16x8_t va1x2 = vbicq_u16(va1, vm2);
      const uint16x8_t va2x2 = vbicq_u16(va2, vm2);
      const uint16x8_t va3x2 = vbicq_u16(va3, vm2);
      const uint16x8_t va4x2 = vbicq_u16(va4, vm2);
      const uint16x8_t va0x3 = vbicq_u16(va0, vm3);
      const uint16x8_t va1x3 = vbicq_u16(va1, vm3);
      const uint16x8_t va2x3 = vbicq_u16(va2, vm3);
      const uint16x8_t va3x3 = vbicq_u16(va3, vm3);
      const uint16x8_t va4x3 = vbicq_u16(va4, vm3);

      const float32x4_t va0x0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x0), 16));
      const float32x4_t va1x0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va1x0), 16));
      const float32x4_t va2x0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va2x0), 16));
      const float32x4_t va3x0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va3x0), 16));
      const float32x4_t va4x0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va4x0), 16));
      const float32x4_t va0x1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x1), 16));
      const float32x4_t va1x1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va1x1), 16));
      const float32x4_t va2x1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va2x1), 16));
      const float32x4_t va3x1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va3x1), 16));
      const float32x4_t va4x1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va4x1), 16));
      const float32x4_t va0x2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x2), 16));
      const float32x4_t va1x2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va1x2), 16));
      const float32x4_t va2x2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va2x2), 16));
      const float32x4_t va3x2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va3x2), 16));
      const float32x4_t va4x2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va4x2), 16));
      const float32x4_t va0x3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x3), 16));
      const float32x4_t va1x3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va1x3), 16));
      const float32x4_t va2x3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va2x3), 16));
      const float32x4_t va3x3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va3x3), 16));
      const float32x4_t va4x3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va4x3), 16));

      vacc0x0 = vfmaq_f32(vacc0x0, va0x0e, vb0e);
      vacc1x0 = vfmaq_f32(vacc1x0, va1x0e, vb0e);
      vacc2x0 = vfmaq_f32(vacc2x0, va2x0e, vb0e);
      vacc3x0 = vfmaq_f32(vacc3x0, va3x0e, vb0e);
      vacc4x0 = vfmaq_f32(vacc4x0, va4x0e, vb0e);
      vacc0x1 = vfmaq_f32(vacc0x1, va0x1e, vb1e);
      vacc1x1 = vfmaq_f32(vacc1x1, va1x1e, vb1e);
      vacc2x1 = vfmaq_f32(vacc2x1, va2x1e, vb1e);
      vacc3x1 = vfmaq_f32(vacc3x1, va3x1e, vb1e);
      vacc4x1 = vfmaq_f32(vacc4x1, va4x1e, vb1e);
      vacc0x2 = vfmaq_f32(vacc0x2, va0x2e, vb2e);
      vacc1x2 = vfmaq_f32(vacc1x2, va1x2e, vb2e);
      vacc2x2 = vfmaq_f32(vacc2x2, va2x2e, vb2e);
      vacc3x2 = vfmaq_f32(vacc3x2, va3x2e, vb2e);
      vacc4x2 = vfmaq_f32(vacc4x2, va4x2e, vb2e);
      vacc0x3 = vfmaq_f32(vacc0x3, va0x3e, vb3e);
      vacc1x3 = vfmaq_f32(vacc1x3, va1x3e, vb3e);
      vacc2x3 = vfmaq_f32(vacc2x3, va2x3e, vb3e);
      vacc3x3 = vfmaq_f32(vacc3x3, va3x3e, vb3e);
      vacc4x3 = vfmaq_f32(vacc4x3, va4x3e, vb3e);

      const float32x4_t vb0o = vreinterpretq_f32_u16(vandq_u16(vb0, vmask));
      const float32x4_t vb1o = vreinterpretq_f32_u16(vandq_u16(vb1, vmask));
      const float32x4_t vb2o = vreinterpretq_f32_u16(vandq_u16(vb2, vmask));
      const float32x4_t vb3o = vreinterpretq_f32_u16(vandq_u16(vb3, vmask));

      const float32x4_t va0x0o = vreinterpretq_f32_u16(vandq_u16(va0x0, vmask));
      const float32x4_t va1x0o = vreinterpretq_f32_u16(vandq_u16(va1x0, vmask));
      const float32x4_t va2x0o = vreinterpretq_f32_u16(vandq_u16(va2x0, vmask));
      const float32x4_t va3x0o = vreinterpretq_f32_u16(vandq_u16(va3x0, vmask));
      const float32x4_t va4x0o = vreinterpretq_f32_u16(vandq_u16(va4x0, vmask));
      const float32x4_t va0x1o = vreinterpretq_f32_u16(vandq_u16(va0x1, vmask));
      const float32x4_t va1x1o = vreinterpretq_f32_u16(vandq_u16(va1x1, vmask));
      const float32x4_t va2x1o = vreinterpretq_f32_u16(vandq_u16(va2x1, vmask));
      const float32x4_t va3x1o = vreinterpretq_f32_u16(vandq_u16(va3x1, vmask));
      const float32x4_t va4x1o = vreinterpretq_f32_u16(vandq_u16(va4x1, vmask));
      const float32x4_t va0x2o = vreinterpretq_f32_u16(vandq_u16(va0x2, vmask));
      const float32x4_t va1x2o = vreinterpretq_f32_u16(vandq_u16(va1x2, vmask));
      const float32x4_t va2x2o = vreinterpretq_f32_u16(vandq_u16(va2x2, vmask));
      const float32x4_t va3x2o = vreinterpretq_f32_u16(vandq_u16(va3x2, vmask));
      const float32x4_t va4x2o = vreinterpretq_f32_u16(vandq_u16(va4x2, vmask));
      const float32x4_t va0x3o = vreinterpretq_f32_u16(vandq_u16(va0x3, vmask));
      const float32x4_t va1x3o = vreinterpretq_f32_u16(vandq_u16(va1x3, vmask));
      const float32x4_t va2x3o = vreinterpretq_f32_u16(vandq_u16(va2x3, vmask));
      const float32x4_t va3x3o = vreinterpretq_f32_u16(vandq_u16(va3x3, vmask));
      const float32x4_t va4x3o = vreinterpretq_f32_u16(vandq_u16(va4x3, vmask));

      vacc0x0 = vfmaq_f32(vacc0x0, va0x0o, vb0o);
      vacc1x0 = vfmaq_f32(vacc1x0, va1x0o, vb0o);
      vacc2x0 = vfmaq_f32(vacc2x0, va2x0o, vb0o);
      vacc3x0 = vfmaq_f32(vacc3x0, va3x0o, vb0o);
      vacc4x0 = vfmaq_f32(vacc4x0, va4x0o, vb0o);
      vacc0x1 = vfmaq_f32(vacc0x1, va0x1o, vb1o);
      vacc1x1 = vfmaq_f32(vacc1x1, va1x1o, vb1o);
      vacc2x1 = vfmaq_f32(vacc2x1, va2x1o, vb1o);
      vacc3x1 = vfmaq_f32(vacc3x1, va3x1o, vb1o);
      vacc4x1 = vfmaq_f32(vacc4x1, va4x1o, vb1o);
      vacc0x2 = vfmaq_f32(vacc0x2, va0x2o, vb2o);
      vacc1x2 = vfmaq_f32(vacc1x2, va1x2o, vb2o);
      vacc2x2 = vfmaq_f32(vacc2x2, va2x2o, vb2o);
      vacc3x2 = vfmaq_f32(vacc3x2, va3x2o, vb2o);
      vacc4x2 = vfmaq_f32(vacc4x2, va4x2o, vb2o);
      vacc0x3 = vfmaq_f32(vacc0x3, va0x3o, vb3o);
      vacc1x3 = vfmaq_f32(vacc1x3, va1x3o, vb3o);
      vacc2x3 = vfmaq_f32(vacc2x3, va2x3o, vb3o);
      vacc3x3 = vfmaq_f32(vacc3x3, va3x3o, vb3o);
      vacc4x3 = vfmaq_f32(vacc4x3, va4x3o, vb3o);
    }

#if XNN_ARCH_ARM64
    const float32x4_t vacc0x01 = vpaddq_f32(vacc0x0, vacc0x1);
    const float32x4_t vacc1x01 = vpaddq_f32(vacc1x0, vacc1x1);
    const float32x4_t vacc2x01 = vpaddq_f32(vacc2x0, vacc2x1);
    const float32x4_t vacc3x01 = vpaddq_f32(vacc3x0, vacc3x1);
    const float32x4_t vacc4x01 = vpaddq_f32(vacc4x0, vacc4x1);
    const float32x4_t vacc0x23 = vpaddq_f32(vacc0x2, vacc0x3);
    const float32x4_t vacc1x23 = vpaddq_f32(vacc1x2, vacc1x3);
    const float32x4_t vacc2x23 = vpaddq_f32(vacc2x2, vacc2x3);
    const float32x4_t vacc3x23 = vpaddq_f32(vacc3x2, vacc3x3);
    const float32x4_t vacc4x23 = vpaddq_f32(vacc4x2, vacc4x3);

    float32x4_t vacc0x0123 = vpaddq_f32(vacc0x01, vacc0x23);
    float32x4_t vacc1x0123 = vpaddq_f32(vacc1x01, vacc1x23);
    float32x4_t vacc2x0123 = vpaddq_f32(vacc2x01, vacc2x23);
    float32x4_t vacc3x0123 = vpaddq_f32(vacc3x01, vacc3x23);
    float32x4_t vacc4x0123 = vpaddq_f32(vacc4x01, vacc4x23);
#else
    const float32x2_t vsum0x0 = vadd_f32(vget_low_f32(vacc0x0), vget_high_f32(vacc0x0));
    const float32x2_t vsum1x0 = vadd_f32(vget_low_f32(vacc1x0), vget_high_f32(vacc1x0));
    const float32x2_t vsum2x0 = vadd_f32(vget_low_f32(vacc2x0), vget_high_f32(vacc2x0));
    const float32x2_t vsum3x0 = vadd_f32(vget_low_f32(vacc3x0), vget_high_f32(vacc3x0));
    const float32x2_t vsum4x0 = vadd_f32(vget_low_f32(vacc4x0), vget_high_f32(vacc4x0));
    const float32x2_t vsum0x1 = vadd_f32(vget_low_f32(vacc0x1), vget_high_f32(vacc0x1));
    const float32x2_t vsum1x1 = vadd_f32(vget_low_f32(vacc1x1), vget_high_f32(vacc1x1));
    const float32x2_t vsum2x1 = vadd_f32(vget_low_f32(vacc2x1), vget_high_f32(vacc2x1));
    const float32x2_t vsum3x1 = vadd_f32(vget_low_f32(vacc3x1), vget_high_f32(vacc3x1));
    const float32x2_t vsum4x1 = vadd_f32(vget_low_f32(vacc4x1), vget_high_f32(vacc4x1));
    const float32x2_t vsum0x2 = vadd_f32(vget_low_f32(vacc0x2), vget_high_f32(vacc0x2));
    const float32x2_t vsum1x2 = vadd_f32(vget_low_f32(vacc1x2), vget_high_f32(vacc1x2));
    const float32x2_t vsum2x2 = vadd_f32(vget_low_f32(vacc2x2), vget_high_f32(vacc2x2));
    const float32x2_t vsum3x2 = vadd_f32(vget_low_f32(vacc3x2), vget_high_f32(vacc3x2));
    const float32x2_t vsum4x2 = vadd_f32(vget_low_f32(vacc4x2), vget_high_f32(vacc4x2));
    const float32x2_t vsum0x3 = vadd_f32(vget_low_f32(vacc0x3), vget_high_f32(vacc0x3));
    const float32x2_t vsum1x3 = vadd_f32(vget_low_f32(vacc1x3), vget_high_f32(vacc1x3));
    const float32x2_t vsum2x3 = vadd_f32(vget_low_f32(vacc2x3), vget_high_f32(vacc2x3));
    const float32x2_t vsum3x3 = vadd_f32(vget_low_f32(vacc3x3), vget_high_f32(vacc3x3));
    const float32x2_t vsum4x3 = vadd_f32(vget_low_f32(vacc4x3), vget_high_f32(vacc4x3));

    float32x4_t vacc0x0123 = vcombine_f32(vpadd_f32(vsum0x0, vsum0x1), vpadd_f32(vsum0x2, vsum0x3));
    float32x4_t vacc1x0123 = vcombine_f32(vpadd_f32(vsum1x0, vsum1x1), vpadd_f32(vsum1x2, vsum1x3));
    float32x4_t vacc2x0123 = vcombine_f32(vpadd_f32(vsum2x0, vsum2x1), vpadd_f32(vsum2x2, vsum2x3));
    float32x4_t vacc3x0123 = vcombine_f32(vpadd_f32(vsum3x0, vsum3x1), vpadd_f32(vsum3x2, vsum3x3));
    float32x4_t vacc4x0123 = vcombine_f32(vpadd_f32(vsum4x0, vsum4x1), vpadd_f32(vsum4x2, vsum4x3));
#endif

    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc4x0123 = vminq_f32(vacc4x0123, vmax);

    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);

    uint16x4_t vout0x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc0x0123), 16);
    uint16x4_t vout1x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc1x0123), 16);
    uint16x4_t vout2x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc2x0123), 16);
    uint16x4_t vout3x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc3x0123), 16);
    uint16x4_t vout4x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc4x0123), 16);

    if XNN_LIKELY(nc >= 4) {
      vst1_u16(c0, vout0x0123);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      vst1_u16(c1, vout1x0123);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1_u16(c2, vout2x0123);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1_u16(c3, vout3x0123);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1_u16(c4, vout4x0123);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_u16(vout0x0123), 0); c0 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_u16(vout1x0123), 0); c1 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_u16(vout2x0123), 0); c2 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_u16(vout3x0123), 0); c3 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_u16(vout4x0123), 0); c4 += 2;

        vout0x0123 = vext_u16(vout0x0123, vout0x0123, 2);
        vout1x0123 = vext_u16(vout1x0123, vout1x0123, 2);
        vout2x0123 = vext_u16(vout2x0123, vout2x0123, 2);
        vout3x0123 = vext_u16(vout3x0123, vout3x0123, 2);
        vout4x0123 = vext_u16(vout4x0123, vout4x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vout0x0123, 0);
        vst1_lane_u16(c1, vout1x0123, 0);
        vst1_lane_u16(c2, vout2x0123, 0);
        vst1_lane_u16(c3, vout3x0123, 0);
        vst1_lane_u16(c4, vout4x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
