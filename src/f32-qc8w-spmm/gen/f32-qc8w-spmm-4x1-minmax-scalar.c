// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_f32_qc8w_spmm_minmax_ukernel_4x1__scalar(
    size_t mc,
    size_t nc,
    const float* input,
    const void* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t output_decrement = output_stride * nc - 4 * sizeof(float);
  while (mc >= 4 * sizeof(float)) {
    const void* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 1) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = unaligned_indexed_load_f32(w, 0);
      w = (const float*) w + 1;
      float vacc1x0 = vacc0x0;
      float vacc2x0 = vacc0x0;
      float vacc3x0 = vacc0x0;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float vi0 = input[0];
          const float vi1 = input[1];
          const float vi2 = input[2];
          const float vi3 = input[3];
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const float vw0 = (float) ((const int8_t*) w)[0];
          w = (const int8_t*) w + 1;
          vacc0x0 += vi0 * vw0;
          vacc1x0 += vi1 * vw0;
          vacc2x0 += vi2 * vw0;
          vacc3x0 += vi3 * vw0;
        } while (--nnz != 0);
      }
      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      w = (const float*) w + 1;
      vacc0x0 *= vscale0;
      vacc1x0 *= vscale0;
      vacc2x0 *= vscale0;
      vacc3x0 *= vscale0;
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      float vout1x0 = math_min_f32(vacc1x0, vmax);
      float vout2x0 = math_min_f32(vacc2x0, vmax);
      float vout3x0 = math_min_f32(vacc3x0, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      vout1x0 = math_max_f32(vout1x0, vmin);
      vout2x0 = math_max_f32(vout2x0, vmin);
      vout3x0 = math_max_f32(vout3x0, vmin);
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 1;
    }
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = unaligned_load_f32(w);
        w = (const float*) w + 1;
        float vacc1 = vacc0;
        float vacc2 = vacc0;
        float vacc3 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw = (float) ((const int8_t*) w)[0];
            w = (const int8_t*) w + 1;
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            vacc2 += vi2 * vw;
            vacc3 += vi3 * vw;
          } while (--nnz != 0);
        }
        float vscale = unaligned_load_f32(w);
        w = (const float*) w + 1;
        vacc0 *= vscale;
        vacc1 *= vscale;
        vacc2 *= vscale;
        vacc3 *= vscale;
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        float vout2 = math_min_f32(vacc2, vmax);
        float vout3 = math_min_f32(vacc3, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        vout2 = math_max_f32(vout2, vmin);
        vout3 = math_max_f32(vout3, vmin);
        output[0] = vout0;
        output[1] = vout1;
        output[2] = vout2;
        output[3] = vout3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 4;
    mc -= 4 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const void* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 1) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = unaligned_indexed_load_f32(w, 0);
        w = (const float*) w + 1;
        float vacc1x0 = vacc0x0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = (float) ((const int8_t*) w)[0];
            w = (const int8_t*) w + 1;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
          } while (--nnz != 0);
        }
        const float vscale0 = unaligned_indexed_load_f32(w, 0);
        w = (const float*) w + 1;
        vacc0x0 *= vscale0;
        vacc1x0 *= vscale0;
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = unaligned_load_f32(w);
          w = (const float*) w + 1;
          float vacc1 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = (float) ((const int8_t*) w)[0];
              w = (const int8_t*) w + 1;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
            } while (--nnz != 0);
          }
          float vscale = unaligned_load_f32(w);
          w = (const float*) w + 1;
          vacc0 *= vscale;
          vacc1 *= vscale;
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const void* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 1) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = unaligned_indexed_load_f32(w, 0);
        w = (const float*) w + 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = (float) ((const int8_t*) w)[0];
            w = (const int8_t*) w + 1;
            vacc0x0 += vi0 * vw0;
          } while (--nnz != 0);
        }
        const float vscale0 = unaligned_indexed_load_f32(w, 0);
        w = (const float*) w + 1;
        vacc0x0 *= vscale0;
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        output[0] = vout0x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = unaligned_load_f32(w);
          w = (const float*) w + 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = (float) ((const int8_t*) w)[0];
              w = (const int8_t*) w + 1;
              vacc0 += vi0 * vw;
            } while (--nnz != 0);
          }
          float vscale = unaligned_load_f32(w);
          w = (const float*) w + 1;
          vacc0 *= vscale;
          float vout0 = math_min_f32(vacc0, vmax);
          vout0 = math_max_f32(vout0, vmin);
          output[0] = vout0;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
