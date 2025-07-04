// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include "src/xnnpack/raddstoreexpminusmax.h"
#include "src/xnnpack/simd/f32-hvx.h"

void xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const HVX_Vector vi_max = xnn_set1_f32(*max);
  const HVX_Vector vlog2e = xnn_set1_f32(0x1.715476p+0f);
  const HVX_Vector vmagic_bias = xnn_set1_f32(0x1.8000FEp23f);
  const HVX_Vector vminus_ln2_hi = xnn_set1_f32(-0x1.62E400p-1f);
  const HVX_Vector vminus_ln2_lo = xnn_set1_f32(-0x1.7F7D1Cp-20f);
  const HVX_Vector vc5 = xnn_set1_f32(0x1.0F9F9Cp-7f);
  const HVX_Vector vc4 = xnn_set1_f32(0x1.573A1Ap-5f);
  const HVX_Vector vc3 = xnn_set1_f32(0x1.555A80p-3f);
  const HVX_Vector vc2 = xnn_set1_f32(0x1.FFFDC6p-2f);
  const HVX_Vector vc1 = xnn_set1_f32(0x1.FFFFF6p-1f);
  const HVX_Vector vdenorm_cutoff = xnn_set1_f32(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const HVX_Vector vi = xnn_loadu_f32(input);
    input += 32;

    const HVX_Vector vx = xnn_sub_f32(vi, vi_max);

    HVX_Vector vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = xnn_sub_f32(vn, vmagic_bias);

    HVX_Vector vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
    vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = xnn_fmadd_f32(vc5, vt, vc4);
    vp = xnn_fmadd_f32(vp, vt, vc3);
    vp = xnn_fmadd_f32(vp, vt, vc2);
    vp = xnn_fmadd_f32(vp, vt, vc1);

    vt = xnn_mul_f32(vt, vs);
    HVX_Vector vf = xnn_fmadd_f32(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    xnn_storeu_f32(output, vf);
    output += 32;

    vacc0 = xnn_add_f32(vacc0, vf);
  }

  if XNN_UNLIKELY(batch != 0) {
    const HVX_Vector vi = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const HVX_Vector vx = xnn_sub_f32(vi, vi_max);

    HVX_Vector vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = xnn_sub_f32(vn, vmagic_bias);

    HVX_Vector vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
    vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = xnn_fmadd_f32(vc5, vt, vc4);
    vp = xnn_fmadd_f32(vp, vt, vc3);
    vp = xnn_fmadd_f32(vp, vt, vc2);
    vp = xnn_fmadd_f32(vp, vt, vc1);

    vt = xnn_mul_f32(vt, vs);
    HVX_Vector vf = xnn_fmadd_f32(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    Q6_V_vstu_variable(output, batch, vf);

    HVX_VectorPred mask = Q6_Q_vsetq_R(batch);
    vacc0 = xnn_add_f32(vacc0, Q6_V_vand_QV(mask, vf));
  }

  *sum = xnn_reduce_add_f32(vacc0);
}
