// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vbinary.h"

void xnn_f32_vadd_ukernel__hvx_u128(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    HVX_Vector va0 = xnn_loadu_f32(input_a + 0);
    HVX_Vector va1 = xnn_loadu_f32(input_a + 32);
    HVX_Vector va2 = xnn_loadu_f32(input_a + 64);
    HVX_Vector va3 = xnn_loadu_f32(input_a + 96);
    HVX_Vector vb0 = xnn_loadu_f32(input_b + 0);
    HVX_Vector vb1 = xnn_loadu_f32(input_b + 32);
    HVX_Vector vb2 = xnn_loadu_f32(input_b + 64);
    HVX_Vector vb3 = xnn_loadu_f32(input_b + 96);
    input_a += 128;
    input_b += 128;

    HVX_Vector vacc0 = xnn_add_f32(va0, vb0);
    HVX_Vector vacc1 = xnn_add_f32(va1, vb1);
    HVX_Vector vacc2 = xnn_add_f32(va2, vb2);
    HVX_Vector vacc3 = xnn_add_f32(va3, vb3);


    xnn_storeu_f32(output + 0, vacc0);
    xnn_storeu_f32(output + 32, vacc1);
    xnn_storeu_f32(output + 64, vacc2);
    xnn_storeu_f32(output + 96, vacc3);
    output += 128;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = xnn_loadu_f32(input_a);
    HVX_Vector vb = xnn_loadu_f32(input_b);
    input_a += 32;
    input_b += 32;

    HVX_Vector vacc = xnn_add_f32(va, vb);

    xnn_storeu_f32(output, vacc);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector va = xnn_load_tail_f32(input_a, batch >> XNN_LOG2_SIZEOF_FLOAT);
    HVX_Vector vb = xnn_load_tail_f32(input_b, batch >> XNN_LOG2_SIZEOF_FLOAT);

    HVX_Vector vacc = xnn_add_f32(va, vb);

    Q6_V_vstu_variable(output, batch, vacc);
  }
}
