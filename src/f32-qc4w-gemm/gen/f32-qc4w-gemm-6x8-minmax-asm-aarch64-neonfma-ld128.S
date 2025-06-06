// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"


# void xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          x4
#     const void* w,            x5
#     float* c,                 x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp, 16] -> (x0)
#     const xnn_f32_qc4w_minmax_params* params)  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0   x3  v0
# A1   x9  v1
# A2  x10  v2
# A3  x11  v3
# A4  x12  v4
# A5   x4  v5
# B    x5  v16 v17 v18 v19
# C    x6  v20 v21
# C   x16  v22 v23
# C   x17  v24 v25
# C   x14  v26 v27
# C   x13  v28 v29
# C    x7  v30 v31
# Clamp v6 v7
# zerop/mask v8 v9
# unused A   v10 v11
# unused B   v12 v13 v14 v15

BEGIN_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128

        # Load params pointer
        LDR         x8, [sp, 8]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x9, x3, x4              // a1 = a0 + a_stride
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x9, x3, x9, LO          //   a1 = a0
        CSEL        x16, x6, x16, LO        //   c1 = c0

        ADD         x10, x9, x4             // a2 = a1 + a_stride
        ADD         x17, x16, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x10, x9, x10, LS        //   a2 = a1
        CSEL        x17, x16, x17, LS       //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x11, x10, x4            // a3 = a2 + a_stride
        ADD         x14, x17, x7            // c3 = c2 + cm_stride
        CSEL        x11, x10, x11, LO       //   a3 = a2
        CSEL        x14, x17, x14, LO       //   c3 = c2

        ADD         x12, x11, x4            // a4 = a3 + a_stride
        ADD         x13, x14, x7            // c4 = c3 + cm_stride
                                            // if mr <= 4
        CSEL        x12, x11, x12, LS       //   a4 = a3
        CSEL        x13, x14, x13, LS       //   c4 = c3

        CMP         x0, 6                   // if mr < 6
        ADD         x4, x12, x4             // a5 = a4 + a_stride
        ADD         x7, x13, x7             // c5 = c4 + cm_stride
        CSEL        x4, x12, x4, LO         //   a5 = a4
        CSEL        x7, x13, x7, LO         //   c5 = c4

        STP         d8,  d9, [sp, -16]!     // Save d8-d9 on stack
        # Load min/max/zerop values
        LD3R        {v6.4s, v7.4s, v8.4s}, [x8]
        NEG         v8.4s, v8.4s
        MOVI        v9.16b, 15

0:
        # Load initial bias from w into accumulators
        LDP         q20, q21, [x5], 32
        MOV         v22.16b, v20.16b
        PRFM        PLDL1KEEP, [x5, 0]      // Prefetch B
        MOV         v23.16b, v21.16b
        PRFM        PLDL1KEEP, [x5, 64]
        MOV         v24.16b, v20.16b
        PRFM        PLDL1KEEP, [x5, 128]
        MOV         v25.16b, v21.16b
        PRFM        PLDL1KEEP, [x5, 192]
        MOV         v26.16b, v20.16b
        PRFM        PLDL1KEEP,  [x3]        // Prefetch A
        MOV         v27.16b, v21.16b
        PRFM        PLDL1KEEP,  [x9]
        MOV         v28.16b, v20.16b
        PRFM        PLDL1KEEP, [x10]
        MOV         v29.16b, v21.16b
        PRFM        PLDL1KEEP, [x11]
        MOV         v30.16b, v20.16b
        PRFM        PLDL1KEEP, [x12]
        MOV         v31.16b, v21.16b
        PRFM        PLDL1KEEP,  [x4]

        # Is there at least 4 floats (16 bytes)?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
        # 48 FMA + 6 ld128 A + 4 LDP B
1:
        LDR         q0,  [x3], 16
        LDR         q18, [x5], 16
        SXTL        v17.8h, v18.8b
        SXTL2       v19.8h, v18.16b
        LDR         q1,  [x9], 16
        SXTL        v16.4s, v17.4h
        SXTL2       v17.4s, v17.8h
        LDR         q2, [x10], 16
        SXTL        v18.4s, v19.4h
        SXTL2       v19.4s, v19.8h
        LDR         q3, [x11], 16
        SCVTF       v16.4s, v16.4s
        SCVTF       v17.4s, v17.4s
        LDR         q4, [x12], 16
        SCVTF       v18.4s, v18.4s
        SCVTF       v19.4s, v19.4s
        LDR         q5,  [x4], 16
        FMLA        v20.4s, v16.4s,  v0.s[0]
        FMLA        v22.4s, v16.4s,  v1.s[0]
        FMLA        v24.4s, v16.4s,  v2.s[0]
        FMLA        v26.4s, v16.4s,  v3.s[0]
        FMLA        v28.4s, v16.4s,  v4.s[0]
        FMLA        v30.4s, v16.4s,  v5.s[0]
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v1.s[0]
        FMLA        v25.4s, v17.4s,  v2.s[0]
        FMLA        v27.4s, v17.4s,  v3.s[0]
        FMLA        v29.4s, v17.4s,  v4.s[0]
        FMLA        v31.4s, v17.4s,  v5.s[0]

        FMLA        v20.4s, v18.4s,  v0.s[1]
        LDR         q17, [x5], 8
        SXTL        v17.8h, v17.8b
        SXTL        v16.4s, v17.4h
        SXTL2       v17.4s, v17.8h
        SCVTF       v16.4s, v16.4s
        SCVTF       v17.4s, v17.4s
        FMLA        v22.4s, v18.4s,  v1.s[1]
        FMLA        v24.4s, v18.4s,  v2.s[1]
        FMLA        v26.4s, v18.4s,  v3.s[1]
        FMLA        v28.4s, v18.4s,  v4.s[1]
        FMLA        v30.4s, v18.4s,  v5.s[1]
        FMLA        v21.4s, v19.4s,  v0.s[1]
        FMLA        v23.4s, v19.4s,  v1.s[1]
        FMLA        v25.4s, v19.4s,  v2.s[1]
        FMLA        v27.4s, v19.4s,  v3.s[1]
        FMLA        v29.4s, v19.4s,  v4.s[1]
        FMLA        v31.4s, v19.4s,  v5.s[1]

        FMLA        v20.4s, v16.4s,  v0.s[2]
        LDR         q19, [x5], 8
        SXTL        v19.8h, v19.8b
        SXTL        v18.4s, v19.4h
        SXTL2       v19.4s, v19.8h
        SCVTF       v18.4s, v18.4s
        SCVTF       v19.4s, v19.4s
        FMLA        v22.4s, v16.4s,  v1.s[2]
        FMLA        v24.4s, v16.4s,  v2.s[2]
        FMLA        v26.4s, v16.4s,  v3.s[2]
        FMLA        v28.4s, v16.4s,  v4.s[2]
        FMLA        v30.4s, v16.4s,  v5.s[2]
        FMLA        v21.4s, v17.4s,  v0.s[2]
        FMLA        v23.4s, v17.4s,  v1.s[2]
        FMLA        v25.4s, v17.4s,  v2.s[2]
        FMLA        v27.4s, v17.4s,  v3.s[2]
        FMLA        v29.4s, v17.4s,  v4.s[2]
        FMLA        v31.4s, v17.4s,  v5.s[2]

        FMLA        v20.4s, v18.4s,  v0.s[3]
        FMLA        v22.4s, v18.4s,  v1.s[3]
        FMLA        v24.4s, v18.4s,  v2.s[3]
        FMLA        v26.4s, v18.4s,  v3.s[3]
        FMLA        v28.4s, v18.4s,  v4.s[3]
        FMLA        v30.4s, v18.4s,  v5.s[3]
        FMLA        v21.4s, v19.4s,  v0.s[3]
        FMLA        v23.4s, v19.4s,  v1.s[3]
        FMLA        v25.4s, v19.4s,  v2.s[3]
        FMLA        v27.4s, v19.4s,  v3.s[3]
        SUBS        x0, x0, 16
        FMLA        v29.4s, v19.4s,  v4.s[3]
        FMLA        v31.4s, v19.4s,  v5.s[3]
        B.HS        1b

        # Is there a remainder?- 2 floats of A (8 bytes) or less
        TST         x0, 15
        B.NE        3f

2:
        # Scale
        LDP         q16, q17, [x5], 32
        FMUL        v20.4s, v20.4s, v16.4s
        FMUL        v21.4s, v21.4s, v17.4s
        FMUL        v22.4s, v22.4s, v16.4s
        FMUL        v23.4s, v23.4s, v17.4s
        FMUL        v24.4s, v24.4s, v16.4s
        FMUL        v25.4s, v25.4s, v17.4s
        FMUL        v26.4s, v26.4s, v16.4s
        FMUL        v27.4s, v27.4s, v17.4s
        FMUL        v28.4s, v28.4s, v16.4s
        FMUL        v29.4s, v29.4s, v17.4s
        FMUL        v30.4s, v30.4s, v16.4s
        FMUL        v31.4s, v31.4s, v17.4s
        # Clamp
        FMAX        v20.4s, v20.4s, v6.4s
        # Load cn_stride
        LDR         x0, [sp, 16]
        FMAX        v21.4s, v21.4s, v6.4s
        FMAX        v22.4s, v22.4s, v6.4s
        FMAX        v23.4s, v23.4s, v6.4s
        FMAX        v24.4s, v24.4s, v6.4s
        FMAX        v25.4s, v25.4s, v6.4s
        FMAX        v26.4s, v26.4s, v6.4s
        FMAX        v27.4s, v27.4s, v6.4s
        FMAX        v28.4s, v28.4s, v6.4s
        FMAX        v29.4s, v29.4s, v6.4s
        FMAX        v30.4s, v30.4s, v6.4s
        FMAX        v31.4s, v31.4s, v6.4s
        SUBS        x1, x1, 8
        FMIN        v20.4s, v20.4s, v7.4s
        FMIN        v21.4s, v21.4s, v7.4s
        FMIN        v22.4s, v22.4s, v7.4s
        FMIN        v23.4s, v23.4s, v7.4s
        FMIN        v24.4s, v24.4s, v7.4s
        FMIN        v25.4s, v25.4s, v7.4s
        FMIN        v26.4s, v26.4s, v7.4s
        FMIN        v27.4s, v27.4s, v7.4s
        FMIN        v28.4s, v28.4s, v7.4s
        FMIN        v29.4s, v29.4s, v7.4s
        FMIN        v30.4s, v30.4s, v7.4s
        FMIN        v31.4s, v31.4s, v7.4s

        # Store full 6 x 8
        B.LO        5f

        ST1         {v20.16b, v21.16b},  [x6], x0
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v22.16b, v23.16b}, [x16], x0
        SUB         x9,  x9, x2             // a1 -= kc
        ST1         {v24.16b, v25.16b}, [x17], x0
        SUB         x10, x10, x2            // a2 -= kc
        ST1         {v26.16b, v27.16b}, [x14], x0
        SUB         x11, x11, x2            // a3 -= kc
        ST1         {v28.16b, v29.16b}, [x13], x0
        SUB         x12, x12, x2            // a4 -= kc
        ST1         {v30.16b, v31.16b},  [x7], x0
        SUB         x4,  x4, x2             // a5 -= kc

        B.HI        0b
        LDP         d8,  d9, [sp], 16
        RET

3:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ         x0, 3, 4f

        # Remainder- 2 floats of A (8 bytes)
        LDR         d0,  [x3], 8
        LDR         q18, [x5], 16
        SXTL        v17.8h, v18.8b
        SXTL2       v19.8h, v18.16b
        LDR         d1,  [x9], 8
        SXTL        v16.4s, v17.4h
        SXTL2       v17.4s, v17.8h
        LDR         d2, [x10], 8
        SXTL        v18.4s, v19.4h
        SXTL2       v19.4s, v19.8h
        LDR         d3, [x11], 8
        SCVTF       v16.4s, v16.4s
        SCVTF       v17.4s, v17.4s
        LDR         d4, [x12], 8
        SCVTF       v18.4s, v18.4s
        SCVTF       v19.4s, v19.4s
        LDR         d5,  [x4], 8
        FMLA        v20.4s, v16.4s,  v0.s[0]
        FMLA        v22.4s, v16.4s,  v1.s[0]
        FMLA        v24.4s, v16.4s,  v2.s[0]
        FMLA        v26.4s, v16.4s,  v3.s[0]
        FMLA        v28.4s, v16.4s,  v4.s[0]
        FMLA        v30.4s, v16.4s,  v5.s[0]
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v1.s[0]
        FMLA        v25.4s, v17.4s,  v2.s[0]
        FMLA        v27.4s, v17.4s,  v3.s[0]
        FMLA        v29.4s, v17.4s,  v4.s[0]
        FMLA        v31.4s, v17.4s,  v5.s[0]

        FMLA        v20.4s, v18.4s,  v0.s[1]
        FMLA        v22.4s, v18.4s,  v1.s[1]
        FMLA        v24.4s, v18.4s,  v2.s[1]
        FMLA        v26.4s, v18.4s,  v3.s[1]
        FMLA        v28.4s, v18.4s,  v4.s[1]
        FMLA        v30.4s, v18.4s,  v5.s[1]
        FMLA        v21.4s, v19.4s,  v0.s[1]
        FMLA        v23.4s, v19.4s,  v1.s[1]
        FMLA        v25.4s, v19.4s,  v2.s[1]
        FMLA        v27.4s, v19.4s,  v3.s[1]
        FMLA        v29.4s, v19.4s,  v4.s[1]
        FMLA        v31.4s, v19.4s,  v5.s[1]

        # Is there a remainder?- 1 float of A (4 bytes)
        TBZ         x0, 2, 2b

        # Remainder- 1 float of A (4 bytes)
4:
        LDR         s0,  [x3], 4
        LDR         d17, [x5], 8
        SXTL        v17.8h, v17.8b
        LDR         s1,  [x9], 4
        SXTL        v16.4s, v17.4h
        LDR         s2, [x10], 4
        SXTL2       v17.4s, v17.8h
        LDR         s3, [x11], 4
        SCVTF       v16.4s, v16.4s
        LDR         s4, [x12], 4
        SCVTF       v17.4s, v17.4s
        LDR         s5,  [x4], 4
        FMLA        v20.4s, v16.4s,  v0.s[0]
        FMLA        v22.4s, v16.4s,  v1.s[0]
        FMLA        v24.4s, v16.4s,  v2.s[0]
        FMLA        v26.4s, v16.4s,  v3.s[0]
        FMLA        v28.4s, v16.4s,  v4.s[0]
        FMLA        v30.4s, v16.4s,  v5.s[0]
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v1.s[0]
        FMLA        v25.4s, v17.4s,  v2.s[0]
        FMLA        v27.4s, v17.4s,  v3.s[0]
        FMLA        v29.4s, v17.4s,  v4.s[0]
        FMLA        v31.4s, v17.4s,  v5.s[0]
        B           2b

        # Store odd width
5:
        TBZ         x1, 2, 6f
        STR         q20,  [x6], 16
        MOV         v20.16b, v21.16b
        STR         q22, [x16], 16
        MOV         v22.16b, v23.16b
        STR         q24, [x17], 16
        MOV         v24.16b, v25.16b
        STR         q26, [x14], 16
        MOV         v26.16b, v27.16b
        STR         q28, [x13], 16
        MOV         v28.16b, v29.16b
        STR         q30,  [x7], 16
        MOV         v30.16b, v31.16b

6:
        TBZ         x1, 1, 7f
        STR         d20,  [x6], 8
        STR         d22, [x16], 8
        DUP         d20, v20.d[1]
        DUP         d22, v22.d[1]
        STR         d24, [x17], 8
        STR         d26, [x14], 8
        DUP         d24, v24.d[1]
        DUP         d26, v26.d[1]
        STR         d28, [x13], 8
        STR         d30,  [x7], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

7:
        TBZ         x1, 0, 8f
        STR         s20,  [x6]
        STR         s22, [x16]
        STR         s24, [x17]
        STR         s26, [x14]
        STR         s28, [x13]
        STR         s30,  [x7]
8:
        LDP         d8,  d9, [sp], 16
        RET

END_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
