# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for aarch64
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AARCH64_ASM_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/f16-gemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S
  src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S
  src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S
  src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/f16-igemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/f16-igemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S
  src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S
  src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S
  src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f32-dwconv/f32-dwconv-9p4c-minmax-asm-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S
  src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a55.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a73.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-igemm/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a55.S
  src/f32-igemm/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a73.S
  src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S
  src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S
  src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S
  src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondotfp16arith-cortex-a55.S
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu16-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu16-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S)

SET(NON_PROD_AARCH64_ASM_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x8-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/f16-gemm-4x8-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/f16-gemm-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/f16-gemm-6x8-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S
  src/f16-gemm/gen/f16-gemm-8x8-minmax-asm-aarch64-neonfp16arith-ld64.S
  src/f16-igemm/f16-igemm-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S
  src/f16-igemm/f16-igemm-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S
  src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S
  src/f32-dwconv/f32-dwconv-9p4c-minmax-asm-aarch64-neonfma.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S
  src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-1x12-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/f32-gemm-1x16-minmax-asm-aarch64-neonfma-ld32.S
  src/f32-gemm/gen/f32-gemm-1x16-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-1x16-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-2x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-2x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-2x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-2x16-minmax-asm-aarch64-neonfma-ld32.S
  src/f32-gemm/gen/f32-gemm-2x16-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-2x16-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-3x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-3x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-3x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-3x16-minmax-asm-aarch64-neonfma-ld32.S
  src/f32-gemm/gen/f32-gemm-3x16-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-3x16-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/f32-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-4x12-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-gemm/gen/f32-gemm-4x16-minmax-asm-aarch64-neonfma-ld32.S
  src/f32-gemm/gen/f32-gemm-4x16-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-4x16-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-5x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-5x16-minmax-asm-aarch64-neonfma-ld32.S
  src/f32-gemm/gen/f32-gemm-5x16-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-5x16-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-7x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-7x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-7x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-gemm/gen/f32-gemm-8x8-minmax-asm-aarch64-neonfma-ld32-2.S
  src/f32-gemm/gen/f32-gemm-8x8-minmax-asm-aarch64-neonfma-ld64-2.S
  src/f32-gemm/gen/f32-gemm-8x8-minmax-asm-aarch64-neonfma-ld128-2.S
  src/f32-igemm/f32-igemm-1x12-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/f32-igemm-4x12-minmax-asm-aarch64-neonfma-cortex-a53.S
  src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S
  src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-igemm/gen/f32-igemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-igemm/gen/f32-igemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-ld128-prfm.S
  src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
  src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-cortex-a75.S
  src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-ld128-prfm.S
  src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-asm-aarch64-neondot-ld32.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-asm-aarch64-neondot-ld64.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-asm-aarch64-neondot-ld128.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld64.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mull.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16-minmax-fp32-asm-aarch64-neondot-ld32.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16-minmax-fp32-asm-aarch64-neondot-ld128.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S
  src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu16-asm-aarch64-neon-mlal-lane-cortex-a53.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S
  src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu16-asm-aarch64-neon-mlal-lane-cortex-a53.S)

SET(ALL_AARCH64_ASM_MICROKERNEL_SRCS ${PROD_AARCH64_ASM_MICROKERNEL_SRCS} + ${NON_PROD_AARCH64_ASM_MICROKERNEL_SRCS})
