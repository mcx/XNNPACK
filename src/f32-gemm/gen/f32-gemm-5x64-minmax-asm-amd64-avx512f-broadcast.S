// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_5x64__asm_amd64_avx512f_broadcast

      .intel_syntax noprefix
      # Free up GP registers.
      # Save register arguments for tail call to msan annotation helper.
      push rdi
      push rsi
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # load params to free up GP registers
      mov r13, [rsp + 96] # params
      vbroadcastss zmm0, dword ptr [r13]
      vbroadcastss zmm1, dword ptr [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13

      # Allocate some space on the stack.
      sub rsp, 192

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      # Clamp a & c pointers if mr <= 2
      mov r15, rax
      add r15, r8
      mov rbx, r13
      add rbx, r11
      cmp rdi, 2
      cmovle r15, rax
      cmovle rbx, r13

      # Clamp a & c pointers if mr <= 3
      mov r14, r15
      add r14, r8
      mov rbp, rbx
      add rbp, r11
      cmp rdi, 3
      cmovle r14, r15
      cmovle rbp, rbx

      # Clamp a & c pointers if mr <= 4
      mov r12, r14
      add r12, r8
      mov r8, rbp
      add r8, r11
      cmp rdi, 4
      cmovle r12, r14
      cmovle r8, rbp

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  zmm11, [r9 + 0]
      vmovaps  zmm16, [r9 + 64]
      vmovaps  zmm21, [r9 + 128]
      vmovaps  zmm26, [r9 + 192]
      vmovaps zmm12, zmm11
      vmovaps zmm13, zmm11
      vmovaps zmm14, zmm11
      vmovaps zmm15, zmm11
      vmovaps zmm17, zmm16
      vmovaps zmm18, zmm16
      vmovaps zmm19, zmm16
      vmovaps zmm20, zmm16
      vmovaps zmm22, zmm21
      vmovaps zmm23, zmm21
      vmovaps zmm24, zmm21
      vmovaps zmm25, zmm21
      vmovaps zmm27, zmm26
      vmovaps zmm28, zmm26
      vmovaps zmm29, zmm26
      vmovaps zmm30, zmm26
      add r9, 256

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      vmovaps  zmm9, [r9 + 128]
      vmovaps  zmm10, [r9 + 192]
      add r9, 256
      vbroadcastss zmm2, dword ptr [rcx + r11]
      vfmadd231ps  zmm11, zmm2, zmm7
      vfmadd231ps  zmm16, zmm2, zmm8
      vfmadd231ps  zmm21, zmm2, zmm9
      vfmadd231ps  zmm26, zmm2, zmm10
      vbroadcastss zmm3, dword ptr [rax + r11]
      vfmadd231ps  zmm12, zmm3, zmm7
      vfmadd231ps  zmm17, zmm3, zmm8
      vfmadd231ps  zmm22, zmm3, zmm9
      vfmadd231ps  zmm27, zmm3, zmm10
      vbroadcastss zmm4, dword ptr [r15 + r11]
      vfmadd231ps  zmm13, zmm4, zmm7
      vfmadd231ps  zmm18, zmm4, zmm8
      vfmadd231ps  zmm23, zmm4, zmm9
      vfmadd231ps  zmm28, zmm4, zmm10
      vbroadcastss zmm5, dword ptr [r14 + r11]
      vfmadd231ps  zmm14, zmm5, zmm7
      vfmadd231ps  zmm19, zmm5, zmm8
      vfmadd231ps  zmm24, zmm5, zmm9
      vfmadd231ps  zmm29, zmm5, zmm10
      vbroadcastss zmm6, dword ptr [r12 + r11]
      vfmadd231ps  zmm15, zmm6, zmm7
      vfmadd231ps  zmm20, zmm6, zmm8
      vfmadd231ps  zmm25, zmm6, zmm9
      vfmadd231ps  zmm30, zmm6, zmm10

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vminps  zmm15, zmm1, zmm15
      vminps  zmm19, zmm1, zmm19
      vminps  zmm23, zmm1, zmm23
      vminps  zmm27, zmm1, zmm27
      vminps  zmm12, zmm1, zmm12
      vminps  zmm16, zmm1, zmm16
      vminps  zmm20, zmm1, zmm20
      vminps  zmm24, zmm1, zmm24
      vminps  zmm28, zmm1, zmm28
      vminps  zmm13, zmm1, zmm13
      vminps  zmm17, zmm1, zmm17
      vminps  zmm21, zmm1, zmm21
      vminps  zmm25, zmm1, zmm25
      vminps  zmm29, zmm1, zmm29
      vminps  zmm14, zmm1, zmm14
      vminps  zmm18, zmm1, zmm18
      vminps  zmm22, zmm1, zmm22
      vminps  zmm26, zmm1, zmm26
      vminps  zmm30, zmm1, zmm30
      vmaxps  zmm11, zmm0, zmm11
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm19, zmm0, zmm19
      vmaxps  zmm23, zmm0, zmm23
      vmaxps  zmm27, zmm0, zmm27
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm20, zmm0, zmm20
      vmaxps  zmm24, zmm0, zmm24
      vmaxps  zmm28, zmm0, zmm28
      vmaxps  zmm13, zmm0, zmm13
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm21, zmm0, zmm21
      vmaxps  zmm25, zmm0, zmm25
      vmaxps  zmm29, zmm0, zmm29
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm22, zmm0, zmm22
      vmaxps  zmm26, zmm0, zmm26
      vmaxps  zmm30, zmm0, zmm30

      # Check whether full or partial store.
      cmp rsi, 64
      jl .Ltail

      vmovups  [r10], zmm11
      vmovups  [r10 + 64], zmm16
      vmovups  [r10 + 128], zmm21
      vmovups  [r10 + 192], zmm26
      vmovups  [r13], zmm12
      vmovups  [r13 + 64], zmm17
      vmovups  [r13 + 128], zmm22
      vmovups  [r13 + 192], zmm27
      vmovups  [rbx], zmm13
      vmovups  [rbx + 64], zmm18
      vmovups  [rbx + 128], zmm23
      vmovups  [rbx + 192], zmm28
      vmovups  [rbp], zmm14
      vmovups  [rbp + 64], zmm19
      vmovups  [rbp + 128], zmm24
      vmovups  [rbp + 192], zmm29
      vmovups  [r8], zmm15
      vmovups  [r8 + 64], zmm20
      vmovups  [r8 + 128], zmm25
      vmovups  [r8 + 192], zmm30
      add r10, 256
      add r13, 256
      add rbx, 256
      add rbp, 256
      add r8, 256

      sub rsi, 64
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      shr r11, 16
      kmovw k2, r11d
      shr r11, 16
      kmovw k3, r11d
      shr r11, 16
      kmovw k4, r11d

      vmovups  zmmword ptr [r10]{k1}, zmm11
      vmovups  zmmword ptr [r10 + 64]{k2}, zmm16
      vmovups  zmmword ptr [r10 + 128]{k3}, zmm21
      vmovups  zmmword ptr [r10 + 192]{k4}, zmm26
      vmovups  zmmword ptr [r13]{k1}, zmm12
      vmovups  zmmword ptr [r13 + 64]{k2}, zmm17
      vmovups  zmmword ptr [r13 + 128]{k3}, zmm22
      vmovups  zmmword ptr [r13 + 192]{k4}, zmm27
      vmovups  zmmword ptr [rbx]{k1}, zmm13
      vmovups  zmmword ptr [rbx + 64]{k2}, zmm18
      vmovups  zmmword ptr [rbx + 128]{k3}, zmm23
      vmovups  zmmword ptr [rbx + 192]{k4}, zmm28
      vmovups  zmmword ptr [rbp]{k1}, zmm14
      vmovups  zmmword ptr [rbp + 64]{k2}, zmm19
      vmovups  zmmword ptr [rbp + 128]{k3}, zmm24
      vmovups  zmmword ptr [rbp + 192]{k4}, zmm29
      vmovups  zmmword ptr [r8]{k1}, zmm15
      vmovups  zmmword ptr [r8 + 64]{k2}, zmm20
      vmovups  zmmword ptr [r8 + 128]{k3}, zmm25
      vmovups  zmmword ptr [r8 + 192]{k4}, zmm30

.Lreturn:
      add rsp, 192
      mov r13, [rsp]
      mov rsp, r13
      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      pop rsi
      pop rdi
      #if XNN_HAS_FEATURE(memory_sanitizer)
      jmp xnn_gemm_ukernel_msan_sizeof_c_4
      #else
      ret
      #endif
END_FUNCTION xnn_f32_gemm_minmax_ukernel_5x64__asm_amd64_avx512f_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_5x64__asm_amd64_avx512f_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_5x64__asm_amd64_avx512f_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__