// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast

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
      sub rsp, 256
      # Write rsi (a pointer) to the stack as we need the register.
      mov [rsp + 16], rcx
      # Write r10 (c pointer) to the stack as we need the register.
      mov [rsp + 24], r10

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 32], rax
      mov [rsp + 40], r13

      # Clamp a & c pointers if mr <= 2
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 2
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 48], rcx
      mov [rsp + 56], r10

      # Clamp a & c pointers if mr <= 3
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 3
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 64], rax
      mov [rsp + 72], r13

      # Clamp a & c pointers if mr <= 4
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 4
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 80], rcx
      mov [rsp + 88], r10

      # Clamp a & c pointers if mr <= 5
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 5
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 96], rax
      mov [rsp + 104], r13

      # Clamp a & c pointers if mr <= 6
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 6
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 112], rcx
      mov [rsp + 120], r10

      # Clamp a & c pointers if mr <= 7
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 7
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 128], rax
      mov [rsp + 136], r13

      # Clamp a & c pointers if mr <= 8
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 8
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 144], rcx
      mov [rsp + 152], r10

      # Clamp a & c pointers if mr <= 9
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 9
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 160], rax
      mov [rsp + 168], r13

      # Clamp a & c pointers if mr <= 10
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 10
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 176], rcx
      mov [rsp + 184], r10

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Read a pointers from stack into GP registers.
      mov rcx, [rsp + 16]
      mov rax, [rsp + 32]
      mov r15, [rsp + 48]
      mov r14, [rsp + 64]
      mov r12, [rsp + 80]
      mov r10, [rsp + 96]
      mov r13, [rsp + 112]
      mov rbx, [rsp + 128]
      mov rbp, [rsp + 144]
      mov r8, [rsp + 160]
      mov rdi, [rsp + 176]

      # Initialize accumulators with the biases.
      vmovaps  zmm11, [r9 + 0]
      vmovaps zmm12, zmm11
      vmovaps zmm13, zmm11
      vmovaps zmm14, zmm11
      vmovaps zmm15, zmm11
      vmovaps zmm16, zmm11
      vmovaps zmm17, zmm11
      vmovaps zmm18, zmm11
      vmovaps zmm19, zmm11
      vmovaps zmm20, zmm11
      vmovaps zmm21, zmm11
      add r9, 64

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      add r9, 64
      vbroadcastss zmm2, dword ptr [rcx + r11]
      vfmadd231ps  zmm11, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [rax + r11]
      vfmadd231ps  zmm12, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [r15 + r11]
      vfmadd231ps  zmm13, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [r14 + r11]
      vfmadd231ps  zmm14, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [r12 + r11]
      vfmadd231ps  zmm15, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [r10 + r11]
      vfmadd231ps  zmm16, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [r13 + r11]
      vfmadd231ps  zmm17, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [rbx + r11]
      vfmadd231ps  zmm18, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [rbp + r11]
      vfmadd231ps  zmm19, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [r8 + r11]
      vfmadd231ps  zmm20, zmm2, zmm7
      vbroadcastss zmm2, dword ptr [rdi + r11]
      vfmadd231ps  zmm21, zmm2, zmm7

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vminps  zmm12, zmm1, zmm12
      vminps  zmm13, zmm1, zmm13
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vminps  zmm18, zmm1, zmm18
      vminps  zmm19, zmm1, zmm19
      vminps  zmm20, zmm1, zmm20
      vminps  zmm21, zmm1, zmm21
      vmaxps  zmm11, zmm0, zmm11
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm13, zmm0, zmm13
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm19, zmm0, zmm19
      vmaxps  zmm20, zmm0, zmm20
      vmaxps  zmm21, zmm0, zmm21

      # Pop output pointers from the stack.
      mov rcx, [rsp + 24]
      mov rax, [rsp + 40]
      mov r15, [rsp + 56]
      mov r14, [rsp + 72]
      mov r12, [rsp + 88]
      mov r10, [rsp + 104]
      mov r13, [rsp + 120]
      mov rbx, [rsp + 136]
      mov rbp, [rsp + 152]
      mov r8, [rsp + 168]
      mov rdi, [rsp + 184]

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [rcx], zmm11
      vmovups  [rax], zmm12
      vmovups  [r15], zmm13
      vmovups  [r14], zmm14
      vmovups  [r12], zmm15
      vmovups  [r10], zmm16
      vmovups  [r13], zmm17
      vmovups  [rbx], zmm18
      vmovups  [rbp], zmm19
      vmovups  [r8], zmm20
      vmovups  [rdi], zmm21
      add rcx, 64
      add rax, 64
      add r15, 64
      add r14, 64
      add r12, 64
      add r10, 64
      add r13, 64
      add rbx, 64
      add rbp, 64
      add r8, 64
      add rdi, 64

      # Write output pointers to the stack.
      mov [rsp + 24], rcx
      mov [rsp + 40], rax
      mov [rsp + 56], r15
      mov [rsp + 72], r14
      mov [rsp + 88], r12
      mov [rsp + 104], r10
      mov [rsp + 120], r13
      mov [rsp + 136], rbx
      mov [rsp + 152], rbp
      mov [rsp + 168], r8
      mov [rsp + 184], rdi

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  zmmword ptr [rcx]{k1}, zmm11
      vmovups  zmmword ptr [rax]{k1}, zmm12
      vmovups  zmmword ptr [r15]{k1}, zmm13
      vmovups  zmmword ptr [r14]{k1}, zmm14
      vmovups  zmmword ptr [r12]{k1}, zmm15
      vmovups  zmmword ptr [r10]{k1}, zmm16
      vmovups  zmmword ptr [r13]{k1}, zmm17
      vmovups  zmmword ptr [rbx]{k1}, zmm18
      vmovups  zmmword ptr [rbp]{k1}, zmm19
      vmovups  zmmword ptr [r8]{k1}, zmm20
      vmovups  zmmword ptr [rdi]{k1}, zmm21

.Lreturn:
      add rsp, 256
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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__