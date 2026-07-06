# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for uint8 x int2 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_uint8_int2_int32(x86):

  def __init__(self, arch, vector_bits, tile_shape):
    super().__init__(
        arch, "uint8_int2_int32", "int32_t", vector_bits, tile_shape
    )
    self.a_type = "uint8_t"
    self.b_type = "int2x4"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def b_tile_size_bytes(self):
    return f"{self.tile_shape[1] * self.tile_shape[2] // 4}"

  def header(self):
    return super().header() + """
#include <immintrin.h>

namespace {

YNN_INTRINSIC int32_t unaligned_load_u8x4(const uint8_t* ptr) {
  int32_t value;
  memcpy(&value, ptr, sizeof(int32_t));
  return value;
}

}  // namespace
"""

  def load_a_tile(self, i, k):
    a = f"unaligned_load_u8x4({self.a_ptr(i, k)})"
    return f"__m{self.bits}i a_{i}_{k} = {self._mm()}_set1_epi32({a});\n"

  def product(self, i, j, k):
    mm = self._mm()
    a_ik = f"a_{i}_{k}"
    return (
        f"c_{i}_{j} = {mm}_add_epi32(c_{i}_{j},"
        f" {mm}_madd_epi16({mm}_maddubs_epi16({a_ik}, b_{k}_{j}),"
        f" {mm}_set1_epi16(1)));\n"
    )


class x86_avx2_uint8_int2_int32(x86_uint8_int2_int32, x86_avx):

  def __init__(self, arch="avx2", vector_bits=256):
    super().__init__(arch, vector_bits, tile_shape=(1, 8, 4))

  def begin_func(self, func_name):
    result = super().begin_func(func_name)
    result += """
const __m256i mask = _mm256_set1_epi8(0x03);
const __m256i sign_lut = _mm256_setr_epi8(
    0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1,
    0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1);
const __m256i shift_amt = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6);
const __m256i perm_mask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
const __m256i shuf_mask = _mm256_setr_epi8(
    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
"""
    return result

  def load_b_tile(self, k, j):
    bits = self.bits
    mm = self._mm()
    return f"""
__m{bits}i b_{k}_{j} = {mm}_broadcastq_epi64(_mm_loadl_epi64({self.b_ptr(k, j, "__m128i")}));
b_{k}_{j} = {mm}_srlv_epi32(b_{k}_{j}, shift_amt);
b_{k}_{j} = {mm}_permutevar8x32_epi32(b_{k}_{j}, perm_mask);
b_{k}_{j} = {mm}_shuffle_epi8(b_{k}_{j}, shuf_mask);
b_{k}_{j} = {mm}_and_si{bits}(b_{k}_{j}, mask);
b_{k}_{j} = {mm}_shuffle_epi8(sign_lut, b_{k}_{j});
"""


class x86_avx512_uint8_int2_int32(x86_uint8_int2_int32, x86_avx512):

  def __init__(self, arch="avx512", vector_bits=512):
    super().__init__(arch, vector_bits, tile_shape=(1, 16, 4))

  def begin_func(self, func_name):
    return super().begin_func(func_name) + """
const __m512i mask = _mm512_set1_epi8(0x03);
const __m512i sign_lut = _mm512_set_epi8(
    -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0,
    -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0,
    -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0,
    -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0);
const __m512i broadcast_x4_perm = _mm512_setr_epi32(
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
const __m512i shift_amt = _mm512_setr_epi32(
    0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
const __m512i shuf_mask = _mm512_set_epi8(
    15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
    15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
    15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
    15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
"""

  def load_b_tile(self, k, j):
    bits = self.bits
    mm = self._mm()
    return f"""
__m{bits}i b_{k}_{j} = {mm}_castsi128_si{bits}(_mm_loadu_si128({self.b_ptr(k, j, "__m128i")}));
b_{k}_{j} = {mm}_permutexvar_epi32(broadcast_x4_perm, b_{k}_{j});
b_{k}_{j} = {mm}_srlv_epi32(b_{k}_{j}, shift_amt);
b_{k}_{j} = {mm}_shuffle_epi8(b_{k}_{j}, shuf_mask);
b_{k}_{j} = {mm}_and_si{bits}(b_{k}_{j}, mask);
b_{k}_{j} = {mm}_shuffle_epi8(sign_lut, b_{k}_{j});
"""


class x86_avx512vnni_uint8_int2_int32(x86_avx512_uint8_int2_int32):

  def __init__(self, arch="avx512vnni"):
    super().__init__(arch)

  def product(self, i, j, k):
    mm = self._mm()
    return f"c_{i}_{j} = {mm}_dpbusd_epi32(c_{i}_{j}, a_{i}_{k}, b_{k}_{j});\n"


generate_dot_kernels(
    x86_avx2_uint8_int2_int32(),
    [
        (1, 16, 8),
        (2, 16, 8),
        (3, 16, 8),
        (1, 8, 8),
        (3, 8, 8),
        (4, 8, 8),
        (8, 8, 8),
    ],
)

generate_dot_kernels(
    x86_avx512_uint8_int2_int32(),
    [
        (1, 32, 8),
        (2, 32, 8),
        (3, 32, 8),
        (4, 32, 8),
        (5, 32, 8),
        (1, 16, 8),
        (5, 16, 8),
        (8, 16, 8),
    ],
)

generate_dot_kernels(
    x86_avx512vnni_uint8_int2_int32(),
    [
        (1, 32, 8),
        (2, 32, 8),
        (3, 32, 8),
        (4, 32, 8),
        (5, 32, 8),
        (1, 16, 8),
        (5, 16, 8),
        (8, 16, 8),
    ],
)
