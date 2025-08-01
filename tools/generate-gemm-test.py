#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import collections
import os
import re
import sys
import zlib
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon

parser = argparse.ArgumentParser(description="XNNPACK generator")
parser.add_argument(
    "-s", "--spec", metavar="FILE", required=True, help="Spec (YAML) file"
)
parser.add_argument(
    "-o",
    "--output-test",
    action="append",
    metavar="FILE",
    required=True,
    help="Test output (C++ source) file(s)",
)
parser.add_argument(
    "-b",
    "--output-bench",
    metavar="FILE",
    required=False,
    help="Benchmark output (C++ source) file(s)",
)
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]
  if "s" in param_spec:
    param_spec, sr = param_spec.split("s", 1)
    sr = int(sr)
  else:
    sr = 1
  if "c" in param_spec:
    param_spec, kr = param_spec.split("c", 1)
    kr = int(kr)
  else:
    kr = 1
  if "v" in param_spec:
    vector_tile = True
    param_spec, _ = param_spec.split("v", 1)
  else:
    vector_tile = False
  mr, nr = map(int, param_spec.split("x"))
  arch, isa, assembly = xnncommon.parse_target_name(target_name)
  mr_packed = re.search(r"mstep([0-9]+)", target_name)
  if mr_packed:
    mr_packed = mr // int(mr_packed.group(1))
  else:
    mr_packed = mr
  if "sme2" in target_name:
    # SME2 kernels have a non-constant mr, nr that we need to use functions to
    # learn the value of. However, we cannot call these functions unless SME2
    # is supported by the hardware.
    mr = """[]() -> size_t {{
      const struct xnn_hardware_config* hardware_config =
            xnn_init_hardware_config();
      if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {{
        return {name}_get_mr();
      }} else {{
        return 0;
      }}
    }}
""".format(name=name)
    nr = """[]() -> size_t {{
      const struct xnn_hardware_config* hardware_config =
            xnn_init_hardware_config();
      if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {{
        return {name}_get_nr();
      }} else {{
        return 0;
      }}
    }}
""".format(name=name)
    mr_packed = mr

  requantization = common_parts[-3]
  if requantization not in ["fp32", "rndnu", "rndnu16"]:
    requantization = None
  return (
      mr,
      nr,
      kr,
      sr,
      mr_packed,
      vector_tile,
      requantization,
      arch,
      isa,
      assembly,
  )


GEMM_BENCH_CODE = """\
$if CPP_CHECK:
  #if ${CPP_CHECK}
static void ${UKERNEL_NAME}(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    ${GEMM},
    $if INIT_PARAMS is not None:
      ${INIT_PARAMS},
    $if PACK_FN is not None:
      ${PACK_FN},
    $if PACKED_STRIDE_FN is not None:
      ${PACKED_STRIDE_FN},
    /*mr=*/${MR}, /*nr=*/${NR}${NR_SCALE}, /*kr=*/${KR}, /*sr=*/${SR},
    $if PACKED_LHS:
      /*mr_packed=*/${MR_PACKED},
    /*arch_flags=*/${ARCH_FLAGS});
}\n
$if WEIGHTS_DATATYPE in ['qb4w']:
  BENCHMARK_GEMM_BL(${UKERNEL_NAME})
$else:
  BENCHMARK_GEMM(${UKERNEL_NAME})
$if CPP_CHECK:
  #endif  // ${CPP_CHECK}
"""

GEMM_CREATE_TESTS_CODE = """\
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
std::vector<GemmTestParams> CreateTests(
    size_t k_block, size_t adj_k_block,
    ConstantOrFunction mr, ConstantOrFunction nr, size_t kr, size_t sr,
    $if PACKED_LHS:
      ConstantOrFunction mr_packed,
    bool is_igemm,
    bool unsigned_inputs,
    uint8_t planes,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    uint64_t arch_flags = 0) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  $if NR_SCALE != "":
    nr = nr${NR_SCALE};
  std::string nrs = std::to_string(nr);

  $if PACKED_LHS:
    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).mr_packed(mr_packed).unsigned_inputs(unsigned_inputs).planes(planes);
  $else:
    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).unsigned_inputs(unsigned_inputs).planes(planes);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if WEIGHTS_DATATYPE in ['qb4w']:
            .bl(32)
      , test_func, arch_flags));
  $if not PACKED_LHS:
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_eq_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr).k(k_block)
              .a_stride(xnnpack::NextPrime(k_block + 1))
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags));
    }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block)
          $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if WEIGHTS_DATATYPE in ['qb4w']:
            .bl(32)
      , test_func, arch_flags)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block)
          $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if WEIGHTS_DATATYPE in ['qb4w']:
            .bl(32)
      , test_func, arch_flags)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block)
          $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if WEIGHTS_DATATYPE in ['qb4w']:
            .bl(32)
      , test_func, arch_flags)
      .loop_n(1, nr));
  $if IS_PIPELINED:
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s,
        tester.clone()
          .m(mr).n(nr).k(k_block * 2)
          $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if WEIGHTS_DATATYPE in ['qb4w']:
            .bl(32)
      , test_func, arch_flags));
    $if not PACKED_LHS:
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_eq_" + kb2s + "_strided_a",
            tester.clone()
                .m(mr).n(nr).k(k_block * 2)
                .a_stride(xnnpack::NextPrime(k_block * 2 + 1))
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
            , test_func, arch_flags));
      }
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s + "_subtile",
        tester.clone()
            .k(k_block * 2)
            $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            $if WEIGHTS_DATATYPE in ['qb4w']:
              .bl(32)
        , test_func, arch_flags)
        .loop_n(1, nr)
        .loop_m(1, mr));
  $if WEIGHTS_DATATYPE not in ['qb4w']:
      if (k_block > 1) {
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs,
            tester.clone()
                .m(mr).n(nr)
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(1, adj_k_block - 1));
        $if not PACKED_LHS:
          if (!is_igemm) {
            gemm_tests.push_back(GemmTestParams(
                "k_lt_" + akbs + "_strided_a",
                tester.clone()
                    .m(mr).n(nr)
                    .a_stride(xnnpack::NextPrime(adj_k_block + 1))
                    $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                      .b_zero_point(8)
                    $if WEIGHTS_DATATYPE in ['qb4w']:
                      .bl(32)
                , test_func, arch_flags)
                .loop_k(1, adj_k_block - 1));
          }
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs + "_subtile",
            tester.clone()
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(1, adj_k_block - 1)
            .loop_n(1, nr)
            .loop_m(1, mr));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs,
          tester.clone()
              .m(mr).n(nr)
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
      $if not PACKED_LHS:
        if (is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "k_gt_" + akbs + "_strided_a",
              tester.clone()
                  .m(mr).n(nr)
                  .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
        }
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs + "_subtile",
          tester.clone()
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
          .loop_n(1, nr)
          .loop_m(1, mr));
      if (k_block > 1) {
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs,
            tester.clone()
                .m(mr).n(nr)
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(adj_k_block + k_block, k_block * 5, k_block));
        $if not PACKED_LHS:
          if (is_igemm) {
            gemm_tests.push_back(GemmTestParams(
                "k_div_" + kbs + "_strided_a",
                tester.clone()
                    .m(mr).n(nr)
                    .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                    $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                      .b_zero_point(8)
                    $if WEIGHTS_DATATYPE in ['qb4w']:
                      .bl(32)
                , test_func, arch_flags)
                .loop_k(adj_k_block + k_block, k_block * 3, k_block));
          }
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs + "_subtile",
            tester.clone()
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(adj_k_block + k_block, k_block * 5, k_block)
            .loop_n(1, nr)
            .loop_m(1, mr));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs,
          tester.clone()
              .m(mr)
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          $if NR_SCALE != "":
            .loop_n(nr + 1, nr * 2 - 1, 4)
          $else:
            .loop_n(nr + 1, nr * 2 - 1)
          .loop_k(1, k_block * 3, k_block + 1));
      $if not PACKED_LHS:
        if (!is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "n_gt_" + nrs + "_strided_a",
              tester.clone()
                  .m(mr)
                  .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if WEIGHTS_DATATYPE in ['qb4w']:
                    .bl(32)
              , test_func, arch_flags)
              $if NR_SCALE != "":
                .loop_n(nr + 1, nr * 2 - 1, 4)
              $else:
                .loop_n(nr + 1, nr * 2 - 1)
              .loop_k(1, k_block * 3, k_block));
        }
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_subtile",
          tester.clone()
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          $if NR_SCALE != "":
            .loop_n(nr + 1, nr * 2 - 1, 4)
          $else:
            .loop_n(nr + 1, nr * 2 - 1)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_m(1, mr));
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs,
          tester.clone()
              .m(mr)
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1));
      $if not PACKED_LHS:
        if (!is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "n_div_" + nrs + "_strided_a",
              tester.clone()
                  .m(mr)
                  .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if WEIGHTS_DATATYPE in ['qb4w']:
                    .bl(32)
              , test_func, arch_flags)
              .loop_n(nr * 2, nr * 3, nr)
              .loop_k(1, k_block * 3, k_block));
        }
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_subtile",
          tester.clone()
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_m(1, mr));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "small_kernel",
            tester.clone()
                .m(mr).n(nr).ks(3)
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "small_kernel_subtile",
            tester.clone()
                .ks(3)
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1)
            .loop_n(1, nr)
            .loop_m(1, mr));
        gemm_tests.push_back(GemmTestParams(
            "n_gt_" + nrs + "_small_kernel",
            tester.clone()
                .m(mr).ks(3)
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            $if NR_SCALE != "":
              .loop_n(nr + 1, nr * 2 - 1, 4)
            $else:
              .loop_n(nr + 1, nr * 2 - 1)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "n_div_" + nrs + "_small_kernel",
            tester.clone()
                .m(mr).ks(3)
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_n(nr * 2, nr * 3, nr)
            .loop_k(1, k_block * 3, k_block + 1));
      }
      gemm_tests.push_back(GemmTestParams(
          "strided_cm_subtile",
          tester.clone()
              .mr(mr).nr(nr).kr(kr).sr(sr)
              .cm_stride(xnnpack::NextPrime(nr + 1))
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "a_offset",
            tester.clone()
                .m(mr).n(nr).ks(3)
                .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "zero",
            tester.clone()
                .m(mr).n(nr).ks(3)
                .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
                $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if WEIGHTS_DATATYPE in ['qb4w']:
                  .bl(32)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1)
            .loop_zi(0, mr - 1));
      }
      $if ACTIVATION == "MINMAX":
        $if OUTPUT_DATATYPE in {'f32', 'f16'}:
          gemm_tests.push_back(GemmTestParams(
              "min",
              tester.clone()
                  .m(mr).n(nr).k(k_block).min(0.0f)
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if WEIGHTS_DATATYPE in ['qb4w']:
                    .bl(32)
              , test_func, arch_flags));
          gemm_tests.push_back(GemmTestParams(
              "max",
              tester.clone()
                  .m(mr).n(nr).k(k_block).max(0.0f)
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if WEIGHTS_DATATYPE in ['qb4w']:
                    .bl(32)
              , test_func, arch_flags));
        $else:
          gemm_tests.push_back(GemmTestParams(
              "qmin",
              tester.clone()
                  .m(mr).n(nr).k(k_block).qmin(128)
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if WEIGHTS_DATATYPE in ['qb4w']:
                    .bl(32)
              , test_func, arch_flags));
          gemm_tests.push_back(GemmTestParams(
              "qmax",
              tester.clone()
                  .m(mr).n(nr).k(k_block).qmax(128)
                  $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if WEIGHTS_DATATYPE in ['qb4w']:
                    .bl(32)
              , test_func, arch_flags));
      gemm_tests.push_back(GemmTestParams(
          "strided_cm",
          tester.clone()
              .m(mr).n(nr).k(k_block)
              .cm_stride(xnnpack::NextPrime(nr + 1))
              $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if WEIGHTS_DATATYPE in ['qb4w']:
                .bl(32)
          , test_func, arch_flags));
      $if INPUT_DATATYPE == "qu8":
        gemm_tests.push_back(GemmTestParams(
            "no_a_zero_point",
            tester.clone()
                .m(mr).n(nr).a_zero_point(0)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1));
      $if INPUT_DATATYPE == "qu8":
        gemm_tests.push_back(GemmTestParams(
            "no_b_zero_point",
            tester.clone()
                .m(mr).n(nr).b_zero_point(0)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "b_zero_point",
            tester.clone()
                .m(mr).n(nr).k(k_block)
            , test_func, arch_flags)
            .loop_bzp(0, 255));
        gemm_tests.push_back(GemmTestParams(
            "no_zero_point",
            tester.clone()
                .m(mr).n(nr)
                .a_zero_point(0)
                .b_zero_point(0)
            , test_func, arch_flags)
            .loop_k(1, k_block * 3, k_block + 1));
  $if WEIGHTS_DATATYPE in ['qb4w']:
    gemm_tests.push_back(GemmTestParams(
        "bl",
        tester.clone()
            .m(mr).n(nr).k(k_block * 12)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
        .loop_bl(32, k_block * 32, 32));

  return gemm_tests;
}
"""

GEMM_TEST_CODE = """\
$if CPP_CHECK:
  #if ${CPP_CHECK}
INSTANTIATE_TEST_SUITE_P(
    ${TEST_NAME}, GemmTest,
    testing::ValuesIn(CreateTests(
        /*k_block=*/${KBLOCK},
        /*adj_k_block=*/${ADJKBLOCK},
        /*mr=*/${MR}, /*nr=*/${NR}, /*kr=*/${KR}, /*sr=*/${SR},
        $if PACKED_LHS:
          /*mr_packed=*/${MR_PACKED},
        /*is_igemm=*/${"true" if UKERNEL_TYPE in ("IGEMM",) else "false"},
        /*unsigned_inputs=*/${"true" if UNSIGNED_INPUTS else "false"},
        /*planes=*/${PLANES},
        [](GemmMicrokernelTester& tester) {
          tester.${TEST_FUN}(${",\\n                      ".join(TEST_ARGS)});
        },
        ${ARCH_FLAGS})),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });

$if TEST_NAME.startswith('GENERATE') and INPUT_DATATYPE in ['f32', 'f16']:
  TEST(${TEST_NAME}, subtile_m_upto_mr) {
    TEST_REQUIRES_ARCH_FLAGS(${ARCH_FLAGS});
    for (uint32_t max_mr = 1; max_mr <= ${MR}; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= ${KBLOCK * 2}; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            $if NR > 1:
              .nr(${NR})
            $if KR > 1:
              .kr(${KR})
            $if SR > 1:
              .sr(${SR})
            .m(m)
            $if NR > 1:
              .n(${NR})
            .k(k)
            $if WEIGHTS_DATATYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            .${TEST_FUN}(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

$if CPP_CHECK:
  #endif  // ${CPP_CHECK}
"""


def generate_test_cases(
    ukernel,
    mr,
    nr,
    kr,
    sr,
    mr_packed,
    k_block,
    unsigned_inputs,
    planes,
    vector_tile,
    init_fn,
    pack_fn,
    pack_lh_fn,
    pack_lh_size_fn,
    packed_stride_fn,
    requantization,
    is_pipelined,
    cpp_check,
    isa,
):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    mr: MR parameter of the GEMM micro-kernel.
    nr: NR parameter of the GEMM micro-kernel.
    kr: KR parameter of the GEMM micro-kernel.
    sr: SR parameter of the GEMM micro-kernel.
    mr_packed: Optional MR parameter for the left-hand packing function.
    k_block: Number of K values processed per one iteration of the main loop of
      the micro-kernel.
    unsigned_inputs: whether the inputs should be converted to unsigned
      integers. Some microkernels are more efficient with unsigned inputs.
    planes: the number of packing planes used by the microkernel.
    vector_tile: Indicates if vector tile for NR is specified in vectors rather
      than elements.
    init_fn: C name of the function to initialize microkernel parameters.
    pack_fn: C name of the function to pack the weights.
    pack_lh_fn: Optional name of a function to pack the LHS for the GEMM.
    pack_lh_size_fn: Optional name of a function to compute the size of a packed
      LHS for the GEMM.
    packed_stride_fn: C name of the function to compute the packed weights
      stride.
    requantization: name of the requantization scheme used by the microkernel.
    is_pipelined: Indicates if the micro-kernel is implemented with software
      pipelining. Additional test cases are generated for software pipelined
      micro-kernels to separately test prologue + epiloque of the pipelined loop
      and iteration of the pipelined loop.
    cpp_check: Optional preprocessor macro to check for the availability of the
      micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, ukernel_name = ukernel.split("_", 1)

  input_datatype = None
  output_datatype = None
  weights_datatype = None
  accum_type = None
  features = None
  ukernel_type = "gemm"
  activation = "linear"
  for token in ukernel.split("ukernel", 1)[0].split("_")[1:]:
    if token in {"minmax", "linear", "relu"}:
      activation = token
    elif token in {
        "f32",
        "f16",
        "qs8",
        "qu8",
        "qd8",
        "qp8",
        "pf32",
        "pf16",
        "pqs8",
        "bf16",
    }:
      if input_datatype is None:
        input_datatype = token
      elif weights_datatype is None:
        weights_datatype = token
      else:
        output_datatype = weights_datatype
        weights_datatype = token
    elif token in {"gemm", "igemm", "gemminc"}:
      ukernel_type = token
    elif token in {"qc4w", "qc8w", "qb4w"}:
      weights_datatype = token
    elif token in {"f32acc"}:
      accum_type = token
    elif token in {"fp32", "fp32acc", "rndnu", "rndnu16", "ppmm"}:
      features = token
    elif token:
      print(f'Unknown token "{token}" in "{ukernel}".')

  weights_datatype = weights_datatype or input_datatype

  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)

  if pack_lh_fn and pack_lh_size_fn:
    test_args.append(f"{pack_lh_fn}")
    test_args.append(f"{pack_lh_size_fn}")
  elif pack_lh_fn or pack_lh_size_fn:
    raise ValueError(
        "Either both or neither of pack_lh_fn and/or pack_lh_size_fn must be"
        " supplied"
    )

  if pack_fn:
    test_args.append(pack_fn)
  if packed_stride_fn:
    test_args.append(packed_stride_fn)

  if init_fn and requantization:
    requantization_datatype = {"qc8": "qs8", "pqs8": "qs8"}.get(
        input_datatype, input_datatype
    )
    test_args.append(
        "xnn_%s_requantize_%s" % (requantization_datatype, requantization)
    )

  output_datatype = init_fn.split("_")[2] if init_fn else output_datatype

  nr_scale = ""
  if vector_tile:
    accum_type = (
        accum_type
        or {
            "qs8": "int32_t",
            "qd8": "int32_t",
            "qp8": "int32_t",
            "qu8": "int32_t",
            "f16": "xnn_float16",
            "f32": "float",
        }[input_datatype]
    )
    nr_scale = {
        "rvv": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % accum_type
    }[isa]
  test_fun_name = "".join(ukernel.split("_")[1:4]).upper()
  if test_fun_name in {"QP8F32QC8W"}:
    test_fun_name = "_".join(["Test", test_fun_name])
  elif input_datatype in {"pf32", "pf16", "pqs8"}:
    test_fun_name = "_".join(["Test", input_datatype.upper()])
  else:
    test_fun_name = "Test"
  test_args = {
      "TEST_NAME": ukernel_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "TEST_FUN": test_fun_name,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "INPUT_DATATYPE": input_datatype,
      "WEIGHTS_DATATYPE": weights_datatype,
      "ACTIVATION": activation.upper(),
      "MR": mr,
      "NR": nr,
      "KR": kr,
      "SR": sr,
      "MR_PACKED": mr_packed,
      "KBLOCK": k_block,
      "UNSIGNED_INPUTS": unsigned_inputs,
      "PLANES": planes,
      "NR_SCALE": nr_scale,
      "ADJKBLOCK": 2 * k_block if is_pipelined else k_block,
      "IS_PIPELINED": is_pipelined,
      "ARCH_FLAGS": xnncommon.get_arch_flags(isa),
      "next_prime": next_prime,
      "CPP_CHECK": cpp_check,
      "OUTPUT_DATATYPE": output_datatype,
      "PACKED_LHS": input_datatype in {"qp8", "pf32", "pf16", "pqs8"},
  }

  create_test_case = xngen.preprocess(GEMM_CREATE_TESTS_CODE, test_args)

  test_case = xngen.preprocess(GEMM_TEST_CODE, test_args)

  benchmark = xngen.preprocess(
      GEMM_BENCH_CODE,
      {
          "UKERNEL_NAME": ukernel_name,
          "GEMM": ukernel,
          "WEIGHTS_DATATYPE": weights_datatype,
          "INPUT_DATATYPE": input_datatype,
          "INIT_PARAMS": init_fn,
          "PACK_FN": pack_fn,
          "PACKED_STRIDE_FN": packed_stride_fn,
          "MR": mr,
          "NR": nr,
          "KR": kr,
          "SR": sr,
          "MR_PACKED": mr_packed,
          "NR_SCALE": nr_scale,
          "ARCH_FLAGS": xnncommon.get_arch_flags(isa),
          "CPP_CHECK": cpp_check,
          "PACKED_LHS": input_datatype in {"qp8", "pf32", "pf16", "pqs8"},
      },
  )
  return create_test_case, test_case, benchmark


def main(args):
  options = parser.parse_args(args)
  num_output_files = len(options.output_test)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    constant_or_function = """\
namespace {

struct ConstantOrFunction {
  ConstantOrFunction(size_t x) : fn([x]() { return x; }) {}  //NOLINT
  ConstantOrFunction(int x) : fn([x]() { return x; }) {}  //NOLINT
  template <typename Fn>
  ConstantOrFunction(Fn fn) : fn(std::move(fn)) {}  //NOLINT

  std::function<size_t()> fn;

  operator size_t() const { return fn(); }  //NOLINT
};

}  // namespace
"""

    tests = """\
// clang-format off
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/ppmm.h"
#include "src/xnnpack/requantization.h"
#include "test/gemm-microkernel-tester.h"
#include "test/next_prime.h"

{constant_or_function}
""".format(specification=options.spec, generator=sys.argv[0], constant_or_function=constant_or_function)

    benches = """\
// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include <cstdint>
#include <functional>

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"

{constant_or_function}
""".format(specification=options.spec, generator=sys.argv[0], constant_or_function=constant_or_function)

    test_outputs = collections.defaultdict(str)
    bench_outputs = benches

    isa_hierarchy = xnncommon.isa_hierarchy_map()

    # Cached `CreateTests` functions.
    idx_from_create_tests_hash = None
    idx_from_create_tests_hash = collections.defaultdict(
        lambda: len(idx_from_create_tests_hash) + 1
    )
    create_tests_from_idx = {}

    benches = [""] * len(isa_hierarchy)
    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      k_block = int(ukernel_spec["k-block"])
      if "unsigned-inputs" in ukernel_spec:
        unsigned_inputs = int(ukernel_spec["unsigned-inputs"])
      else:
        unsigned_inputs = False
      if "planes" in ukernel_spec:
        planes = int(ukernel_spec["planes"])
      else:
        planes = 1
      init_fn = ukernel_spec.get("init")
      pack_fn = ukernel_spec.get("pack")
      pack_lh_fn = ukernel_spec.get("pack-lh-fn")
      pack_lh_size_fn = ukernel_spec.get("pack-lh-size-fn")
      packed_stride_fn = ukernel_spec.get("packed-stride")
      pipelined = bool(ukernel_spec.get("pipelined", False))
      cpp_check = ukernel_spec.get("cpp-check", False)
      (
          mr,
          nr,
          kr,
          sr,
          mr_packed,
          vector_tile,
          requantization,
          arch,
          isa,
          assembly,
      ) = split_ukernel_name(name)

      if k_block < kr * sr:
        print(f"Error: k_block ({k_block}) must be >= kr * sr ({kr} * {sr}) for kernel {name} in {options.spec}")
        exit(1)

      create_tests, test_case, bench_case = generate_test_cases(
          name,
          mr,
          nr,
          kr,
          sr,
          mr_packed,
          k_block,
          unsigned_inputs,
          planes,
          vector_tile,
          init_fn,
          pack_fn,
          pack_lh_fn,
          pack_lh_size_fn,
          packed_stride_fn,
          requantization,
          pipelined,
          cpp_check,
          isa,
      )

      # Store or reuse the `CreateTests` function?
      create_tests_hash = hash(create_tests)
      create_tests_idx = idx_from_create_tests_hash[create_tests_hash]
      if create_tests_idx not in create_tests_from_idx:
        create_tests_from_idx[create_tests_idx] = create_tests.replace(
            "CreateTests(", f"CreateTests{create_tests_idx}("
        )
        if isa == "rvv":
          create_tests_from_idx[create_tests_idx] = (
              xnncommon.postprocess_test_case(
                  create_tests_from_idx[create_tests_idx], arch, isa, assembly
              )
          )
      test_case = test_case.replace(
          "CreateTests(", f"CreateTests{create_tests_idx}("
      )

      # Hash the name of each microkernel and figure out which output file to
      # write it to.
      output_index = zlib.crc32(bytes(name, "utf-8")) % num_output_files
      test_outputs[
          options.output_test[output_index]
      ] += "\n\n" + xnncommon.postprocess_test_case(
          test_case, arch, isa, assembly
      )
      benches[
          isa_hierarchy.get(isa, 0)
      ] += "\n\n" + xnncommon.postprocess_test_case(
          bench_case, arch, isa, assembly
      )

    for arch_idx in reversed(range(len(isa_hierarchy))):
      bench_outputs += benches[arch_idx]

    bench_outputs += """\n
#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
"""

    if options.output_bench:
      # Strip out consecutive preprocessor `endif`/`if` pairs.
      for _ in range(2):
        bench_outputs = re.sub(
            r"^ *\#endif  // ([^\n]+)\n+ *\#if \1\n",
            "\n",
            bench_outputs,
            flags=re.MULTILINE,
        )
      output_name = options.output_bench
      xnncommon.overwrite_if_changed(output_name, bench_outputs)

    create_tests = (
        "namespace {\n\n"
        + "\n".join(create_tests_from_idx.values())
        + "\n}  // namespace\n"
    )
    test_outputs = {
        k: tests + "\n" + create_tests + v + "\n"
        for k, v in test_outputs.items()
    }

    # Strip out consecutive preprocessor `endif`/`if` pairs.
    for _ in range(2):
      test_outputs = {
          k: re.sub(
              r"^ *\#endif  // ([^\n]+)\n+ *\#if \1\n",
              "\n",
              v,
              flags=re.MULTILINE,
          )
          for k, v in test_outputs.items()
      }

    for output_name in options.output_test:
      xnncommon.overwrite_if_changed(output_name, test_outputs[output_name])


if __name__ == "__main__":
  main(sys.argv[1:])
