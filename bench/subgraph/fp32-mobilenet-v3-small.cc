// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "bench/subgraph/models.h"
#include "include/xnnpack.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t FP32MobileNetV3Small() {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;  // NOLINT(runtime/random_device)
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v0_dims = {{1, 224, 224, 3}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v1_dims = {{1, 112, 112, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v1_dims.size(), v1_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v2_dims = {{1, 112, 112, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v2_dims.size(), v2_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  uint32_t v3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v3_dims = {{1, 56, 56, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v3_dims.size(), v3_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v3" << std::endl;
    return nullptr;
  }

  uint32_t v4 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v4_dims = {{1, 1, 1, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v4_dims.size(), v4_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v4" << std::endl;
    return nullptr;
  }

  uint32_t v5 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v5_dims = {{1, 1, 1, 8}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v5_dims.size(), v5_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v5" << std::endl;
    return nullptr;
  }

  uint32_t v6 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v6_dims = {{1, 1, 1, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v6_dims.size(), v6_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v6" << std::endl;
    return nullptr;
  }

  uint32_t v7 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v7_dims = {{1, 1, 1, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v7_dims.size(), v7_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v7" << std::endl;
    return nullptr;
  }

  uint32_t v8 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v8_dims = {{1, 56, 56, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v8_dims.size(), v8_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v8" << std::endl;
    return nullptr;
  }

  uint32_t v9 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v9_dims = {{1, 56, 56, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v9_dims.size(), v9_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v9" << std::endl;
    return nullptr;
  }

  uint32_t v10 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v10_dims = {{1, 56, 56, 72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v10_dims.size(), v10_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v10" << std::endl;
    return nullptr;
  }

  uint32_t v11 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v11_dims = {{1, 28, 28, 72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v11_dims.size(), v11_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v11" << std::endl;
    return nullptr;
  }

  uint32_t v12 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v12_dims = {{1, 28, 28, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v12_dims.size(), v12_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v12" << std::endl;
    return nullptr;
  }

  uint32_t v13 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v13_dims = {{1, 28, 28, 88}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v13_dims.size(), v13_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v13" << std::endl;
    return nullptr;
  }

  uint32_t v14 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v14_dims = {{1, 28, 28, 88}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v14_dims.size(), v14_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v14" << std::endl;
    return nullptr;
  }

  uint32_t v15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v15_dims = {{1, 28, 28, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v15_dims.size(), v15_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v15" << std::endl;
    return nullptr;
  }

  uint32_t v16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v16_dims = {{1, 28, 28, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v16_dims.size(), v16_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v16" << std::endl;
    return nullptr;
  }

  uint32_t v17 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v17_dims = {{1, 28, 28, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v17_dims.size(), v17_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v17" << std::endl;
    return nullptr;
  }

  uint32_t v18 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v18_dims = {{1, 28, 28, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v18_dims.size(), v18_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v18" << std::endl;
    return nullptr;
  }

  uint32_t v19 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v19_dims = {{1, 14, 14, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v19_dims.size(), v19_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v19" << std::endl;
    return nullptr;
  }

  uint32_t v20 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v20_dims = {{1, 14, 14, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v20_dims.size(), v20_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v20" << std::endl;
    return nullptr;
  }

  uint32_t v21 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v21_dims = {{1, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v21_dims.size(), v21_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v21" << std::endl;
    return nullptr;
  }

  uint32_t v22 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v22_dims = {{1, 1, 1, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v22_dims.size(), v22_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v22" << std::endl;
    return nullptr;
  }

  uint32_t v23 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v23_dims = {{1, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v23_dims.size(), v23_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v23" << std::endl;
    return nullptr;
  }

  uint32_t v24 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v24_dims = {{1, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v24_dims.size(), v24_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v24" << std::endl;
    return nullptr;
  }

  uint32_t v25 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v25_dims = {{1, 14, 14, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v25_dims.size(), v25_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v25" << std::endl;
    return nullptr;
  }

  uint32_t v26 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v26_dims = {{1, 14, 14, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v26_dims.size(), v26_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v26" << std::endl;
    return nullptr;
  }

  uint32_t v27 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v27_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v27_dims.size(), v27_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v27" << std::endl;
    return nullptr;
  }

  uint32_t v28 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v28_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v28_dims.size(), v28_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v28" << std::endl;
    return nullptr;
  }

  uint32_t v29 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v29_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v29_dims.size(), v29_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v29" << std::endl;
    return nullptr;
  }

  uint32_t v30 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v30_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v30_dims.size(), v30_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v30" << std::endl;
    return nullptr;
  }

  uint32_t v31 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v31_dims = {{1, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v31_dims.size(), v31_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v31" << std::endl;
    return nullptr;
  }

  uint32_t v32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v32_dims = {{1, 1, 1, 64}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v32_dims.size(), v32_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v32" << std::endl;
    return nullptr;
  }

  uint32_t v33 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v33_dims = {{1, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v33_dims.size(), v33_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v33" << std::endl;
    return nullptr;
  }

  uint32_t v34 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v34_dims = {{1, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v34_dims.size(), v34_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v34" << std::endl;
    return nullptr;
  }

  uint32_t v35 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v35_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v35_dims.size(), v35_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v35" << std::endl;
    return nullptr;
  }

  uint32_t v36 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v36_dims = {{1, 14, 14, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v36_dims.size(), v36_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v36" << std::endl;
    return nullptr;
  }

  uint32_t v37 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v37_dims = {{1, 14, 14, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v37_dims.size(), v37_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v37" << std::endl;
    return nullptr;
  }

  uint32_t v38 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v38_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v38_dims.size(), v38_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v38" << std::endl;
    return nullptr;
  }

  uint32_t v39 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v39_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v39_dims.size(), v39_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v39" << std::endl;
    return nullptr;
  }

  uint32_t v40 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v40_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v40_dims.size(), v40_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v40" << std::endl;
    return nullptr;
  }

  uint32_t v41 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v41_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v41_dims.size(), v41_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v41" << std::endl;
    return nullptr;
  }

  uint32_t v42 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v42_dims = {{1, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v42_dims.size(), v42_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v42" << std::endl;
    return nullptr;
  }

  uint32_t v43 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v43_dims = {{1, 1, 1, 64}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v43_dims.size(), v43_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v43" << std::endl;
    return nullptr;
  }

  uint32_t v44 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v44_dims = {{1, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v44_dims.size(), v44_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v44" << std::endl;
    return nullptr;
  }

  uint32_t v45 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v45_dims = {{1, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v45_dims.size(), v45_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v45" << std::endl;
    return nullptr;
  }

  uint32_t v46 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v46_dims = {{1, 14, 14, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v46_dims.size(), v46_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v46" << std::endl;
    return nullptr;
  }

  uint32_t v47 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v47_dims = {{1, 14, 14, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v47_dims.size(), v47_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v47" << std::endl;
    return nullptr;
  }

  uint32_t v48 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v48_dims = {{1, 14, 14, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v48_dims.size(), v48_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v48" << std::endl;
    return nullptr;
  }

  uint32_t v49 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v49_dims = {{1, 14, 14, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v49_dims.size(), v49_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v49" << std::endl;
    return nullptr;
  }

  uint32_t v50 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v50_dims = {{1, 14, 14, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v50_dims.size(), v50_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v50" << std::endl;
    return nullptr;
  }

  uint32_t v51 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v51_dims = {{1, 14, 14, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v51_dims.size(), v51_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v51" << std::endl;
    return nullptr;
  }

  uint32_t v52 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v52_dims = {{1, 14, 14, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v52_dims.size(), v52_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v52" << std::endl;
    return nullptr;
  }

  uint32_t v53 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v53_dims = {{1, 1, 1, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v53_dims.size(), v53_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v53" << std::endl;
    return nullptr;
  }

  uint32_t v54 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v54_dims = {{1, 1, 1, 32}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v54_dims.size(), v54_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v54" << std::endl;
    return nullptr;
  }

  uint32_t v55 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v55_dims = {{1, 1, 1, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v55_dims.size(), v55_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v55" << std::endl;
    return nullptr;
  }

  uint32_t v56 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v56_dims = {{1, 1, 1, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v56_dims.size(), v56_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v56" << std::endl;
    return nullptr;
  }

  uint32_t v57 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v57_dims = {{1, 14, 14, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v57_dims.size(), v57_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v57" << std::endl;
    return nullptr;
  }

  uint32_t v58 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v58_dims = {{1, 14, 14, 48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v58_dims.size(), v58_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v58" << std::endl;
    return nullptr;
  }

  uint32_t v59 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v59_dims = {{1, 14, 14, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v59_dims.size(), v59_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v59" << std::endl;
    return nullptr;
  }

  uint32_t v60 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v60_dims = {{1, 14, 14, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v60_dims.size(), v60_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v60" << std::endl;
    return nullptr;
  }

  uint32_t v61 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v61_dims = {{1, 14, 14, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v61_dims.size(), v61_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v61" << std::endl;
    return nullptr;
  }

  uint32_t v62 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v62_dims = {{1, 14, 14, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v62_dims.size(), v62_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v62" << std::endl;
    return nullptr;
  }

  uint32_t v63 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v63_dims = {{1, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v63_dims.size(), v63_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v63" << std::endl;
    return nullptr;
  }

  uint32_t v64 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v64_dims = {{1, 1, 1, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v64_dims.size(), v64_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v64" << std::endl;
    return nullptr;
  }

  uint32_t v65 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v65_dims = {{1, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v65_dims.size(), v65_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v65" << std::endl;
    return nullptr;
  }

  uint32_t v66 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v66_dims = {{1, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v66_dims.size(), v66_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v66" << std::endl;
    return nullptr;
  }

  uint32_t v67 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v67_dims = {{1, 14, 14, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v67_dims.size(), v67_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v67" << std::endl;
    return nullptr;
  }

  uint32_t v68 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v68_dims = {{1, 14, 14, 48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v68_dims.size(), v68_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v68" << std::endl;
    return nullptr;
  }

  uint32_t v69 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v69_dims = {{1, 14, 14, 48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v69_dims.size(), v69_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v69" << std::endl;
    return nullptr;
  }

  uint32_t v70 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v70_dims = {{1, 14, 14, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v70_dims.size(), v70_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v70" << std::endl;
    return nullptr;
  }

  uint32_t v71 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v71_dims = {{1, 14, 14, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v71_dims.size(), v71_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v71" << std::endl;
    return nullptr;
  }

  uint32_t v72 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v72_dims = {{1, 7, 7, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v72_dims.size(), v72_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v72" << std::endl;
    return nullptr;
  }

  uint32_t v73 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v73_dims = {{1, 7, 7, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v73_dims.size(), v73_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v73" << std::endl;
    return nullptr;
  }

  uint32_t v74 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v74_dims = {{1, 1, 1, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v74_dims.size(), v74_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v74" << std::endl;
    return nullptr;
  }

  uint32_t v75 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v75_dims = {{1, 1, 1, 72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v75_dims.size(), v75_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v75" << std::endl;
    return nullptr;
  }

  uint32_t v76 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v76_dims = {{1, 1, 1, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v76_dims.size(), v76_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v76" << std::endl;
    return nullptr;
  }

  uint32_t v77 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v77_dims = {{1, 1, 1, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v77_dims.size(), v77_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v77" << std::endl;
    return nullptr;
  }

  uint32_t v78 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v78_dims = {{1, 7, 7, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v78_dims.size(), v78_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v78" << std::endl;
    return nullptr;
  }

  uint32_t v79 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v79_dims = {{1, 7, 7, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v79_dims.size(), v79_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v79" << std::endl;
    return nullptr;
  }

  uint32_t v80 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v80_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v80_dims.size(), v80_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v80" << std::endl;
    return nullptr;
  }

  uint32_t v81 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v81_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v81_dims.size(), v81_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v81" << std::endl;
    return nullptr;
  }

  uint32_t v82 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v82_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v82_dims.size(), v82_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v82" << std::endl;
    return nullptr;
  }

  uint32_t v83 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v83_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v83_dims.size(), v83_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v83" << std::endl;
    return nullptr;
  }

  uint32_t v84 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v84_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v84_dims.size(), v84_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v84" << std::endl;
    return nullptr;
  }

  uint32_t v85 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v85_dims = {{1, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v85_dims.size(), v85_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v85" << std::endl;
    return nullptr;
  }

  uint32_t v86 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v86_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v86_dims.size(), v86_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v86" << std::endl;
    return nullptr;
  }

  uint32_t v87 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v87_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v87_dims.size(), v87_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v87" << std::endl;
    return nullptr;
  }

  uint32_t v88 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v88_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v88_dims.size(), v88_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v88" << std::endl;
    return nullptr;
  }

  uint32_t v89 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v89_dims = {{1, 7, 7, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v89_dims.size(), v89_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v89" << std::endl;
    return nullptr;
  }

  uint32_t v90 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v90_dims = {{1, 7, 7, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v90_dims.size(), v90_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v90" << std::endl;
    return nullptr;
  }

  uint32_t v91 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v91_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v91_dims.size(), v91_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v91);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v91" << std::endl;
    return nullptr;
  }

  uint32_t v92 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v92_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v92_dims.size(), v92_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v92" << std::endl;
    return nullptr;
  }

  uint32_t v93 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v93_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v93_dims.size(), v93_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v93" << std::endl;
    return nullptr;
  }

  uint32_t v94 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v94_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v94_dims.size(), v94_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v94" << std::endl;
    return nullptr;
  }

  uint32_t v95 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v95_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v95_dims.size(), v95_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v95" << std::endl;
    return nullptr;
  }

  uint32_t v96 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v96_dims = {{1, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v96_dims.size(), v96_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v96" << std::endl;
    return nullptr;
  }

  uint32_t v97 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v97_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v97_dims.size(), v97_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v97" << std::endl;
    return nullptr;
  }

  uint32_t v98 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v98_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v98_dims.size(), v98_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v98" << std::endl;
    return nullptr;
  }

  uint32_t v99 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v99_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v99_dims.size(), v99_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v99);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v99" << std::endl;
    return nullptr;
  }

  uint32_t v100 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v100_dims = {{1, 7, 7, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v100_dims.size(), v100_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v100);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v100" << std::endl;
    return nullptr;
  }

  uint32_t v101 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v101_dims = {{1, 7, 7, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v101_dims.size(), v101_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v101);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v101" << std::endl;
    return nullptr;
  }

  uint32_t v102 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v102_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v102_dims.size(), v102_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v102);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v102" << std::endl;
    return nullptr;
  }

  uint32_t v103 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v103_dims = {{1, 7, 7, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v103_dims.size(), v103_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v103);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v103" << std::endl;
    return nullptr;
  }

  uint32_t v104 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v104_dims = {{1, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v104_dims.size(), v104_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v104);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v104" << std::endl;
    return nullptr;
  }

  uint32_t v105 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v105_dims = {{1, 1, 1, 1024}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v105_dims.size(), v105_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v105);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v105" << std::endl;
    return nullptr;
  }

  uint32_t v106 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v106_dims = {{1, 1, 1, 1024}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v106_dims.size(), v106_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v106);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v106" << std::endl;
    return nullptr;
  }

  uint32_t v107 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v107_dims = {{1, 1, 1, 1024}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v107_dims.size(), v107_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v107);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v107" << std::endl;
    return nullptr;
  }

  uint32_t v108 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v108_dims = {{1, 1, 1, 1001}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v108_dims.size(), v108_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v108);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v108" << std::endl;
    return nullptr;
  }

  uint32_t v109 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v109_dims = {{1, 1001}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v109_dims.size(), v109_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v109);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v109" << std::endl;
    return nullptr;
  }

  uint32_t v110 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v110_dims = {{1, 1001}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v110_dims.size(), v110_dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v110);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v110" << std::endl;
    return nullptr;
  }

  static std::vector<float> w111_data;
  w111_data.resize(XNN_PAD_EXTRA_BYTES(432, float));
  uint32_t w111 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w111_dims = {{16, 3, 3, 3}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w111_dims.size(), w111_dims.data(),
      /*data=*/w111_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w111);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w111" << std::endl;
    return nullptr;
  }

  static std::vector<float> w112_data;
  w112_data.resize(XNN_PAD_EXTRA_BYTES(16, float));
  uint32_t w112 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w112_dims = {{16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w112_dims.size(), w112_dims.data(),
      /*data=*/w112_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w112);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w112" << std::endl;
    return nullptr;
  }

  static std::vector<float> w113_data;
  w113_data.resize(XNN_PAD_EXTRA_BYTES(144, float));
  uint32_t w113 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w113_dims = {{1, 3, 3, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w113_dims.size(), w113_dims.data(),
      /*data=*/w113_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w113);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w113" << std::endl;
    return nullptr;
  }

  static std::vector<float> w114_data;
  w114_data.resize(XNN_PAD_EXTRA_BYTES(16, float));
  uint32_t w114 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w114_dims = {{16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w114_dims.size(), w114_dims.data(),
      /*data=*/w114_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w114);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w114" << std::endl;
    return nullptr;
  }

  static std::vector<float> w115_data;
  w115_data.resize(XNN_PAD_EXTRA_BYTES(128, float));
  uint32_t w115 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w115_dims = {{8, 1, 1, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w115_dims.size(), w115_dims.data(),
      /*data=*/w115_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w115);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w115" << std::endl;
    return nullptr;
  }

  static std::vector<float> w116_data;
  w116_data.resize(XNN_PAD_EXTRA_BYTES(8, float));
  uint32_t w116 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w116_dims = {{8}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w116_dims.size(), w116_dims.data(),
      /*data=*/w116_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w116);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w116" << std::endl;
    return nullptr;
  }

  static std::vector<float> w117_data;
  w117_data.resize(XNN_PAD_EXTRA_BYTES(128, float));
  uint32_t w117 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w117_dims = {{16, 1, 1, 8}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w117_dims.size(), w117_dims.data(),
      /*data=*/w117_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w117);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w117" << std::endl;
    return nullptr;
  }

  static std::vector<float> w118_data;
  w118_data.resize(XNN_PAD_EXTRA_BYTES(16, float));
  uint32_t w118 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w118_dims = {{16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w118_dims.size(), w118_dims.data(),
      /*data=*/w118_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w118);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w118" << std::endl;
    return nullptr;
  }

  static std::vector<float> w119_data;
  w119_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w119 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w119_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w119_dims.size(), w119_dims.data(),
      /*data=*/w119_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w119);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w119" << std::endl;
    return nullptr;
  }

  static std::vector<float> w120_data;
  w120_data.resize(XNN_PAD_EXTRA_BYTES(256, float));
  uint32_t w120 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w120_dims = {{16, 1, 1, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w120_dims.size(), w120_dims.data(),
      /*data=*/w120_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w120);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w120" << std::endl;
    return nullptr;
  }

  static std::vector<float> w121_data;
  w121_data.resize(XNN_PAD_EXTRA_BYTES(16, float));
  uint32_t w121 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w121_dims = {{16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w121_dims.size(), w121_dims.data(),
      /*data=*/w121_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w121);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w121" << std::endl;
    return nullptr;
  }

  static std::vector<float> w122_data;
  w122_data.resize(XNN_PAD_EXTRA_BYTES(1152, float));
  uint32_t w122 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w122_dims = {{72, 1, 1, 16}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w122_dims.size(), w122_dims.data(),
      /*data=*/w122_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w122);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w122" << std::endl;
    return nullptr;
  }

  static std::vector<float> w123_data;
  w123_data.resize(XNN_PAD_EXTRA_BYTES(72, float));
  uint32_t w123 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w123_dims = {{72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w123_dims.size(), w123_dims.data(),
      /*data=*/w123_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w123);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w123" << std::endl;
    return nullptr;
  }

  static std::vector<float> w124_data;
  w124_data.resize(XNN_PAD_EXTRA_BYTES(648, float));
  uint32_t w124 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w124_dims = {{1, 3, 3, 72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w124_dims.size(), w124_dims.data(),
      /*data=*/w124_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w124);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w124" << std::endl;
    return nullptr;
  }

  static std::vector<float> w125_data;
  w125_data.resize(XNN_PAD_EXTRA_BYTES(72, float));
  uint32_t w125 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w125_dims = {{72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w125_dims.size(), w125_dims.data(),
      /*data=*/w125_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w125);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w125" << std::endl;
    return nullptr;
  }

  static std::vector<float> w126_data;
  w126_data.resize(XNN_PAD_EXTRA_BYTES(1728, float));
  uint32_t w126 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w126_dims = {{24, 1, 1, 72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w126_dims.size(), w126_dims.data(),
      /*data=*/w126_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w126);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w126" << std::endl;
    return nullptr;
  }

  static std::vector<float> w127_data;
  w127_data.resize(XNN_PAD_EXTRA_BYTES(24, float));
  uint32_t w127 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w127_dims = {{24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w127_dims.size(), w127_dims.data(),
      /*data=*/w127_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w127);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w127" << std::endl;
    return nullptr;
  }

  static std::vector<float> w128_data;
  w128_data.resize(XNN_PAD_EXTRA_BYTES(2112, float));
  uint32_t w128 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w128_dims = {{88, 1, 1, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w128_dims.size(), w128_dims.data(),
      /*data=*/w128_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w128);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w128" << std::endl;
    return nullptr;
  }

  static std::vector<float> w129_data;
  w129_data.resize(XNN_PAD_EXTRA_BYTES(88, float));
  uint32_t w129 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w129_dims = {{88}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w129_dims.size(), w129_dims.data(),
      /*data=*/w129_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w129);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w129" << std::endl;
    return nullptr;
  }

  static std::vector<float> w130_data;
  w130_data.resize(XNN_PAD_EXTRA_BYTES(792, float));
  uint32_t w130 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w130_dims = {{1, 3, 3, 88}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w130_dims.size(), w130_dims.data(),
      /*data=*/w130_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w130);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w130" << std::endl;
    return nullptr;
  }

  static std::vector<float> w131_data;
  w131_data.resize(XNN_PAD_EXTRA_BYTES(88, float));
  uint32_t w131 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w131_dims = {{88}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w131_dims.size(), w131_dims.data(),
      /*data=*/w131_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w131);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w131" << std::endl;
    return nullptr;
  }

  static std::vector<float> w132_data;
  w132_data.resize(XNN_PAD_EXTRA_BYTES(2112, float));
  uint32_t w132 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w132_dims = {{24, 1, 1, 88}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w132_dims.size(), w132_dims.data(),
      /*data=*/w132_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w132);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w132" << std::endl;
    return nullptr;
  }

  static std::vector<float> w133_data;
  w133_data.resize(XNN_PAD_EXTRA_BYTES(24, float));
  uint32_t w133 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w133_dims = {{24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w133_dims.size(), w133_dims.data(),
      /*data=*/w133_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w133);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w133" << std::endl;
    return nullptr;
  }

  static std::vector<float> w134_data;
  w134_data.resize(XNN_PAD_EXTRA_BYTES(2304, float));
  uint32_t w134 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w134_dims = {{96, 1, 1, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w134_dims.size(), w134_dims.data(),
      /*data=*/w134_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w134);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w134" << std::endl;
    return nullptr;
  }

  static std::vector<float> w135_data;
  w135_data.resize(XNN_PAD_EXTRA_BYTES(96, float));
  uint32_t w135 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w135_dims = {{96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w135_dims.size(), w135_dims.data(),
      /*data=*/w135_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w135);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w135" << std::endl;
    return nullptr;
  }

  static std::vector<float> w136_data;
  w136_data.resize(XNN_PAD_EXTRA_BYTES(2400, float));
  uint32_t w136 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w136_dims = {{1, 5, 5, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w136_dims.size(), w136_dims.data(),
      /*data=*/w136_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w136);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w136" << std::endl;
    return nullptr;
  }

  static std::vector<float> w137_data;
  w137_data.resize(XNN_PAD_EXTRA_BYTES(96, float));
  uint32_t w137 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w137_dims = {{96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w137_dims.size(), w137_dims.data(),
      /*data=*/w137_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w137);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w137" << std::endl;
    return nullptr;
  }

  static std::vector<float> w138_data;
  w138_data.resize(XNN_PAD_EXTRA_BYTES(2304, float));
  uint32_t w138 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w138_dims = {{24, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w138_dims.size(), w138_dims.data(),
      /*data=*/w138_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w138);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w138" << std::endl;
    return nullptr;
  }

  static std::vector<float> w139_data;
  w139_data.resize(XNN_PAD_EXTRA_BYTES(24, float));
  uint32_t w139 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w139_dims = {{24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w139_dims.size(), w139_dims.data(),
      /*data=*/w139_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w139);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w139" << std::endl;
    return nullptr;
  }

  static std::vector<float> w140_data;
  w140_data.resize(XNN_PAD_EXTRA_BYTES(2304, float));
  uint32_t w140 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w140_dims = {{96, 1, 1, 24}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w140_dims.size(), w140_dims.data(),
      /*data=*/w140_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w140);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w140" << std::endl;
    return nullptr;
  }

  static std::vector<float> w141_data;
  w141_data.resize(XNN_PAD_EXTRA_BYTES(96, float));
  uint32_t w141 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w141_dims = {{96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w141_dims.size(), w141_dims.data(),
      /*data=*/w141_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w141);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w141" << std::endl;
    return nullptr;
  }

  static std::vector<float> w142_data;
  w142_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w142 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w142_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w142_dims.size(), w142_dims.data(),
      /*data=*/w142_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w142);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w142" << std::endl;
    return nullptr;
  }

  static std::vector<float> w143_data;
  w143_data.resize(XNN_PAD_EXTRA_BYTES(3840, float));
  uint32_t w143 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w143_dims = {{40, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w143_dims.size(), w143_dims.data(),
      /*data=*/w143_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w143);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w143" << std::endl;
    return nullptr;
  }

  static std::vector<float> w144_data;
  w144_data.resize(XNN_PAD_EXTRA_BYTES(40, float));
  uint32_t w144 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w144_dims = {{40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w144_dims.size(), w144_dims.data(),
      /*data=*/w144_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w144);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w144" << std::endl;
    return nullptr;
  }

  static std::vector<float> w145_data;
  w145_data.resize(XNN_PAD_EXTRA_BYTES(9600, float));
  uint32_t w145 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w145_dims = {{240, 1, 1, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w145_dims.size(), w145_dims.data(),
      /*data=*/w145_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w145);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w145" << std::endl;
    return nullptr;
  }

  static std::vector<float> w146_data;
  w146_data.resize(XNN_PAD_EXTRA_BYTES(240, float));
  uint32_t w146 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w146_dims = {{240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w146_dims.size(), w146_dims.data(),
      /*data=*/w146_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w146);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w146" << std::endl;
    return nullptr;
  }

  static std::vector<float> w147_data;
  w147_data.resize(XNN_PAD_EXTRA_BYTES(6000, float));
  uint32_t w147 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w147_dims = {{1, 5, 5, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w147_dims.size(), w147_dims.data(),
      /*data=*/w147_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w147);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w147" << std::endl;
    return nullptr;
  }

  static std::vector<float> w148_data;
  w148_data.resize(XNN_PAD_EXTRA_BYTES(240, float));
  uint32_t w148 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w148_dims = {{240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w148_dims.size(), w148_dims.data(),
      /*data=*/w148_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w148);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w148" << std::endl;
    return nullptr;
  }

  static std::vector<float> w149_data;
  w149_data.resize(XNN_PAD_EXTRA_BYTES(15360, float));
  uint32_t w149 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w149_dims = {{64, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w149_dims.size(), w149_dims.data(),
      /*data=*/w149_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w149);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w149" << std::endl;
    return nullptr;
  }

  static std::vector<float> w150_data;
  w150_data.resize(XNN_PAD_EXTRA_BYTES(64, float));
  uint32_t w150 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w150_dims = {{64}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w150_dims.size(), w150_dims.data(),
      /*data=*/w150_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w150);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w150" << std::endl;
    return nullptr;
  }

  static std::vector<float> w151_data;
  w151_data.resize(XNN_PAD_EXTRA_BYTES(15360, float));
  uint32_t w151 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w151_dims = {{240, 1, 1, 64}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w151_dims.size(), w151_dims.data(),
      /*data=*/w151_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w151);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w151" << std::endl;
    return nullptr;
  }

  static std::vector<float> w152_data;
  w152_data.resize(XNN_PAD_EXTRA_BYTES(240, float));
  uint32_t w152 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w152_dims = {{240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w152_dims.size(), w152_dims.data(),
      /*data=*/w152_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w152);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w152" << std::endl;
    return nullptr;
  }

  static std::vector<float> w153_data;
  w153_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w153 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w153_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w153_dims.size(), w153_dims.data(),
      /*data=*/w153_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w153);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w153" << std::endl;
    return nullptr;
  }

  static std::vector<float> w154_data;
  w154_data.resize(XNN_PAD_EXTRA_BYTES(9600, float));
  uint32_t w154 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w154_dims = {{40, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w154_dims.size(), w154_dims.data(),
      /*data=*/w154_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w154);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w154" << std::endl;
    return nullptr;
  }

  static std::vector<float> w155_data;
  w155_data.resize(XNN_PAD_EXTRA_BYTES(40, float));
  uint32_t w155 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w155_dims = {{40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w155_dims.size(), w155_dims.data(),
      /*data=*/w155_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w155);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w155" << std::endl;
    return nullptr;
  }

  static std::vector<float> w156_data;
  w156_data.resize(XNN_PAD_EXTRA_BYTES(9600, float));
  uint32_t w156 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w156_dims = {{240, 1, 1, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w156_dims.size(), w156_dims.data(),
      /*data=*/w156_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w156);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w156" << std::endl;
    return nullptr;
  }

  static std::vector<float> w157_data;
  w157_data.resize(XNN_PAD_EXTRA_BYTES(240, float));
  uint32_t w157 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w157_dims = {{240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w157_dims.size(), w157_dims.data(),
      /*data=*/w157_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w157);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w157" << std::endl;
    return nullptr;
  }

  static std::vector<float> w158_data;
  w158_data.resize(XNN_PAD_EXTRA_BYTES(6000, float));
  uint32_t w158 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w158_dims = {{1, 5, 5, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w158_dims.size(), w158_dims.data(),
      /*data=*/w158_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w158);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w158" << std::endl;
    return nullptr;
  }

  static std::vector<float> w159_data;
  w159_data.resize(XNN_PAD_EXTRA_BYTES(240, float));
  uint32_t w159 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w159_dims = {{240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w159_dims.size(), w159_dims.data(),
      /*data=*/w159_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w159);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w159" << std::endl;
    return nullptr;
  }

  static std::vector<float> w160_data;
  w160_data.resize(XNN_PAD_EXTRA_BYTES(15360, float));
  uint32_t w160 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w160_dims = {{64, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w160_dims.size(), w160_dims.data(),
      /*data=*/w160_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w160);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w160" << std::endl;
    return nullptr;
  }

  static std::vector<float> w161_data;
  w161_data.resize(XNN_PAD_EXTRA_BYTES(64, float));
  uint32_t w161 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w161_dims = {{64}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w161_dims.size(), w161_dims.data(),
      /*data=*/w161_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w161);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w161" << std::endl;
    return nullptr;
  }

  static std::vector<float> w162_data;
  w162_data.resize(XNN_PAD_EXTRA_BYTES(15360, float));
  uint32_t w162 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w162_dims = {{240, 1, 1, 64}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w162_dims.size(), w162_dims.data(),
      /*data=*/w162_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w162);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w162" << std::endl;
    return nullptr;
  }

  static std::vector<float> w163_data;
  w163_data.resize(XNN_PAD_EXTRA_BYTES(240, float));
  uint32_t w163 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w163_dims = {{240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w163_dims.size(), w163_dims.data(),
      /*data=*/w163_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w163);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w163" << std::endl;
    return nullptr;
  }

  static std::vector<float> w164_data;
  w164_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w164 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w164_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w164_dims.size(), w164_dims.data(),
      /*data=*/w164_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w164);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w164" << std::endl;
    return nullptr;
  }

  static std::vector<float> w165_data;
  w165_data.resize(XNN_PAD_EXTRA_BYTES(9600, float));
  uint32_t w165 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w165_dims = {{40, 1, 1, 240}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w165_dims.size(), w165_dims.data(),
      /*data=*/w165_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w165);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w165" << std::endl;
    return nullptr;
  }

  static std::vector<float> w166_data;
  w166_data.resize(XNN_PAD_EXTRA_BYTES(40, float));
  uint32_t w166 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w166_dims = {{40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w166_dims.size(), w166_dims.data(),
      /*data=*/w166_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w166);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w166" << std::endl;
    return nullptr;
  }

  static std::vector<float> w167_data;
  w167_data.resize(XNN_PAD_EXTRA_BYTES(4800, float));
  uint32_t w167 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w167_dims = {{120, 1, 1, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w167_dims.size(), w167_dims.data(),
      /*data=*/w167_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w167);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w167" << std::endl;
    return nullptr;
  }

  static std::vector<float> w168_data;
  w168_data.resize(XNN_PAD_EXTRA_BYTES(120, float));
  uint32_t w168 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w168_dims = {{120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w168_dims.size(), w168_dims.data(),
      /*data=*/w168_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w168);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w168" << std::endl;
    return nullptr;
  }

  static std::vector<float> w169_data;
  w169_data.resize(XNN_PAD_EXTRA_BYTES(3000, float));
  uint32_t w169 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w169_dims = {{1, 5, 5, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w169_dims.size(), w169_dims.data(),
      /*data=*/w169_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w169);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w169" << std::endl;
    return nullptr;
  }

  static std::vector<float> w170_data;
  w170_data.resize(XNN_PAD_EXTRA_BYTES(120, float));
  uint32_t w170 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w170_dims = {{120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w170_dims.size(), w170_dims.data(),
      /*data=*/w170_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w170);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w170" << std::endl;
    return nullptr;
  }

  static std::vector<float> w171_data;
  w171_data.resize(XNN_PAD_EXTRA_BYTES(3840, float));
  uint32_t w171 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w171_dims = {{32, 1, 1, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w171_dims.size(), w171_dims.data(),
      /*data=*/w171_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w171);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w171" << std::endl;
    return nullptr;
  }

  static std::vector<float> w172_data;
  w172_data.resize(XNN_PAD_EXTRA_BYTES(32, float));
  uint32_t w172 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w172_dims = {{32}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w172_dims.size(), w172_dims.data(),
      /*data=*/w172_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w172);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w172" << std::endl;
    return nullptr;
  }

  static std::vector<float> w173_data;
  w173_data.resize(XNN_PAD_EXTRA_BYTES(3840, float));
  uint32_t w173 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w173_dims = {{120, 1, 1, 32}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w173_dims.size(), w173_dims.data(),
      /*data=*/w173_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w173);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w173" << std::endl;
    return nullptr;
  }

  static std::vector<float> w174_data;
  w174_data.resize(XNN_PAD_EXTRA_BYTES(120, float));
  uint32_t w174 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w174_dims = {{120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w174_dims.size(), w174_dims.data(),
      /*data=*/w174_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w174);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w174" << std::endl;
    return nullptr;
  }

  static std::vector<float> w175_data;
  w175_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w175 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w175_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w175_dims.size(), w175_dims.data(),
      /*data=*/w175_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w175);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w175" << std::endl;
    return nullptr;
  }

  static std::vector<float> w176_data;
  w176_data.resize(XNN_PAD_EXTRA_BYTES(5760, float));
  uint32_t w176 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w176_dims = {{48, 1, 1, 120}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w176_dims.size(), w176_dims.data(),
      /*data=*/w176_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w176);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w176" << std::endl;
    return nullptr;
  }

  static std::vector<float> w177_data;
  w177_data.resize(XNN_PAD_EXTRA_BYTES(48, float));
  uint32_t w177 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w177_dims = {{48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w177_dims.size(), w177_dims.data(),
      /*data=*/w177_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w177);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w177" << std::endl;
    return nullptr;
  }

  static std::vector<float> w178_data;
  w178_data.resize(XNN_PAD_EXTRA_BYTES(6912, float));
  uint32_t w178 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w178_dims = {{144, 1, 1, 48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w178_dims.size(), w178_dims.data(),
      /*data=*/w178_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w178);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w178" << std::endl;
    return nullptr;
  }

  static std::vector<float> w179_data;
  w179_data.resize(XNN_PAD_EXTRA_BYTES(144, float));
  uint32_t w179 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w179_dims = {{144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w179_dims.size(), w179_dims.data(),
      /*data=*/w179_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w179);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w179" << std::endl;
    return nullptr;
  }

  static std::vector<float> w180_data;
  w180_data.resize(XNN_PAD_EXTRA_BYTES(3600, float));
  uint32_t w180 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w180_dims = {{1, 5, 5, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w180_dims.size(), w180_dims.data(),
      /*data=*/w180_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w180);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w180" << std::endl;
    return nullptr;
  }

  static std::vector<float> w181_data;
  w181_data.resize(XNN_PAD_EXTRA_BYTES(144, float));
  uint32_t w181 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w181_dims = {{144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w181_dims.size(), w181_dims.data(),
      /*data=*/w181_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w181);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w181" << std::endl;
    return nullptr;
  }

  static std::vector<float> w182_data;
  w182_data.resize(XNN_PAD_EXTRA_BYTES(5760, float));
  uint32_t w182 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w182_dims = {{40, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w182_dims.size(), w182_dims.data(),
      /*data=*/w182_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w182);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w182" << std::endl;
    return nullptr;
  }

  static std::vector<float> w183_data;
  w183_data.resize(XNN_PAD_EXTRA_BYTES(40, float));
  uint32_t w183 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w183_dims = {{40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w183_dims.size(), w183_dims.data(),
      /*data=*/w183_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w183);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w183" << std::endl;
    return nullptr;
  }

  static std::vector<float> w184_data;
  w184_data.resize(XNN_PAD_EXTRA_BYTES(5760, float));
  uint32_t w184 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w184_dims = {{144, 1, 1, 40}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w184_dims.size(), w184_dims.data(),
      /*data=*/w184_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w184);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w184" << std::endl;
    return nullptr;
  }

  static std::vector<float> w185_data;
  w185_data.resize(XNN_PAD_EXTRA_BYTES(144, float));
  uint32_t w185 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w185_dims = {{144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w185_dims.size(), w185_dims.data(),
      /*data=*/w185_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w185);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w185" << std::endl;
    return nullptr;
  }

  static std::vector<float> w186_data;
  w186_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w186 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w186_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w186_dims.size(), w186_dims.data(),
      /*data=*/w186_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w186);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w186" << std::endl;
    return nullptr;
  }

  static std::vector<float> w187_data;
  w187_data.resize(XNN_PAD_EXTRA_BYTES(6912, float));
  uint32_t w187 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w187_dims = {{48, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w187_dims.size(), w187_dims.data(),
      /*data=*/w187_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w187);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w187" << std::endl;
    return nullptr;
  }

  static std::vector<float> w188_data;
  w188_data.resize(XNN_PAD_EXTRA_BYTES(48, float));
  uint32_t w188 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w188_dims = {{48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w188_dims.size(), w188_dims.data(),
      /*data=*/w188_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w188);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w188" << std::endl;
    return nullptr;
  }

  static std::vector<float> w189_data;
  w189_data.resize(XNN_PAD_EXTRA_BYTES(13824, float));
  uint32_t w189 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w189_dims = {{288, 1, 1, 48}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w189_dims.size(), w189_dims.data(),
      /*data=*/w189_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w189);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w189" << std::endl;
    return nullptr;
  }

  static std::vector<float> w190_data;
  w190_data.resize(XNN_PAD_EXTRA_BYTES(288, float));
  uint32_t w190 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w190_dims = {{288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w190_dims.size(), w190_dims.data(),
      /*data=*/w190_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w190);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w190" << std::endl;
    return nullptr;
  }

  static std::vector<float> w191_data;
  w191_data.resize(XNN_PAD_EXTRA_BYTES(7200, float));
  uint32_t w191 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w191_dims = {{1, 5, 5, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w191_dims.size(), w191_dims.data(),
      /*data=*/w191_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w191);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w191" << std::endl;
    return nullptr;
  }

  static std::vector<float> w192_data;
  w192_data.resize(XNN_PAD_EXTRA_BYTES(288, float));
  uint32_t w192 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w192_dims = {{288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w192_dims.size(), w192_dims.data(),
      /*data=*/w192_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w192);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w192" << std::endl;
    return nullptr;
  }

  static std::vector<float> w193_data;
  w193_data.resize(XNN_PAD_EXTRA_BYTES(20736, float));
  uint32_t w193 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w193_dims = {{72, 1, 1, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w193_dims.size(), w193_dims.data(),
      /*data=*/w193_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w193);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w193" << std::endl;
    return nullptr;
  }

  static std::vector<float> w194_data;
  w194_data.resize(XNN_PAD_EXTRA_BYTES(72, float));
  uint32_t w194 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w194_dims = {{72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w194_dims.size(), w194_dims.data(),
      /*data=*/w194_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w194);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w194" << std::endl;
    return nullptr;
  }

  static std::vector<float> w195_data;
  w195_data.resize(XNN_PAD_EXTRA_BYTES(20736, float));
  uint32_t w195 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w195_dims = {{288, 1, 1, 72}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w195_dims.size(), w195_dims.data(),
      /*data=*/w195_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w195);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w195" << std::endl;
    return nullptr;
  }

  static std::vector<float> w196_data;
  w196_data.resize(XNN_PAD_EXTRA_BYTES(288, float));
  uint32_t w196 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w196_dims = {{288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w196_dims.size(), w196_dims.data(),
      /*data=*/w196_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w196);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w196" << std::endl;
    return nullptr;
  }

  static std::vector<float> w197_data;
  w197_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w197 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w197_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w197_dims.size(), w197_dims.data(),
      /*data=*/w197_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w197);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w197" << std::endl;
    return nullptr;
  }

  static std::vector<float> w198_data;
  w198_data.resize(XNN_PAD_EXTRA_BYTES(27648, float));
  uint32_t w198 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w198_dims = {{96, 1, 1, 288}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w198_dims.size(), w198_dims.data(),
      /*data=*/w198_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w198);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w198" << std::endl;
    return nullptr;
  }

  static std::vector<float> w199_data;
  w199_data.resize(XNN_PAD_EXTRA_BYTES(96, float));
  uint32_t w199 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w199_dims = {{96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w199_dims.size(), w199_dims.data(),
      /*data=*/w199_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w199);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w199" << std::endl;
    return nullptr;
  }

  static std::vector<float> w200_data;
  w200_data.resize(XNN_PAD_EXTRA_BYTES(55296, float));
  uint32_t w200 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w200_dims = {{576, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w200_dims.size(), w200_dims.data(),
      /*data=*/w200_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w200);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w200" << std::endl;
    return nullptr;
  }

  static std::vector<float> w201_data;
  w201_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w201 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w201_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w201_dims.size(), w201_dims.data(),
      /*data=*/w201_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w201);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w201" << std::endl;
    return nullptr;
  }

  static std::vector<float> w202_data;
  w202_data.resize(XNN_PAD_EXTRA_BYTES(14400, float));
  uint32_t w202 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w202_dims = {{1, 5, 5, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w202_dims.size(), w202_dims.data(),
      /*data=*/w202_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w202);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w202" << std::endl;
    return nullptr;
  }

  static std::vector<float> w203_data;
  w203_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w203 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w203_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w203_dims.size(), w203_dims.data(),
      /*data=*/w203_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w203);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w203" << std::endl;
    return nullptr;
  }

  static std::vector<float> w204_data;
  w204_data.resize(XNN_PAD_EXTRA_BYTES(82944, float));
  uint32_t w204 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w204_dims = {{144, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w204_dims.size(), w204_dims.data(),
      /*data=*/w204_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w204);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w204" << std::endl;
    return nullptr;
  }

  static std::vector<float> w205_data;
  w205_data.resize(XNN_PAD_EXTRA_BYTES(144, float));
  uint32_t w205 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w205_dims = {{144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w205_dims.size(), w205_dims.data(),
      /*data=*/w205_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w205);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w205" << std::endl;
    return nullptr;
  }

  static std::vector<float> w206_data;
  w206_data.resize(XNN_PAD_EXTRA_BYTES(82944, float));
  uint32_t w206 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w206_dims = {{576, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w206_dims.size(), w206_dims.data(),
      /*data=*/w206_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w206);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w206" << std::endl;
    return nullptr;
  }

  static std::vector<float> w207_data;
  w207_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w207 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w207_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w207_dims.size(), w207_dims.data(),
      /*data=*/w207_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w207);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w207" << std::endl;
    return nullptr;
  }

  static std::vector<float> w208_data;
  w208_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w208 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w208_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w208_dims.size(), w208_dims.data(),
      /*data=*/w208_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w208);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w208" << std::endl;
    return nullptr;
  }

  static std::vector<float> w209_data;
  w209_data.resize(XNN_PAD_EXTRA_BYTES(55296, float));
  uint32_t w209 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w209_dims = {{96, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w209_dims.size(), w209_dims.data(),
      /*data=*/w209_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w209);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w209" << std::endl;
    return nullptr;
  }

  static std::vector<float> w210_data;
  w210_data.resize(XNN_PAD_EXTRA_BYTES(96, float));
  uint32_t w210 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w210_dims = {{96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w210_dims.size(), w210_dims.data(),
      /*data=*/w210_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w210);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w210" << std::endl;
    return nullptr;
  }

  static std::vector<float> w211_data;
  w211_data.resize(XNN_PAD_EXTRA_BYTES(55296, float));
  uint32_t w211 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w211_dims = {{576, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w211_dims.size(), w211_dims.data(),
      /*data=*/w211_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w211);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w211" << std::endl;
    return nullptr;
  }

  static std::vector<float> w212_data;
  w212_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w212 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w212_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w212_dims.size(), w212_dims.data(),
      /*data=*/w212_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w212);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w212" << std::endl;
    return nullptr;
  }

  static std::vector<float> w213_data;
  w213_data.resize(XNN_PAD_EXTRA_BYTES(14400, float));
  uint32_t w213 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w213_dims = {{1, 5, 5, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w213_dims.size(), w213_dims.data(),
      /*data=*/w213_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w213);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w213" << std::endl;
    return nullptr;
  }

  static std::vector<float> w214_data;
  w214_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w214 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w214_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w214_dims.size(), w214_dims.data(),
      /*data=*/w214_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w214);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w214" << std::endl;
    return nullptr;
  }

  static std::vector<float> w215_data;
  w215_data.resize(XNN_PAD_EXTRA_BYTES(82944, float));
  uint32_t w215 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w215_dims = {{144, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w215_dims.size(), w215_dims.data(),
      /*data=*/w215_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w215);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w215" << std::endl;
    return nullptr;
  }

  static std::vector<float> w216_data;
  w216_data.resize(XNN_PAD_EXTRA_BYTES(144, float));
  uint32_t w216 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w216_dims = {{144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w216_dims.size(), w216_dims.data(),
      /*data=*/w216_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w216);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w216" << std::endl;
    return nullptr;
  }

  static std::vector<float> w217_data;
  w217_data.resize(XNN_PAD_EXTRA_BYTES(82944, float));
  uint32_t w217 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w217_dims = {{576, 1, 1, 144}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w217_dims.size(), w217_dims.data(),
      /*data=*/w217_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w217);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w217" << std::endl;
    return nullptr;
  }

  static std::vector<float> w218_data;
  w218_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w218 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w218_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w218_dims.size(), w218_dims.data(),
      /*data=*/w218_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w218);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w218" << std::endl;
    return nullptr;
  }

  static std::vector<float> w219_data;
  w219_data.resize(XNN_PAD_EXTRA_BYTES(1, float));
  uint32_t w219 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w219_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w219_dims.size(), w219_dims.data(),
      /*data=*/w219_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w219);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w219" << std::endl;
    return nullptr;
  }

  static std::vector<float> w220_data;
  w220_data.resize(XNN_PAD_EXTRA_BYTES(55296, float));
  uint32_t w220 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w220_dims = {{96, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w220_dims.size(), w220_dims.data(),
      /*data=*/w220_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w220);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w220" << std::endl;
    return nullptr;
  }

  static std::vector<float> w221_data;
  w221_data.resize(XNN_PAD_EXTRA_BYTES(96, float));
  uint32_t w221 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w221_dims = {{96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w221_dims.size(), w221_dims.data(),
      /*data=*/w221_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w221);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w221" << std::endl;
    return nullptr;
  }

  static std::vector<float> w222_data;
  w222_data.resize(XNN_PAD_EXTRA_BYTES(55296, float));
  uint32_t w222 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w222_dims = {{576, 1, 1, 96}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w222_dims.size(), w222_dims.data(),
      /*data=*/w222_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w222);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w222" << std::endl;
    return nullptr;
  }

  static std::vector<float> w223_data;
  w223_data.resize(XNN_PAD_EXTRA_BYTES(576, float));
  uint32_t w223 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w223_dims = {{576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w223_dims.size(), w223_dims.data(),
      /*data=*/w223_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w223);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w223" << std::endl;
    return nullptr;
  }

  static std::vector<float> w224_data;
  w224_data.resize(XNN_PAD_EXTRA_BYTES(589824, float));
  uint32_t w224 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w224_dims = {{1024, 1, 1, 576}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w224_dims.size(), w224_dims.data(),
      /*data=*/w224_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w224);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w224" << std::endl;
    return nullptr;
  }

  static std::vector<float> w225_data;
  w225_data.resize(XNN_PAD_EXTRA_BYTES(1024, float));
  uint32_t w225 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w225_dims = {{1024}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w225_dims.size(), w225_dims.data(),
      /*data=*/w225_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w225);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w225" << std::endl;
    return nullptr;
  }

  static std::vector<float> w226_data;
  w226_data.resize(XNN_PAD_EXTRA_BYTES(1025024, float));
  uint32_t w226 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w226_dims = {{1001, 1, 1, 1024}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w226_dims.size(), w226_dims.data(),
      /*data=*/w226_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w226);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w226" << std::endl;
    return nullptr;
  }

  static std::vector<float> w227_data;
  w227_data.resize(XNN_PAD_EXTRA_BYTES(1001, float));
  uint32_t w227 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w227_dims = {{1001}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w227_dims.size(), w227_dims.data(),
      /*data=*/w227_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w227);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w227" << std::endl;
    return nullptr;
  }

  static std::vector<int32_t> w228_data;
  w228_data.resize(XNN_PAD_EXTRA_BYTES(2, int32_t));
  uint32_t w228 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w228_dims = {{2}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w228_dims.size(), w228_dims.data(),
      /*data=*/w228_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w228);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w228" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));
  auto i32rng = std::bind(std::uniform_int_distribution<int>(-10000, 10000),
                          std::ref(rng));
  std::generate(w111_data.begin(), w111_data.end(), std::ref(f32rng));
  std::generate(w112_data.begin(), w112_data.end(), std::ref(f32rng));
  std::generate(w113_data.begin(), w113_data.end(), std::ref(f32rng));
  std::generate(w114_data.begin(), w114_data.end(), std::ref(f32rng));
  std::generate(w115_data.begin(), w115_data.end(), std::ref(f32rng));
  std::generate(w116_data.begin(), w116_data.end(), std::ref(f32rng));
  std::generate(w117_data.begin(), w117_data.end(), std::ref(f32rng));
  std::generate(w118_data.begin(), w118_data.end(), std::ref(f32rng));
  std::generate(w119_data.begin(), w119_data.end(), std::ref(f32rng));
  std::generate(w120_data.begin(), w120_data.end(), std::ref(f32rng));
  std::generate(w121_data.begin(), w121_data.end(), std::ref(f32rng));
  std::generate(w122_data.begin(), w122_data.end(), std::ref(f32rng));
  std::generate(w123_data.begin(), w123_data.end(), std::ref(f32rng));
  std::generate(w124_data.begin(), w124_data.end(), std::ref(f32rng));
  std::generate(w125_data.begin(), w125_data.end(), std::ref(f32rng));
  std::generate(w126_data.begin(), w126_data.end(), std::ref(f32rng));
  std::generate(w127_data.begin(), w127_data.end(), std::ref(f32rng));
  std::generate(w128_data.begin(), w128_data.end(), std::ref(f32rng));
  std::generate(w129_data.begin(), w129_data.end(), std::ref(f32rng));
  std::generate(w130_data.begin(), w130_data.end(), std::ref(f32rng));
  std::generate(w131_data.begin(), w131_data.end(), std::ref(f32rng));
  std::generate(w132_data.begin(), w132_data.end(), std::ref(f32rng));
  std::generate(w133_data.begin(), w133_data.end(), std::ref(f32rng));
  std::generate(w134_data.begin(), w134_data.end(), std::ref(f32rng));
  std::generate(w135_data.begin(), w135_data.end(), std::ref(f32rng));
  std::generate(w136_data.begin(), w136_data.end(), std::ref(f32rng));
  std::generate(w137_data.begin(), w137_data.end(), std::ref(f32rng));
  std::generate(w138_data.begin(), w138_data.end(), std::ref(f32rng));
  std::generate(w139_data.begin(), w139_data.end(), std::ref(f32rng));
  std::generate(w140_data.begin(), w140_data.end(), std::ref(f32rng));
  std::generate(w141_data.begin(), w141_data.end(), std::ref(f32rng));
  std::generate(w142_data.begin(), w142_data.end(), std::ref(f32rng));
  std::generate(w143_data.begin(), w143_data.end(), std::ref(f32rng));
  std::generate(w144_data.begin(), w144_data.end(), std::ref(f32rng));
  std::generate(w145_data.begin(), w145_data.end(), std::ref(f32rng));
  std::generate(w146_data.begin(), w146_data.end(), std::ref(f32rng));
  std::generate(w147_data.begin(), w147_data.end(), std::ref(f32rng));
  std::generate(w148_data.begin(), w148_data.end(), std::ref(f32rng));
  std::generate(w149_data.begin(), w149_data.end(), std::ref(f32rng));
  std::generate(w150_data.begin(), w150_data.end(), std::ref(f32rng));
  std::generate(w151_data.begin(), w151_data.end(), std::ref(f32rng));
  std::generate(w152_data.begin(), w152_data.end(), std::ref(f32rng));
  std::generate(w153_data.begin(), w153_data.end(), std::ref(f32rng));
  std::generate(w154_data.begin(), w154_data.end(), std::ref(f32rng));
  std::generate(w155_data.begin(), w155_data.end(), std::ref(f32rng));
  std::generate(w156_data.begin(), w156_data.end(), std::ref(f32rng));
  std::generate(w157_data.begin(), w157_data.end(), std::ref(f32rng));
  std::generate(w158_data.begin(), w158_data.end(), std::ref(f32rng));
  std::generate(w159_data.begin(), w159_data.end(), std::ref(f32rng));
  std::generate(w160_data.begin(), w160_data.end(), std::ref(f32rng));
  std::generate(w161_data.begin(), w161_data.end(), std::ref(f32rng));
  std::generate(w162_data.begin(), w162_data.end(), std::ref(f32rng));
  std::generate(w163_data.begin(), w163_data.end(), std::ref(f32rng));
  std::generate(w164_data.begin(), w164_data.end(), std::ref(f32rng));
  std::generate(w165_data.begin(), w165_data.end(), std::ref(f32rng));
  std::generate(w166_data.begin(), w166_data.end(), std::ref(f32rng));
  std::generate(w167_data.begin(), w167_data.end(), std::ref(f32rng));
  std::generate(w168_data.begin(), w168_data.end(), std::ref(f32rng));
  std::generate(w169_data.begin(), w169_data.end(), std::ref(f32rng));
  std::generate(w170_data.begin(), w170_data.end(), std::ref(f32rng));
  std::generate(w171_data.begin(), w171_data.end(), std::ref(f32rng));
  std::generate(w172_data.begin(), w172_data.end(), std::ref(f32rng));
  std::generate(w173_data.begin(), w173_data.end(), std::ref(f32rng));
  std::generate(w174_data.begin(), w174_data.end(), std::ref(f32rng));
  std::generate(w175_data.begin(), w175_data.end(), std::ref(f32rng));
  std::generate(w176_data.begin(), w176_data.end(), std::ref(f32rng));
  std::generate(w177_data.begin(), w177_data.end(), std::ref(f32rng));
  std::generate(w178_data.begin(), w178_data.end(), std::ref(f32rng));
  std::generate(w179_data.begin(), w179_data.end(), std::ref(f32rng));
  std::generate(w180_data.begin(), w180_data.end(), std::ref(f32rng));
  std::generate(w181_data.begin(), w181_data.end(), std::ref(f32rng));
  std::generate(w182_data.begin(), w182_data.end(), std::ref(f32rng));
  std::generate(w183_data.begin(), w183_data.end(), std::ref(f32rng));
  std::generate(w184_data.begin(), w184_data.end(), std::ref(f32rng));
  std::generate(w185_data.begin(), w185_data.end(), std::ref(f32rng));
  std::generate(w186_data.begin(), w186_data.end(), std::ref(f32rng));
  std::generate(w187_data.begin(), w187_data.end(), std::ref(f32rng));
  std::generate(w188_data.begin(), w188_data.end(), std::ref(f32rng));
  std::generate(w189_data.begin(), w189_data.end(), std::ref(f32rng));
  std::generate(w190_data.begin(), w190_data.end(), std::ref(f32rng));
  std::generate(w191_data.begin(), w191_data.end(), std::ref(f32rng));
  std::generate(w192_data.begin(), w192_data.end(), std::ref(f32rng));
  std::generate(w193_data.begin(), w193_data.end(), std::ref(f32rng));
  std::generate(w194_data.begin(), w194_data.end(), std::ref(f32rng));
  std::generate(w195_data.begin(), w195_data.end(), std::ref(f32rng));
  std::generate(w196_data.begin(), w196_data.end(), std::ref(f32rng));
  std::generate(w197_data.begin(), w197_data.end(), std::ref(f32rng));
  std::generate(w198_data.begin(), w198_data.end(), std::ref(f32rng));
  std::generate(w199_data.begin(), w199_data.end(), std::ref(f32rng));
  std::generate(w200_data.begin(), w200_data.end(), std::ref(f32rng));
  std::generate(w201_data.begin(), w201_data.end(), std::ref(f32rng));
  std::generate(w202_data.begin(), w202_data.end(), std::ref(f32rng));
  std::generate(w203_data.begin(), w203_data.end(), std::ref(f32rng));
  std::generate(w204_data.begin(), w204_data.end(), std::ref(f32rng));
  std::generate(w205_data.begin(), w205_data.end(), std::ref(f32rng));
  std::generate(w206_data.begin(), w206_data.end(), std::ref(f32rng));
  std::generate(w207_data.begin(), w207_data.end(), std::ref(f32rng));
  std::generate(w208_data.begin(), w208_data.end(), std::ref(f32rng));
  std::generate(w209_data.begin(), w209_data.end(), std::ref(f32rng));
  std::generate(w210_data.begin(), w210_data.end(), std::ref(f32rng));
  std::generate(w211_data.begin(), w211_data.end(), std::ref(f32rng));
  std::generate(w212_data.begin(), w212_data.end(), std::ref(f32rng));
  std::generate(w213_data.begin(), w213_data.end(), std::ref(f32rng));
  std::generate(w214_data.begin(), w214_data.end(), std::ref(f32rng));
  std::generate(w215_data.begin(), w215_data.end(), std::ref(f32rng));
  std::generate(w216_data.begin(), w216_data.end(), std::ref(f32rng));
  std::generate(w217_data.begin(), w217_data.end(), std::ref(f32rng));
  std::generate(w218_data.begin(), w218_data.end(), std::ref(f32rng));
  std::generate(w219_data.begin(), w219_data.end(), std::ref(f32rng));
  std::generate(w220_data.begin(), w220_data.end(), std::ref(f32rng));
  std::generate(w221_data.begin(), w221_data.end(), std::ref(f32rng));
  std::generate(w222_data.begin(), w222_data.end(), std::ref(f32rng));
  std::generate(w223_data.begin(), w223_data.end(), std::ref(f32rng));
  std::generate(w224_data.begin(), w224_data.end(), std::ref(f32rng));
  std::generate(w225_data.begin(), w225_data.end(), std::ref(f32rng));
  std::generate(w226_data.begin(), w226_data.end(), std::ref(f32rng));
  std::generate(w227_data.begin(), w227_data.end(), std::ref(f32rng));
  std::generate(w228_data.begin(), w228_data.end(), std::ref(i32rng));

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/3, /*kernel_width=*/3,
      /*subsampling_height=*/2, /*subsampling_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/3,
      /*group_output_channels=*/16,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v0, w111, w112, v1,
      /*flags=*/XNN_FLAG_TENSORFLOW_SAME_PADDING);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v1, v2,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/3, /*kernel_width=*/3,
      /*subsampling_height=*/2, /*subsampling_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/16,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v2, w113, w114, v3,
      /*flags=*/XNN_FLAG_TENSORFLOW_SAME_PADDING);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/56, /*pooling_width=*/56,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v3, v4,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #3" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/16,
      /*group_output_channels=*/8,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v4, w115, w116, v5,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/8,
      /*group_output_channels=*/16,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v5, w117, w118, v6,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  xnn_binary_params v7_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v7_params,
                             /*input1_id=*/v6,
                             /*input2_id=*/w119,
                             /*output_id=*/v7,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  xnn_binary_params v8_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v8_params,
                             /*input1_id=*/v3,
                             /*input2_id=*/v7,
                             /*output_id=*/v8,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/16,
      /*group_output_channels=*/16,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v8, w120, w121, v9,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #8" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/16,
      /*group_output_channels=*/72,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v9, w122, w123,
      v10,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/3, /*kernel_width=*/3,
      /*subsampling_height=*/2, /*subsampling_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/72,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v10, w124, w125,
      v11,
      /*flags=*/XNN_FLAG_TENSORFLOW_SAME_PADDING);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #10" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/72,
      /*group_output_channels=*/24,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v11, w126, w127,
      v12,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #11" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/24,
      /*group_output_channels=*/88,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v12, w128, w129,
      v13,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #12" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1,
      /*padding_left=*/1,
      /*kernel_height=*/3, /*kernel_width=*/3,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/88,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v13, w130, w131,
      v14,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #13" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/88,
      /*group_output_channels=*/24,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v14, w132, w133,
      v15,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #14" << std::endl;
    return nullptr;
  }

  xnn_binary_params v16_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v16_params,
                             /*input1_id=*/v15,
                             /*input2_id=*/v12,
                             /*output_id=*/v16,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #15" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/24,
      /*group_output_channels=*/96,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v16, w134, w135,
      v17,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #16" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v17, v18,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #17" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/2, /*subsampling_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/96,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v18, w136, w137,
      v19,
      /*flags=*/XNN_FLAG_TENSORFLOW_SAME_PADDING);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #18" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v19, v20,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #19" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/14, /*pooling_width=*/14,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v20, v21,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #20" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/96,
      /*group_output_channels=*/24,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v21, w138, w139,
      v22,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #21" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/24,
      /*group_output_channels=*/96,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v22, w140, w141, v23,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #22" << std::endl;
    return nullptr;
  }

  xnn_binary_params v24_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v24_params,
                             /*input1_id=*/v23,
                             /*input2_id=*/w142,
                             /*output_id=*/v24,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #23" << std::endl;
    return nullptr;
  }

  xnn_binary_params v25_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v25_params,
                             /*input1_id=*/v20,
                             /*input2_id=*/v24,
                             /*output_id=*/v25,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #24" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/96,
      /*group_output_channels=*/40,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v25, w143, w144,
      v26,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #25" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/40,
      /*group_output_channels=*/240,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v26, w145, w146,
      v27,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #26" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v27, v28,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #27" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/2, /*padding_right=*/2, /*padding_bottom=*/2,
      /*padding_left=*/2,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/240,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v28, w147, w148,
      v29,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #28" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v29, v30,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #29" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/14, /*pooling_width=*/14,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v30, v31,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #30" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/240,
      /*group_output_channels=*/64,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v31, w149, w150,
      v32,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #31" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/64,
      /*group_output_channels=*/240,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v32, w151, w152, v33,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #32" << std::endl;
    return nullptr;
  }

  xnn_binary_params v34_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v34_params,
                             /*input1_id=*/v33,
                             /*input2_id=*/w153,
                             /*output_id=*/v34,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #33" << std::endl;
    return nullptr;
  }

  xnn_binary_params v35_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v35_params,
                             /*input1_id=*/v30,
                             /*input2_id=*/v34,
                             /*output_id=*/v35,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #34" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/240,
      /*group_output_channels=*/40,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v35, w154, w155,
      v36,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #35" << std::endl;
    return nullptr;
  }

  xnn_binary_params v37_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v37_params,
                             /*input1_id=*/v36,
                             /*input2_id=*/v26,
                             /*output_id=*/v37,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #36" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/40,
      /*group_output_channels=*/240,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v37, w156, w157,
      v38,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #37" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v38, v39,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #38" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/2, /*padding_right=*/2, /*padding_bottom=*/2,
      /*padding_left=*/2,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/240,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v39, w158, w159,
      v40,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #39" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v40, v41,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #40" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/14, /*pooling_width=*/14,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v41, v42,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #41" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/240,
      /*group_output_channels=*/64,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v42, w160, w161,
      v43,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #42" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/64,
      /*group_output_channels=*/240,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v43, w162, w163, v44,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #43" << std::endl;
    return nullptr;
  }

  xnn_binary_params v45_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v45_params,
                             /*input1_id=*/v44,
                             /*input2_id=*/w164,
                             /*output_id=*/v45,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #44" << std::endl;
    return nullptr;
  }

  xnn_binary_params v46_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v46_params,
                             /*input1_id=*/v41,
                             /*input2_id=*/v45,
                             /*output_id=*/v46,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #45" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/240,
      /*group_output_channels=*/40,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v46, w165, w166,
      v47,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #46" << std::endl;
    return nullptr;
  }

  xnn_binary_params v48_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v48_params,
                             /*input1_id=*/v47,
                             /*input2_id=*/v37,
                             /*output_id=*/v48,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #47" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/40,
      /*group_output_channels=*/120,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v48, w167, w168,
      v49,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #48" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v49, v50,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #49" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/2, /*padding_right=*/2, /*padding_bottom=*/2,
      /*padding_left=*/2,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/120,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v50, w169, w170,
      v51,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #50" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v51, v52,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #51" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/14, /*pooling_width=*/14,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v52, v53,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #52" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/120,
      /*group_output_channels=*/32,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v53, w171, w172,
      v54,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #53" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/32,
      /*group_output_channels=*/120,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v54, w173, w174, v55,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #54" << std::endl;
    return nullptr;
  }

  xnn_binary_params v56_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v56_params,
                             /*input1_id=*/v55,
                             /*input2_id=*/w175,
                             /*output_id=*/v56,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #55" << std::endl;
    return nullptr;
  }

  xnn_binary_params v57_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v57_params,
                             /*input1_id=*/v52,
                             /*input2_id=*/v56,
                             /*output_id=*/v57,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #56" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/120,
      /*group_output_channels=*/48,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v57, w176, w177,
      v58,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #57" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/48,
      /*group_output_channels=*/144,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v58, w178, w179,
      v59,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #58" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v59, v60,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #59" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/2, /*padding_right=*/2, /*padding_bottom=*/2,
      /*padding_left=*/2,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/144,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v60, w180, w181,
      v61,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #60" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v61, v62,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #61" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/14, /*pooling_width=*/14,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v62, v63,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #62" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/144,
      /*group_output_channels=*/40,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v63, w182, w183,
      v64,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #63" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/40,
      /*group_output_channels=*/144,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v64, w184, w185, v65,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #64" << std::endl;
    return nullptr;
  }

  xnn_binary_params v66_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v66_params,
                             /*input1_id=*/v65,
                             /*input2_id=*/w186,
                             /*output_id=*/v66,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #65" << std::endl;
    return nullptr;
  }

  xnn_binary_params v67_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v67_params,
                             /*input1_id=*/v62,
                             /*input2_id=*/v66,
                             /*output_id=*/v67,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #66" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/144,
      /*group_output_channels=*/48,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v67, w187, w188,
      v68,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #67" << std::endl;
    return nullptr;
  }

  xnn_binary_params v69_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v69_params,
                             /*input1_id=*/v68,
                             /*input2_id=*/v58,
                             /*output_id=*/v69,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #68" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/48,
      /*group_output_channels=*/288,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v69, w189, w190,
      v70,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #69" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v70, v71,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #70" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/2, /*subsampling_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/288,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v71, w191, w192,
      v72,
      /*flags=*/XNN_FLAG_TENSORFLOW_SAME_PADDING);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #71" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v72, v73,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #72" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/7, /*pooling_width=*/7,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v73, v74,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #73" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/288,
      /*group_output_channels=*/72,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v74, w193, w194,
      v75,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #74" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/72,
      /*group_output_channels=*/288,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v75, w195, w196, v76,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #75" << std::endl;
    return nullptr;
  }

  xnn_binary_params v77_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v77_params,
                             /*input1_id=*/v76,
                             /*input2_id=*/w197,
                             /*output_id=*/v77,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #76" << std::endl;
    return nullptr;
  }

  xnn_binary_params v78_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v78_params,
                             /*input1_id=*/v73,
                             /*input2_id=*/v77,
                             /*output_id=*/v78,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #77" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/288,
      /*group_output_channels=*/96,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v78, w198, w199,
      v79,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #78" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/96,
      /*group_output_channels=*/576,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v79, w200, w201,
      v80,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #79" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v80, v81,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #80" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/2, /*padding_right=*/2, /*padding_bottom=*/2,
      /*padding_left=*/2,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/576,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v81, w202, w203,
      v82,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #81" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v82, v83,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #82" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/7, /*pooling_width=*/7,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v83, v84,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #83" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/576,
      /*group_output_channels=*/144,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v84, w204, w205,
      v85,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #84" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/144,
      /*group_output_channels=*/576,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v85, w206, w207, v86,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #85" << std::endl;
    return nullptr;
  }

  xnn_binary_params v87_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v87_params,
                             /*input1_id=*/v86,
                             /*input2_id=*/w208,
                             /*output_id=*/v87,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #86" << std::endl;
    return nullptr;
  }

  xnn_binary_params v88_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v88_params,
                             /*input1_id=*/v83,
                             /*input2_id=*/v87,
                             /*output_id=*/v88,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #87" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/576,
      /*group_output_channels=*/96,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v88, w209, w210,
      v89,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #88" << std::endl;
    return nullptr;
  }

  xnn_binary_params v90_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v90_params,
                             /*input1_id=*/v89,
                             /*input2_id=*/v79,
                             /*output_id=*/v90,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #89" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/96,
      /*group_output_channels=*/576,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v90, w211, w212,
      v91,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #90" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v91, v92,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #91" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/2, /*padding_right=*/2, /*padding_bottom=*/2,
      /*padding_left=*/2,
      /*kernel_height=*/5, /*kernel_width=*/5,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/576,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v92, w213, w214,
      v93,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #92" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v93, v94,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #93" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/7, /*pooling_width=*/7,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v94, v95,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #94" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/576,
      /*group_output_channels=*/144,
      /*output_min=*/0.0f,
      /*output_max=*/std::numeric_limits<float>::infinity(), v95, w215, w216,
      v96,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #95" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/144,
      /*group_output_channels=*/576,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v96, w217, w218, v97,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #96" << std::endl;
    return nullptr;
  }

  xnn_binary_params v98_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v98_params,
                             /*input1_id=*/v97,
                             /*input2_id=*/w219,
                             /*output_id=*/v98,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #97" << std::endl;
    return nullptr;
  }

  xnn_binary_params v99_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v99_params,
                             /*input1_id=*/v94,
                             /*input2_id=*/v98,
                             /*output_id=*/v99,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #98" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/576,
      /*group_output_channels=*/96,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v99, w220, w221,
      v100,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #99" << std::endl;
    return nullptr;
  }

  xnn_binary_params v101_params = {-std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v101_params,
                             /*input1_id=*/v100,
                             /*input2_id=*/v90,
                             /*output_id=*/v101,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #100" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/96,
      /*group_output_channels=*/576,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v101, w222, w223,
      v102,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #101" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v102, v103,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #102" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/7, /*pooling_width=*/7,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v103, v104,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #103" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/576,
      /*group_output_channels=*/1024,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v104, w224, w225,
      v105,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #104" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_hardswish,
                            /*params=*/nullptr, v105, v106,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #105" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*pooling_height=*/1, /*pooling_width=*/1,
      /*stride_height=*/1, /*stride_width=*/1,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v106, v107,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #106" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/1024,
      /*group_output_channels=*/1001,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v107, w226, w227,
      v108,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #107" << std::endl;
    return nullptr;
  }

  status = xnn_define_copy(subgraph,
                           /*input_id=*/v108,
                           /*output_id=*/v109,
                           /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #108" << std::endl;
    return nullptr;
  }

  status = xnn_define_softmax(subgraph, v109, v110,
                              /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #109" << std::endl;
    return nullptr;
  }

  return subgraph;
}  // NOLINT(readability/fn_size)

}  // namespace models
