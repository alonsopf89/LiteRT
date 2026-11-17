// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/test/generators/graph_helpers.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_layout.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::litert::internal::GetTflOpCodeInd;
using ::litert::internal::GetTflOpCodes;
using ::litert::internal::GetTflOptions;

TEST(GraphHelpersTest, TestSingleOpModel) {
  TensorDetails lhs = {{1, 2, 3}, kLiteRtElementTypeInt32, "lhs"};

  TensorDetails rhs = {
      {}, kLiteRtElementTypeInt32, "cst", MakeBufferRef<int32_t>({1})};

  TensorDetails output = {{1, 2, 3}, kLiteRtElementTypeInt32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflAdd>(
                      {std::move(lhs), std::move(rhs)}, {std::move(output)},
                      tflite::ActivationFunctionType_NONE, false));

  auto& sg = model->Subgraph(0);
  {
    EXPECT_EQ(sg.Inputs().size(), 1);
    EXPECT_EQ(sg.Inputs()[0]->Name(), "lhs");

    EXPECT_EQ(sg.Outputs().size(), 1);
    EXPECT_EQ(sg.Outputs()[0]->Name(), "output");

    ASSERT_EQ(sg.Tensors().size(), 3);
  }

  {
    ASSERT_EQ(sg.Ops().size(), 1);
    const auto& op = sg.Op(0);
    EXPECT_EQ(op.OpCode(), kLiteRtOpCodeTflAdd);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_AddOptions);
    const auto* add_opts = tfl_opts.AsAddOptions();
    ASSERT_NE(add_opts, nullptr);
    EXPECT_EQ(add_opts->fused_activation_function,
              tflite::ActivationFunctionType_NONE);
    EXPECT_EQ(add_opts->pot_scale_int16, false);
    EXPECT_EQ(op.NumInputs(), 2);
    EXPECT_EQ(op.NumOutputs(), 1);
  }

  {
    const auto& lhs_tensor = sg.Tensor(0);
    EXPECT_EQ(lhs_tensor.Name(), "lhs");
  }

  {
    const auto& rhs_tensor = sg.Tensor(1);
    EXPECT_EQ(rhs_tensor.Name(), "cst");
    EXPECT_EQ(rhs_tensor.Weights().Buffer().Size(), sizeof(int32_t));
  }

  {
    const auto& output_tensor = sg.Tensor(2);
    EXPECT_EQ(output_tensor.Name(), "output");
  }
}

TEST(GraphHelpersTest, SingleOpModelPropagatesOpVersion) {
  std::vector<int32_t> target_shape = {2, 3};
  TensorDetails input = {{1, 3}, kLiteRtElementTypeFloat32, "input"};
  TensorDetails shape = {
      {2}, kLiteRtElementTypeInt32, "shape", MakeOwningBufferRef(target_shape)};
  TensorDetails output = {{2, 3}, kLiteRtElementTypeFloat32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflBroadcastTo>(
                      {std::move(input), std::move(shape)}, {std::move(output)},
                      OpVersion{2}));

  const auto& sg = model->Subgraph(0);
  ASSERT_EQ(sg.Ops().size(), 1);
  const auto& op = sg.Op(0);
  EXPECT_EQ(op.OpCode(), kLiteRtOpCodeTflBroadcastTo);
  const auto& codes = GetTflOpCodes(*model);
  const int32_t code_idx = GetTflOpCodeInd(op);
  ASSERT_GE(code_idx, 0);
  ASSERT_LT(static_cast<size_t>(code_idx), codes.size());
  EXPECT_EQ(codes[code_idx]->version, 2);
}

TEST(GraphHelpersTest, SingleOpModelWithOptionsSetsTypedOptionsAndVersion) {
  std::vector<int32_t> target_size = {4, 4};
  TensorDetails input = {{1, 2, 2, 1}, kLiteRtElementTypeFloat32, "input"};
  TensorDetails size = {
      {2}, kLiteRtElementTypeInt32, "size", MakeOwningBufferRef(target_size)};
  TensorDetails output = {{1, 4, 4, 1}, kLiteRtElementTypeFloat32, "output"};

  tflite::ResizeBilinearOptionsT opts;
  opts.align_corners = false;
  opts.half_pixel_centers = true;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModelWithOptions<kLiteRtOpCodeTflResizeBilinear>(
                      {std::move(input), std::move(size)}, {std::move(output)},
                      std::move(opts), OpVersion{3}));

  const auto& sg = model->Subgraph(0);
  ASSERT_EQ(sg.Ops().size(), 1);
  const auto& op = sg.Op(0);
  EXPECT_EQ(op.OpCode(), kLiteRtOpCodeTflResizeBilinear);
  const auto& codes = GetTflOpCodes(*model);
  const int32_t code_idx = GetTflOpCodeInd(op);
  ASSERT_GE(code_idx, 0);
  ASSERT_LT(static_cast<size_t>(code_idx), codes.size());
  EXPECT_EQ(codes[code_idx]->version, 3);
  const auto* resize_opts = GetTflOptions(op).AsResizeBilinearOptions();
  ASSERT_NE(resize_opts, nullptr);
  EXPECT_FALSE(resize_opts->align_corners);
  EXPECT_TRUE(resize_opts->half_pixel_centers);
}

TEST(GraphHelpersTest, SampleBroadcastShapesRespectsInvariants) {
  std::mt19937 rng(12345);

  for (int iter = 0; iter < 50; ++iter) {
    std::array<Layout::Dim, 3> s1;
    std::array<Layout::Dim, 3> s2;
    std::array<Layout::Dim, 3> out;
    SampleBroadcastShapes<3, 3>(rng, s1, s2, out, /*min_dim=*/2, /*max_dim=*/5,
                                /*broadcast_prob=*/0.2,
                                /*force_broadcast=*/true);
    bool differed = false;
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_TRUE(s1[i] == s2[i] || s1[i] == 1 || s2[i] == 1);
      EXPECT_EQ(out[i], std::max(s1[i], s2[i]));
      EXPECT_GE(out[i], 2);
      EXPECT_LE(out[i], 5);
      if (s1[i] != s2[i]) {
        differed = true;
      }
    }
    EXPECT_TRUE(differed);
  }

  for (int iter = 0; iter < 50; ++iter) {
    std::array<Layout::Dim, 4> lhs;
    std::array<Layout::Dim, 2> rhs;
    std::array<Layout::Dim, 4> out;
    SampleBroadcastShapes<4, 2>(rng, lhs, rhs, out, /*min_dim=*/2,
                                /*max_dim=*/5, /*broadcast_prob=*/0.5,
                                /*force_broadcast=*/false,
                                /*unidirectional_rhs_to_lhs=*/true);
    EXPECT_EQ(out, lhs);
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_TRUE(rhs[i] == 1 || rhs[i] == lhs[i + 2]);
    }
  }
}

}  // namespace
}  // namespace litert::testing
