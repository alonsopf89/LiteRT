// Copyright 2026 Google LLC.
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

#include "litert/core/model/ops/broadcast_to.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/op_shape_inference_utils.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(BroadcastToOpTest, SimpleTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<int32_t> shape_data = {2, 2};
  SetWeightsFromUnownedBuffer(
      shape_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(shape_data.data()),
                                 shape_data.size() * sizeof(int32_t)));
  shape_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeInt32, {static_cast<int32_t>(shape_data.size())}));

  ASSERT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(BroadcastToOpTest, ScalarBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<int32_t> shape_data = {3, 1};
  SetWeightsFromUnownedBuffer(
      shape_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(shape_data.data()),
                                 shape_data.size() * sizeof(int32_t)));
  shape_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeInt32, {static_cast<int32_t>(shape_data.size())}));

  ASSERT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3, 1));
}

TEST(BroadcastToOpTest, InvalidBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<int32_t> shape_data = {2, 3};
  SetWeightsFromUnownedBuffer(
      shape_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(shape_data.data()),
                                 shape_data.size() * sizeof(int32_t)));
  shape_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeInt32, {static_cast<int32_t>(shape_data.size())}));

  EXPECT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

struct BroadcastToErrorCase {
  std::string test_name;
  bool omit_inputs;
  LiteRtElementType shape_type;
  std::vector<uint8_t> shape_bytes;
};

using BroadcastToErrorTest = ::testing::TestWithParam<BroadcastToErrorCase>;

TEST_P(BroadcastToErrorTest, RejectsMissingInputTensorOrInvalidTargetDims) {
  const auto& param = GetParam();
  std::vector<Dims> input_shapes = {{1, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtOpT op;
  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  if (!param.omit_inputs) {
    shape_tensor.SetType(MakeRankedTensorType(param.shape_type, {2}));
    SetWeightsFromOwnedBuffer(
        shape_tensor.Weights(),
        OwningBufferRef<uint8_t>(param.shape_bytes.data(),
                                 param.shape_bytes.size()));
    op.Inputs() = {&input_tensor, &shape_tensor};
  }

  EXPECT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastToErrorTests, BroadcastToErrorTest,
    ::testing::Values(
        BroadcastToErrorCase{
            "MissingInputs", /*omit_inputs=*/true, kLiteRtElementTypeInt32, {}},
        BroadcastToErrorCase{"NegativeDimInInt32Shape", /*omit_inputs=*/false,
                             kLiteRtElementTypeInt32,
                             ToBytes<int32_t>({-2, 2})},
        BroadcastToErrorCase{
            "DimExceedingInt32MaxInInt64Shape",
            /*omit_inputs=*/false, kLiteRtElementTypeInt64,
            ToBytes<int64_t>(
                {2, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) +
                        1})}),
    [](const ::testing::TestParamInfo<BroadcastToErrorCase>& info) {
      return info.param.test_name;
    });

TEST(BroadcastToOpTest, ComputeBroadcastStridesCheckedValidCases) {
  std::vector<size_t> strides;
  size_t total_out = 0;

  // Scalar -> 2D.
  ASSERT_EQ(ComputeBroadcastStridesChecked({}, {2, 3}, strides, total_out),
            kLiteRtStatusOk);
  EXPECT_THAT(strides, ElementsAre(0, 0));
  EXPECT_EQ(total_out, 6);

  // Scalar -> scalar.
  ASSERT_EQ(ComputeBroadcastStridesChecked({}, {}, strides, total_out),
            kLiteRtStatusOk);
  EXPECT_TRUE(strides.empty());
  EXPECT_EQ(total_out, 1);

  // Leading rank expansion + singleton dim: [1, 3] -> [2, 4, 3].
  ASSERT_EQ(
      ComputeBroadcastStridesChecked({1, 3}, {2, 4, 3}, strides, total_out),
      kLiteRtStatusOk);
  EXPECT_THAT(strides, ElementsAre(0, 0, 1));
  EXPECT_EQ(total_out, 24);

  // [2, 1, 3] -> [2, 4, 3].
  ASSERT_EQ(
      ComputeBroadcastStridesChecked({2, 1, 3}, {2, 4, 3}, strides, total_out),
      kLiteRtStatusOk);
  EXPECT_THAT(strides, ElementsAre(3, 0, 1));
  EXPECT_EQ(total_out, 24);

  // Zero-sized tensor with huge trailing dims must NOT overflow stride math.
  ASSERT_EQ(ComputeBroadcastStridesChecked({0, 65536, 65536, 65536, 65536},
                                           {0, 65536, 65536, 65536, 65536},
                                           strides, total_out),
            kLiteRtStatusOk);
  EXPECT_EQ(total_out, 0);
  EXPECT_THAT(strides, ElementsAre(0, 0, 0, 0, 0));
}

TEST(BroadcastToOpTest, ComputeBroadcastStridesCheckedRejectsInvalidInputs) {
  std::vector<size_t> strides;
  size_t total_out = 0;

  // Input rank exceeds output rank.
  EXPECT_EQ(ComputeBroadcastStridesChecked({2, 3}, {3}, strides, total_out),
            kLiteRtStatusErrorInvalidArgument);
  // Incompatible non-singleton dimension.
  EXPECT_EQ(ComputeBroadcastStridesChecked({2, 3}, {2, 4}, strides, total_out),
            kLiteRtStatusErrorInvalidArgument);
  // Incompatible dimension hidden behind leading zero.
  EXPECT_EQ(ComputeBroadcastStridesChecked({0, 3}, {0, 4}, strides, total_out),
            kLiteRtStatusErrorInvalidArgument);
  // Dynamic (-1) or negative dimensions, even when another dimension is 0.
  EXPECT_EQ(ComputeBroadcastStridesChecked({-1, 3}, {2, 3}, strides, total_out),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ComputeBroadcastStridesChecked({0, -1}, {0, 3}, strides, total_out),
            kLiteRtStatusErrorInvalidArgument);
  // Element-count overflow (65536^4 == 2^64).
  EXPECT_EQ(ComputeBroadcastStridesChecked({65536, 65536, 65536, 65536},
                                           {65536, 65536, 65536, 65536},
                                           strides, total_out),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(BroadcastToOpTest, ReferenceBroadcastToValuesAndByteOverflow) {
  // [2, 1, 3] -> [2, 2, 3].
  const std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  const int32_t in_dims[] = {2, 1, 3};
  const int32_t out_dims[] = {2, 2, 3};
  std::vector<int32_t> out(12, 0);
  ASSERT_EQ(
      ReferenceBroadcastTo(input.data(), in_dims, 3, out_dims, 3, out.data()),
      kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6));

  // Scalar -> [2, 2].
  const int32_t scalar = 42;
  const int32_t s_out_dims[] = {2, 2};
  std::vector<int32_t> s_out(4, 0);
  ASSERT_EQ(
      ReferenceBroadcastTo(&scalar, nullptr, 0, s_out_dims, 2, s_out.data()),
      kLiteRtStatusOk);
  EXPECT_THAT(s_out, ElementsAre(42, 42, 42, 42));

  // Element count fits in size_t (2^63) while byte count for int32_t overflows.
  const int32_t huge_dims[] = {1 << 30, 1 << 30, 8};
  EXPECT_EQ(
      ReferenceBroadcastTo<int32_t>(nullptr, nullptr, 0, huge_dims, 3, nullptr),
      kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::internal
