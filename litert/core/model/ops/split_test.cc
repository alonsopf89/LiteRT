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

#include "litert/core/model/ops/split.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

void SetSplitOptions(LiteRtOpT& op, int num_splits) {
  auto options = std::make_unique<tflite::SplitOptionsT>();
  options->num_splits = num_splits;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SplitOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
}

TEST(SplitOpTest, TwoSplits) {
  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  op.Inputs() = {&axis_tensor, &input_tensor};

  std::vector<int32_t> axis_data = {0};
  SetWeightsFromUnownedBuffer(
      axis_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(axis_data.data()),
                                 axis_data.size() * sizeof(int32_t)));

  std::vector<Dims> input_shapes = {{}, {4, 2}};  // Axis scalar, Input
  std::vector<Dims> output_shapes(2);
  SetSplitOptions(op, 2);

  ASSERT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(2, 2));
  }
}

TEST(SplitOpTest, TwoSplitsAxisOne) {
  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  op.Inputs() = {&axis_tensor, &input_tensor};

  std::vector<int32_t> axis_data = {1};
  SetWeightsFromUnownedBuffer(
      axis_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(axis_data.data()),
                                 axis_data.size() * sizeof(int32_t)));

  std::vector<Dims> input_shapes = {{}, {4, 2}};
  std::vector<Dims> output_shapes(2);
  SetSplitOptions(op, 2);

  ASSERT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(4, 1));
  }
}

TEST(SplitOpTest, NegativeAxis) {
  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  op.Inputs() = {&axis_tensor, &input_tensor};

  std::vector<int32_t> axis_data = {-1};
  SetWeightsFromUnownedBuffer(
      axis_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(axis_data.data()),
                                 axis_data.size() * sizeof(int32_t)));

  std::vector<Dims> input_shapes = {{}, {4, 2}};
  std::vector<Dims> output_shapes(2);
  SetSplitOptions(op, 2);

  ASSERT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(4, 1));
  }
}

struct SplitErrorCase {
  std::string test_name;
  bool omit_inputs;
  int num_splits;
  int32_t axis;
};

using SplitErrorTest = ::testing::TestWithParam<SplitErrorCase>;

TEST_P(SplitErrorTest, RejectsMissingInputsNonPositiveSplitsAndAxisUnderflow) {
  const auto& param = GetParam();
  std::vector<Dims> input_shapes = {{}, {4, 2}};
  std::vector<Dims> output_shapes(2);

  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  if (!param.omit_inputs) {
    op.Inputs() = {&axis_tensor, &input_tensor};
    SetSplitOptions(op, param.num_splits);
    SetWeightsFromOwnedBuffer(
        axis_tensor.Weights(),
        OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(&param.axis),
                                 sizeof(int32_t)));
  }

  EXPECT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    SplitErrorTests, SplitErrorTest,
    ::testing::Values(
        SplitErrorCase{"MissingInputs", /*omit_inputs=*/true, 2, 0},
        SplitErrorCase{"NonPositiveSplits", /*omit_inputs=*/false, 0, 0},
        SplitErrorCase{"AxisUnderflow", /*omit_inputs=*/false, 2,
                       std::numeric_limits<int32_t>::min()}),
    [](const ::testing::TestParamInfo<SplitErrorCase>& info) {
      return info.param.test_name;
    });

TEST(SplitOpTest, ReferenceSplitAndSplitVValidCases) {
  const std::vector<int32_t> input = {1, 2, 3, 4, 5, 6, 7, 8};
  const int32_t dims[] = {2, 4};

  // Equal split along negative axis (-1 -> axis 1).
  std::vector<int32_t> out0(4, -1);
  std::vector<int32_t> out1(4, -1);
  std::vector<int32_t*> outputs = {out0.data(), out1.data()};
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/-1,
                          outputs);
  EXPECT_THAT(out0, ElementsAre(1, 2, 5, 6));
  EXPECT_THAT(out1, ElementsAre(3, 4, 7, 8));

  // Unequal split with a zero-sized chunk (nullptr allowed for 0-sized output).
  std::vector<int32_t> v_out0(2, -1);
  std::vector<int32_t> v_out2(6, -1);
  std::vector<int32_t*> v_outputs = {v_out0.data(), nullptr, v_out2.data()};
  const std::vector<int32_t> split_sizes = {1, 0, 3};
  ReferenceSplitV<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/1,
                           split_sizes, v_outputs);
  EXPECT_THAT(v_out0, ElementsAre(1, 5));
  EXPECT_THAT(v_out2, ElementsAre(2, 3, 4, 6, 7, 8));
}

TEST(SplitOpTest, ReferenceSplitRejectsInvalidRankAxisAndDivisibility) {
  const std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  const int32_t dims[] = {2, 3};
  std::vector<int32_t> out0(3, -99);
  std::vector<int32_t> out1(3, -99);
  std::vector<int32_t*> outputs = {out0.data(), out1.data()};

  // Non-positive rank, out-of-range positive axis, double-normalization range
  // [-2*rank, -rank-1], INT32_MIN axis, and non-divisible axis_size.
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/0, /*axis=*/0,
                          outputs);
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/-1, /*axis=*/0,
                          outputs);
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/2,
                          outputs);
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/-3,
                          outputs);
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/-4,
                          outputs);
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/2,
                          std::numeric_limits<int32_t>::min(), outputs);
  ReferenceSplit<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/1,
                          outputs);

  EXPECT_THAT(out0, ElementsAre(-99, -99, -99));
  EXPECT_THAT(out1, ElementsAre(-99, -99, -99));
}

TEST(SplitOpTest, ReferenceSplitVRejectsMismatchedOutputsAndInvalidSplits) {
  const std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  const int32_t dims[] = {2, 3};
  std::vector<int32_t> out0(4, -77);
  std::vector<int32_t> out1(4, -77);

  // Fewer outputs than split_sizes.
  const std::vector<int32_t> two_splits = {1, 2};
  std::vector<int32_t*> one_output = {out0.data()};
  ReferenceSplitV<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/1,
                           two_splits, one_output);

  // Negative split size.
  std::vector<int32_t*> two_outputs = {out0.data(), out1.data()};
  const std::vector<int32_t> neg_splits = {-1, 4};
  ReferenceSplitV<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/1,
                           neg_splits, two_outputs);

  // Sum of split sizes exceeds axis_size.
  const std::vector<int32_t> excess_splits = {2, 2};
  ReferenceSplitV<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/1,
                           excess_splits, two_outputs);

  // Null output for non-zero split chunk.
  std::vector<int32_t*> null_second = {out0.data(), nullptr};
  ReferenceSplitV<int32_t>(input.data(), dims, /*input_rank=*/2, /*axis=*/1,
                           two_splits, null_second);

  EXPECT_THAT(out0, ElementsAre(-77, -77, -77, -77));
  EXPECT_THAT(out1, ElementsAre(-77, -77, -77, -77));
}

}  // namespace
}  // namespace litert::internal
