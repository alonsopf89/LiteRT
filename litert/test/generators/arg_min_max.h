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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ARG_MIN_MAX_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ARG_MIN_MAX_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reductions.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

// Generator for ARG_MAX (fp32) and ARG_MIN (fp32, int8, int16, int32) -> int32
// on rank >= 2 so output is at least 1D.
template <typename Rank, typename T, typename OpCode,
          typename MaxTensorSize = SizeC<1024>>
class ArgMinMax : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    int32_t axis;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<int32_t>, Params>;
  using Ptr = std::unique_ptr<ArgMinMax>;

  static constexpr absl::string_view Name() { return "ArgMinMax"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kInElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.input_shape.begin());
    std::uniform_int_distribution<int32_t> axis_dist(0, kRank - 1);
    const int32_t pos_axis = axis_dist(rng);
    p.axis =
        (rng() % 2 == 0) ? pos_axis : (pos_axis - static_cast<int32_t>(kRank));
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<ArgMinMax>(std::move(params), std::move(model));
  }

  ArgMinMax(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    RandomTensorDataBuilder b = data_builder;
    if constexpr (std::is_floating_point_v<T>) {
      if (!b.IsFloatDummy()) {
        b.SetFloatRange(-10.0f, 10.0f);
      }
    } else if constexpr (std::is_same_v<T, int32_t>) {
      if (!b.IsIntDummy()) {
        b.SetIntRange(-10000, 10000);
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR((input.template WriteRandom<T>(b, device)));
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    std::array<int32_t, kRank> dims;
    for (size_t i = 0; i < kRank; ++i) {
      dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    if (!litert::internal::ReferenceArgMinMax<T>(
            input.data.data(), dims.data(), static_cast<int>(kRank),
            params_.axis, kOpCode == kLiteRtOpCodeTflArgMax,
            output.data.data())) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "ReferenceArgMinMax failed.");
    }
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr size_t kRank = Rank::value;
  static_assert(kRank >= 2, "ArgMinMax generator expects Rank >= 2.");
  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflArgMax ||
                kOpCode == kLiteRtOpCodeTflArgMin);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kInElementType = GetElementType<T>();

  static constexpr AffineQuantSpec GetInputQuantSpec() {
    if constexpr (std::is_same_v<T, int8_t>) {
      return {0.02f, -5};
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return {0.002f, 0};
    } else {
      return {1.0f, 0};
    }
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> in_dims(params.input_shape.begin(),
                                 params.input_shape.end());
    constexpr AffineQuantSpec kInputQuant = GetInputQuantSpec();
    if constexpr (kOpCode == kLiteRtOpCodeTflArgMax) {
      using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
      TensorTf input = litert::tensor::Create(
          "input", litert::tensor::ApiType<T>::value, in_dims);
      MaybeSetQuant<T>(input, kInputQuant);
      TensorTf output = litert::tensor::ArgMax(input, params.axis,
                                               litert::tensor::Type::kI32);
      output.SetName("output");
      return SaveTensorGraph({input}, {output});
    } else {
      int32_t pos_axis = params.axis;
      if (pos_axis < 0) {
        pos_axis += static_cast<int32_t>(kRank);
      }
      if (pos_axis < 0 || pos_axis >= static_cast<int32_t>(kRank)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "ArgMin axis out of bounds.");
      }
      std::vector<int32_t> out_dims;
      out_dims.reserve(kRank - 1);
      for (size_t i = 0; i < kRank; ++i) {
        if (static_cast<int32_t>(i) != pos_axis) {
          out_dims.push_back(in_dims[i]);
        }
      }
      std::vector<int32_t> axis_vals = {params.axis};
      std::vector<TensorDetails> op_inputs(2);
      op_inputs[0] =
          TensorDetails{in_dims, LiteRtElementType(kInElementType), "input"};
      op_inputs[1] = TensorDetails{
          {1}, kLiteRtElementTypeInt32, "axis", MakeOwningBufferRef(axis_vals)};
      std::vector<TensorDetails> op_outputs(1);
      op_outputs[0] =
          TensorDetails{out_dims, kLiteRtElementTypeInt32, "output"};
      MaybeSetTensorDetailsQuant<T>(op_inputs[0], kInputQuant);
      tflite::ArgMinOptionsT am_opts;
      am_opts.output_type = tflite::TensorType_INT32;
      return SingleOpModelWithOptions<kLiteRtOpCodeTflArgMin>(
          op_inputs, op_outputs, std::move(am_opts));
    }
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ARG_MIN_MAX_H_
