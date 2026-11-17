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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_LOGICAL_BITWISE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_LOGICAL_BITWISE_H_

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
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

// Generator for LOGICAL_NOT (bool -> bool).
template <typename Rank, typename MaxTensorSize = SizeC<1024>>
class LogicalNot : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
  };

  using Traits = TestLogicTraits<TypeList<bool>, TypeList<bool>, Params>;
  using Ptr = std::unique_ptr<LogicalNot>;

  static constexpr absl::string_view Name() { return "LogicalNot"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<LogicalNot>(std::move(params), std::move(model));
  }

  LogicalNot(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<bool>(params_.shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    litert::internal::ReferenceLogicalNot(input.data.data(), output.data.data(),
                                          input.data.size());
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<bool>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<bool>::value, dims);
    TensorTf output = litert::tensor::LogicalNot(input);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

// Generator for LOGICAL_AND and LOGICAL_OR with broadcasting.
template <typename Rank1, typename Rank2, typename OpCode>
class LogicalBinary : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank1::value> input1_shape;
    std::array<Layout::Dim, Rank2::value> input2_shape;
    std::array<Layout::Dim, std::max(Rank1::value, Rank2::value)> output_shape;
  };

  using Traits = TestLogicTraits<TypeList<bool, bool>, TypeList<bool>, Params>;
  using Ptr = std::unique_ptr<LogicalBinary>;

  static constexpr absl::string_view Name() { return "LogicalBinary"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    SampleBroadcastShapes<kRank1, kRank2>(rng, p.input1_shape, p.input2_shape,
                                          p.output_shape, /*min_dim=*/1,
                                          /*max_dim=*/6);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<LogicalBinary>(std::move(params), std::move(model));
  }

  LogicalBinary(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);
    LITERT_ASSIGN_OR_RETURN(
        auto lhs,
        MakeRandomBuffer<bool>(params_.input1_shape, device, data_builder));
    LITERT_ASSIGN_OR_RETURN(
        auto rhs,
        MakeRandomBuffer<bool>(params_.input2_shape, device, data_builder));
    inputs.push_back(std::move(lhs));
    inputs.push_back(std::move(rhs));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [lhs, rhs] = ref_in;
    auto [output] = ref_out;
    if constexpr (kOpCode == kLiteRtOpCodeTflLogicalAnd) {
      litert::internal::ReferenceLogicalAnd(
          lhs.data.data(), params_.input1_shape.data(), kRank1, rhs.data.data(),
          params_.input2_shape.data(), kRank2, output.data.data(),
          params_.output_shape.data(), kMaxRank);
    } else {
      litert::internal::ReferenceLogicalOr(
          lhs.data.data(), params_.input1_shape.data(), kRank1, rhs.data.data(),
          params_.input2_shape.data(), kRank2, output.data.data(),
          params_.output_shape.data(), kMaxRank);
    }
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank1::value_type, size_t>);
  static_assert(std::is_same_v<typename Rank2::value_type, size_t>);
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr size_t kRank1 = Rank1::value;
  static constexpr size_t kRank2 = Rank2::value;
  static constexpr size_t kMaxRank = std::max(kRank1, kRank2);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflLogicalAnd ||
                kOpCode == kLiteRtOpCodeTflLogicalOr);

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims1(params.input1_shape.begin(),
                               params.input1_shape.end());
    std::vector<int32_t> dims2(params.input2_shape.begin(),
                               params.input2_shape.end());
    TensorTf lhs = litert::tensor::Create(
        "lhs", litert::tensor::ApiType<bool>::value, dims1);
    TensorTf rhs = litert::tensor::Create(
        "rhs", litert::tensor::ApiType<bool>::value, dims2);
    TensorTf output = (kOpCode == kLiteRtOpCodeTflLogicalAnd)
                          ? litert::tensor::LogicalAnd(lhs, rhs)
                          : litert::tensor::LogicalOr(lhs, rhs);
    output.SetName("output");
    return SaveTensorGraph({lhs, rhs}, {output});
  }

  Params params_;
};

// Generator for INT32 BITWISE_XOR with broadcasting.
template <typename Rank1, typename Rank2>
class BitwiseXor : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank1::value> input1_shape;
    std::array<Layout::Dim, Rank2::value> input2_shape;
    std::array<Layout::Dim, std::max(Rank1::value, Rank2::value)> output_shape;
  };

  using Traits =
      TestLogicTraits<TypeList<int32_t, int32_t>, TypeList<int32_t>, Params>;
  using Ptr = std::unique_ptr<BitwiseXor>;

  static constexpr absl::string_view Name() { return "BitwiseXor"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    SampleBroadcastShapes<kRank1, kRank2>(rng, p.input1_shape, p.input2_shape,
                                          p.output_shape, /*min_dim=*/1,
                                          /*max_dim=*/6);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<BitwiseXor>(std::move(params), std::move(model));
  }

  BitwiseXor(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);
    LITERT_ASSIGN_OR_RETURN(
        auto lhs,
        MakeRandomBuffer<int32_t>(params_.input1_shape, device, data_builder));
    LITERT_ASSIGN_OR_RETURN(
        auto rhs,
        MakeRandomBuffer<int32_t>(params_.input2_shape, device, data_builder));
    inputs.push_back(std::move(lhs));
    inputs.push_back(std::move(rhs));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [lhs, rhs] = ref_in;
    auto [output] = ref_out;
    litert::internal::ReferenceBitwiseXor<int32_t>(
        lhs.data.data(), params_.input1_shape.data(), kRank1, rhs.data.data(),
        params_.input2_shape.data(), kRank2, output.data.data(),
        params_.output_shape.data(), kMaxRank);
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank1::value_type, size_t>);
  static_assert(std::is_same_v<typename Rank2::value_type, size_t>);
  static constexpr size_t kRank1 = Rank1::value;
  static constexpr size_t kRank2 = Rank2::value;
  static constexpr size_t kMaxRank = std::max(kRank1, kRank2);

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims1(params.input1_shape.begin(),
                               params.input1_shape.end());
    std::vector<int32_t> dims2(params.input2_shape.begin(),
                               params.input2_shape.end());
    TensorTf lhs = litert::tensor::Create(
        "lhs", litert::tensor::ApiType<int32_t>::value, dims1);
    TensorTf rhs = litert::tensor::Create(
        "rhs", litert::tensor::ApiType<int32_t>::value, dims2);
    TensorTf output = litert::tensor::BitwiseXor(lhs, rhs);
    output.SetName("output");
    return SaveTensorGraph({lhs, rhs}, {output});
  }

  Params params_;
};

// Generator for UINT32 RIGHT_SHIFT with constant shift in [0, 31].
template <typename Rank, typename MaxTensorSize = SizeC<1024>>
class RightShift : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
    uint32_t shift = 1;
  };

  using Traits =
      TestLogicTraits<TypeList<uint32_t>, TypeList<uint32_t>, Params>;
  using Ptr = std::unique_ptr<RightShift>;

  static constexpr absl::string_view Name() { return "RightShift"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    std::uniform_int_distribution<uint32_t> shift_dist(0, 31);
    p.shift = shift_dist(rng);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<RightShift>(std::move(params), std::move(model));
  }

  RightShift(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<uint32_t>(params_.shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    std::array<int, kRank> dims;
    for (size_t i = 0; i < kRank; ++i) {
      dims[i] = static_cast<int>(params_.shape[i]);
    }
    const int shift_dims[1] = {1};
    litert::internal::ReferenceRightShift<uint32_t>(
        input.data.data(), dims.data(), static_cast<int>(kRank), &params_.shift,
        shift_dims, 1, output.data.data(), dims.data(),
        static_cast<int>(kRank));
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<uint32_t>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    TensorTf lhs = litert::tensor::Create(
        "lhs", litert::tensor::ApiType<uint32_t>::value, dims);
    TensorTf shift(
        {.type = litert::tensor::Type::kU32,
         .shape = {1},
         .buffer =
             litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kU32>(
                 {params.shift})});
    shift.SetName("shift");
    TensorTf output = litert::tensor::RightShift(lhs, shift);
    output.SetName("output");
    return SaveTensorGraph({lhs}, {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_LOGICAL_BITWISE_H_
