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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ARITHMETIC_OPS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ARITHMETIC_OPS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
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
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

// Generator for ADD_N with N inputs (fp32, int8).
template <typename Rank, typename T, typename NumInputsC,
          typename MaxTensorSize = SizeC<1024>>
class AddN : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
  };

  using Traits =
      TestLogicTraits<typename RepeatTypeList<T, NumInputsC::value>::Type,
                      TypeList<T>, Params>;
  using Ptr = std::unique_ptr<AddN>;

  static constexpr absl::string_view Name() { return "AddN"; }

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
    return std::make_unique<AddN>(std::move(params), std::move(model));
  }

  AddN(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    if constexpr (std::is_integral_v<T>) {
      ConformanceSpec spec;
      spec.comparator_kind = ConformanceComparatorKind::kQuantizedBucket;
      spec.bucket_tolerance = 1;
      return spec;
    }
    return TestGraph::GetConformanceSpec();
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(kNumInputs);
    for (size_t i = 0; i < kNumInputs; ++i) {
      LITERT_ASSIGN_OR_RETURN(
          auto buf, MakeRandomBuffer<T>(params_.shape, device, data_builder,
                                        -2.0f, 2.0f));
      inputs.push_back(std::move(buf));
    }
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [output] = ref_out;
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    std::vector<const T*> ptrs;
    ptrs.reserve(kNumInputs);
    for (size_t k = 0; k < kNumInputs; ++k) {
      ptrs.push_back(inputs[k].template Span<T>().data());
    }
    litert::internal::ReferenceAddN<T>(ptrs, output.data.data(),
                                       output.data.size(), kQuantSpec.input,
                                       kQuantSpec.output);
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static_assert(std::is_same_v<typename NumInputsC::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kNumInputs = NumInputsC::value;
  static_assert(kNumInputs >= 2);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  struct QuantSpec {
    AffineQuantSpec input;
    AffineQuantSpec output;
  };

  static constexpr QuantSpec GetQuantSpec() {
    if constexpr (std::is_same_v<T, int8_t>) {
      return {{0.02f, 0}, {0.08f, 0}};
    } else {
      return {{1.0f, 0}, {1.0f, 0}};
    }
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    std::vector<TensorDetails> op_inputs(kNumInputs);
    for (size_t i = 0; i < kNumInputs; ++i) {
      op_inputs[i] = TensorDetails{dims, LiteRtElementType(kElementType),
                                   absl::StrCat("input_", i)};
      MaybeSetTensorDetailsQuant<T>(op_inputs[i], kQuantSpec.input);
    }
    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{dims, LiteRtElementType(kElementType), "output"};
    MaybeSetTensorDetailsQuant<T>(op_outputs[0], kQuantSpec.output);
    return SingleOpModel<kLiteRtOpCodeTflAddN>(op_inputs, op_outputs);
  }

  Params params_;
};

// Generator for INT32 FLOOR_DIV and FLOOR_MOD with positive constant divisor.
template <typename Rank, typename T, typename OpCode,
          typename MaxTensorSize = SizeC<1024>>
class FloorConstDiv : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
    int32_t divisor = 2;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<FloorConstDiv>;

  static constexpr absl::string_view Name() { return "FloorConstDiv"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    std::uniform_int_distribution<int32_t> div_dist(2, 16);
    p.divisor = div_dist(rng);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<FloorConstDiv>(std::move(params), std::move(model));
  }

  FloorConstDiv(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    RandomTensorDataBuilder b = data_builder;
    if (!b.IsIntDummy()) {
      b.SetIntRange(-1000, 1000);
    }
    return MakeSingleRandomInput<T>(params_.shape, device, b);
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
    const int div_dims[1] = {1};
    if constexpr (kOpCode == kLiteRtOpCodeTflFloorDiv) {
      litert::internal::ReferenceFloorDiv<T>(
          input.data.data(), dims.data(), static_cast<int>(kRank),
          &params_.divisor, div_dims, 1, output.data.data(), dims.data(),
          static_cast<int>(kRank));
    } else {
      litert::internal::ReferenceFloorMod<T>(
          input.data.data(), dims.data(), static_cast<int>(kRank),
          &params_.divisor, div_dims, 1, output.data.data(), dims.data(),
          static_cast<int>(kRank));
    }
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr size_t kRank = Rank::value;
  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflFloorDiv ||
                kOpCode == kLiteRtOpCodeTflFloorMod);
  static_assert(std::is_same_v<T, int32_t>);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    TensorTf lhs =
        litert::tensor::Create("lhs", litert::tensor::ApiType<T>::value, dims);
    TensorTf rhs(
        {.type = litert::tensor::Type::kI32,
         .shape = {1},
         .buffer =
             litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
                 {params.divisor})});
    rhs.SetName("rhs");
    TensorTf output = (kOpCode == kLiteRtOpCodeTflFloorDiv)
                          ? litert::tensor::FloorDiv(lhs, rhs)
                          : litert::tensor::FloorMod(lhs, rhs);
    output.SetName("output");
    return SaveTensorGraph({lhs}, {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ARITHMETIC_OPS_H_
