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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_QUANTIZE_CAST_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_QUANTIZE_CAST_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename InT, typename OutT, typename Rank,
          typename MaxTensorSize = SizeC<1024>>
class Quantize : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
  };

  using Traits = TestLogicTraits<TypeList<InT>, TypeList<OutT>, Params>;
  using Ptr = std::unique_ptr<Quantize>;

  static constexpr absl::string_view Name() { return "Quantize"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kInElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Quantize>(std::move(params), std::move(model));
  }

  Quantize(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kQuantizedBucket;
    spec.bucket_tolerance = 1;
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<InT>(params_.shape, device, data_builder,
                                      -2.0f, 2.0f);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    litert::internal::ReferenceQuantize<InT, OutT>(
        input.data.data(), input.data.size(), kQuantSpec.input.scale,
        kQuantSpec.input.zero_point, kQuantSpec.output.scale,
        kQuantSpec.output.zero_point, output.data.data());
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kInElementType = GetElementType<InT>();

  struct QuantSpec {
    AffineQuantSpec input;
    AffineQuantSpec output;
  };

  static constexpr QuantSpec GetQuantSpec() {
    AffineQuantSpec in_spec{1.0f, 0};
    if constexpr (std::is_same_v<InT, int8_t>) {
      in_spec = {0.02f, -5};
    } else if constexpr (std::is_same_v<InT, int16_t>) {
      in_spec = {0.002f, 0};
    }
    AffineQuantSpec out_spec{0.04f, 10};
    if constexpr (std::is_same_v<OutT, int16_t>) {
      out_spec = {0.004f, 0};
    }
    return {in_spec, out_spec};
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<InT>::value, dims);
    MaybeSetQuant<InT>(input, kQuantSpec.input);
    TensorTf output = litert::tensor::Quantize(
        input, litert::tensor::ApiType<OutT>::value, {kQuantSpec.output.scale},
        {kQuantSpec.output.zero_point});
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename InT, typename OutT, typename Rank,
          typename MaxTensorSize = SizeC<1024>>
class Cast : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
  };

  using Traits = TestLogicTraits<TypeList<InT>, TypeList<OutT>, Params>;
  using Ptr = std::unique_ptr<Cast>;

  static constexpr absl::string_view Name() { return "Cast"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kInElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Cast>(std::move(params), std::move(model));
  }

  Cast(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    RandomTensorDataBuilder b = data_builder;
    if constexpr (std::is_floating_point_v<InT>) {
      if (!b.IsFloatDummy()) {
        b.SetFloatRange(-1000.0f, 1000.0f);
      }
    } else if constexpr (std::is_same_v<OutT, bool>) {
      if (!b.IsIntDummy()) {
        b.SetIntRange(-5, 5);
      }
    } else if constexpr (std::is_same_v<InT, int32_t> &&
                         std::is_same_v<OutT, int8_t>) {
      if (!b.IsIntDummy()) {
        b.SetIntRange(-300, 300);
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<InT>(params_.shape));
    LITERT_RETURN_IF_ERROR((input.template WriteRandom<InT>(b, device)));
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
    litert::internal::ReferenceCast(input.data.data(), input.data.size(),
                                    output.data.data());
    return {};
  }

 private:
  static_assert(!std::is_same_v<InT, OutT>,
                "Cast generator requires InT != OutT.");
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kInElementType = GetElementType<InT>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<InT>::value, dims);
    TensorTf output =
        litert::tensor::Cast(input, litert::tensor::ApiType<OutT>::value);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_QUANTIZE_CAST_H_
