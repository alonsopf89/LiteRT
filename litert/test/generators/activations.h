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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ACTIVATIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ACTIVATIONS_H_

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
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

// Generator for LEAKY_RELU (fp32, int8, int16).
template <typename Rank, typename T, typename MaxTensorSize = SizeC<1024>>
class LeakyRelu : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
    float alpha = 0.25f;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<LeakyRelu>;

  static constexpr absl::string_view Name() { return "LeakyRelu"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    p.alpha = 0.25f;
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<LeakyRelu>(std::move(params), std::move(model));
  }

  LeakyRelu(Params params, LiteRtModelT::Ptr model)
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
    return MakeSingleRandomInput<T>(params_.shape, device, data_builder, -4.0f,
                                    4.0f);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    if constexpr (std::is_floating_point_v<T>) {
      litert::internal::ReferenceLeakyRelu(input.data.data(),
                                           output.data.data(),
                                           input.data.size(), params_.alpha);
    } else {
      constexpr QuantSpec kQuantSpec = GetQuantSpec();
      litert::internal::ReferenceLeakyReluQuantized(
          input.data.data(), output.data.data(), input.data.size(),
          params_.alpha, kQuantSpec.input, kQuantSpec.output);
    }
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  struct QuantSpec {
    AffineQuantSpec input;
    AffineQuantSpec output;
  };

  static constexpr QuantSpec GetQuantSpec() {
    if constexpr (std::is_same_v<T, int8_t>) {
      return {{0.02f, -5}, {0.04f, 10}};
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return {{0.002f, 0}, {0.004f, 0}};
    } else {
      return {{1.0f, 0}, {1.0f, 0}};
    }
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input, kQuantSpec.input);
    TensorTf output = litert::tensor::LeakyRelu(input, params.alpha);
    MaybeSetQuant<T>(output, kQuantSpec.output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

// Generator for PRELU (fp32, int8, int16) with channel-broadcast positive
// alpha.
template <typename Rank, typename T, typename MaxTensorSize = SizeC<1024>>
class Prelu : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
    std::array<Layout::Dim, Rank::value> alpha_shape;
  };

  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Prelu>;

  static constexpr absl::string_view Name() { return "Prelu"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.shape.begin());
    p.alpha_shape.fill(1);
    p.alpha_shape[kRank - 1] = p.shape[kRank - 1];
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Prelu>(std::move(params), std::move(model));
  }

  Prelu(Params params, LiteRtModelT::Ptr model)
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
    inputs.reserve(2);

    RandomTensorDataBuilder in_b = data_builder;
    RandomTensorDataBuilder alpha_b = data_builder;
    if constexpr (std::is_floating_point_v<T>) {
      if (!in_b.IsFloatDummy()) in_b.SetFloatRange(-4.0f, 4.0f);
      if (!alpha_b.IsFloatDummy()) alpha_b.SetFloatRange(0.05f, 0.5f);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      if (!alpha_b.IsIntDummy()) alpha_b.SetIntRange(5, 50);
    } else if constexpr (std::is_same_v<T, int16_t>) {
      if (!alpha_b.IsIntDummy()) alpha_b.SetIntRange(50, 500);
    }

    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(params_.shape));
    LITERT_RETURN_IF_ERROR((input.template WriteRandom<T>(in_b, device)));
    LITERT_ASSIGN_OR_RETURN(auto alpha,
                            SimpleBuffer::Create<T>(params_.alpha_shape));
    LITERT_RETURN_IF_ERROR((alpha.template WriteRandom<T>(alpha_b, device)));
    inputs.push_back(std::move(input));
    inputs.push_back(std::move(alpha));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input, alpha] = ref_in;
    auto [output] = ref_out;
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    litert::internal::ReferencePrelu(input.data.data(), alpha.data.data(),
                                     output.data.data(), input.data.size(),
                                     alpha.data.size(), kQuantSpec.input,
                                     kQuantSpec.alpha, kQuantSpec.output);
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  struct QuantSpec {
    AffineQuantSpec input;
    AffineQuantSpec alpha;
    AffineQuantSpec output;
  };

  static constexpr QuantSpec GetQuantSpec() {
    if constexpr (std::is_same_v<T, int8_t>) {
      return {{0.02f, -5}, {0.01f, 0}, {0.04f, 10}};
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return {{0.002f, 0}, {0.001f, 0}, {0.004f, 0}};
    } else {
      return {{1.0f, 0}, {1.0f, 0}, {1.0f, 0}};
    }
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());
    std::vector<int32_t> alpha_dims(params.alpha_shape.begin(),
                                    params.alpha_shape.end());
    constexpr QuantSpec kQuantSpec = GetQuantSpec();
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    TensorTf alpha = litert::tensor::Create(
        "alpha", litert::tensor::ApiType<T>::value, alpha_dims);
    MaybeSetQuant<T>(input, kQuantSpec.input);
    MaybeSetQuant<T>(alpha, kQuantSpec.alpha);
    TensorTf output = litert::tensor::PRelu(input, alpha);
    MaybeSetQuant<T>(output, kQuantSpec.output);
    output.SetName("output");
    return SaveTensorGraph({input, alpha}, {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ACTIVATIONS_H_
