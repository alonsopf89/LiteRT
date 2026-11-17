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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SHAPE_MANIP_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SHAPE_MANIP_H_

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
#include "litert/core/model/ops/broadcast_to.h"
#include "litert/core/model/ops/expand_dims.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/ops/squeeze.h"
#include "litert/core/model/ops/tile.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename Rank, typename T, typename MaxTensorSize = SizeC<1024>>
class ExpandDims : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    int32_t axis;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<ExpandDims>;

  static constexpr absl::string_view Name() { return "ExpandDims"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.input_shape.begin());
    p.axis = static_cast<int32_t>(rng() % (kRank + 1));
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<ExpandDims>(std::move(params), std::move(model));
  }

  ExpandDims(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.input_shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    litert::internal::ReferenceExpandDims(input.data.data(), input.data.size(),
                                          output.data.data());
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    TensorTf output = litert::tensor::ExpandDims(input, params.axis);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename Rank, typename T>
class Squeeze : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    int32_t axis;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Squeeze>;

  static constexpr absl::string_view Name() { return "Squeeze"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(2, 5);
    for (size_t i = 0; i < kRank; ++i) {
      p.input_shape[i] = dim_dist(rng);
    }
    p.axis = static_cast<int32_t>(rng() % kRank);
    p.input_shape[p.axis] = 1;
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Squeeze>(std::move(params), std::move(model));
  }

  Squeeze(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.input_shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    litert::internal::ReferenceSqueeze(input.data.data(), input.data.size(),
                                       output.data.data());
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static_assert(kRank >= 2);

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    TensorTf output = litert::tensor::Squeeze(input, {params.axis});
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename Rank, typename T>
class Tile : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    std::array<int32_t, Rank::value> multiples;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Tile>;

  static constexpr absl::string_view Name() { return "Tile"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(1, 3);
    std::uniform_int_distribution<int> mult_dist(1, 3);
    for (size_t i = 0; i < kRank; ++i) {
      p.input_shape[i] = dim_dist(rng);
      p.multiples[i] = mult_dist(rng);
    }
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Tile>(std::move(params), std::move(model));
  }

  Tile(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.input_shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    std::array<int32_t, kRank> in_dims;
    for (size_t i = 0; i < kRank; ++i) {
      in_dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    litert::internal::ReferenceTile(
        input.data.data(), in_dims.data(), params_.multiples.data(),
        static_cast<int>(kRank), output.data.data());
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    std::vector<int> multiples(params.multiples.begin(),
                               params.multiples.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    TensorTf output = litert::tensor::Tile(input, multiples);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename Rank, typename T, typename MaxTensorSize = SizeC<1024>>
class ReverseV2 : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    int32_t axis;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<ReverseV2>;

  static constexpr absl::string_view Name() { return "ReverseV2"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.input_shape.begin());
    p.axis = static_cast<int32_t>(rng() % kRank);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<ReverseV2>(std::move(params), std::move(model));
  }

  ReverseV2(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.input_shape, device, data_builder);
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
    litert::internal::ReferenceReverseV2(input.data.data(), dims.data(),
                                         static_cast<int>(kRank), &params_.axis,
                                         1, output.data.data());
    return {};
  }

 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    TensorTf axes(
        {.type = litert::tensor::Type::kI32,
         .shape = {1},
         .buffer =
             litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
                 {params.axis})});
    TensorTf output = litert::tensor::Reverse(input, axes);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename InputRank, typename OutputRank, typename T>
class BroadcastTo : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, InputRank::value> input_shape;
    std::array<Layout::Dim, OutputRank::value> output_shape;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<BroadcastTo>;

  static constexpr absl::string_view Name() { return "BroadcastTo"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::array<Layout::Dim, kOutputRank> target_shape;
    SampleBroadcastShapes<kOutputRank, kInputRank>(
        rng, target_shape, p.input_shape, p.output_shape,
        /*min_dim=*/2, /*max_dim=*/4, /*broadcast_prob=*/0.5,
        /*force_broadcast=*/false, /*unidirectional_rhs_to_lhs=*/true);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<BroadcastTo>(std::move(params), std::move(model));
  }

  BroadcastTo(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.input_shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [input] = ref_in;
    auto [output] = ref_out;
    std::array<int32_t, kInputRank> in_dims;
    for (size_t i = 0; i < kInputRank; ++i) {
      in_dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    std::array<int32_t, kOutputRank> out_dims;
    for (size_t i = 0; i < kOutputRank; ++i) {
      out_dims[i] = static_cast<int32_t>(params_.output_shape[i]);
    }
    LITERT_RETURN_IF_ERROR(litert::internal::ReferenceBroadcastTo(
        input.data.data(), in_dims.data(), static_cast<int>(kInputRank),
        out_dims.data(), static_cast<int>(kOutputRank), output.data.data()));
    return {};
  }

 private:
  static constexpr size_t kInputRank = InputRank::value;
  static constexpr size_t kOutputRank = OutputRank::value;
  static_assert(kOutputRank >= kInputRank);
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> in_dims(params.input_shape.begin(),
                                 params.input_shape.end());
    std::vector<int32_t> out_dims(params.output_shape.begin(),
                                  params.output_shape.end());
    std::vector<TensorDetails> op_inputs(2);
    op_inputs[0] =
        TensorDetails{in_dims, LiteRtElementType(kElementType), "input"};
    op_inputs[1] = TensorDetails{{static_cast<int32_t>(kOutputRank)},
                                 kLiteRtElementTypeInt32,
                                 "shape",
                                 MakeOwningBufferRef(out_dims)};
    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{out_dims, LiteRtElementType(kElementType), "output"};
    MaybeSetTensorDetailsQuant<T>(op_inputs[0]);
    MaybeSetTensorDetailsQuant<T>(op_outputs[0]);
    return SingleOpModel<kLiteRtOpCodeTflBroadcastTo>(op_inputs, op_outputs,
                                                      OpVersion{2});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SHAPE_MANIP_H_
