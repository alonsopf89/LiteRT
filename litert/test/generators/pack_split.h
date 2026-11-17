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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PACK_SPLIT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PACK_SPLIT_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
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
#include "litert/core/model/ops/pack.h"
#include "litert/core/model/ops/split.h"
#include "litert/core/model/ops/unpack.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

template <typename Rank, typename T, typename Axis = SizeC<0>,
          typename MaxTensorSize = SizeC<512>>
class Pack : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
  };

  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Pack>;

  static constexpr absl::string_view Name() { return "Pack"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy_n(std::cbegin(tensor_type.layout.dimensions), kRank,
                p.input_shape.begin());
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Pack>(std::move(params), std::move(model));
  }

  Pack(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);
    for (int i = 0; i < 2; ++i) {
      LITERT_ASSIGN_OR_RETURN(
          auto buf,
          MakeRandomBuffer<T>(params_.input_shape, device, data_builder));
      inputs.push_back(std::move(buf));
    }
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [in0, in1] = ref_in;
    auto [output] = ref_out;
    std::array<int32_t, kRank> dims;
    for (size_t i = 0; i < kRank; ++i) {
      dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    const std::array<const T*, 2> ptrs = {in0.data.data(), in1.data.data()};
    litert::internal::ReferencePack<T>(
        ptrs, dims.data(), static_cast<int>(kRank), kAxis, output.data.data());
    return {};
  }

 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr int32_t kAxis = Axis::value;
  static_assert(kAxis >= 0 && kAxis <= static_cast<int32_t>(kRank));
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf in0 = litert::tensor::Create(
        "input_0", litert::tensor::ApiType<T>::value, dims);
    TensorTf in1 = litert::tensor::Create(
        "input_1", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(in0);
    MaybeSetQuant<T>(in1);
    TensorTf output = litert::tensor::Pack({in0, in1}, kAxis);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({in0, in1}, {output});
  }

  Params params_;
};

template <typename Rank, typename T, typename Axis = SizeC<0>>
class Unpack : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T, T>, Params>;
  using Ptr = std::unique_ptr<Unpack>;

  static constexpr absl::string_view Name() { return "Unpack"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(2, 4);
    for (size_t i = 0; i < kRank; ++i) {
      p.input_shape[i] = dim_dist(rng);
    }
    p.input_shape[kAxis] = 2;
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Unpack>(std::move(params), std::move(model));
  }

  Unpack(Params params, LiteRtModelT::Ptr model)
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
    auto [out0, out1] = ref_out;
    std::array<int32_t, kRank> dims;
    for (size_t i = 0; i < kRank; ++i) {
      dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    const std::array<T*, 2> ptrs = {out0.data.data(), out1.data.data()};
    litert::internal::ReferenceUnpack<T>(input.data.data(), dims.data(),
                                         static_cast<int>(kRank), kAxis, ptrs);
    return {};
  }

 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr int32_t kAxis = Axis::value;
  static_assert(kRank >= 2 && kAxis >= 0 &&
                kAxis < static_cast<int32_t>(kRank));

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    std::vector<TensorTf> outputs = litert::tensor::Unpack(input, 2, kAxis);
    for (size_t i = 0; i < outputs.size(); ++i) {
      MaybeSetQuant<T>(outputs[i]);
      outputs[i].SetName(i == 0 ? "output_0" : "output_1");
    }
    return SaveTensorGraph({input}, {outputs[0], outputs[1]});
  }

  Params params_;
};

template <typename Rank, typename T, typename Axis = SizeC<0>>
class Split : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T, T>, Params>;
  using Ptr = std::unique_ptr<Split>;

  static constexpr absl::string_view Name() { return "Split"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(2, 4);
    for (size_t i = 0; i < kRank; ++i) {
      p.input_shape[i] = dim_dist(rng);
    }
    p.input_shape[kAxis] = 2 * dim_dist(rng);
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Split>(std::move(params), std::move(model));
  }

  Split(Params params, LiteRtModelT::Ptr model)
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
    auto [out0, out1] = ref_out;
    std::array<int32_t, kRank> dims;
    for (size_t i = 0; i < kRank; ++i) {
      dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    const std::array<T*, 2> ptrs = {out0.data.data(), out1.data.data()};
    litert::internal::ReferenceSplit<T>(input.data.data(), dims.data(),
                                        static_cast<int>(kRank), kAxis, ptrs);
    return {};
  }

 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr int32_t kAxis = Axis::value;
  static_assert(kAxis >= 0 && kAxis < static_cast<int32_t>(kRank));

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    std::vector<TensorTf> outputs = litert::tensor::Split(input, kAxis, 2);
    for (size_t i = 0; i < outputs.size(); ++i) {
      MaybeSetQuant<T>(outputs[i]);
      outputs[i].SetName(i == 0 ? "output_0" : "output_1");
    }
    return SaveTensorGraph({input}, {outputs[0], outputs[1]});
  }

  Params params_;
};

template <typename Rank, typename T, typename Axis = SizeC<0>>
class SplitV : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    std::array<int32_t, 2> split_sizes;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T, T>, Params>;
  using Ptr = std::unique_ptr<SplitV>;

  static constexpr absl::string_view Name() { return "SplitV"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(2, 4);
    for (size_t i = 0; i < kRank; ++i) {
      p.input_shape[i] = dim_dist(rng);
    }
    p.split_sizes[0] = dim_dist(rng);
    p.split_sizes[1] = dim_dist(rng) + 1;
    p.input_shape[kAxis] = p.split_sizes[0] + p.split_sizes[1];
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<SplitV>(std::move(params), std::move(model));
  }

  SplitV(Params params, LiteRtModelT::Ptr model)
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
    auto [out0, out1] = ref_out;
    std::array<int32_t, kRank> dims;
    for (size_t i = 0; i < kRank; ++i) {
      dims[i] = static_cast<int32_t>(params_.input_shape[i]);
    }
    const std::array<T*, 2> ptrs = {out0.data.data(), out1.data.data()};
    litert::internal::ReferenceSplitV<T>(input.data.data(), dims.data(),
                                         static_cast<int>(kRank), kAxis,
                                         params_.split_sizes, ptrs);
    return {};
  }

 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr int32_t kAxis = Axis::value;
  static_assert(kAxis >= 0 && kAxis < static_cast<int32_t>(kRank));
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    std::vector<int32_t> sizes(params.split_sizes.begin(),
                               params.split_sizes.end());
    std::vector<int32_t> axis_data = {kAxis};

    std::vector<TensorDetails> op_inputs(3);
    op_inputs[0] =
        TensorDetails{dims, LiteRtElementType(kElementType), "input"};
    op_inputs[1] = TensorDetails{{2},
                                 kLiteRtElementTypeInt32,
                                 "size_splits",
                                 MakeOwningBufferRef(sizes)};
    op_inputs[2] = TensorDetails{{},
                                 kLiteRtElementTypeInt32,
                                 "split_dim",
                                 MakeOwningBufferRef(axis_data)};

    std::vector<int32_t> out0_dims = dims;
    out0_dims[kAxis] = sizes[0];
    std::vector<int32_t> out1_dims = dims;
    out1_dims[kAxis] = sizes[1];
    std::vector<TensorDetails> op_outputs(2);
    op_outputs[0] =
        TensorDetails{out0_dims, LiteRtElementType(kElementType), "output_0"};
    op_outputs[1] =
        TensorDetails{out1_dims, LiteRtElementType(kElementType), "output_1"};
    MaybeSetTensorDetailsQuant<T>(op_inputs[0]);
    MaybeSetTensorDetailsQuant<T>(op_outputs[0]);
    MaybeSetTensorDetailsQuant<T>(op_outputs[1]);
    tflite::SplitVOptionsT options;
    options.num_splits = 2;
    return SingleOpModelWithOptions<kLiteRtOpCodeTflSplitV>(
        op_inputs, op_outputs, std::move(options));
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PACK_SPLIT_H_
