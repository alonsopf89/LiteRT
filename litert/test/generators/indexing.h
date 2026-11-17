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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_INDEXING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_INDEXING_H_

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
#include "litert/core/model/ops/gather.h"
#include "litert/core/model/ops/slice.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

template <typename ParamsRank, typename IndicesRank, typename T>
class Gather : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, ParamsRank::value> params_shape;
    std::array<Layout::Dim, IndicesRank::value> indices_shape;
    std::vector<int32_t> indices_data;
    int32_t axis;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Gather>;

  static constexpr absl::string_view Name() { return "Gather"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(2, 5);
    for (size_t i = 0; i < kParamsRank; ++i) {
      p.params_shape[i] = dim_dist(rng);
    }
    size_t num_indices = 1;
    std::uniform_int_distribution<int> idx_dim_dist(1, 3);
    for (size_t i = 0; i < kIndicesRank; ++i) {
      p.indices_shape[i] = idx_dim_dist(rng);
      num_indices *= static_cast<size_t>(p.indices_shape[i]);
    }
    p.axis = static_cast<int32_t>(rng() % kParamsRank);
    std::uniform_int_distribution<int32_t> val_dist(
        0, static_cast<int32_t>(p.params_shape[p.axis]) - 1);
    p.indices_data.resize(num_indices);
    for (size_t i = 0; i < num_indices; ++i) {
      p.indices_data[i] = val_dist(rng);
    }
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Gather>(std::move(params), std::move(model));
  }

  Gather(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.params_shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [params_buf] = ref_in;
    auto [output] = ref_out;
    std::array<int32_t, kParamsRank> p_dims;
    for (size_t i = 0; i < kParamsRank; ++i) {
      p_dims[i] = static_cast<int32_t>(params_.params_shape[i]);
    }
    std::array<int32_t, kIndicesRank> i_dims;
    for (size_t i = 0; i < kIndicesRank; ++i) {
      i_dims[i] = static_cast<int32_t>(params_.indices_shape[i]);
    }
    litert::internal::ReferenceGather(
        params_buf.data.data(), p_dims.data(), static_cast<int>(kParamsRank),
        params_.indices_data.data(), i_dims.data(),
        static_cast<int>(kIndicesRank), params_.axis, /*batch_dims=*/0,
        output.data.data());
    return {};
  }

 private:
  static constexpr size_t kParamsRank = ParamsRank::value;
  static constexpr size_t kIndicesRank = IndicesRank::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> p_dims(params.params_shape.begin(),
                                params.params_shape.end());
    std::vector<int> i_dims(params.indices_shape.begin(),
                            params.indices_shape.end());
    TensorTf input = litert::tensor::Create(
        "params", litert::tensor::ApiType<T>::value, p_dims);
    MaybeSetQuant<T>(input);
    TensorTf indices(
        {.type = litert::tensor::Type::kI32,
         .shape = i_dims,
         .buffer =
             litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
                 params.indices_data)});
    TensorTf output =
        litert::tensor::Gather(input, indices, params.axis, /*batch_dims=*/0);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename ParamsRank, typename IndicesRank, typename Depth, typename T>
class GatherNd : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, ParamsRank::value> params_shape;
    std::array<Layout::Dim, IndicesRank::value> indices_shape;
    std::vector<int32_t> indices_data;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<GatherNd>;

  static constexpr absl::string_view Name() { return "GatherNd"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(2, 5);
    for (size_t i = 0; i < kParamsRank; ++i) {
      p.params_shape[i] = dim_dist(rng);
    }
    size_t num_rows = 1;
    std::uniform_int_distribution<int> row_dist(1, 3);
    for (size_t i = 0; i + 1 < kIndicesRank; ++i) {
      p.indices_shape[i] = row_dist(rng);
      num_rows *= static_cast<size_t>(p.indices_shape[i]);
    }
    p.indices_shape[kIndicesRank - 1] = kDepth;
    p.indices_data.resize(num_rows * kDepth);
    for (size_t r = 0; r < num_rows; ++r) {
      for (size_t d = 0; d < kDepth; ++d) {
        p.indices_data[r * kDepth + d] =
            static_cast<int32_t>(rng() % p.params_shape[d]);
      }
    }
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<GatherNd>(std::move(params), std::move(model));
  }

  GatherNd(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override { return ExactSpec(); }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return MakeSingleRandomInput<T>(params_.params_shape, device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_in, Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_out,
                            Traits::MakeReferenceOutputs(outputs));
    auto [params_buf] = ref_in;
    auto [output] = ref_out;
    std::array<int32_t, kParamsRank> p_dims;
    for (size_t i = 0; i < kParamsRank; ++i) {
      p_dims[i] = static_cast<int32_t>(params_.params_shape[i]);
    }
    std::array<int32_t, kIndicesRank> i_dims;
    for (size_t i = 0; i < kIndicesRank; ++i) {
      i_dims[i] = static_cast<int32_t>(params_.indices_shape[i]);
    }
    litert::internal::ReferenceGatherNd(
        params_buf.data.data(), p_dims.data(), static_cast<int>(kParamsRank),
        params_.indices_data.data(), i_dims.data(),
        static_cast<int>(kIndicesRank), output.data.data());
    return {};
  }

 private:
  static constexpr size_t kParamsRank = ParamsRank::value;
  static constexpr size_t kIndicesRank = IndicesRank::value;
  static constexpr size_t kDepth = Depth::value;
  static_assert(kIndicesRank >= 1 && kDepth >= 1 && kDepth <= kParamsRank);
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> p_dims(params.params_shape.begin(),
                                params.params_shape.end());
    std::vector<int> i_dims(params.indices_shape.begin(),
                            params.indices_shape.end());
    TensorTf input = litert::tensor::Create(
        "params", litert::tensor::ApiType<T>::value, p_dims);
    MaybeSetQuant<T>(input);
    TensorTf indices(
        {.type = litert::tensor::Type::kI32,
         .shape = i_dims,
         .buffer =
             litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
                 params.indices_data)});
    TensorTf output = litert::tensor::GatherNd(input, indices);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename Rank, typename T, typename ShrinkMask = SizeC<0>>
class StridedSlice : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, Rank::value> input_shape;
    std::array<int32_t, Rank::value> begin;
    std::array<int32_t, Rank::value> end;
    std::array<int32_t, Rank::value> strides;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<StridedSlice>;

  static constexpr absl::string_view Name() { return "StridedSlice"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> dim_dist(3, 6);
    for (size_t d = 0; d < kRank; ++d) {
      const int n = dim_dist(rng);
      p.input_shape[d] = n;
      if ((kShrinkMask & (1 << d)) != 0) {
        p.begin[d] = static_cast<int32_t>(rng() % n);
        p.end[d] = p.begin[d] + 1;
        p.strides[d] = 1;
      } else {
        p.begin[d] = static_cast<int32_t>(rng() % (n - 1));
        std::uniform_int_distribution<int32_t> end_dist(p.begin[d] + 1, n);
        p.end[d] = end_dist(rng);
        p.strides[d] = static_cast<int32_t>(1 + (rng() % 2));
      }
    }
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<StridedSlice>(std::move(params), std::move(model));
  }

  StridedSlice(Params params, LiteRtModelT::Ptr model)
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
    LITERT_RETURN_IF_ERROR(litert::internal::ReferenceStridedSlice(
        input.data.data(), dims.data(), static_cast<int>(kRank),
        params_.begin.data(), params_.end.data(), params_.strides.data(),
        /*begin_mask=*/0, /*end_mask=*/0, /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0, kShrinkMask, output.data.data()));
    return {};
  }

 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr int32_t kShrinkMask = ShrinkMask::value;
  static_assert(kShrinkMask == 0 || kRank >= 2);
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    std::vector<int32_t> begin(params.begin.begin(), params.begin.end());
    std::vector<int32_t> end(params.end.begin(), params.end.end());
    std::vector<int32_t> strides(params.strides.begin(), params.strides.end());

    std::vector<int32_t> out_dims;
    for (size_t d = 0; d < kRank; ++d) {
      if ((kShrinkMask & (1 << d)) != 0) {
        continue;
      }
      out_dims.push_back((end[d] - begin[d] + strides[d] - 1) / strides[d]);
    }

    std::vector<TensorDetails> op_inputs(4);
    op_inputs[0] =
        TensorDetails{dims, LiteRtElementType(kElementType), "input"};
    op_inputs[1] = TensorDetails{{static_cast<int32_t>(kRank)},
                                 kLiteRtElementTypeInt32,
                                 "begin",
                                 MakeOwningBufferRef(begin)};
    op_inputs[2] = TensorDetails{{static_cast<int32_t>(kRank)},
                                 kLiteRtElementTypeInt32,
                                 "end",
                                 MakeOwningBufferRef(end)};
    op_inputs[3] = TensorDetails{{static_cast<int32_t>(kRank)},
                                 kLiteRtElementTypeInt32,
                                 "strides",
                                 MakeOwningBufferRef(strides)};

    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{out_dims, LiteRtElementType(kElementType), "output"};
    MaybeSetTensorDetailsQuant<T>(op_inputs[0]);
    MaybeSetTensorDetailsQuant<T>(op_outputs[0]);

    tflite::StridedSliceOptionsT options;
    options.begin_mask = 0;
    options.end_mask = 0;
    options.ellipsis_mask = 0;
    options.new_axis_mask = 0;
    options.shrink_axis_mask = kShrinkMask;
    options.offset = false;
    return SingleOpModelWithOptions<kLiteRtOpCodeTflStridedSlice>(
        op_inputs, op_outputs, std::move(options));
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_INDEXING_H_
