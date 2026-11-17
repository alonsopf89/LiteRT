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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SPATIAL_OPS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SPATIAL_OPS_H_

#include <array>
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
#include "litert/core/model/ops/spatial.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename OpCodeC, typename T, typename AlignCornersC,
          typename HalfPixelCentersC>
class ResizeOp : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, 4> input_shape;
    std::array<int32_t, 2> target_size;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<ResizeOp>;

  static constexpr absl::string_view Name() {
    return kBilinear ? "ResizeBilinear" : "ResizeNearestNeighbor";
  }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> batch_dist(1, 2);
    std::uniform_int_distribution<int> hw_dist(2, 5);
    std::uniform_int_distribution<int> out_hw_dist(2, 8);
    std::uniform_int_distribution<int> ch_dist(1, 4);
    p.input_shape = {batch_dist(rng), hw_dist(rng), hw_dist(rng), ch_dist(rng)};
    p.target_size = {out_hw_dist(rng), out_hw_dist(rng)};
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<ResizeOp>(std::move(params), std::move(model));
  }

  ResizeOp(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    if constexpr (kBilinear) {
      if constexpr (std::is_floating_point_v<T>) {
        spec.comparator_kind = ConformanceComparatorKind::kFloatElementwise;
        spec.absolute_tolerance = 1e-4;
        spec.relative_tolerance = 1e-4;
      } else {
        // TFLite's ResizeBilinearInteger uses 10-bit fixed-point coordinate
        // scaling (`((1 << 10) * in + out / 2) / out`), which can differ from
        // exact floating-point interpolation coordinates by up to ~0.004 over
        // 8 output pixels. Across a full-range int16 step (65535), that
        // coordinate delta accounts for up to ~262-320 buckets (~1-2 for int8).
        spec.comparator_kind = ConformanceComparatorKind::kQuantizedBucket;
        spec.bucket_tolerance = std::is_same_v<T, int16_t> ? 320 : 2;
      }
    } else {
      spec.comparator_kind = ConformanceComparatorKind::kExact;
    }
    return spec;
  }

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
    if constexpr (kBilinear) {
      litert::internal::ReferenceResizeBilinear(
          input.data.data(), params_.input_shape[0], params_.input_shape[1],
          params_.input_shape[2], params_.input_shape[3],
          params_.target_size[0], params_.target_size[1], kAlign, kHalf,
          output.data.data());
    } else {
      litert::internal::ReferenceResizeNearestNeighbor(
          input.data.data(), params_.input_shape[0], params_.input_shape[1],
          params_.input_shape[2], params_.input_shape[3],
          params_.target_size[0], params_.target_size[1], kAlign, kHalf,
          output.data.data());
    }
    return {};
  }

 private:
  static constexpr LiteRtOpCode kOpCode = OpCodeC::value;
  static constexpr bool kBilinear = kOpCode == kLiteRtOpCodeTflResizeBilinear;
  static constexpr bool kAlign = AlignCornersC::value;
  static constexpr bool kHalf = HalfPixelCentersC::value;
  static_assert(!(kAlign && kHalf));
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> in_dims(params.input_shape.begin(),
                                 params.input_shape.end());
    std::vector<int32_t> size_data(params.target_size.begin(),
                                   params.target_size.end());
    std::vector<int32_t> out_dims = {in_dims[0], size_data[0], size_data[1],
                                     in_dims[3]};
    std::vector<TensorDetails> op_inputs(2);
    op_inputs[0] =
        TensorDetails{in_dims, LiteRtElementType(kElementType), "input"};
    op_inputs[1] = TensorDetails{
        {2}, kLiteRtElementTypeInt32, "size", MakeOwningBufferRef(size_data)};
    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{out_dims, LiteRtElementType(kElementType), "output"};
    MaybeSetTensorDetailsQuant<T>(op_inputs[0]);
    MaybeSetTensorDetailsQuant<T>(op_outputs[0]);
    // TODO(b/586803505): Migrate to TensorAPI once per-op operator version
    // propagation is available in TensorAPI's TFLite export (which currently
    // emits version 1, whereas Resize* with half_pixel_centers requires
    // version 3).
    typename FbOpTypes<kOpCode>::OptionsT opts;
    opts.align_corners = kAlign;
    opts.half_pixel_centers = kHalf;
    return SingleOpModelWithOptions<kOpCode>(
        op_inputs, op_outputs, std::move(opts), OpVersion{kHalf ? 3 : 1});
  }

  Params params_;
};

template <typename OpCodeC, typename T>
class SpaceDepthOp : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, 4> input_shape;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<SpaceDepthOp>;

  static constexpr absl::string_view Name() {
    return kSpaceToDepth ? "SpaceToDepth" : "DepthToSpace";
  }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> batch_dist(1, 2);
    std::uniform_int_distribution<int> mul_dist(1, 3);
    if constexpr (kSpaceToDepth) {
      p.input_shape = {batch_dist(rng), kBlockSize * mul_dist(rng),
                       kBlockSize * mul_dist(rng), mul_dist(rng)};
    } else {
      p.input_shape = {batch_dist(rng), mul_dist(rng) + 1, mul_dist(rng) + 1,
                       kBlockSize * kBlockSize * mul_dist(rng)};
    }
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<SpaceDepthOp>(std::move(params), std::move(model));
  }

  SpaceDepthOp(Params params, LiteRtModelT::Ptr model)
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
    if constexpr (kSpaceToDepth) {
      litert::internal::ReferenceSpaceToDepth(
          input.data.data(), params_.input_shape[0], params_.input_shape[1],
          params_.input_shape[2], params_.input_shape[3], kBlockSize,
          output.data.data());
    } else {
      litert::internal::ReferenceDepthToSpace(
          input.data.data(), params_.input_shape[0], params_.input_shape[1],
          params_.input_shape[2], params_.input_shape[3], kBlockSize,
          output.data.data());
    }
    return {};
  }

 private:
  static constexpr LiteRtOpCode kOpCode = OpCodeC::value;
  static constexpr bool kSpaceToDepth = kOpCode == kLiteRtOpCodeTflSpaceToDepth;
  static constexpr int32_t kBlockSize = 2;

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, dims);
    MaybeSetQuant<T>(input);
    TensorTf output = kSpaceToDepth
                          ? litert::tensor::SpaceToDepth(input, kBlockSize)
                          : litert::tensor::DepthToSpace(input, kBlockSize);
    MaybeSetQuant<T>(output);
    output.SetName("output");
    return SaveTensorGraph({input}, {output});
  }

  Params params_;
};

template <typename OpCodeC, typename T>
class SpaceBatchNdOp : public TestGraph {
 public:
  struct Params {
    std::array<Layout::Dim, 4> input_shape;
    std::array<int32_t, 2> block_shape;
    std::array<int32_t, 4> paddings_or_crops;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<SpaceBatchNdOp>;

  static constexpr absl::string_view Name() {
    return kSpaceToBatch ? "SpaceToBatchNd" : "BatchToSpaceNd";
  }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    p.block_shape = {2, 2};
    std::uniform_int_distribution<int> batch_dist(1, 2);
    std::uniform_int_distribution<int> hw_dist(2, 4);
    std::uniform_int_distribution<int> ch_dist(1, 4);
    if constexpr (kSpaceToBatch) {
      const int h = hw_dist(rng);
      const int w = hw_dist(rng);
      const int ph0 = rng() % 2;
      const int ph1 = (2 - (h + ph0) % 2) % 2;
      const int pw0 = rng() % 2;
      const int pw1 = (2 - (w + pw0) % 2) % 2;
      p.input_shape = {batch_dist(rng), h, w, ch_dist(rng)};
      p.paddings_or_crops = {ph0, ph1, pw0, pw1};
    } else {
      const int h = hw_dist(rng);
      const int w = hw_dist(rng);
      p.input_shape = {batch_dist(rng) * 4, h, w, ch_dist(rng)};
      p.paddings_or_crops = {
          static_cast<int32_t>(rng() % 2), static_cast<int32_t>(rng() % 2),
          static_cast<int32_t>(rng() % 2), static_cast<int32_t>(rng() % 2)};
    }
    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<SpaceBatchNdOp>(std::move(params),
                                            std::move(model));
  }

  SpaceBatchNdOp(Params params, LiteRtModelT::Ptr model)
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
    if constexpr (kSpaceToBatch) {
      litert::internal::ReferenceSpaceToBatchNd(
          input.data.data(), params_.input_shape[0], params_.input_shape[1],
          params_.input_shape[2], params_.input_shape[3],
          params_.block_shape[0], params_.block_shape[1],
          params_.paddings_or_crops[0], params_.paddings_or_crops[1],
          params_.paddings_or_crops[2], params_.paddings_or_crops[3],
          static_cast<T>(0), output.data.data());
    } else {
      litert::internal::ReferenceBatchToSpaceNd(
          input.data.data(), params_.input_shape[0], params_.input_shape[1],
          params_.input_shape[2], params_.input_shape[3],
          params_.block_shape[0], params_.block_shape[1],
          params_.paddings_or_crops[0], params_.paddings_or_crops[1],
          params_.paddings_or_crops[2], params_.paddings_or_crops[3],
          output.data.data());
    }
    return {};
  }

 private:
  static constexpr LiteRtOpCode kOpCode = OpCodeC::value;
  static constexpr bool kSpaceToBatch =
      kOpCode == kLiteRtOpCodeTflSpaceToBatchNd;
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> in_dims(params.input_shape.begin(),
                                 params.input_shape.end());
    std::vector<int32_t> block_data(params.block_shape.begin(),
                                    params.block_shape.end());
    std::vector<int32_t> pad_data(params.paddings_or_crops.begin(),
                                  params.paddings_or_crops.end());
    std::vector<int32_t> out_dims(4);
    const int32_t area = block_data[0] * block_data[1];
    if constexpr (kSpaceToBatch) {
      out_dims = {in_dims[0] * area,
                  (in_dims[1] + pad_data[0] + pad_data[1]) / block_data[0],
                  (in_dims[2] + pad_data[2] + pad_data[3]) / block_data[1],
                  in_dims[3]};
    } else {
      out_dims = {in_dims[0] / area,
                  in_dims[1] * block_data[0] - pad_data[0] - pad_data[1],
                  in_dims[2] * block_data[1] - pad_data[2] - pad_data[3],
                  in_dims[3]};
    }
    std::vector<TensorDetails> op_inputs(3);
    op_inputs[0] =
        TensorDetails{in_dims, LiteRtElementType(kElementType), "input"};
    op_inputs[1] = TensorDetails{{2},
                                 kLiteRtElementTypeInt32,
                                 "block_shape",
                                 MakeOwningBufferRef(block_data)};
    op_inputs[2] = TensorDetails{{2, 2},
                                 kLiteRtElementTypeInt32,
                                 kSpaceToBatch ? "paddings" : "crops",
                                 MakeOwningBufferRef(pad_data)};
    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{out_dims, LiteRtElementType(kElementType), "output"};
    MaybeSetTensorDetailsQuant<T>(op_inputs[0]);
    MaybeSetTensorDetailsQuant<T>(op_outputs[0]);
    return SingleOpModel<kOpCode>(op_inputs, op_outputs);
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SPATIAL_OPS_H_
