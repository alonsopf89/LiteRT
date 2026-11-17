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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_CONV_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_CONV_H_

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
#include "litert/core/model/ops/convolution.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

template <typename T, typename PaddingC, typename StrideC, typename FaC>
class TransposeConv : public TestGraph {
 public:
  using FilterT = T;
  using BiasT = std::conditional_t<std::is_integral_v<T>, int32_t, T>;

  struct QuantSpec {
    AffineQuantSpec input{0.05f, -3};
    float weight_scale = 0.04f;
    AffineQuantSpec output{0.1f, -2};
  };

  struct Params {
    std::array<Layout::Dim, 4> input_shape;
    std::array<Layout::Dim, 4> filter_shape;
    std::array<Layout::Dim, 4> output_shape;
    std::vector<T> filter_data;
    std::vector<BiasT> bias_data;
    QuantSpec quant;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<TransposeConv>;

  static constexpr absl::string_view Name() { return "TransposeConv"; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    Params p;
    std::uniform_int_distribution<int> batch_dist(1, 2);
    std::uniform_int_distribution<int> hw_dist(2, 4);
    std::uniform_int_distribution<int> ch_dist(1, 4);
    const int b = batch_dist(rng);
    const int in_h = hw_dist(rng);
    const int in_w = hw_dist(rng);
    const int in_c = ch_dist(rng);
    const int out_c = ch_dist(rng);
    const int out_h = kPadding == tflite::Padding_SAME
                          ? in_h * kStride
                          : (in_h - 1) * kStride + kKernelSize;
    const int out_w = kPadding == tflite::Padding_SAME
                          ? in_w * kStride
                          : (in_w - 1) * kStride + kKernelSize;

    p.input_shape = {b, in_h, in_w, in_c};
    p.filter_shape = {out_c, kKernelSize, kKernelSize, in_c};
    p.output_shape = {b, out_h, out_w, out_c};
    const size_t num_weights =
        static_cast<size_t>(out_c) * kKernelSize * kKernelSize * in_c;
    p.filter_data.resize(num_weights);
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<int> w_dist(-12, 12);
      for (auto& w : p.filter_data) w = static_cast<T>(w_dist(rng));
      p.bias_data.resize(out_c);
      std::uniform_int_distribution<int32_t> b_dist(-40, 40);
      for (auto& bv : p.bias_data) bv = b_dist(rng);
      if constexpr (kFa != tflite::ActivationFunctionType_NONE) {
        p.quant.output.zero_point = -128;
      }
    } else {
      std::uniform_real_distribution<float> w_dist(-0.5f, 0.5f);
      for (auto& w : p.filter_data) w = w_dist(rng);
      p.bias_data.resize(out_c);
      for (auto& bv : p.bias_data) bv = w_dist(rng);
    }

    return Create(std::move(p));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<TransposeConv>(std::move(params), std::move(model));
  }

  TransposeConv(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    if constexpr (std::is_integral_v<T>) {
      spec.comparator_kind = ConformanceComparatorKind::kQuantizedBucket;
      spec.bucket_tolerance = 1;
    } else {
      spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
      spec.absolute_tolerance = 1e-4;
      spec.relative_tolerance = 1e-4;
      spec.accumulation_depth =
          params_.input_shape[3] * kKernelSize * kKernelSize;
    }
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    RandomTensorDataBuilder b = data_builder;
    if constexpr (std::is_floating_point_v<T>) {
      if (!b.IsFloatDummy()) {
        b.SetFloatRange(-2.0f, 2.0f);
      }
    } else {
      if (!b.IsIntDummy()) {
        b.SetIntRange(-20, 20);
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
    const int batch = params_.input_shape[0];
    const int in_h = params_.input_shape[1];
    const int in_w = params_.input_shape[2];
    const int in_c = params_.input_shape[3];
    const int out_h = params_.output_shape[1];
    const int out_w = params_.output_shape[2];
    const int out_c = params_.output_shape[3];
    const int pad_t =
        kPadding == tflite::Padding_SAME
            ? std::max(0, ((in_h - 1) * kStride + kKernelSize - out_h) / 2)
            : 0;
    const int pad_l =
        kPadding == tflite::Padding_SAME
            ? std::max(0, ((in_w - 1) * kStride + kKernelSize - out_w) / 2)
            : 0;

    if constexpr (std::is_integral_v<T>) {
      litert::internal::ReferenceTransposeConvQuantized<T, FilterT, BiasT>(
          input.data.data(), params_.filter_data.data(),
          params_.bias_data.data(), output.data.data(), batch, in_h, in_w, in_c,
          out_h, out_w, out_c, kKernelSize, kKernelSize, kStride, kStride,
          pad_t, pad_l, kFa, params_.quant.input, params_.quant.weight_scale,
          params_.quant.output);
      return {};
    } else {
      litert::internal::ReferenceTransposeConv(
          input.data.data(), params_.filter_data.data(),
          params_.bias_data.data(), output.data.data(), batch, in_h, in_w, in_c,
          out_h, out_w, out_c, kKernelSize, kKernelSize, kStride, kStride,
          pad_t, pad_l, kFa);
      return {};
    }
  }

 private:
  static constexpr tflite::Padding kPadding = PaddingC::value;
  static constexpr int32_t kStride = StrideC::value;
  static constexpr tflite::ActivationFunctionType kFa = FaC::value;
  static constexpr int32_t kKernelSize = 3;
  static_assert(kStride >= 1 && kStride <= kKernelSize);
  static constexpr ElementType kElementType = GetElementType<T>();

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> out_shape(params.output_shape.begin(),
                                   params.output_shape.end());
    std::vector<int32_t> filter_dims(params.filter_shape.begin(),
                                     params.filter_shape.end());
    std::vector<int32_t> in_dims(params.input_shape.begin(),
                                 params.input_shape.end());
    std::vector<int32_t> bias_dims = {out_shape[3]};

    std::vector<TensorDetails> op_inputs(4);
    op_inputs[0] = TensorDetails{{4},
                                 kLiteRtElementTypeInt32,
                                 "output_shape",
                                 MakeOwningBufferRef(out_shape)};
    op_inputs[1] =
        TensorDetails{filter_dims, LiteRtElementType(kElementType), "weights",
                      MakeOwningBufferRef(params.filter_data)};
    op_inputs[2] =
        TensorDetails{in_dims, LiteRtElementType(kElementType), "input"};
    op_inputs[3] =
        TensorDetails{bias_dims,
                      std::is_integral_v<T> ? kLiteRtElementTypeInt32
                                            : kLiteRtElementTypeFloat32,
                      "bias", MakeOwningBufferRef(params.bias_data)};
    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{out_shape, LiteRtElementType(kElementType), "output"};
    if constexpr (std::is_integral_v<T>) {
      op_inputs[1].quantization =
          MakePerTensorQuantization(params.quant.weight_scale, 0);
      op_inputs[2].quantization = MakePerTensorQuantization(
          params.quant.input.scale, params.quant.input.zero_point);
      op_inputs[3].quantization = MakePerTensorQuantization(
          params.quant.input.scale * params.quant.weight_scale, 0);
      op_outputs[0].quantization = MakePerTensorQuantization(
          params.quant.output.scale, params.quant.output.zero_point);
    }

    // TensorAPI's TransposeConv lowers bias via a separate Add op instead of a
    // single fused 4-input TRANSPOSE_CONV op (version 4 with fused activation).
    // Retain the typed low-level builder so this generator exercises the
    // single fused operator directly.
    tflite::TransposeConvOptionsT opts;
    opts.padding = kPadding;
    opts.stride_h = kStride;
    opts.stride_w = kStride;
    opts.fused_activation_function = kFa;
    return SingleOpModelWithOptions<kLiteRtOpCodeTflTransposeConv>(
        op_inputs, op_outputs, std::move(opts), OpVersion{4});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_CONV_H_
