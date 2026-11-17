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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SPLIT_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SPLIT_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/op_shape_inference_utils.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferSplit(const LiteRtOpT& op,
                               absl::Span<const Dims> input_shapes,
                               std::vector<Dims>& output_shapes) {
  constexpr int kSplitMinArgs = 2;
  constexpr int kAxisArgIndex = 0;
  constexpr int kInputArgIndex = 1;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  // Inputs: Axis, Input.
  if (input_shapes.size() < kSplitMinArgs || output_shapes.empty() ||
      op.Inputs().size() < kSplitMinArgs ||
      op.Inputs()[kAxisArgIndex] == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& axis_tensor = op.Input(kAxisArgIndex);
  const auto& input_shape = input_shapes[kInputArgIndex];

  const auto& opts = GetTflOptions(op);
  const auto* split_opts = opts.AsSplitOptions();
  int32_t num_splits = split_opts ? split_opts->num_splits : op.NumOutputs();
  if (num_splits <= 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (axis_tensor.Weights().Buffer().Size() >= kShapeTensorSize) {
    auto buf = axis_tensor.Weights().Buffer();
    int32_t axis = *reinterpret_cast<const int32_t*>(buf.Data());
    if ((axis < 0 && !SafeAdd(axis, input_shape.size(), axis)) || axis < 0 ||
        static_cast<size_t>(axis) >= input_shape.size()) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    Dims out_shape = input_shape;
    if (out_shape[axis] != -1) {
      out_shape[axis] /= num_splits;
    }

    output_shapes.resize(num_splits);
    for (size_t i = 0; i < output_shapes.size(); ++i)
      output_shapes[i] = out_shape;
    return kLiteRtStatusOk;
  }

  // Assume generic split if axis unknown.
  output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

template <typename T>
inline void ReferenceSplitV(const T* input_data, const int32_t* input_dims,
                            int input_rank, int32_t axis,
                            absl::Span<const int32_t> split_sizes,
                            absl::Span<T* const> outputs) {
  if (input_rank <= 0 || input_dims == nullptr ||
      outputs.size() < split_sizes.size()) {
    return;
  }
  AxisLayout layout;
  if (ComputeAxisLayout(absl::MakeConstSpan(input_dims, input_rank), axis,
                        layout) != kLiteRtStatusOk) {
    return;
  }
  size_t total_split = 0;
  for (size_t s = 0; s < split_sizes.size(); ++s) {
    if (split_sizes[s] < 0 ||
        !SafeAdd(total_split, static_cast<size_t>(split_sizes[s]),
                 total_split) ||
        total_split > layout.axis_size) {
      return;
    }
  }
  if (layout.outer_size == 0 || layout.inner_size == 0 || split_sizes.empty()) {
    return;
  }
  if (total_split > 0 && input_data == nullptr) {
    return;
  }
  for (size_t s = 0; s < split_sizes.size(); ++s) {
    if (split_sizes[s] > 0 && outputs[s] == nullptr) {
      return;
    }
  }
  for (size_t o = 0; o < layout.outer_size; ++o) {
    size_t in_offset = o * layout.axis_size * layout.inner_size;
    for (size_t s = 0; s < split_sizes.size(); ++s) {
      const size_t chunk =
          static_cast<size_t>(split_sizes[s]) * layout.inner_size;
      if (chunk > 0) {
        std::memcpy(outputs[s] + o * chunk, input_data + in_offset,
                    chunk * sizeof(T));
      }
      in_offset += chunk;
    }
  }
}

template <typename T>
inline void ReferenceSplit(const T* input_data, const int32_t* input_dims,
                           int input_rank, int32_t axis,
                           absl::Span<T* const> outputs) {
  if (input_rank <= 0 || input_dims == nullptr || outputs.empty()) {
    return;
  }
  AxisLayout layout;
  if (ComputeAxisLayout(absl::MakeConstSpan(input_dims, input_rank), axis,
                        layout) != kLiteRtStatusOk) {
    return;
  }
  if (layout.axis_size % outputs.size() != 0) {
    return;
  }
  const int32_t split_size =
      static_cast<int32_t>(layout.axis_size / outputs.size());
  std::vector<int32_t> split_sizes(outputs.size(), split_size);
  ReferenceSplitV<T>(input_data, input_dims, input_rank, axis, split_sizes,
                     outputs);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SPLIT_H_
