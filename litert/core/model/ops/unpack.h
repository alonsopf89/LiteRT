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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_UNPACK_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_UNPACK_H_

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

inline LiteRtStatus InferUnpack(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  if (input_shapes.empty()) return kLiteRtStatusErrorShapeInferenceFailed;

  constexpr int kFirstInputIndex = 0;
  const auto& s0 = input_shapes[kFirstInputIndex];

  const auto& opts = GetTflOptions(op);
  const auto* unpack_opts = opts.AsUnpackOptions();
  constexpr int kDefaultAxis = 0;
  int32_t axis = unpack_opts ? unpack_opts->axis : kDefaultAxis;
  int32_t num = unpack_opts ? unpack_opts->num : op.NumOutputs();
  if (num <= 0) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = s0;
  if ((axis < 0 && !SafeAdd(axis, out_shape.size(), axis)) || axis < 0 ||
      static_cast<size_t>(axis) >= out_shape.size())
    return kLiteRtStatusErrorShapeInferenceFailed;

  out_shape.erase(out_shape.begin() + axis);

  if (output_shapes.size() != static_cast<size_t>(num)) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes[i] = out_shape;
  }
  return kLiteRtStatusOk;
}

template <typename T>
inline void ReferenceUnpack(const T* input_data, const int32_t* input_dims,
                            int input_rank, int32_t axis,
                            absl::Span<T* const> outputs) {
  if (input_rank <= 0) return;
  AxisLayout layout;
  if (ComputeAxisLayout(absl::MakeConstSpan(input_dims, input_rank), axis,
                        layout) != kLiteRtStatusOk) {
    return;
  }
  for (size_t o = 0; o < layout.outer_size; ++o) {
    for (size_t n = 0; n < layout.axis_size; ++n) {
      std::memcpy(outputs[n] + o * layout.inner_size,
                  input_data + (o * layout.axis_size + n) * layout.inner_size,
                  layout.inner_size * sizeof(T));
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_UNPACK_H_
