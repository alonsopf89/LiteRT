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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/util/tensor_type_util.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

// Returns true if the product of all positive dimensions (> 1) in `shape` fits
// in a signed 32-bit integer. Dynamic (-1) and zero (0) dimensions are skipped
// so that static sub-products are still validated even when another dimension
// is dynamic or zero.
inline bool CheckShapePositiveProductFitsInInt32(
    absl::Span<const int32_t> shape) {
  int32_t product = 1;
  for (int32_t dim : shape) {
    if (dim > 1 && !SafeMultiply(product, dim, product)) {
      return false;
    }
  }
  return true;
}

inline bool ComputeEffectiveFilterSize(int32_t filter_size,
                                       int32_t dilation_rate,
                                       int32_t& effective_filter_size) {
  if (filter_size <= 0 || dilation_rate <= 0) {
    return false;
  }
  int32_t dilated_filter_size = 0;
  return SafeSubtract(filter_size, 1, dilated_filter_size) &&
         SafeMultiply(dilated_filter_size, dilation_rate,
                      dilated_filter_size) &&
         SafeAdd(dilated_filter_size, 1, effective_filter_size);
}

// Computes the output size of a convolution or pooling operation.
inline int32_t ComputeOutputSize(tflite::Padding padding, int32_t image_size,
                                 int32_t filter_size, int32_t stride,
                                 int32_t dilation_rate = 1) {
  if (image_size == -1) return -1;
  if (image_size < 0 || filter_size <= 0 || stride <= 0 || dilation_rate <= 0) {
    return -1;
  }

  int32_t effective_filter_size = 0;
  if (!ComputeEffectiveFilterSize(filter_size, dilation_rate,
                                  effective_filter_size)) {
    return -1;
  }

  int32_t value = 0;
  switch (padding) {
    case tflite::Padding_SAME: {
      int32_t stride_minus_one = 0;
      if (!SafeSubtract(stride, 1, stride_minus_one) ||
          !SafeAdd(image_size, stride_minus_one, value) ||
          !SafeDivide(value, stride, value)) {
        return -1;
      }
      break;
    }
    case tflite::Padding_VALID:
      if (!SafeSubtract(image_size, effective_filter_size, value) ||
          !SafeAdd(value, stride, value) || !SafeDivide(value, stride, value)) {
        return -1;
      }
      break;
    default:
      return -1;
  }

  return value >= 0 ? value : -1;
}

// Per-tensor affine quantization parameters shared by core reference kernels
// and test generators.
struct AffineQuantSpec {
  float scale = 1.0f;
  int64_t zero_point = 0;
};

template <typename T>
inline float DequantizeValue(T val, float scale, int64_t zero_point) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<float>(val);
  } else {
    return (static_cast<float>(val) - static_cast<float>(zero_point)) * scale;
  }
}

template <typename T>
inline float DequantizeValue(T val, const AffineQuantSpec& spec) {
  return DequantizeValue<T>(val, spec.scale, spec.zero_point);
}

template <typename T>
inline T QuantizeValue(float val, float scale, int64_t zero_point) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(val);
  } else {
    const int64_t q = std::llround(val / scale) + zero_point;
    const int64_t lo = static_cast<int64_t>(std::numeric_limits<T>::min());
    const int64_t hi = static_cast<int64_t>(std::numeric_limits<T>::max());
    return static_cast<T>(std::clamp(q, lo, hi));
  }
}

template <typename T>
inline T QuantizeValue(float val, const AffineQuantSpec& spec) {
  return QuantizeValue<T>(val, spec.scale, spec.zero_point);
}

template <typename T>
inline T QuantizeValueFromDouble(double val, const AffineQuantSpec& spec) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(val);
  } else {
    const int64_t q =
        std::llround(val / static_cast<double>(spec.scale)) + spec.zero_point;
    const int64_t lo = static_cast<int64_t>(std::numeric_limits<T>::min());
    const int64_t hi = static_cast<int64_t>(std::numeric_limits<T>::max());
    return static_cast<T>(std::clamp(q, lo, hi));
  }
}

struct AxisLayout {
  size_t outer_size = 1;
  size_t axis_size = 1;
  size_t inner_size = 1;
};

// Computes checked outer/axis/inner element counts around `axis` in
// [-rank, rank - 1] for a tensor with shape `dims`.
inline LiteRtStatus ComputeAxisLayout(absl::Span<const int32_t> dims,
                                      int32_t axis, AxisLayout& out) {
  if (dims.empty() ||
      dims.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const int32_t rank = static_cast<int32_t>(dims.size());
  if (axis < 0 && !SafeAdd(axis, rank, axis)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (axis < 0 || axis >= rank) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto total_elems = GetNumElements(dims);
  if (!total_elems) {
    return total_elems.Error().Status();
  }
  if (*total_elems == 0) {
    out = {0, static_cast<size_t>(dims[axis]), 0};
    return kLiteRtStatusOk;
  }
  size_t outer = 1;
  for (int32_t i = 0; i < axis; ++i) {
    outer *= static_cast<size_t>(dims[i]);
  }
  size_t inner = 1;
  for (int32_t i = axis + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(dims[i]);
  }
  out = {outer, static_cast<size_t>(dims[axis]), inner};
  return kLiteRtStatusOk;
}

// Computes checked outer/inner element counts when inserting a new axis at
// `axis` in [-(rank + 1), rank] (e.g., Pack).
inline LiteRtStatus ComputeInsertAxisLayout(absl::Span<const int32_t> dims,
                                            int32_t axis, size_t& outer_size,
                                            size_t& inner_size) {
  const int32_t rank = static_cast<int32_t>(dims.size());
  if (axis < 0) {
    axis += rank + 1;
  }
  if (axis < 0 || axis > rank) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto total_elems = GetNumElements(dims);
  if (!total_elems) {
    return total_elems.Error().Status();
  }
  if (*total_elems == 0) {
    outer_size = 0;
    inner_size = 0;
    return kLiteRtStatusOk;
  }
  outer_size = 1;
  for (int32_t i = 0; i < axis; ++i) {
    outer_size *= static_cast<size_t>(dims[i]);
  }
  inner_size = 1;
  for (int32_t i = axis; i < rank; ++i) {
    inner_size *= static_cast<size_t>(dims[i]);
  }
  return kLiteRtStatusOk;
}

// Validates broadcast compatibility and computes checked contiguous input
// strides aligned to `output_dims`, along with checked `total_out`.
// Rejects input_rank > output_rank, negative/dynamic dimensions, incompatible
// non-singleton dimensions, and element-count overflow. Empty tensors are
// fully validated before returning `total_out = 0` with zeroed strides so
// trailing dimensions cannot overflow stride products.
inline LiteRtStatus ComputeBroadcastStridesChecked(
    absl::Span<const int32_t> input_dims, absl::Span<const int32_t> output_dims,
    std::vector<size_t>& strides, size_t& total_out) {
  if (input_dims.size() > output_dims.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto in_elems = GetNumElements(input_dims);
  if (!in_elems) {
    return in_elems.Error().Status();
  }
  auto out_elems = GetNumElements(output_dims);
  if (!out_elems) {
    return out_elems.Error().Status();
  }
  const size_t rank_diff = output_dims.size() - input_dims.size();
  for (size_t d = 0; d < input_dims.size(); ++d) {
    const int32_t in_dim = input_dims[d];
    const int32_t out_dim = output_dims[d + rank_diff];
    if (in_dim != 1 && in_dim != out_dim) {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  total_out = *out_elems;
  strides.assign(output_dims.size(), 0);
  if (total_out == 0 || output_dims.empty()) {
    return kLiteRtStatusOk;
  }
  // Because `total_out > 0` and every aligned input dimension is either 1 or
  // `out_dim`, every input dimension is strictly positive and `*in_elems`
  // bounds every suffix product computed below.
  size_t current_stride = 1;
  for (size_t i = input_dims.size(); i > 0; --i) {
    const size_t d = i - 1;
    const size_t out_d = d + rank_diff;
    strides[out_d] = (input_dims[d] == 1) ? 0 : current_stride;
    current_stride *= static_cast<size_t>(input_dims[d]);
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_
