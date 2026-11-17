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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferDepthToSpace(const LiteRtOpT& op,
                                      absl::Span<const Dims> input_shapes,
                                      std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kHeightDimIndex = 1;
  constexpr int kWidthDimIndex = 2;
  constexpr int kDepthDimIndex = 3;
  constexpr int kSpatialRank = 4;

  if (input_shapes.empty() || output_shapes.size() != 1) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  if (input_shape.size() != kSpatialRank) {
    return kLiteRtStatusErrorShapeInferenceFailed;  // NHWC
  }
  for (int32_t dim : input_shape) {
    if (dim < -1) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  const auto& opts = GetTflOptions(op);
  const auto* d2s_opts = opts.AsDepthToSpaceOptions();
  int32_t block_size = d2s_opts ? d2s_opts->block_size : 0;
  int32_t block_area = 0;
  if (block_size <= 0 || !SafeMultiply(block_size, block_size, block_area)) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = input_shape;
  int32_t height = input_shape[kHeightDimIndex];
  int32_t width = input_shape[kWidthDimIndex];
  int32_t depth = input_shape[kDepthDimIndex];

  if (depth != -1) {
    if (depth % block_area != 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    out_shape[kDepthDimIndex] = depth / block_area;
  }
  if (height != -1 &&
      !SafeMultiply(height, block_size, out_shape[kHeightDimIndex])) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if (width != -1 &&
      !SafeMultiply(width, block_size, out_shape[kWidthDimIndex])) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferSpaceToDepth(const LiteRtOpT& op,
                                      absl::Span<const Dims> input_shapes,
                                      std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kHeightDimIndex = 1;
  constexpr int kWidthDimIndex = 2;
  constexpr int kDepthDimIndex = 3;
  constexpr int kSpatialRank = 4;

  if (input_shapes.empty() || output_shapes.size() != 1) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  if (input_shape.size() != kSpatialRank) {
    return kLiteRtStatusErrorShapeInferenceFailed;  // NHWC
  }
  for (int32_t dim : input_shape) {
    if (dim < -1) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  const auto& opts = GetTflOptions(op);
  const auto* s2d_opts = opts.AsSpaceToDepthOptions();
  int32_t block_size = s2d_opts ? s2d_opts->block_size : 0;
  int32_t block_area = 0;
  if (block_size <= 0 || !SafeMultiply(block_size, block_size, block_area)) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = input_shape;
  int32_t height = input_shape[kHeightDimIndex];
  int32_t width = input_shape[kWidthDimIndex];
  int32_t depth = input_shape[kDepthDimIndex];

  if (depth != -1 &&
      !SafeMultiply(depth, block_area, out_shape[kDepthDimIndex])) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if (height != -1) {
    if (height % block_size != 0) return kLiteRtStatusErrorShapeInferenceFailed;
    out_shape[kHeightDimIndex] = height / block_size;
  }
  if (width != -1) {
    if (width % block_size != 0) return kLiteRtStatusErrorShapeInferenceFailed;
    out_shape[kWidthDimIndex] = width / block_size;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferResizeOp(const LiteRtOpT& op,
                                  absl::Span<const Dims> input_shapes,
                                  std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kSizeArgIndex = 1;
  constexpr int kResizeMinArgs = 2;
  constexpr int kSpatialRank = 4;
  constexpr int kResizeSizeTensorSize = 2 * sizeof(int32_t);

  // Inputs: Input, Size.
  if (input_shapes.size() < kResizeMinArgs || output_shapes.size() != 1 ||
      op.Inputs().size() < kResizeMinArgs ||
      op.Inputs()[kSizeArgIndex] == nullptr) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& size_tensor = op.Input(kSizeArgIndex);

  if (input_shape.size() != kSpatialRank)
    return kLiteRtStatusErrorShapeInferenceFailed;  // Expect NHWC

  Dims out_shape = input_shape;

  if (size_tensor.Weights().Buffer().Size() == kResizeSizeTensorSize) {
    auto buf = size_tensor.Weights().Buffer();
    const int32_t* size_data = reinterpret_cast<const int32_t*>(buf.Data());
    if (size_data[0] <= 0 || size_data[1] <= 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }

    out_shape[1] = size_data[0];  // New Height
    out_shape[2] = size_data[1];  // New Width
  } else {
    // Dynamic size
    out_shape[1] = -1;
    out_shape[2] = -1;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferResizeBilinear(const LiteRtOpT& op,
                                        absl::Span<const Dims> input_shapes,
                                        std::vector<Dims>& output_shapes) {
  return InferResizeOp(op, input_shapes, output_shapes);
}

inline LiteRtStatus InferResizeNearestNeighbor(
    const LiteRtOpT& op, absl::Span<const Dims> input_shapes,
    std::vector<Dims>& output_shapes) {
  return InferResizeOp(op, input_shapes, output_shapes);
}

inline LiteRtStatus InferSpaceToBatchNd(const ShapeInferenceContext& ctx,
                                        InferenceResult& result) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kBlockShapeArgIndex = 1;
  constexpr size_t kPaddingsArgIndex = 2;

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(kInputArgIndex);
  Dims block_shape_shape = ctx.GetInputShape(kBlockShapeArgIndex);
  Dims paddings_shape = ctx.GetInputShape(kPaddingsArgIndex);

  if (input_shape.size() < 3) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const size_t spatial_dims_num = input_shape.size() - 2;

  if (!block_shape_shape.empty()) {
    if (block_shape_shape.size() != 1 ||
        (block_shape_shape[0] != -1 &&
         block_shape_shape[0] != static_cast<int32_t>(spatial_dims_num))) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }
  if (!paddings_shape.empty()) {
    if (paddings_shape.size() != 2 ||
        (paddings_shape[0] != -1 &&
         paddings_shape[0] != static_cast<int32_t>(spatial_dims_num)) ||
        (paddings_shape[1] != -1 && paddings_shape[1] != 2)) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  auto block_shape_buf = ctx.GetInputData(kBlockShapeArgIndex);
  auto paddings_buf = ctx.GetInputData(kPaddingsArgIndex);

  if (!block_shape_buf.empty() &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int32_t) &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!paddings_buf.empty() &&
      paddings_buf.size() != spatial_dims_num * 2 * sizeof(int32_t) &&
      paddings_buf.size() != spatial_dims_num * 2 * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (block_shape_buf.empty() || paddings_buf.empty()) {
    Dims out_shape = input_shape;
    out_shape[0] = -1;
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      out_shape[i + 1] = -1;
    }
    result.output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  std::vector<int64_t> block_shape_vals(spatial_dims_num);
  if (block_shape_buf.size() == spatial_dims_num * sizeof(int32_t)) {
    const uint8_t* data = block_shape_buf.data();
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      block_shape_vals[i] = val;
    }
  } else if (block_shape_buf.size() == spatial_dims_num * sizeof(int64_t)) {
    const uint8_t* data = block_shape_buf.data();
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      int64_t val;
      std::memcpy(&val, data + i * sizeof(int64_t), sizeof(int64_t));
      block_shape_vals[i] = val;
    }
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::vector<int64_t> paddings_vals(spatial_dims_num * 2);
  if (paddings_buf.size() == spatial_dims_num * 2 * sizeof(int32_t)) {
    const uint8_t* data = paddings_buf.data();
    for (size_t i = 0; i < spatial_dims_num * 2; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      paddings_vals[i] = val;
    }
  } else if (paddings_buf.size() == spatial_dims_num * 2 * sizeof(int64_t)) {
    const uint8_t* data = paddings_buf.data();
    for (size_t i = 0; i < spatial_dims_num * 2; ++i) {
      int64_t val;
      std::memcpy(&val, data + i * sizeof(int64_t), sizeof(int64_t));
      paddings_vals[i] = val;
    }
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape = input_shape;
  for (int32_t dim : input_shape) {
    if (dim < -1) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }
  int64_t output_batch_size = input_shape[0];
  for (size_t dim = 0; dim < spatial_dims_num; ++dim) {
    int64_t bs = block_shape_vals[dim];
    if (bs <= 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (output_batch_size != -1 &&
        !SafeMultiply(output_batch_size, bs, output_batch_size)) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    int64_t pad_before = paddings_vals[dim * 2];
    int64_t pad_after = paddings_vals[dim * 2 + 1];
    if (pad_before < 0 || pad_after < 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (input_shape[dim + 1] != -1) {
      int64_t final_dim_size = 0;
      if (!SafeAdd(static_cast<int64_t>(input_shape[dim + 1]), pad_before,
                   final_dim_size) ||
          !SafeAdd(final_dim_size, pad_after, final_dim_size) ||
          final_dim_size % bs != 0) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      int64_t scaled_dim = final_dim_size / bs;
      if (scaled_dim > std::numeric_limits<int32_t>::max()) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      out_shape[dim + 1] = static_cast<int32_t>(scaled_dim);
    } else {
      out_shape[dim + 1] = -1;
    }
  }
  if (output_batch_size != -1 &&
      output_batch_size > std::numeric_limits<int32_t>::max()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  out_shape[0] =
      (output_batch_size == -1) ? -1 : static_cast<int32_t>(output_batch_size);
  result.output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferBatchToSpaceNd(const ShapeInferenceContext& ctx,
                                        InferenceResult& result) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kBlockShapeArgIndex = 1;
  constexpr size_t kCropsArgIndex = 2;

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(kInputArgIndex);
  Dims block_shape_shape = ctx.GetInputShape(kBlockShapeArgIndex);
  Dims crops_shape = ctx.GetInputShape(kCropsArgIndex);

  if (input_shape.size() < 3) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const size_t spatial_dims_num = input_shape.size() - 2;

  if (!block_shape_shape.empty()) {
    if (block_shape_shape.size() != 1 ||
        (block_shape_shape[0] != -1 &&
         block_shape_shape[0] != static_cast<int32_t>(spatial_dims_num))) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }
  if (!crops_shape.empty()) {
    if (crops_shape.size() != 2 ||
        (crops_shape[0] != -1 &&
         crops_shape[0] != static_cast<int32_t>(spatial_dims_num)) ||
        (crops_shape[1] != -1 && crops_shape[1] != 2)) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  auto block_shape_buf = ctx.GetInputData(kBlockShapeArgIndex);
  auto crops_buf = ctx.GetInputData(kCropsArgIndex);

  if (!block_shape_buf.empty() &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int32_t) &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!crops_buf.empty() &&
      crops_buf.size() != spatial_dims_num * 2 * sizeof(int32_t) &&
      crops_buf.size() != spatial_dims_num * 2 * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (block_shape_buf.empty() || crops_buf.empty()) {
    Dims out_shape = input_shape;
    out_shape[0] = -1;
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      out_shape[i + 1] = -1;
    }
    result.output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  std::vector<int64_t> block_shape_vals(spatial_dims_num);
  if (block_shape_buf.size() == spatial_dims_num * sizeof(int32_t)) {
    const uint8_t* data = block_shape_buf.data();
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      block_shape_vals[i] = val;
    }
  } else if (block_shape_buf.size() == spatial_dims_num * sizeof(int64_t)) {
    const uint8_t* data = block_shape_buf.data();
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      int64_t val;
      std::memcpy(&val, data + i * sizeof(int64_t), sizeof(int64_t));
      block_shape_vals[i] = val;
    }
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::vector<int64_t> crops_vals(spatial_dims_num * 2);
  if (crops_buf.size() == spatial_dims_num * 2 * sizeof(int32_t)) {
    const uint8_t* data = crops_buf.data();
    for (size_t i = 0; i < spatial_dims_num * 2; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      crops_vals[i] = val;
    }
  } else if (crops_buf.size() == spatial_dims_num * 2 * sizeof(int64_t)) {
    const uint8_t* data = crops_buf.data();
    for (size_t i = 0; i < spatial_dims_num * 2; ++i) {
      int64_t val;
      std::memcpy(&val, data + i * sizeof(int64_t), sizeof(int64_t));
      crops_vals[i] = val;
    }
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape = input_shape;
  for (int32_t dim : input_shape) {
    if (dim < -1) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }
  int64_t output_batch_size = input_shape[0];
  for (size_t dim = 0; dim < spatial_dims_num; ++dim) {
    int64_t bs = block_shape_vals[dim];
    if (bs <= 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (output_batch_size != -1) {
      if (output_batch_size % bs != 0) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      output_batch_size /= bs;
    }
    int64_t crop_before = crops_vals[dim * 2];
    int64_t crop_after = crops_vals[dim * 2 + 1];
    if (crop_before < 0 || crop_after < 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (input_shape[dim + 1] != -1) {
      int64_t final_dim_size = 0;
      if (!SafeMultiply(static_cast<int64_t>(input_shape[dim + 1]), bs,
                        final_dim_size) ||
          !SafeSubtract(final_dim_size, crop_before, final_dim_size) ||
          !SafeSubtract(final_dim_size, crop_after, final_dim_size) ||
          final_dim_size < 0 ||
          final_dim_size > std::numeric_limits<int32_t>::max()) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      out_shape[dim + 1] = static_cast<int32_t>(final_dim_size);
    } else {
      out_shape[dim + 1] = -1;
    }
  }
  if (output_batch_size != -1 &&
      output_batch_size > std::numeric_limits<int32_t>::max()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  out_shape[0] =
      (output_batch_size == -1) ? -1 : static_cast<int32_t>(output_batch_size);
  result.output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

template <typename T>
inline void ReferenceResizeBilinear(const T* input_data, int batch, int in_h,
                                    int in_w, int channels, int out_h,
                                    int out_w, bool align_corners,
                                    bool half_pixel_centers, T* output_data) {
  float height_scale = static_cast<float>(in_h) / out_h;
  float width_scale = static_cast<float>(in_w) / out_w;
  if (align_corners && out_h > 1) {
    height_scale = static_cast<float>(in_h - 1) / (out_h - 1);
  }
  if (align_corners && out_w > 1) {
    width_scale = static_cast<float>(in_w - 1) / (out_w - 1);
  }
  auto compute_interp = [](int idx, float scale, bool half_pixel, int in_size,
                           float& scaled, int32_t& lo, int32_t& hi) {
    scaled = half_pixel ? ((idx + 0.5f) * scale - 0.5f) : (idx * scale);
    lo = std::max(static_cast<int32_t>(std::floor(scaled)), int32_t{0});
    hi = std::min(static_cast<int32_t>(std::ceil(scaled)), in_size - 1);
  };
  for (int b = 0; b < batch; ++b) {
    for (int y = 0; y < out_h; ++y) {
      float in_y;
      int32_t y0, y1;
      compute_interp(y, height_scale, half_pixel_centers, in_h, in_y, y0, y1);
      const double wy1 = static_cast<double>(in_y) - static_cast<double>(y0);
      const double wy0 = 1.0 - wy1;
      for (int x = 0; x < out_w; ++x) {
        float in_x;
        int32_t x0, x1;
        compute_interp(x, width_scale, half_pixel_centers, in_w, in_x, x0, x1);
        const double wx1 = static_cast<double>(in_x) - static_cast<double>(x0);
        const double wx0 = 1.0 - wx1;
        for (int c = 0; c < channels; ++c) {
          const double v00 = static_cast<double>(
              input_data[((b * in_h + y0) * in_w + x0) * channels + c]);
          const double v10 = static_cast<double>(
              input_data[((b * in_h + y1) * in_w + x0) * channels + c]);
          const double v01 = static_cast<double>(
              input_data[((b * in_h + y0) * in_w + x1) * channels + c]);
          const double v11 = static_cast<double>(
              input_data[((b * in_h + y1) * in_w + x1) * channels + c]);
          const double val = v00 * wy0 * wx0 + v10 * wy1 * wx0 +
                             v01 * wy0 * wx1 + v11 * wy1 * wx1;
          T out_val;
          if constexpr (std::numeric_limits<T>::is_integer) {
            // Round half away from zero symmetrically for signed and unsigned
            // integer types, then saturate to [min<T>, max<T>].
            const int64_t rounded = std::llround(val);
            const int64_t lo =
                static_cast<int64_t>(std::numeric_limits<T>::min());
            const int64_t hi =
                static_cast<int64_t>(std::numeric_limits<T>::max());
            out_val = static_cast<T>(std::clamp(rounded, lo, hi));
          } else {
            out_val = static_cast<T>(val);
          }
          output_data[((b * out_h + y) * out_w + x) * channels + c] = out_val;
        }
      }
    }
  }
}

template <typename T>
inline void ReferenceResizeNearestNeighbor(
    const T* input_data, int batch, int in_h, int in_w, int channels, int out_h,
    int out_w, bool align_corners, bool half_pixel_centers, T* output_data) {
  auto nearest = [](int idx, int in_size, int out_size, bool align,
                    bool half_pixel) -> int32_t {
    const float scale = (align && out_size > 1)
                            ? (in_size - 1) / static_cast<float>(out_size - 1)
                            : in_size / static_cast<float>(out_size);
    const float offset = half_pixel ? 0.5f : 0.0f;
    int32_t out = std::min(
        align ? static_cast<int32_t>(std::round((idx + offset) * scale))
              : static_cast<int32_t>(std::floor((idx + offset) * scale)),
        in_size - 1);
    return half_pixel ? std::max(int32_t{0}, out) : out;
  };
  for (int b = 0; b < batch; ++b) {
    for (int y = 0; y < out_h; ++y) {
      const int32_t in_y =
          nearest(y, in_h, out_h, align_corners, half_pixel_centers);
      for (int x = 0; x < out_w; ++x) {
        const int32_t in_x =
            nearest(x, in_w, out_w, align_corners, half_pixel_centers);
        std::memcpy(output_data + ((b * out_h + y) * out_w + x) * channels,
                    input_data + ((b * in_h + in_y) * in_w + in_x) * channels,
                    static_cast<size_t>(channels) * sizeof(T));
      }
    }
  }
}

template <typename T>
inline void ReferenceSpaceToDepth(const T* input_data, int batch, int in_h,
                                  int in_w, int in_c, int block_size,
                                  T* output_data) {
  const int out_h = in_h / block_size;
  const int out_w = in_w / block_size;
  const int out_c = in_c * block_size * block_size;
  for (int b = 0; b < batch; ++b) {
    for (int ih = 0; ih < in_h; ++ih) {
      const int oh = ih / block_size;
      const int by = ih % block_size;
      for (int iw = 0; iw < in_w; ++iw) {
        const int ow = iw / block_size;
        const int bx = iw % block_size;
        const int oc_base = (by * block_size + bx) * in_c;
        std::memcpy(
            output_data + ((b * out_h + oh) * out_w + ow) * out_c + oc_base,
            input_data + ((b * in_h + ih) * in_w + iw) * in_c,
            static_cast<size_t>(in_c) * sizeof(T));
      }
    }
  }
}

template <typename T>
inline void ReferenceDepthToSpace(const T* input_data, int batch, int in_h,
                                  int in_w, int in_c, int block_size,
                                  T* output_data) {
  const int out_h = in_h * block_size;
  const int out_w = in_w * block_size;
  const int out_c = in_c / (block_size * block_size);
  for (int b = 0; b < batch; ++b) {
    for (int ih = 0; ih < in_h; ++ih) {
      for (int iw = 0; iw < in_w; ++iw) {
        for (int by = 0; by < block_size; ++by) {
          for (int bx = 0; bx < block_size; ++bx) {
            const int oh = ih * block_size + by;
            const int ow = iw * block_size + bx;
            const int ic_base = (by * block_size + bx) * out_c;
            std::memcpy(
                output_data + ((b * out_h + oh) * out_w + ow) * out_c,
                input_data + ((b * in_h + ih) * in_w + iw) * in_c + ic_base,
                static_cast<size_t>(out_c) * sizeof(T));
          }
        }
      }
    }
  }
}

template <typename T>
inline void ReferenceSpaceToBatchNd(const T* input_data, int batch, int in_h,
                                    int in_w, int channels, int block_h,
                                    int block_w, int pad_t, int pad_b,
                                    int pad_l, int pad_r, T pad_val,
                                    T* output_data) {
  const int out_h = (in_h + pad_t + pad_b) / block_h;
  const int out_w = (in_w + pad_l + pad_r) / block_w;
  const int out_batch = batch * block_h * block_w;
  for (int ob = 0; ob < out_batch; ++ob) {
    const int in_b = ob % batch;
    const int shift_w = (ob / batch) % block_w;
    const int shift_h = (ob / batch) / block_w;
    for (int oy = 0; oy < out_h; ++oy) {
      const int ih = oy * block_h + shift_h - pad_t;
      for (int ox = 0; ox < out_w; ++ox) {
        const int iw = ox * block_w + shift_w - pad_l;
        T* out = output_data + ((ob * out_h + oy) * out_w + ox) * channels;
        if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
          std::fill_n(out, channels, pad_val);
        } else {
          const T* in =
              input_data + ((in_b * in_h + ih) * in_w + iw) * channels;
          std::memcpy(out, in, static_cast<size_t>(channels) * sizeof(T));
        }
      }
    }
  }
}

template <typename T>
inline void ReferenceBatchToSpaceNd(const T* input_data, int batch, int in_h,
                                    int in_w, int channels, int block_h,
                                    int block_w, int crop_t, int crop_b,
                                    int crop_l, int crop_r, T* output_data) {
  const int out_batch = batch / (block_h * block_w);
  const int out_h = in_h * block_h - crop_t - crop_b;
  const int out_w = in_w * block_w - crop_l - crop_r;
  for (int ib = 0; ib < batch; ++ib) {
    const int out_b = ib % out_batch;
    const int spatial_offset = ib / out_batch;
    const int shift_h = spatial_offset / block_w;
    const int shift_w = spatial_offset % block_w;
    for (int iy = 0; iy < in_h; ++iy) {
      const int oh = iy * block_h + shift_h - crop_t;
      if (oh < 0 || oh >= out_h) continue;
      for (int ix = 0; ix < in_w; ++ix) {
        const int ow = ix * block_w + shift_w - crop_l;
        if (ow < 0 || ow >= out_w) continue;
        std::memcpy(
            output_data + ((out_b * out_h + oh) * out_w + ow) * channels,
            input_data + ((ib * in_h + iy) * in_w + ix) * channels,
            static_cast<size_t>(channels) * sizeof(T));
      }
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_
