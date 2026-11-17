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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_PAD_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_PAD_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferPad(const LiteRtOpT& op,
                             absl::Span<const Dims> input_shapes,
                             std::vector<Dims>& output_shapes) {
  constexpr size_t kPadMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kPaddingsArgIndex = 1;
  constexpr size_t kPairsPerDim = 2;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  if (input_shapes.size() < kPadMinArgs || output_shapes.size() != 1 ||
      op.Inputs().size() < kPadMinArgs ||
      op.Inputs()[kPaddingsArgIndex] == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& paddings_tensor = op.Input(kPaddingsArgIndex);

  if (paddings_tensor.Weights().Buffer().Size() < kShapeTensorSize) {
    // Dynamic padding.
    output_shapes[0] = Dims(input_shape.size(), -1);
    return kLiteRtStatusErrorUnsupported;
  }

  auto buf = paddings_tensor.Weights().Buffer();
  // Padding tensor is [Rank, 2] int32 or int64.
  const size_t num_elements = input_shape.size() * kPairsPerDim;
  std::vector<int64_t> paddings(num_elements);

  if (buf.Size() == num_elements * sizeof(int32_t)) {
    const uint8_t* data = buf.Data();
    for (size_t i = 0; i < num_elements; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      paddings[i] = val;
    }
  } else if (buf.Size() == num_elements * sizeof(int64_t)) {
    std::memcpy(paddings.data(), buf.Data(), num_elements * sizeof(int64_t));
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape = input_shape;
  for (int i = 0; i < input_shape.size(); ++i) {
    int64_t before = paddings[i * kPairsPerDim];
    int64_t after = paddings[i * kPairsPerDim + 1];
    if (before < 0 || after < 0 || out_shape[i] < -1) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (out_shape[i] != -1) {
      int32_t padded_dim = 0;
      if (!SafeAdd(out_shape[i], before, padded_dim) ||
          !SafeAdd(padded_dim, after, padded_dim)) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      out_shape[i] = padded_dim;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferPadv2(const LiteRtOpT& op,
                               absl::Span<const Dims> input_shapes,
                               std::vector<Dims>& output_shapes) {
  return InferPad(op, input_shapes, output_shapes);
}

inline LiteRtStatus InferMirrorPad(const LiteRtOpT& op,
                                   absl::Span<const Dims> input_shapes,
                                   std::vector<Dims>& output_shapes) {
  return InferPad(op, input_shapes, output_shapes);
}

template <typename T>
inline void ReferencePadv2(const T* input_data, const int32_t* input_dims,
                           int rank, const int32_t* paddings, T constant_value,
                           T* output_data) {
  if (rank <= 0) {
    if (rank == 0) output_data[0] = input_data[0];
    return;
  }
  std::vector<int32_t> out_dims(rank);
  std::vector<size_t> in_strides(rank, 1);
  size_t total_out = 1;
  for (int d = rank - 1; d >= 0; --d) {
    out_dims[d] = input_dims[d] + paddings[2 * d] + paddings[2 * d + 1];
    total_out *= static_cast<size_t>(out_dims[d]);
    if (d + 1 < rank) {
      in_strides[d] =
          in_strides[d + 1] * static_cast<size_t>(input_dims[d + 1]);
    }
  }
  std::vector<int32_t> pos(rank, 0);
  for (size_t out_idx = 0; out_idx < total_out; ++out_idx) {
    bool in_bounds = true;
    size_t in_idx = 0;
    for (int d = 0; d < rank; ++d) {
      const int32_t c = pos[d] - paddings[2 * d];
      if (c < 0 || c >= input_dims[d]) {
        in_bounds = false;
        break;
      }
      in_idx += static_cast<size_t>(c) * in_strides[d];
    }
    output_data[out_idx] = in_bounds ? input_data[in_idx] : constant_value;
    for (int d = rank - 1; d >= 0; --d) {
      if (++pos[d] < out_dims[d]) break;
      pos[d] = 0;
    }
  }
}

template <typename T>
inline void ReferenceMirrorPad(const T* input_data, const int32_t* input_dims,
                               int rank, const int32_t* paddings,
                               bool symmetric, T* output_data) {
  if (rank <= 0) {
    if (rank == 0) output_data[0] = input_data[0];
    return;
  }
  auto mirror_idx = [symmetric](int32_t idx, int32_t dim) -> int32_t {
    if (dim <= 1) return 0;
    while (idx < 0 || idx >= dim) {
      if (idx < 0) {
        idx = symmetric ? (-idx - 1) : -idx;
      } else {
        idx = symmetric ? (2 * dim - 1 - idx) : (2 * dim - 2 - idx);
      }
    }
    return idx;
  };
  std::vector<int32_t> out_dims(rank);
  std::vector<size_t> in_strides(rank, 1);
  size_t total_out = 1;
  for (int d = rank - 1; d >= 0; --d) {
    out_dims[d] = input_dims[d] + paddings[2 * d] + paddings[2 * d + 1];
    total_out *= static_cast<size_t>(out_dims[d]);
    if (d + 1 < rank) {
      in_strides[d] =
          in_strides[d + 1] * static_cast<size_t>(input_dims[d + 1]);
    }
  }
  std::vector<int32_t> pos(rank, 0);
  for (size_t out_idx = 0; out_idx < total_out; ++out_idx) {
    size_t in_idx = 0;
    for (int d = 0; d < rank; ++d) {
      const int32_t c = mirror_idx(pos[d] - paddings[2 * d], input_dims[d]);
      in_idx += static_cast<size_t>(c) * in_strides[d];
    }
    output_data[out_idx] = input_data[in_idx];
    for (int d = rank - 1; d >= 0; --d) {
      if (++pos[d] < out_dims[d]) break;
      pos[d] = 0;
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_PAD_H_
