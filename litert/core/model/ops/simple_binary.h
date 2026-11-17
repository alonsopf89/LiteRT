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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/op_shape_inference_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/converter/schema/schema_generated.h"
#include "tflite/kernels/internal/runtime_shape.h"

namespace litert::internal {

inline LiteRtStatus InferElementwiseBinary(absl::Span<Dims> input_shapes,
                                           std::vector<Dims>& output_shapes) {
  if (input_shapes.size() != 2 || output_shapes.size() != 1) {
    LITERT_LOG(LITERT_ERROR,
               "Invalid number of input/output shapes for binary op.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& s1 = input_shapes[0];
  const auto& s2 = input_shapes[1];

  size_t rank1 = s1.size();
  size_t rank2 = s2.size();
  size_t max_rank = std::max(rank1, rank2);
  Dims out_shape(max_rank);

  for (int i = 0; i < max_rank; ++i) {
    // Reverse index
    int idx1 = rank1 - 1 - i;
    int idx2 = rank2 - 1 - i;
    int32_t d1 = (idx1 >= 0) ? s1[idx1] : 1;
    int32_t d2 = (idx2 >= 0) ? s2[idx2] : 1;

    if (d1 == -1 || d2 == -1) {
      out_shape[max_rank - 1 - i] = -1;
    } else if (d1 == d2) {
      out_shape[max_rank - 1 - i] = d1;
    } else if (d1 == 1) {
      out_shape[max_rank - 1 - i] = d2;
    } else if (d2 == 1) {
      out_shape[max_rank - 1 - i] = d1;
    } else {
      LITERT_LOG(LITERT_ERROR, "Incompatible dims for binary op.");
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

#define DEFINE_SIMPLE_BINARY_INFER(name)                              \
  inline LiteRtStatus Infer##name(const LiteRtOpT& op,                \
                                  absl::Span<Dims> input_shapes,      \
                                  std::vector<Dims>& output_shapes) { \
    return InferElementwiseBinary(input_shapes, output_shapes);       \
  }

DEFINE_SIMPLE_BINARY_INFER(Add)
DEFINE_SIMPLE_BINARY_INFER(Atan2)
DEFINE_SIMPLE_BINARY_INFER(Div)
DEFINE_SIMPLE_BINARY_INFER(Equal)
DEFINE_SIMPLE_BINARY_INFER(FloorDiv)
DEFINE_SIMPLE_BINARY_INFER(FloorMod)
DEFINE_SIMPLE_BINARY_INFER(Greater)
DEFINE_SIMPLE_BINARY_INFER(GreaterEqual)
DEFINE_SIMPLE_BINARY_INFER(Less)
DEFINE_SIMPLE_BINARY_INFER(LessEqual)
DEFINE_SIMPLE_BINARY_INFER(LogicalAnd)
DEFINE_SIMPLE_BINARY_INFER(LogicalOr)
DEFINE_SIMPLE_BINARY_INFER(Maximum)
DEFINE_SIMPLE_BINARY_INFER(Minimum)
DEFINE_SIMPLE_BINARY_INFER(Mul)
DEFINE_SIMPLE_BINARY_INFER(NotEqual)
DEFINE_SIMPLE_BINARY_INFER(Pow)
DEFINE_SIMPLE_BINARY_INFER(Prelu)
DEFINE_SIMPLE_BINARY_INFER(SquaredDifference)
DEFINE_SIMPLE_BINARY_INFER(Sub)

#undef DEFINE_SIMPLE_BINARY_INFER

inline void ComputeBroadcastStrides(const int32_t* dims, int input_rank,
                                    int output_rank, size_t* strides) {
  if (input_rank < 0 || output_rank < 0 || input_rank > output_rank) {
    for (int i = 0; i < output_rank; ++i) {
      strides[i] = 0;
    }
    return;
  }
  std::vector<int32_t> synthetic_out(output_rank, 1);
  const int rank_diff = output_rank - input_rank;
  for (int i = 0; i < input_rank; ++i) {
    synthetic_out[i + rank_diff] = dims[i];
  }
  const auto in_span = input_rank == 0 ? absl::Span<const int32_t>{}
                                       : absl::MakeConstSpan(dims, input_rank);
  std::vector<size_t> checked_strides;
  size_t total_out = 0;
  if (ComputeBroadcastStridesChecked(in_span, synthetic_out, checked_strides,
                                     total_out) != kLiteRtStatusOk) {
    for (int i = 0; i < output_rank; ++i) {
      strides[i] = 0;
    }
    return;
  }
  for (int i = 0; i < output_rank; ++i) {
    strides[i] = checked_strides[i];
  }
}

template <typename InT, typename OutT, typename BinaryOp>
void RunBinaryOp(const InT* a, const InT* b, OutT* output,
                 const size_t* a_stride, const size_t* b_stride,
                 const size_t* output_stride, const size_t* output_shape,
                 int rank, BinaryOp op) {
  if (rank <= 0) {
    *output = op(*a, *b);
  } else if (rank == 1) {
    assert(output_stride[0] == 1);
    if (a_stride[0] == 0) {
      const InT a_val = *a;
      for (size_t i = 0; i < output_shape[0]; ++i) {
        output[i] = op(a_val, b[i * b_stride[0]]);
      }
    } else if (b_stride[0] == 0) {
      const InT b_val = *b;
      for (size_t i = 0; i < output_shape[0]; ++i) {
        output[i] = op(a[i], b_val);
      }
    } else {
      assert(a_stride[0] == 1 && b_stride[0] == 1);
      for (size_t i = 0; i < output_shape[0]; ++i) {
        output[i] = op(a[i], b[i]);
      }
    }
  } else {
    for (size_t i = 0; i < output_shape[0]; ++i) {
      RunBinaryOp(a + i * a_stride[0], b + i * b_stride[0],
                  output + i * output_stride[0], a_stride + 1, b_stride + 1,
                  output_stride + 1, output_shape + 1, rank - 1, op);
    }
  }
}

template <typename InT, typename OutT, typename BinaryOp>
inline void ReferenceBinaryGeneric(
    const InT* input1_data, const int32_t* input1_dims, int input1_rank,
    const InT* input2_data, const int32_t* input2_dims, int input2_rank,
    OutT* output_data, const int32_t* output_dims, int rank, BinaryOp op) {
  constexpr int kMaxRank = tflite::RuntimeShape::kMaxSmallSize;
  assert(rank <= kMaxRank);
  size_t a_stride[kMaxRank];
  size_t b_stride[kMaxRank];
  size_t o_stride[kMaxRank];
  size_t o_shape[kMaxRank];

  ComputeBroadcastStrides(input1_dims, input1_rank, rank, a_stride);
  ComputeBroadcastStrides(input2_dims, input2_rank, rank, b_stride);

  size_t current_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    o_stride[i] = current_stride;
    o_shape[i] = output_dims[i];
    current_stride *= output_dims[i];
  }

  RunBinaryOp(input1_data, input2_data, output_data, a_stride, b_stride,
              o_stride, o_shape, rank, op);
}

template <typename T>
inline void ApplyActivation(T* output_data, size_t num_elements,
                            tflite::ActivationFunctionType faf) {
  if (faf == tflite::ActivationFunctionType_RELU) {
    for (size_t i = 0; i < num_elements; ++i) {
      if (output_data[i] < static_cast<T>(0.0f)) {
        output_data[i] = static_cast<T>(0.0f);
      }
    }
  } else if (faf == tflite::ActivationFunctionType_RELU6) {
    for (size_t i = 0; i < num_elements; ++i) {
      if (output_data[i] < static_cast<T>(0.0f)) {
        output_data[i] = static_cast<T>(0.0f);
      } else if (output_data[i] > static_cast<T>(6.0f)) {
        output_data[i] = static_cast<T>(6.0f);
      }
    }
  }
}

template <typename T>
inline void ApplyActivation(T* output_data, const int32_t* output_dims,
                            int rank, tflite::ActivationFunctionType faf) {
  if (faf == tflite::ActivationFunctionType_NONE) {
    return;
  }
  size_t num_elements = 1;
  for (int i = 0; i < rank; ++i) {
    num_elements *= output_dims[i];
  }
  ApplyActivation(output_data, num_elements, faf);
}

inline void ReferenceAdd(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::plus<float>());
  ApplyActivation(output_data, output_dims, rank, faf);
}

inline void ReferenceSub(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::minus<float>());
  ApplyActivation(output_data, output_dims, rank, faf);
}

inline void ReferenceMul(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank,
                         std::multiplies<float>());
  ApplyActivation(output_data, output_dims, rank, faf);
}

inline void ReferenceDiv(
    const float* a_data, const int32_t* a_dims, int a_rank, const float* b_data,
    const int32_t* b_dims, int b_rank, float* output_data,
    const int32_t* output_dims, int rank,
    tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::divides<float>());
  ApplyActivation(output_data, output_dims, rank, faf);
}

inline void ReferencePow(const float* a_data, const int32_t* a_dims, int a_rank,
                         const float* b_data, const int32_t* b_dims, int b_rank,
                         float* output_data, const int32_t* output_dims,
                         int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank,
                         [](float a, float b) { return std::pow(a, b); });
}

template <typename T, typename OutT = bool>
inline void ReferenceEqual(const T* a_data, const int32_t* a_dims, int a_rank,
                           const T* b_data, const int32_t* b_dims, int b_rank,
                           OutT* output_data, const int32_t* output_dims,
                           int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::equal_to<T>());
}

template <typename T, typename OutT = bool>
inline void ReferenceNotEqual(const T* a_data, const int32_t* a_dims,
                              int a_rank, const T* b_data,
                              const int32_t* b_dims, int b_rank,
                              OutT* output_data, const int32_t* output_dims,
                              int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank,
                         std::not_equal_to<T>());
}

template <typename T, typename OutT = bool>
inline void ReferenceGreater(const T* a_data, const int32_t* a_dims, int a_rank,
                             const T* b_data, const int32_t* b_dims, int b_rank,
                             OutT* output_data, const int32_t* output_dims,
                             int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::greater<T>());
}

template <typename T, typename OutT = bool>
inline void ReferenceGreaterEqual(const T* a_data, const int32_t* a_dims,
                                  int a_rank, const T* b_data,
                                  const int32_t* b_dims, int b_rank,
                                  OutT* output_data, const int32_t* output_dims,
                                  int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank,
                         std::greater_equal<T>());
}

template <typename T, typename OutT = bool>
inline void ReferenceLess(const T* a_data, const int32_t* a_dims, int a_rank,
                          const T* b_data, const int32_t* b_dims, int b_rank,
                          OutT* output_data, const int32_t* output_dims,
                          int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::less<T>());
}

template <typename T, typename OutT = bool>
inline void ReferenceLessEqual(const T* a_data, const int32_t* a_dims,
                               int a_rank, const T* b_data,
                               const int32_t* b_dims, int b_rank,
                               OutT* output_data, const int32_t* output_dims,
                               int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::less_equal<T>());
}

template <typename T>
inline void ReferenceFloorDiv(const T* a_data, const int32_t* a_dims,
                              int a_rank, const T* b_data,
                              const int32_t* b_dims, int b_rank, T* output_data,
                              const int32_t* output_dims, int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, [](T a, T b) -> T {
                           if constexpr (std::is_floating_point_v<T>) {
                             return static_cast<T>(std::floor(a / b));
                           } else {
                             T q = a / b;
                             T r = a % b;
                             if (r != 0 && ((a < 0) != (b < 0))) {
                               --q;
                             }
                             return q;
                           }
                         });
}

template <typename T>
inline void ReferenceFloorMod(const T* a_data, const int32_t* a_dims,
                              int a_rank, const T* b_data,
                              const int32_t* b_dims, int b_rank, T* output_data,
                              const int32_t* output_dims, int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, [](T a, T b) -> T {
                           if constexpr (std::is_floating_point_v<T>) {
                             return static_cast<T>(a - std::floor(a / b) * b);
                           } else {
                             T q = a / b;
                             T r = a % b;
                             if (r != 0 && ((a < 0) != (b < 0))) {
                               --q;
                             }
                             return a - q * b;
                           }
                         });
}

inline void ReferenceLogicalAnd(const bool* a_data, const int32_t* a_dims,
                                int a_rank, const bool* b_data,
                                const int32_t* b_dims, int b_rank,
                                bool* output_data, const int32_t* output_dims,
                                int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::logical_and<>());
}

inline void ReferenceLogicalOr(const bool* a_data, const int32_t* a_dims,
                               int a_rank, const bool* b_data,
                               const int32_t* b_dims, int b_rank,
                               bool* output_data, const int32_t* output_dims,
                               int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::logical_or<>());
}

template <typename T>
inline void ReferenceBitwiseXor(const T* a_data, const int32_t* a_dims,
                                int a_rank, const T* b_data,
                                const int32_t* b_dims, int b_rank,
                                T* output_data, const int32_t* output_dims,
                                int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank, std::bit_xor<T>());
}

template <typename T>
inline void ReferenceRightShift(const T* a_data, const int32_t* a_dims,
                                int a_rank, const T* b_data,
                                const int32_t* b_dims, int b_rank,
                                T* output_data, const int32_t* output_dims,
                                int rank) {
  ReferenceBinaryGeneric(a_data, a_dims, a_rank, b_data, b_dims, b_rank,
                         output_data, output_dims, rank,
                         [](T a, T b) -> T { return a >> b; });
}

template <typename T>
inline void ReferencePrelu(const T* input_data, const T* alpha_data,
                           T* output_data, size_t num_elements, size_t channels,
                           const AffineQuantSpec& input_quant = {},
                           const AffineQuantSpec& alpha_quant = {},
                           const AffineQuantSpec& output_quant = {}) {
  for (size_t i = 0; i < num_elements; ++i) {
    const float x = DequantizeValue(input_data[i], input_quant);
    const float a = DequantizeValue(alpha_data[i % channels], alpha_quant);
    const float y = x >= 0.0f ? x : a * x;
    output_data[i] = QuantizeValue<T>(y, output_quant);
  }
}

template <typename T>
inline void ReferenceAddN(absl::Span<const T* const> inputs, T* output_data,
                          size_t num_elements,
                          const AffineQuantSpec& input_quant = {},
                          const AffineQuantSpec& output_quant = {}) {
  for (size_t i = 0; i < num_elements; ++i) {
    float sum = 0.0f;
    for (const T* in_ptr : inputs) {
      sum += DequantizeValue(in_ptr[i], input_quant);
    }
    output_data[i] = QuantizeValue<T>(sum, output_quant);
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_
