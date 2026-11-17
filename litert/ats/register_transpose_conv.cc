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

#include "litert/ats/register_transpose_conv.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/transpose_conv.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using SameC = std::integral_constant<tflite::Padding, tflite::Padding_SAME>;
using ValidC = std::integral_constant<tflite::Padding, tflite::Padding_VALID>;

template <typename Fixture>
void RegisterTransposeConvImpl(const AtsConf& options, size_t& test_id,
                               size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture, TransposeConv,
      TypeList<float, int8_t>,
      TypeList<SameC, ValidC>,
      SizeListC<1, 2>,
      FaListC<tflite::ActivationFunctionType_NONE,
              tflite::ActivationFunctionType_RELU,
              tflite::ActivationFunctionType_RELU6>>(iters, test_id, options,
                                                      cap);
  // clang-format on
}

}  // namespace

void RegisterTransposeConv(const AtsConf& options, size_t& test_id,
                           size_t iters, AtsInferenceTest::Capture& cap) {
  RegisterTransposeConvImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterTransposeConv(const AtsConf& options, size_t& test_id,
                           size_t iters, AtsCompileTest::Capture& cap) {
  RegisterTransposeConvImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
