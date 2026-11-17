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

#include "litert/ats/register_pad.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/pad.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

using ReflectC = std::integral_constant<tflite::MirrorPadMode,
                                        tflite::MirrorPadMode_REFLECT>;
using SymmetricC = std::integral_constant<tflite::MirrorPadMode,
                                          tflite::MirrorPadMode_SYMMETRIC>;

template <typename Fixture>
void RegisterPadImpl(const AtsConf& options, size_t& test_id, size_t iters,
                     typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      Pad,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, tflite::half>>
    (iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, Padv2,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, int8_t, int16_t>>(iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, MirrorPad,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, int8_t>,
      TypeList<ReflectC, SymmetricC>>(iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterPad(const AtsConf& options, size_t& test_id, size_t iters,
                 AtsInferenceTest::Capture& cap) {
  RegisterPadImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterPad(const AtsConf& options, size_t& test_id, size_t iters,
                 AtsCompileTest::Capture& cap) {
  RegisterPadImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
