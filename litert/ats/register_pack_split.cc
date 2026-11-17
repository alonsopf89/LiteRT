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

#include "litert/ats/register_pack_split.h"

#include <cstddef>
#include <cstdint>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/pack_split.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterPackSplitImpl(const AtsConf& options, size_t& test_id,
                           size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture, Pack,
      SizeListC<1, 2, 3>,
      TypeList<float, int8_t, int16_t>,
      SizeListC<0, 1>>(iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, Unpack,
      SizeListC<2, 3, 4>,
      TypeList<float, int8_t, int16_t>,
      SizeListC<0, 1>>(iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, Split,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, int8_t, int16_t>,
      SizeListC<0>>(iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, Split,
      SizeListC<2, 3, 4>,
      TypeList<float, int8_t, int16_t>,
      SizeListC<1>>(iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, SplitV,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, int8_t, int16_t>,
      SizeListC<0>>(iters, test_id, options, cap);
  RegisterCombinations<
      Fixture, SplitV,
      SizeListC<2, 3, 4>,
      TypeList<float, int8_t, int16_t>,
      SizeListC<1>>(iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterPackSplit(const AtsConf& options, size_t& test_id, size_t iters,
                       AtsInferenceTest::Capture& cap) {
  RegisterPackSplitImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterPackSplit(const AtsConf& options, size_t& test_id, size_t iters,
                       AtsCompileTest::Capture& cap) {
  RegisterPackSplitImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
