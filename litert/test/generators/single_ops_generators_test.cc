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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_layout.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/activations.h"
#include "litert/test/generators/arg_min_max.h"
#include "litert/test/generators/arithmetic_ops.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/indexing.h"
#include "litert/test/generators/logical_bitwise.h"
#include "litert/test/generators/pack_split.h"
#include "litert/test/generators/pad.h"
#include "litert/test/generators/quantize_cast.h"
#include "litert/test/generators/reduction.h"
#include "litert/test/generators/shape_manip.h"
#include "litert/test/generators/spatial_ops.h"
#include "litert/test/generators/transpose_conv.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::litert::internal::GetTflOpCodeInd;
using ::litert::internal::GetTflOpCodes;
using ::litert::internal::GetTflOptions;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

template <typename T>
SimpleBuffer MakeBuffer(absl::Span<const Layout::Dim> dims,
                        std::initializer_list<T> values) {
  auto buf = SimpleBuffer::Create<T>(dims);
  ABSL_CHECK(buf.HasValue());
  auto view = buf->template AsView<T>();
  ABSL_CHECK_EQ(view.data.size(), values.size());
  std::copy(values.begin(), values.end(), view.data.begin());
  return std::move(*buf);
}

int32_t GetOpVersion(const LiteRtModelT& model, const LiteRtOpT& op) {
  const auto& codes = GetTflOpCodes(model);
  const int32_t ind = GetTflOpCodeInd(op);
  ABSL_CHECK_GE(ind, 0);
  ABSL_CHECK_LT(static_cast<size_t>(ind), codes.size());
  return codes[ind]->version;
}

TEST(SingleOpsGeneratorsTest, LeakyReluFloatAndInt8ExportAndReference) {
  using GenF32 = LeakyRelu<SizeC<1>, float>;
  typename GenF32::Params p_f32;
  p_f32.shape = {4};
  p_f32.alpha = 0.25f;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_f32, GenF32::Create(p_f32));

  const auto& sg = gen_f32->Graph().Subgraph(0);
  ASSERT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Op(0).OpCode(), kLiteRtOpCodeTflLeakyRelu);
  ASSERT_EQ(sg.Inputs().size(), 1);
  ASSERT_EQ(sg.Outputs().size(), 1);
  EXPECT_EQ(sg.Inputs()[0]->Name(), "input");
  EXPECT_EQ(sg.Outputs()[0]->Name(), "output");

  VarBuffers inputs;
  inputs.push_back(MakeBuffer<float>({4}, {-4.0f, 0.0f, 2.0f, -1.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  LITERT_ASSERT_OK(gen_f32->Reference(inputs, outputs));
  EXPECT_THAT(outputs[0].AsView<float>().data,
              ElementsAre(-1.0f, 0.0f, 2.0f, -0.25f));

  using GenI8 = LeakyRelu<SizeC<1>, int8_t>;
  typename GenI8::Params p_i8;
  p_i8.shape = {3};
  p_i8.alpha = 0.2f;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_i8, GenI8::Create(p_i8));
  const auto& sg_i8 = gen_i8->Graph().Subgraph(0);
  ASSERT_EQ(sg_i8.Ops().size(), 1);
  EXPECT_EQ(sg_i8.Op(0).OpCode(), kLiteRtOpCodeTflLeakyRelu);
  EXPECT_EQ(sg_i8.Inputs()[0]->Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(sg_i8.Outputs()[0]->Qparams().first, kLiteRtQuantizationPerTensor);
}

TEST(SingleOpsGeneratorsTest, PreluPreservesExplicitInputOrderAndReference) {
  using Gen = Prelu<SizeC<2>, float>;
  typename Gen::Params p;
  p.shape = {2, 2};
  p.alpha_shape = {1, 2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, Gen::Create(p));

  const auto& sg = gen->Graph().Subgraph(0);
  ASSERT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Op(0).OpCode(), kLiteRtOpCodeTflPrelu);
  ASSERT_EQ(sg.Inputs().size(), 2);
  EXPECT_EQ(sg.Inputs()[0]->Name(), "input");
  EXPECT_EQ(sg.Inputs()[1]->Name(), "alpha");
  ASSERT_EQ(sg.Outputs().size(), 1);
  EXPECT_EQ(sg.Outputs()[0]->Name(), "output");

  // Verify MakeInputs() matches the subgraph input order.
  DefaultDevice device(123);
  RandomTensorDataBuilder data_builder;
  LITERT_ASSERT_OK_AND_ASSIGN(auto rand_inputs,
                              gen->MakeInputs(device, data_builder));
  ASSERT_EQ(rand_inputs.size(), 2);
  EXPECT_EQ(rand_inputs[0].AsView<float>().NumElements(), 4);
  EXPECT_EQ(rand_inputs[1].AsView<float>().NumElements(), 2);

  VarBuffers inputs;
  inputs.push_back(MakeBuffer<float>({2, 2}, {-2.0f, 3.0f, -4.0f, -1.0f}));
  inputs.push_back(MakeBuffer<float>({1, 2}, {0.5f, 0.25f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));
  EXPECT_THAT(outputs[0].AsView<float>().data,
              ElementsAre(-1.0f, 3.0f, -2.0f, -0.25f));
}

TEST(SingleOpsGeneratorsTest, AddNAndFloorConstDivExportAndReference) {
  using GenAddN = AddN<SizeC<1>, float, SizeC<3>>;
  typename GenAddN::Params p_add;
  p_add.shape = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_add, GenAddN::Create(p_add));
  const auto& sg_add = gen_add->Graph().Subgraph(0);
  ASSERT_EQ(sg_add.Ops().size(), 1);
  EXPECT_EQ(sg_add.Op(0).OpCode(), kLiteRtOpCodeTflAddN);
  ASSERT_EQ(sg_add.Inputs().size(), 3);

  VarBuffers add_in;
  add_in.push_back(MakeBuffer<float>({3}, {1.0f, 2.0f, 3.0f}));
  add_in.push_back(MakeBuffer<float>({3}, {4.0f, -2.0f, 0.5f}));
  add_in.push_back(MakeBuffer<float>({3}, {-1.0f, 5.0f, 1.5f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto add_out, SimpleBuffer::LikeSignature(sg_add.Outputs().begin(),
                                                sg_add.Outputs().end()));
  LITERT_ASSERT_OK(gen_add->Reference(add_in, add_out));
  EXPECT_THAT(add_out[0].AsView<float>().data, ElementsAre(4.0f, 5.0f, 5.0f));

  using GenFDiv =
      FloorConstDiv<SizeC<1>, int32_t, OpCodeC<kLiteRtOpCodeTflFloorDiv>>;
  typename GenFDiv::Params p_div;
  p_div.shape = {4};
  p_div.divisor = 3;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_div, GenFDiv::Create(p_div));
  const auto& sg_div = gen_div->Graph().Subgraph(0);
  ASSERT_EQ(sg_div.Ops().size(), 1);
  EXPECT_EQ(sg_div.Op(0).OpCode(), kLiteRtOpCodeTflFloorDiv);
  EXPECT_EQ(sg_div.Inputs().size(), 1);
  EXPECT_EQ(sg_div.Op(0).NumInputs(), 2);

  VarBuffers div_in;
  div_in.push_back(MakeBuffer<int32_t>({4}, {-7, -6, 0, 7}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto div_out, SimpleBuffer::LikeSignature(sg_div.Outputs().begin(),
                                                sg_div.Outputs().end()));
  LITERT_ASSERT_OK(gen_div->Reference(div_in, div_out));
  EXPECT_THAT(div_out[0].AsView<int32_t>().data, ElementsAre(-3, -2, 0, 2));

  using GenFMod =
      FloorConstDiv<SizeC<1>, int32_t, OpCodeC<kLiteRtOpCodeTflFloorMod>>;
  typename GenFMod::Params p_mod;
  p_mod.shape = {4};
  p_mod.divisor = 3;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_mod, GenFMod::Create(p_mod));
  const auto& sg_mod = gen_mod->Graph().Subgraph(0);
  ASSERT_EQ(sg_mod.Ops().size(), 1);
  EXPECT_EQ(sg_mod.Op(0).OpCode(), kLiteRtOpCodeTflFloorMod);
  EXPECT_EQ(sg_mod.Inputs().size(), 1);
  EXPECT_EQ(sg_mod.Op(0).NumInputs(), 2);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mod_out, SimpleBuffer::LikeSignature(sg_mod.Outputs().begin(),
                                                sg_mod.Outputs().end()));
  LITERT_ASSERT_OK(gen_mod->Reference(div_in, mod_out));
  EXPECT_THAT(mod_out[0].AsView<int32_t>().data, ElementsAre(2, 0, 0, 1));
}

TEST(SingleOpsGeneratorsTest, LogicalAndBitwiseExportAndReference) {
  using GenNot = LogicalNot<SizeC<1>>;
  typename GenNot::Params p_not;
  p_not.shape = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_not, GenNot::Create(p_not));
  EXPECT_EQ(gen_not->Graph().Subgraph(0).Ops().size(), 1);
  EXPECT_EQ(gen_not->Graph().Subgraph(0).Op(0).OpCode(),
            kLiteRtOpCodeTflLogicalNot);

  using GenAnd =
      LogicalBinary<SizeC<2>, SizeC<1>, OpCodeC<kLiteRtOpCodeTflLogicalAnd>>;
  typename GenAnd::Params p_and;
  p_and.input1_shape = {2, 2};
  p_and.input2_shape = {2};
  p_and.output_shape = {2, 2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_and, GenAnd::Create(p_and));
  const auto& sg_and = gen_and->Graph().Subgraph(0);
  ASSERT_EQ(sg_and.Ops().size(), 1);
  EXPECT_EQ(sg_and.Op(0).OpCode(), kLiteRtOpCodeTflLogicalAnd);
  ASSERT_EQ(sg_and.Inputs().size(), 2);
  EXPECT_EQ(sg_and.Inputs()[0]->Name(), "lhs");
  EXPECT_EQ(sg_and.Inputs()[1]->Name(), "rhs");

  VarBuffers and_in;
  and_in.push_back(MakeBuffer<bool>({2, 2}, {true, false, false, true}));
  and_in.push_back(MakeBuffer<bool>({2}, {true, true}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto and_out, SimpleBuffer::LikeSignature(sg_and.Outputs().begin(),
                                                sg_and.Outputs().end()));
  LITERT_ASSERT_OK(gen_and->Reference(and_in, and_out));
  EXPECT_THAT(and_out[0].AsView<bool>().data,
              ElementsAre(true, false, false, true));

  using GenXor = BitwiseXor<SizeC<1>, SizeC<1>>;
  typename GenXor::Params p_xor;
  p_xor.input1_shape = {2};
  p_xor.input2_shape = {2};
  p_xor.output_shape = {2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_xor, GenXor::Create(p_xor));
  const auto& sg_xor = gen_xor->Graph().Subgraph(0);
  ASSERT_EQ(sg_xor.Ops().size(), 1);
  EXPECT_EQ(sg_xor.Op(0).OpCode(), kLiteRtOpCodeTflBitwiseXor);
  VarBuffers xor_in;
  xor_in.push_back(MakeBuffer<int32_t>({2}, {0b1010, -1}));
  xor_in.push_back(MakeBuffer<int32_t>({2}, {0b1100, 1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto xor_out, SimpleBuffer::LikeSignature(sg_xor.Outputs().begin(),
                                                sg_xor.Outputs().end()));
  LITERT_ASSERT_OK(gen_xor->Reference(xor_in, xor_out));
  EXPECT_THAT(xor_out[0].AsView<int32_t>().data, ElementsAre(0b0110, -2));

  using GenRShift = RightShift<SizeC<1>>;
  typename GenRShift::Params p_rshift;
  p_rshift.shape = {3};
  p_rshift.shift = 2;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_rs, GenRShift::Create(p_rshift));
  const auto& sg_rs = gen_rs->Graph().Subgraph(0);
  ASSERT_EQ(sg_rs.Ops().size(), 1);
  EXPECT_EQ(sg_rs.Op(0).OpCode(), kLiteRtOpCodeTflRightShift);
  EXPECT_EQ(sg_rs.Inputs().size(), 1);
  EXPECT_EQ(sg_rs.Op(0).NumInputs(), 2);
  VarBuffers rs_in;
  rs_in.push_back(MakeBuffer<uint32_t>({3}, {16u, 9u, 0x80000000u}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rs_out, SimpleBuffer::LikeSignature(sg_rs.Outputs().begin(),
                                               sg_rs.Outputs().end()));
  LITERT_ASSERT_OK(gen_rs->Reference(rs_in, rs_out));
  EXPECT_THAT(rs_out[0].AsView<uint32_t>().data,
              ElementsAre(4u, 2u, 0x20000000u));
}

TEST(SingleOpsGeneratorsTest, Padv2AndMirrorPadExportAndReference) {
  using GenPadv2 = Padv2<SizeC<1>, int8_t>;
  typename GenPadv2::Params p_pv2;
  p_pv2.input_shape = {2};
  p_pv2.paddings = {1, 2};
  p_pv2.constant_value = -7;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_pv2, GenPadv2::Create(p_pv2));
  const auto& sg_pv2 = gen_pv2->Graph().Subgraph(0);
  ASSERT_EQ(sg_pv2.Ops().size(), 1);
  EXPECT_EQ(sg_pv2.Op(0).OpCode(), kLiteRtOpCodeTflPadv2);
  EXPECT_EQ(sg_pv2.Inputs().size(), 1);
  EXPECT_EQ(sg_pv2.Op(0).NumInputs(), 3);
  EXPECT_EQ(sg_pv2.Inputs()[0]->Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(sg_pv2.Outputs()[0]->Qparams().first, kLiteRtQuantizationPerTensor);

  VarBuffers pv2_in;
  pv2_in.push_back(MakeBuffer<int8_t>({2}, {10, 20}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto pv2_out, SimpleBuffer::LikeSignature(sg_pv2.Outputs().begin(),
                                                sg_pv2.Outputs().end()));
  LITERT_ASSERT_OK(gen_pv2->Reference(pv2_in, pv2_out));
  EXPECT_THAT(pv2_out[0].AsView<int8_t>().data,
              ElementsAre(-7, 10, 20, -7, -7));

  using GenMirror =
      MirrorPad<SizeC<1>, float,
                std::integral_constant<tflite::MirrorPadMode,
                                       tflite::MirrorPadMode_REFLECT>>;
  typename GenMirror::Params p_mp;
  p_mp.input_shape = {3};
  p_mp.paddings = {1, 1};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_mp, GenMirror::Create(p_mp));
  const auto& sg_mp = gen_mp->Graph().Subgraph(0);
  ASSERT_EQ(sg_mp.Ops().size(), 1);
  EXPECT_EQ(sg_mp.Op(0).OpCode(), kLiteRtOpCodeTflMirrorPad);
  VarBuffers mp_in;
  mp_in.push_back(MakeBuffer<float>({3}, {1.0f, 2.0f, 3.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mp_out, SimpleBuffer::LikeSignature(sg_mp.Outputs().begin(),
                                               sg_mp.Outputs().end()));
  LITERT_ASSERT_OK(gen_mp->Reference(mp_in, mp_out));
  EXPECT_THAT(mp_out[0].AsView<float>().data,
              ElementsAre(2.0f, 1.0f, 2.0f, 3.0f, 2.0f));
}

TEST(SingleOpsGeneratorsTest, ArgMaxTensorApiAndArgMinLowLevel) {
  using GenMax = ArgMinMax<SizeC<2>, float, OpCodeC<kLiteRtOpCodeTflArgMax>>;
  typename GenMax::Params p;
  p.input_shape = {2, 3};
  p.axis = 1;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_max, GenMax::Create(p));
  const auto& sg_max = gen_max->Graph().Subgraph(0);
  ASSERT_EQ(sg_max.Ops().size(), 1);
  EXPECT_EQ(sg_max.Op(0).OpCode(), kLiteRtOpCodeTflArgMax);
  EXPECT_EQ(sg_max.Outputs()[0]->Type().second.ranked_tensor_type.element_type,
            kLiteRtElementTypeInt32);

  VarBuffers in;
  in.push_back(MakeBuffer<float>({2, 3}, {1.0f, 5.0f, 2.0f, 9.0f, 0.0f, 3.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto max_out, SimpleBuffer::LikeSignature(sg_max.Outputs().begin(),
                                                sg_max.Outputs().end()));
  LITERT_ASSERT_OK(gen_max->Reference(in, max_out));
  EXPECT_THAT(max_out[0].AsView<int32_t>().data, ElementsAre(1, 0));

  using GenMin = ArgMinMax<SizeC<2>, float, OpCodeC<kLiteRtOpCodeTflArgMin>>;
  typename GenMin::Params p_min;
  p_min.input_shape = {2, 3};
  p_min.axis = 1;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_min, GenMin::Create(p_min));
  const auto& sg_min = gen_min->Graph().Subgraph(0);
  ASSERT_EQ(sg_min.Ops().size(), 1);
  EXPECT_EQ(sg_min.Op(0).OpCode(), kLiteRtOpCodeTflArgMin);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto min_out, SimpleBuffer::LikeSignature(sg_min.Outputs().begin(),
                                                sg_min.Outputs().end()));
  LITERT_ASSERT_OK(gen_min->Reference(in, min_out));
  EXPECT_THAT(min_out[0].AsView<int32_t>().data, ElementsAre(0, 1));
}

TEST(SingleOpsGeneratorsTest, ReductionBoolExactAndNegativeAxis) {
  using GenAll = Reduction<SizeC<2>, bool, OpCodeC<kLiteRtOpCodeTflReduceAll>,
                           /*KeepDims=*/std::false_type,
                           /*AllowScalarOutput=*/std::false_type>;
  typename GenAll::Params p;
  p.input_shape = {2, 2};
  p.axes = {-1};
  p.output_shape = {2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, GenAll::Create(p));
  EXPECT_EQ(gen->GetConformanceSpec().comparator_kind,
            ConformanceComparatorKind::kExact);
  const auto& sg = gen->Graph().Subgraph(0);
  ASSERT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Op(0).OpCode(), kLiteRtOpCodeTflReduceAll);

  VarBuffers in;
  in.push_back(MakeBuffer<bool>({2, 2}, {true, true, true, false}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(in, out));
  EXPECT_THAT(out[0].AsView<bool>().data, ElementsAre(true, false));
}

TEST(SingleOpsGeneratorsTest, ResizeOpVersionsAndSignedIdentityGolden) {
  using GenBilinearHalf = ResizeOp<OpCodeC<kLiteRtOpCodeTflResizeBilinear>,
                                   int8_t, std::false_type, std::true_type>;
  typename GenBilinearHalf::Params p;
  p.input_shape = {1, 2, 2, 1};
  p.target_size = {2, 2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_half, GenBilinearHalf::Create(p));
  const auto& sg_half = gen_half->Graph().Subgraph(0);
  ASSERT_EQ(sg_half.Ops().size(), 1);
  EXPECT_EQ(sg_half.Op(0).OpCode(), kLiteRtOpCodeTflResizeBilinear);
  EXPECT_EQ(GetOpVersion(gen_half->Graph(), sg_half.Op(0)), 3);
  const auto* opts = GetTflOptions(sg_half.Op(0)).AsResizeBilinearOptions();
  ASSERT_NE(opts, nullptr);
  EXPECT_FALSE(opts->align_corners);
  EXPECT_TRUE(opts->half_pixel_centers);

  // Verify signed int8 identity resize preserves exact negative values.
  using GenBilinearI8 = ResizeOp<OpCodeC<kLiteRtOpCodeTflResizeBilinear>,
                                 int8_t, std::false_type, std::false_type>;
  typename GenBilinearI8::Params p_i8{{1, 2, 2, 1}, {2, 2}};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_i8, GenBilinearI8::Create(p_i8));
  EXPECT_EQ(GetOpVersion(gen_i8->Graph(), gen_i8->Graph().Subgraph(0).Op(0)),
            1);
  VarBuffers in_i8;
  in_i8.push_back(MakeBuffer<int8_t>({1, 2, 2, 1}, {-10, -20, -30, -40}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_i8,
      SimpleBuffer::LikeSignature(gen_i8->Graph().Subgraph(0).Outputs().begin(),
                                  gen_i8->Graph().Subgraph(0).Outputs().end()));
  LITERT_ASSERT_OK(gen_i8->Reference(in_i8, out_i8));
  EXPECT_THAT(out_i8[0].AsView<int8_t>().data, ElementsAre(-10, -20, -30, -40));

  using GenBilinearI16 = ResizeOp<OpCodeC<kLiteRtOpCodeTflResizeBilinear>,
                                  int16_t, std::false_type, std::false_type>;
  typename GenBilinearI16::Params p_i16{{1, 2, 2, 1}, {2, 2}};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_i16, GenBilinearI16::Create(p_i16));
  VarBuffers in_i16;
  in_i16.push_back(MakeBuffer<int16_t>({1, 2, 2, 1}, {-10, -20, -30, -40}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_i16, SimpleBuffer::LikeSignature(
                        gen_i16->Graph().Subgraph(0).Outputs().begin(),
                        gen_i16->Graph().Subgraph(0).Outputs().end()));
  LITERT_ASSERT_OK(gen_i16->Reference(in_i16, out_i16));
  EXPECT_THAT(out_i16[0].AsView<int16_t>().data,
              ElementsAre(-10, -20, -30, -40));

  using GenNearestHalf =
      ResizeOp<OpCodeC<kLiteRtOpCodeTflResizeNearestNeighbor>, float,
               std::false_type, std::true_type>;
  typename GenNearestHalf::Params p_nn{{1, 2, 2, 1}, {2, 2}};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_nn, GenNearestHalf::Create(p_nn));
  EXPECT_EQ(GetOpVersion(gen_nn->Graph(), gen_nn->Graph().Subgraph(0).Op(0)),
            3);
}

TEST(SingleOpsGeneratorsTest, TransposeConvIsSingleFusedVersion4Op) {
  using Gen = TransposeConv<
      float, std::integral_constant<tflite::Padding, tflite::Padding_VALID>,
      SizeC<1>, FaC<tflite::ActivationFunctionType_RELU>>;
  typename Gen::Params p;
  p.input_shape = {1, 1, 1, 1};
  p.filter_shape = {1, 3, 3, 1};
  p.output_shape = {1, 3, 3, 1};
  p.filter_data = {1.0f, -2.0f, 3.0f, 0.0f, 1.0f, -1.0f, 2.0f, -3.0f, 0.5f};
  p.bias_data = {0.5f};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, Gen::Create(p));

  const auto& sg = gen->Graph().Subgraph(0);
  ASSERT_EQ(sg.Ops().size(), 1);
  EXPECT_EQ(sg.Op(0).OpCode(), kLiteRtOpCodeTflTransposeConv);
  EXPECT_EQ(GetOpVersion(gen->Graph(), sg.Op(0)), 4);
  EXPECT_EQ(sg.Op(0).NumInputs(), 4);
  ASSERT_EQ(sg.Inputs().size(), 1);
  EXPECT_EQ(sg.Inputs()[0]->Name(), "input");
  const auto* tc_opts = GetTflOptions(sg.Op(0)).AsTransposeConvOptions();
  ASSERT_NE(tc_opts, nullptr);
  EXPECT_EQ(tc_opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);

  VarBuffers in;
  in.push_back(MakeBuffer<float>({1, 1, 1, 1}, {1.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out,
      SimpleBuffer::LikeSignature(sg.Outputs().begin(), sg.Outputs().end()));
  LITERT_ASSERT_OK(gen->Reference(in, out));
  EXPECT_THAT(out[0].AsView<float>().data,
              Pointwise(FloatNear(1e-6f), {1.5f, 0.0f, 3.5f, 0.5f, 1.5f, 0.0f,
                                           2.5f, 0.0f, 1.0f}));
}

TEST(SingleOpsGeneratorsTest, PackOrderBroadcastToVersionAndSliceReference) {
  using GenPack = Pack<SizeC<1>, int32_t, SizeC<0>>;
  typename GenPack::Params p_pack;
  p_pack.input_shape = {2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_pack, GenPack::Create(p_pack));
  const auto& sg_pack = gen_pack->Graph().Subgraph(0);
  ASSERT_EQ(sg_pack.Ops().size(), 1);
  EXPECT_EQ(sg_pack.Op(0).OpCode(), kLiteRtOpCodeTflPack);
  ASSERT_EQ(sg_pack.Inputs().size(), 2);
  EXPECT_EQ(sg_pack.Inputs()[0]->Name(), "input_0");
  EXPECT_EQ(sg_pack.Inputs()[1]->Name(), "input_1");

  using GenBcast = BroadcastTo<SizeC<1>, SizeC<2>, int32_t>;
  typename GenBcast::Params p_bcast;
  p_bcast.input_shape = {2};
  p_bcast.output_shape = {2, 2};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_bcast, GenBcast::Create(p_bcast));
  const auto& sg_bcast = gen_bcast->Graph().Subgraph(0);
  ASSERT_EQ(sg_bcast.Ops().size(), 1);
  EXPECT_EQ(sg_bcast.Op(0).OpCode(), kLiteRtOpCodeTflBroadcastTo);
  EXPECT_EQ(GetOpVersion(gen_bcast->Graph(), sg_bcast.Op(0)), 2);
  VarBuffers bcast_in;
  bcast_in.push_back(MakeBuffer<int32_t>({2}, {7, 9}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto bcast_out, SimpleBuffer::LikeSignature(sg_bcast.Outputs().begin(),
                                                  sg_bcast.Outputs().end()));
  LITERT_ASSERT_OK(gen_bcast->Reference(bcast_in, bcast_out));
  EXPECT_THAT(bcast_out[0].AsView<int32_t>().data, ElementsAre(7, 9, 7, 9));

  using GenSlice = StridedSlice<SizeC<2>, int32_t, SizeC<0>>;
  typename GenSlice::Params p_slice;
  p_slice.input_shape = {2, 3};
  p_slice.begin = {0, 1};
  p_slice.end = {2, 3};
  p_slice.strides = {1, 1};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_slice, GenSlice::Create(p_slice));
  const auto& sg_slice = gen_slice->Graph().Subgraph(0);
  ASSERT_EQ(sg_slice.Ops().size(), 1);
  EXPECT_EQ(sg_slice.Op(0).OpCode(), kLiteRtOpCodeTflStridedSlice);
  VarBuffers slice_in;
  slice_in.push_back(MakeBuffer<int32_t>({2, 3}, {0, 1, 2, 3, 4, 5}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto slice_out, SimpleBuffer::LikeSignature(sg_slice.Outputs().begin(),
                                                  sg_slice.Outputs().end()));
  LITERT_ASSERT_OK(gen_slice->Reference(slice_in, slice_out));
  EXPECT_THAT(slice_out[0].AsView<int32_t>().data, ElementsAre(1, 2, 4, 5));
}

TEST(SingleOpsGeneratorsTest, QuantizeAndCastExportAndReference) {
  using GenQuant = Quantize<float, int8_t, SizeC<1>>;
  typename GenQuant::Params p_q;
  p_q.shape = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_q, GenQuant::Create(p_q));
  const auto& sg_q = gen_q->Graph().Subgraph(0);
  ASSERT_EQ(sg_q.Ops().size(), 1);
  EXPECT_EQ(sg_q.Op(0).OpCode(), kLiteRtOpCodeTflQuantize);
  EXPECT_EQ(sg_q.Outputs()[0]->Qparams().first, kLiteRtQuantizationPerTensor);

  VarBuffers q_in;
  q_in.push_back(MakeBuffer<float>({3}, {-0.4f, 0.0f, 0.4f}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_out, SimpleBuffer::LikeSignature(sg_q.Outputs().begin(),
                                              sg_q.Outputs().end()));
  LITERT_ASSERT_OK(gen_q->Reference(q_in, q_out));
  EXPECT_THAT(q_out[0].AsView<int8_t>().data, ElementsAre(0, 10, 20));

  using GenCast = Cast<int32_t, float, SizeC<1>>;
  typename GenCast::Params p_c;
  p_c.shape = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(auto gen_c, GenCast::Create(p_c));
  const auto& sg_c = gen_c->Graph().Subgraph(0);
  ASSERT_EQ(sg_c.Ops().size(), 1);
  EXPECT_EQ(sg_c.Op(0).OpCode(), kLiteRtOpCodeTflCast);
  VarBuffers c_in;
  c_in.push_back(MakeBuffer<int32_t>({3}, {-3, 0, 7}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto c_out, SimpleBuffer::LikeSignature(sg_c.Outputs().begin(),
                                              sg_c.Outputs().end()));
  LITERT_ASSERT_OK(gen_c->Reference(c_in, c_out));
  EXPECT_THAT(c_out[0].AsView<float>().data, ElementsAre(-3.0f, 0.0f, 7.0f));
}

}  // namespace
}  // namespace litert::testing
