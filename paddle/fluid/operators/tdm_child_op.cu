/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#pragma once

#include <cuda.h>
#include <vector>
#include "paddle/fluid/operators/tdm_child_op.h"

namespace paddle {
namespace operators {

template <typename T, typename InfoT = int, typename OutT = int>
__global__ void Kernel_TDMChildInner(const size_t N, const T *input_data,
                                     const InfoT *tree_info_data,
                                     const int child_nums, const int length,
                                     OutT *child_data, OutT *leaf_mask_data) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    const int input_ids = idx / child_nums;
    const int child_ids = idx % child_nums;

    int start_tree_id = static_cast<int>(input_data[input_ids]) * length + 3;
    if ((input_data[input_ids] == 0 || tree_info_data[start_tree_id] == 0)) {
      child_data[idx] = 0;
      leaf_mask_data[idx] = 0;
    } else {
      OutT child_id =
          static_cast<OutT>(tree_info_data[start_tree_id + child_ids]);
      child_data[idx] = child_id;
      leaf_mask_data[idx] = static_cast<OutT>(
          tree_info_data[static_cast<int>(child_id) * length] == 0 ? 0 : 1);
    }
  }
}

template <typename T, typename InfoT = int, typename OutT = int>
void TDMChildInnerCUDA(const framework::ExecutionContext &context,
                       const LoDTensor &input, const LoDTensor &tree_info,
                       LoDTensor *child, LoDTensor *mask) {
  auto child_nums = context.Attr<int>("child_nums");
  auto info_dims = tree_info.dims();
  int node_nums = info_dims[0];
  int length = info_dims[1];

  int input_ids_num = input.numel();
  VLOG(4) << "TDM child op: input numel ->  " << input_ids_num;

  auto *input_data = input.data<T>();
  auto *tree_info_data = tree_info.data<InfoT>();

  auto *child_data = child->mutable_data<OutT>(context.GetPlace());
  auto *leaf_mask_data = mask->mutable_data<OutT>(context.GetPlace());

  auto stream = context.cuda_device_context().stream();

  size_t N = input_ids_num * child_nums;
  // kernel
  Kernel_TDMChildInner<T, InfoT, OutT><<<(N + 512 - 1) / 512, 512, 0, stream>>>(
      N, input_data, tree_info_data, child_nums, length, child_data,
      leaf_mask_data);
}

template <typename DeviceContext, typename T>
class TDMChildCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *tree_info_var = ctx.InputVar("TreeInfo");

    auto &input_tensor = input_var->Get<LoDTensor>();
    const auto &input_type = input_tensor.type();
    bool input_type_match = input_type == framework::proto::VarType::INT32 ||
                            input_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(input_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(X) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(input_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto &tree_info_tensor = tree_info_var->Get<LoDTensor>();
    const auto &info_type = tree_info_tensor.type();
    bool info_type_match = info_type == framework::proto::VarType::INT32 ||
                           info_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(
        info_type_match, true,
        platform::errors::InvalidArgument(
            "Input(TreeInfo) holds the wrong type, it holds %s, but "
            "desires to be %s or %s",
            paddle::framework::DataTypeToString(info_type),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64)));

    auto *child_var = ctx.OutputVar("Child");
    auto *leaf_mask_var = ctx.OutputVar("LeafMask");
    auto *child_tensor = child_var->GetMutable<framework::LoDTensor>();
    auto *leaf_mask_tensor = leaf_mask_var->GetMutable<framework::LoDTensor>();

    auto output_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    bool out_type_match = output_type == framework::proto::VarType::INT32 ||
                          output_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(out_type_match, true,
                      platform::errors::InvalidArgument(
                          "Ouput(Child) & Output(LeafMask) holds the wrong "
                          "type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(output_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    if (info_type == framework::proto::VarType::INT32 &&
        output_type == framework::proto::VarType::INT32) {
      TDMChildInnerCUDA<T, int, int>(ctx, input_tensor, tree_info_tensor,
                                     child_tensor, leaf_mask_tensor);
    } else if (info_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT32) {
      TDMChildInnerCUDA<T, int64_t, int>(ctx, input_tensor, tree_info_tensor,
                                         child_tensor, leaf_mask_tensor);
    } else if (info_type == framework::proto::VarType::INT32 &&
               output_type == framework::proto::VarType::INT64) {
      TDMChildInnerCUDA<T, int, int64_t>(ctx, input_tensor, tree_info_tensor,
                                         child_tensor, leaf_mask_tensor);
    } else if (info_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT64) {
      TDMChildInnerCUDA<T, int64_t, int64_t>(
          ctx, input_tensor, tree_info_tensor, child_tensor, leaf_mask_tensor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    tdm_child,
    paddle::operators::TDMChildCUDAKernel<paddle::platform::CUDADeviceContext,
                                          float>,
    paddle::operators::TDMChildCUDAKernel<paddle::platform::CUDADeviceContext,
                                          double>,
    paddle::operators::TDMChildCUDAKernel<paddle::platform::CUDADeviceContext,
                                          int>,
    paddle::operators::TDMChildCUDAKernel<paddle::platform::CUDADeviceContext,
                                          int64_t>);
