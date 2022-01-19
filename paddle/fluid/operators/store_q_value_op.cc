/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <sstream>
#include <vector>
#include <memory>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/tensor.h"
//#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

//using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoDTensorArray = framework::LoDTensorArray;

template <typename T>
class StoreQValueOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("Ids"); //std::vector<const Tensor*>
    auto ctx_place = ctx.GetPlace();
    int device_id = boost::get<platform::CUDAPlace>(ctx_place).GetDeviceId();

    const auto qvalues_size = inputs.size();
    std::vector<framework::Tensor> cpu_qvalues(qvalues_size);

    for (size_t i = 0; i < qvalues_size; ++i) {
      framework::TensorCopy(*inputs[i], platform::CPUPlace(), &cpu_qvalues[i]);
    }
    framework::BatchGpuPackMgr().store_qvalue(device_id, cpu_qvalues);
  }
};

class StoreQValueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("Ids").size(), 1UL,
                      "Inputs(Ids) of StoreQValueOp should not be empty.");
  }
};

class StoreQValueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids", "The input tensor list .").AsDuplicable();
    AddComment(R"DOC(
store a value Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    store_q_value, ops::StoreQValueOp, ops::StoreQValueOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(store_q_value, ops::StoreQValueOpKernel<float>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(store_q_value, ops::StoreQValueOpKernel<float>);
#endif
