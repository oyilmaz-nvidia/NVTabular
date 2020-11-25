/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "time_two.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <embed.h>
#include <Python.h>
#include <iostream>

namespace py = pybind11;

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void TimeTwoCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

template <typename T>
void create_cupy(int size, const T* in, T* out) {
  
  py::gil_scoped_acquire acquire;
  auto va = py::module::import("vector_add");

  py::dict cai_in;
  std::tuple<long> shape((long)size);
  cai_in["shape"] = shape;
  std::tuple<long, bool> data_in((long)*(&in), true);
  cai_in["data"] = data_in; 

  py::dict cai_out;
  cai_out["shape"] = shape;
  std::tuple<long, bool> data_out((long)*(&out), false);
  cai_out["data"] = data_out; 

  py::object result2 = va.attr("run_inference")(cai_in, cai_out);
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct TimeTwoFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.

    create_cupy(size, in, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct TimeTwoFunctor<GPUDevice, float>;
template struct TimeTwoFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
