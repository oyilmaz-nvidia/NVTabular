## NVTabular | Inference documentation

This document includes the details of the PoC that tests the idea of calling nvtabular or any
python modules in a custom tensorflow op. The goal of this PoC is to see whether we can utilize
the existing nvtabular code base or not for the inference operations that are served
using a tool such as tf serving or triton.

We first test the idea using tensowflow tools. Tensorflow has this tool called tf serving to serve
trained model as servince for inference operations. Users can make a REST request to run inference
for a given data. And, tf server runs the inference on the served model and returns the results to user.

Typically, tf server serves the models that are developed using existing tf functions. tf server also
allows developers to use custom ops. These custom ops are developed only using C++ for now.

This document includes the steps to help you to create a custom op for tensorflow that uses python 
modules and frameworks for GPU operations, and use that custom op in a model that is served with tf serving.

First section shows how to create a custom op using cupy. Then, in second section, we'll show how to
save a model that uses this created op. Lastly, we'll show the steps to build the tf server to serve
this model using tf serving.

### Creating a Tensorflow Custom Op

To build the custom op source code, we'll use docker container. But, to call cupy or cudf functions,
we'll need to install conda to get the libraries and modules into the container.

#### Install Libraries and Modules in the Docker Container
To create a tensorflow custom op, we mainly follow the steps on this [custom op github repo](https://github.com/tensorflow/custom-op), and
added the necessary code and updates on bazel files. Please go through this repo if you'd like to learn more about
the available repos and custom op development. This repo suggests using the available docker containers
for developing a tensorflow custom op. Also, this [repo](https://github.com/tensorflow/custom-op.git) provides
a great custom op example to start with.

First clone this example using;

```
git clone https://github.com/tensorflow/custom-op.git my_op
cd my_op
```

You can find the updated version of this code in this repo as well. We just included them in case you need it.
We'll be using conda to install the necessary libraries and modules. So, you can download the miniconda script file into
this folder for convenience;

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```


There are two main docker containers for developing gpu based custom op. We used the one has ubuntu16. Pull the GPU image using;

```
docker pull tensorflow/tensorflow:custom-op-gpu-ubuntu16
```

You might want to use Docker volumes to map a work_dir from host to the container, so that you can edit files on the host, 
and build with the latest changes in the Docker container. To do so, run the following for GPU

```
docker run --runtime=nvidia --privileged  -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op-gpu-ubuntu16
```

Last step before starting implementing the ops, you want to set up the build environment. 
The custom ops will need to depend on TensorFlow headers and shared library libtensorflow_framework.so, 
which are distributed with TensorFlow official pip package. If you would like to use Bazel to build your ops, 
you might also want to set a few action_envs so that Bazel can find the installed TensorFlow. 
We provide a configure script that does these for you. Simply run ./configure.sh in the docker container and you are good to go.
(Copied directly from [here](https://github.com/tensorflow/custom-op))

```
./configure.sh
```

To test if everything works well, you can build this CPU custom op example follows;

```
bazel build tensorflow_zero_out:python/ops/_zero_out_ops.so
bazel test tensorflow_zero_out:zero_out_ops_py_test
```

For GPU example;

```
bazel build tensorflow_time_two:python/ops/_time_two_ops.so
bazel test tensorflow_time_two:time_two_ops_py_test
```

Now, we need to install the libraries and modules. To do that, we'll use conda. You install the full conda but
we used the miniconda for this PoC. Run the miniconda script;

```
bash Miniconda3-latest-Linux-x86_64.sh
```

When the miniconda is installed, you can run;

```
source ~/.bashrc
```

to run conda in the container.

Then, create a new environment and activate it;

```
conda create -n tf python=3.8
conda activate tf
```

In this PoC, we used cupy. We'll try with cudf as well later. You can install cudf (installs cupy as well) as follows;

```
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=0.16 python=3.8 cudatoolkit=10.1
```

Also install pybind11 using;

```
conda install -c conda-forge pybind11
```

You might want to commit the changes in this container to use it later;

```
docker ps
```

To get the container id (<container_id>) and then run commit;

```
docker commit <container_id> tensorflow/tensorflow:custom-op-gpu-ubuntu16-v1
```

#### Update the Bazel Files

Bazel has two main files, namely WORKSPACE and BUILD. There is only one WORKSPACE file that is located in the
main folder. Then, for every code folder in the project, there is a BUILD file. We'll update these files
to be able to build our code.

First, we need to add the path of libraries into bashrc. Assuming miniconda is installed in root folder, add the following into
bashrc;

```
export LD_LIBRARY_PATH=/root/miniconda3/envs/tf/lib/:$LD_LIBRARY_PATH
```

You should change the path if you installed the miniconda in a different location.

We need to add the path of the libraries in WORKSPACE file. Add the following into WORKSPACE file;

```
new_local_repository(
    name = "conda_python",
    path = "/root/miniconda3/envs/tf/include/python3.7m",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"]),
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "pybind11",
    path = "/root/miniconda3/envs/tf/include/pybind11",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"]),
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "conda_lib",
    path = "/root/miniconda3/envs/tf/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "python37",
    srcs = ["libpython3.7m.so"],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "conda_lib_python",
    path = "/root/miniconda3/envs/tf/lib/python3.7/config-3.7m-x86_64-linux-gnu",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "files",
    srcs = glob(["*.so"]),
    visibility = ["//visibility:public"],
)
""",
)
```

You should change the path

```
/root/miniconda3/envs/tf/
``` 

in these local repos if you installed miniconda in a different location.

In this PoC, we directly updated the code of tensorflow_time_two. You can actually copy and paste
this code and work on the copid files if you like. Now, we'll update the BUILD file of tensorflow_time_two example.

"time_two_ops_gpu" rule in tensorflow_time_two/BUILD file should be updated as follows;

```
cc_library(
    name = "time_two_ops_gpu",
    srcs = ["cc/kernels/time_two.h", "cc/kernels/time_two_kernels.cu.cc"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@conda_python//:headers",
        "@pybind11//:headers",
        "@conda_lib//:python37",
    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
    alwayslink = 1,
    linkopts = ["-lpython3.7m",
                "-lpthread",
                "-lcrypt",
                "-ldl",
                "-lutil",
                "-lrt",
                "-lm",
                ],
    copts = ["-Iexternal/conda_python", 
             "-Iexternal/pybind11", 
             "-pthread", 
             "-std=c++11",
             "-D_GLIBCXX_USE_CXX11_ABI=0",
             "-Wall",
             "-Wno-unused-result",
             "-Wsign-compare", 
             "-march=nocona", 
             "-mtune=haswell", 
             "-ftree-vectorize", 
             "-fPIC", 
             "-fstack-protector-strong",
             "-O3", 
             "-ffunction-sections", 
             "-pipe", 
             "-isystem", 
             "-fuse-linker-plugin", 
             "-ffat-lto-objects", 
             "-flto-partition=none", 
             "-flto", 
             "-DNDEBUG", 
             "-fwrapv"] + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
            
)
```

We suggest copying and pasting the content of BUILD file in our repo into yours since you might miss some of the updates.


#### Update the Code

We'll update the code in tensorflow_time_two/cc/kernels/time_two_kernels.cu.cc to replace the cuda kernel with cupy functions.
Here is the new code;

```
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
```

This code calls run_inference function in vector_add.py file. vector_add.py file 
is located under tensorflow_time_two/python/ops. You can copy and paste it directly
from our repo.

#### Build the Code and Run the Test

Run the following to build the code and run the test;

```
bazel build tensorflow_time_two:python/ops/_time_two_ops.so
bazel test tensorflow_time_two:time_two_ops_py_test
```

The code should build without any error and the tests should pass. If you'd like to see
the messages on the terminal, you can run this as well;

```
bazel-bin/tensorflow_time_two/time_two_ops_py_test
```


### TF Serving

TF serving typically serves models that are developed with the tf functions. It also allows
developers to serve models with custom functions. To do that, tf server has to be built with
the source code of the custom function.

We went through the content in this [tf serving page](https://www.tensorflow.org/tfx/guide/serving)
to learn more about the tf serving. Please run the examples on this page before building the tf server.

#### Install Libraries and Modules in the Docker Container

There is also a docker container for tf server. Same libraries has to be installed in this container.

