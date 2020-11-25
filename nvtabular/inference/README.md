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
(Copied directly from [here]([custom op github repo](https://github.com/tensorflow/custom-op))

```
./configure.sh
```

To test if everything works well, you can build this CPU custome op example follows;

```
bazel build tensorflow_zero_out:python/ops/_zero_out_ops.so
bazel test tensorflow_zero_out:zero_out_ops_py_test
```

For GPU example;

```
bazel build tensorflow_time_two:python/ops/_time_two_ops.so
bazel test tensorflow_time_two:time_two_ops_py_test
```


