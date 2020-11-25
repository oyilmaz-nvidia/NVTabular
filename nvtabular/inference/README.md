## NVTabular | Inference documentation

This document is prepared to help you to create a custom op for tensorflow that uses python 
modules and frameworks for GPU operations, and use that custom op in model that is served with tf serving.


### Getting Started

[RAPIDS Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) library.
NVTabular is available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

The beta (0.3) container is currently available. You can pull the container by running the following command:

```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE nvcr.io/nvidia/nvtabular:0.2 /bin/bash
```


#### Conda

NVTabular can be installed with Anaconda from the ```nvidia``` channel:

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=10.2
```

