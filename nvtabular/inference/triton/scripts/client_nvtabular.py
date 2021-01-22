# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "nvtabular"
shape = [2, 5]

with httpclient.InferenceServerClient("localhost:8000") as client:
    cont_data = np.random.rand(*shape).astype(np.float32)

    #in0 = np.arange(start=1001, stop=1003, dtype=np.int32)
    in0 = np.array([[4, 89], [13, 1034]])
    in0 = np.expand_dims(in0, axis=0)
    in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
    #in0n[0][0] = 'Onur Yilmaz'
    
    cat_data = in0n.reshape(in0.shape)
    cat_data = cat_data[0]
    print("")
    print(cat_data)
    print("")
    
    inputs = [
        httpclient.InferInput("CONT", cont_data.shape,
                              np_to_triton_dtype(cont_data.dtype)),
        httpclient.InferInput("CAT", cat_data.shape, "BYTES")
    ]

    inputs[0].set_data_from_numpy(cont_data, binary_data=False)
    inputs[1].set_data_from_numpy(cat_data, binary_data=True)
    print(inputs[1]._get_binary_data())
    
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print("CONT ({}), CAT ({}) = OUTPUT0 ({})".format(
        cont_data, cat_data, response.as_numpy("OUTPUT0")))
