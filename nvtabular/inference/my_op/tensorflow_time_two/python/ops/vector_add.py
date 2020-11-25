import cupy as cp

def add(a, b):
  return a + b

class cupy_holder(object):
    pass


def run_inference(cai_in, cai_out):
  cai_in['strides'] = None
  cai_in['descr'] = [('', '<i8')]
  cai_in["typestr"] = "<i8"; 
  cai_in["version"] = 2; 

  cai_out['strides'] = None
  cai_out['descr'] = [('', '<i8')]
  cai_out["typestr"] = "<i8"; 
  cai_out["version"] = 2; 

  holder_in = cupy_holder()
  holder_in.__cuda_array_interface__ = cai_in
  data_in = cp.array(holder_in, copy=False)
  
  holder_out = cupy_holder()
  holder_out.__cuda_array_interface__ = cai_out
  data_out = cp.array(holder_out, copy=False)
  
  data_res = data_in * 2

  assign = cp.ElementwiseKernel('int64 x', 'int64 z','z = x','assign')
  assign(data_res, data_out)

  #print(data_in)
  #print(data_out)
