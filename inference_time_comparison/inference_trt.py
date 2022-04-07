import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import time


# Load TensorRT file
tensorrt_file_name = "mobilenet_v3_small.plan"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

with open(tensorrt_file_name, 'rb') as f:
    engine_data = f.read()

engine = trt_runtime.deserialize_cuda_engine(engine_data)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)


    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))

context = engine.create_execution_context()

# Convert torch.tensor to readable TensorRT data format, and allocate it to input.
input_ids = torch.randn(1, 3, 224, 224)

tensor_input = [input_ids]
hosts = [input_.host for input_ in inputs]
trt_types = [trt.int32]

for tensor_array, host, trt_type in zip(tensor_input, hosts, trt_types):
    numpy_array = np.asarray(tensor_array).astype(trt.nptype(trt_type)).ravel()
    np.copyto(host, numpy_array)

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream.
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


start_time = time.time()
for i in range(1000):
    trt_outputs = do_inference(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream
    )
end_time = time.time()

print('Total Inference Time:', end_time - start_time)
print('Inference Time per 1 Iteration:', (end_time - start_time) / 1000)