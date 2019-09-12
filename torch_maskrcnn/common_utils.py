import pycuda.driver as cuda
import tensorrt as trt
from torch import nn
TRT_LOGGER = trt.Logger()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "fc"
    DTYPE = trt.float32


def populate_network(network, net):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    tensor = input_tensor

    k = 0
    for name, module in net.named_modules():
        k += 1
        print('current layer is ', name)
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.cpu().numpy()
            b = module.bias.data.cpu().numpy()
            shape = w.shape
            print(shape)
            if k == 1:
                conv = network.add_convolution(input=input_tensor,
                                               num_output_maps=shape[0], kernel_shape=(shape[2], shape[3]), kernel=w,
                                               bias=b)
            else:
                conv = network.add_convolution(input=tensor.get_output(0),
                                               num_output_maps=shape[0], kernel_shape=(shape[2], shape[3]), kernel=w,
                                               bias=b)
            conv.get_output(0).name = name
            tensor = conv
        elif isinstance(module, nn.MaxPool2d):
            pool = network.add_pooling(input=tensor.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
            pool.stride = (2, 2)
            pool.get_output(0).name = name
            tensor = pool
        elif isinstance(module, nn.Linear):
            w = module.weight.data.cpu().numpy()
            b = module.bias.data.cpu().numpy()
            shape = w.shape
            print(shape)

            ipt = tensor.get_output(0)
            fc = network.add_fully_connected(input=ipt, num_outputs=shape[0], kernel=w, bias=b)
            fc.get_output(0).name = name
            tensor = fc
        elif isinstance(module, nn.ReLU):
            relu = network.add_activation(input=tensor.get_output(0), type=trt.ActivationType.RELU)
            relu.get_output(0).name = name
            tensor = relu

    tensor.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=tensor.get_output(0))


def build_engine(model):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 1 << 30
        # Populate the network using weights from the PyTorch model.
        populate_network(network, model)
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
        with open('./models/mnist_new.trt', "wb") as f:
            f.write(engine.serialize())
        return engine


def allocate_buffers1(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference1(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()