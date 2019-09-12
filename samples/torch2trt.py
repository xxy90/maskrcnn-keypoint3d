from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import torch
from torch import nn
import sys, os
import model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

def populate_network(network, path):
    net=model.Net()

    #net.load_state_dict(torch.load(path))
    dicts=torch.load(path)
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    tensor=input_tensor
    dict={k: v for k, v in dicts.items()}
    list_keys=list(dict.keys())
    print('all keys are',list_keys)
    for name,module in net.named_modules():
        print('current layer is ',name)
        if  isinstance(module,nn.Conv2d):
            w = module.weight.data.cpu().numpy()
            b=module.bias.data.cpu().numpy()
            shape=w.shape
            print(shape)
            if name in list_keys[0]:
                conv=network.add_convolution(input=input_tensor,
                                        num_output_maps=shape[0], kernel_shape=(shape[2], shape[3]), kernel=w,bias=b)
            else:
                conv = network.add_convolution(input=tensor.get_output(0),
                                               num_output_maps=shape[0], kernel_shape=(shape[2], shape[3]), kernel=w,
                                               bias=b)
            conv.stride = (1, 1)
            pool = network.add_pooling(input=conv.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
            pool.stride = (2, 2)
            tensor=pool
        elif isinstance(module,nn.MaxPool2d):
            pool = network.add_pooling(input=tensor.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
            pool.stride = (2, 2)
            pool.get_output(0).name = name
            tensor=pool
        elif isinstance(module,nn.Linear):
            w = module.weight.data.cpu().numpy()
            b = module.bias.data.cpu().numpy()
            shape = w.shape
            print(shape)

            ipt=tensor.get_output(0)
            fc = network.add_fully_connected(input=ipt, num_outputs=shape[0], kernel=w, bias=b)
            fc.get_output(0).name = name
            tensor=fc
        elif isinstance(module,nn.ReLU):
            relu= network.add_activation(input=tensor.get_output(0), type=trt.ActivationType.RELU)
            relu.get_output(0).name=name
            tensor=relu

    tensor.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=tensor.get_output(0))
def build_engine(path):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 1 << 30
        # Populate the network using weights from the PyTorch model.
        populate_network(network, path)
        # Build and return an engine.
        engine=builder.build_cuda_engine(network)
        with open('./models/mnist_new.trt', "wb") as f:
            f.write(engine.serialize())
        return engine

if __name__=='__main__':
    path='./models/mnist_4.pth'
    build_engine(path)
