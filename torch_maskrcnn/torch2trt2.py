import torch
from torch.autograd import Variable
import tensorrt as trt

from tensorrt import OnnxParser as onnxparser
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm, tqdm_notebook



#import calib as calibrator

args = ArgumentParser().parse_args()
args.input_size = 224
args.input_channel = 3
args.fc_num = 5
args.batch_size = 4


args.onnx_model_name = "./models/resnetnew.onnx"
args.trt_model_name = "./models/resnetnew.trt"




def infer(context, input_img, output_size, batch_size):
    # Load engine
    engine = context.get_engine()
    assert (engine.get_nb_bindings() == 2)
    # Convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Create output array to receive data
    output = np.empty(output_size, dtype=np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.nbytes)
    d_output = cuda.mem_alloc(batch_size * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # Return predictions
    return output





def onnx_infer():
    apex = onnxparser.create_onnxconfig()
    apex.set_model_file_name(args.onnx_model_name)
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)

    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)
    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    engine = builder.build_cuda_engine(trt_network)
    with open('./models/resnetnew.trt', "wb") as f:
        f.write(engine.serialize())
    context = engine.create_execution_context()

    print ("Start ONNX Test...")



def onnx_to_int8():
    apex = onnxparser.create_onnxconfig()

    apex.set_model_file_name(args.onnx_model_name)
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)
    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)

    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    # calibration_files = create_calibration_dataset()
    batchstream = calibrator.ImageBatchStream(args)
    int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    builder.set_max_workspace_size(1 << 20)
    builder.set_int8_calibrator(int8_calibrator)
    builder.set_int8_mode(True)
    engine = builder.build_cuda_engine(trt_network)
    modelstream = engine.serialize()
    trt.utils.write_engine_to_file(args.trt_model_name, modelstream)
    engine.destroy()
    builder.destroy()


def trt_infer():
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    engine = trt.utils.load_engine(G_LOGGER, args.trt_model_name)
    context = engine.create_execution_context()

    print ("Start TensorRT Test...")



if __name__ == '__main__':


    onnx_infer()

    if not os.path.exists(args.trt_model_name):
        onnx_to_int8()

    #trt_infer()