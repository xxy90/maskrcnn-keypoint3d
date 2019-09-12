import torch
from torch import nn
from torchvision.transforms import functional as F
import os
import cv2
import torchvision.models as models
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from uff import from_tensorflow_frozen_model
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import onnx
import pycuda.driver as cuda
#import onnx_caffe2.backend
import onnxruntime
from torch.autograd import Variable

TRT_LOGGER = trt.Logger()
os.environ['CUDA_VISIBLE_DEVICES']='4'
def main1():
    device=torch.device('cuda:0')
    pth='04.jpg'
    model = models.resnet50(pretrained=True).to(device)
    img_path = os.path.join('./examples/' + pth)
    img = cv2.imread(img_path)
    img = img[np.newaxis, :]
    img=np.transpose(img,(0,3,1,2)).astype(np.float32)
    input=torch.Tensor(img).to(device)

    nodes=[k for k,v in model.named_modules()]
    output_node=nodes[-1]
    print(output_node)
    torch.onnx.export(model,(input),'./models/resnetnew.onnx',output_names=[output_node])
    start1 = time.time()



    sess=onnxruntime.InferenceSession('./encodings/resnet_onnx.onnx')
    inname = [input.name for input in sess.get_inputs()][0]
    outname = [output.name for output in sess.get_outputs()]
    img=np.array(img,dtype=np.float32)
    output = sess.run(outname, {inname: img})

    print('processing time1 is', time.time() - start1,np.argmax(output))

    return output
def get_engine(onnx_file_path,engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        # print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    else:
        return build_engine()
def load_normalized_test_case(path, pagelocked_buffer):
    test_case_path = os.path.join('./examples/' + path)
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)

def main2():
    pth='04.jpg'

    img_path = os.path.join('./examples/' + pth)
    img = cv2.imread(img_path)
    img = img[np.newaxis, :]
    img = torch.Tensor(img)
    img = img.permute(0, 3, 1, 2)#.cuda()


    model1 = models.resnet50(pretrained=True)#.cuda()
    start2 = time.time()
    output1 = model1(img)
    print('processing time2 is', time.time() - start2,torch.argmax(output1))
    #print('class is ',output1)
    return output1

def main3():
    device = torch.device('cuda:0')
    path = '04.jpg'
    
    
    onnx_file_path = './models/ResNet50.onnx'
    engine_file_path = './models/ResNet50.trt'


    with get_engine(onnx_file_path, engine_file_path, ) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            load_normalized_test_case(path,inputs[0].host)
            start1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    output = trt_outputs


    print('processing time3 is', time.time() - start1, np.argmax(output))

    return output

def main4():
    device = torch.device('cuda:0')
    path = '04.jpg'
    onnx_file_path = './models/resnetnew.onnx'
    engine_file_path='./models/resnetnew.trt'
    
    start1 = time.time()
    

    model = models.resnet50(pretrained=True).to(device)
    img_path = os.path.join('./examples/' + path)
    img = cv2.imread(img_path)
    img = img[np.newaxis, :]
    img=np.transpose(img,(0,3,1,2)).astype(np.float32)
    input=torch.Tensor(img).to(device)

    nodes=[k for k,v in model.named_modules()]
    output_node=nodes[-1]
    print(output_node)
    torch.onnx.export(model,input,onnx_file_path,output_names=['output'])
    

    #with build_engine(model ) as engine:
    with get_engine(onnx_file_path, engine_file_path, ) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            load_normalized_test_case(path,inputs[0].host)
            start1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    output = trt_outputs


    print('processing time4 is', time.time() - start1, np.argmax(output))

    return output




if __name__=='__main__':


    



    main2()
    main4()


    
