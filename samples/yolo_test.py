import cv2
import numpy as np
import tensorrt as trt
import common_utils
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import time
engine_file_path='./models/yolov3.trt'
import sys
sys.path.append('/home/ai/Person_Pose_SVN')
TRT_LOGGER = trt.Logger()
from django_pose.YOLOv3 import YOLOv3

def init_multi():
    yolo=YOLOv3()
    f=open(engine_file_path, "rb")
    runtime=trt.Runtime(TRT_LOGGER)
    engine= runtime.deserialize_cuda_engine(f.read())
    context=engine.create_execution_context()

    return yolo,engine,context
def process_multi(img_path, yolo, engine,context):
    start_tf=time.time()
    image=cv2.imread(img_path)

    img_persons_new, boxes_new, trans=yolo.process_image(image)
    start_tf = time.time()
    img_persons_new, boxes_new, trans = yolo.process_image(image)
    img=draw(image,boxes_new)
    cv2.imwrite('img.jpg',img)
    print('process time for tf is',time.time()-start_tf)

    start_trt=time.time()



    input_resolution_yolov3_HW = (608, 608)
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    image_raw, image = preprocessor.process(img_path)
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    # Do inference with TensorRT
    trt_outputs = []

    inputs, outputs, bindings, stream = common_utils.allocate_buffers(engine)
    inputs[0].host = image
    trt_outputs = common_utils.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    start_trt = time.time()
    trt_outputs = common_utils.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    print('process time for trt is', time.time() - start_trt)
    post_trt=time.time()
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                          # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                           # A list of 9 two-dimensional tuples for the YOLO anchors
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,  # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.5,
                          # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    obj_detected_img.save('out_boxes.png', 'PNG')
    print('process time for trt post is', time.time() - post_trt)
def draw(img,boxes):
    for box in boxes:

        x1,y1,x2,y2=box
        cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,0),2)
    return img
def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def load_normalized_test_case(path, pagelocked_buffer):

    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = cv2.imread(path)
    img_0 = cv2.resize(img, (656, 368))
    img = img_0.ravel()
    #img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)

    return img_0

if __name__=='__main__':
    yolo, engine, context=init_multi()
    img_path='./images/11.jpg'
    process_multi(img_path, yolo, engine, context)