import time
import numpy as np
import tensorrt as trt
from PIL import Image
import sys
import tensorflow as tf
import common_utils
import onnx
import pycuda.driver as cuda
import os


sys.path.append('/data/ai/JF/pose_estimation/multi_pose_estimator')
import cv2
import ast
import numpy as np
import os
import shutil
import time
import PIL.Image as Image
import matplotlib.pyplot as plt

from tf_pose import common
from matplotlib.backends.backend_agg import FigureCanvasAgg
from estimator import TfPoseEstimator,Human,BodyPart,PoseEstimator
from tf_pose.networks import get_graph_path
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
sys.path.append('/data/ai/JF/pose_estimation/multi_pose_estimator/tf_pose/pafprocess')
import pafprocess

from np_Smoother import Smoother
from np_conv import pooling

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
engine_file_path='/home/ai/tensorrt_tar/TensorRT-5.1.5.0/data/models/graph_opt.trt'
TRT_LOGGER = trt.Logger()
def load_normalized_test_case(path, pagelocked_buffer):

    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = cv2.imread(path)
    img_0 = cv2.resize(img, (656, 368))
    img = img_0.ravel()
    #img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)

    return img_0




def init_multi():
    model_path = get_graph_path(model_name='mobilenet_thin')  #'mobilenet_thin/mobilenet_v2_large/mobilenet_v2_small/cmu'
    model_1 = TfPoseEstimator(model_path, target_size=(432, 368), device='0')
    model_2 = TfPoseEstimator(model_path, target_size=(656, 368), device='0')
    f=open(engine_file_path, "rb")
    runtime=trt.Runtime(TRT_LOGGER)
    engine= runtime.deserialize_cuda_engine(f.read())
    context=engine.create_execution_context()

    return model_1, model_2,engine,context

def process_multi(img_path, model_1, model_2,engine,context):
    start_tf=time.time()
    image = common.read_imgfile(img_path, None, None)

    filename = os.path.split(img_path)[1].split('.')[0]
    #scales = ast.literal_eval(node_or_string='[None]')
    humans_ = model_2.inference(image, resize_to_default = True, upsample_size = 4.0)
    if len(humans_) >= 2:
        humans = humans_
    else:
        humans = model_1.inference(image, resize_to_default = True, upsample_size = 4.0)
    image = model_1.draw_humans(image, humans, imgcopy=False)
    path_save_2d =  './results/' + filename + '_2d.jpg'
    cv2.imwrite(path_save_2d, image)

    print('process time for tf is',time.time()-start_tf)

    start_trt=time.time()

    inputs, outputs, bindings, stream = common_utils.allocate_buffers(engine)
    #with engine.create_execution_context() as context:

    img=load_normalized_test_case(img_path, inputs[0].host)
    output=common_utils.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    peaks, heatMat_up, pafMat_up=post_process(output)
    humans = estimate_paf(peaks[0], heatMat_up[0], pafMat_up[0])
    image = draw_points(img, humans)
    path_save_2d = './results/' + filename + '_2d_trt.jpg'
    cv2.imwrite(path_save_2d, image)
    print('process time for trt is', time.time() - start_trt)

def post_process(output):
    ## input is (3,656,368),output is (1,46,82,57)
    output=np.reshape(output,(1,46,82,57))



    heatMat = output[:, :, :, :19]
    pafMat = output[:, :, :, 19:]

    #upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
    sw,sh=int(368/2),int(656/2)

    heatMat_up=[]
    pafMat_up=[]

    for i in range(heatMat.shape[0]):

        heatMat_up.append(cv2.resize(heatMat[i],(sh,sw)))
        pafMat_up.append(cv2.resize(pafMat[i], (sh, sw)))
    heatMat_up=np.array(heatMat_up)
    pafMat_up=np.array(pafMat_up)
    ss=time.time()
    smoother = Smoother({'data': heatMat_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()
    print('the shape of guassian_heatmat is',gaussian_heatMat.shape)

    max_pooled = pooling(gaussian_heatMat,  window_size=3,  padding='SAME')
    print('the shape of max_pooled is ',max_pooled.shape)
    peaks = np.where(np.equal(gaussian_heatMat, max_pooled), gaussian_heatMat,
                                 np.zeros_like(gaussian_heatMat))

    heatMat = pafMat = None



    print('the shape of p,h,f',peaks.shape, heatMat_up.shape, pafMat_up.shape,time.time()-ss)
    peaks, heatMat_up,pafMat_up=peaks.astype('float32'), heatMat_up.astype('float32'),pafMat_up.astype('float32')
    return peaks, heatMat_up, pafMat_up






def get_engine(engine_file_path):

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.

        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        raise Exception('engine is no existed !')
def draw_points(npimg, humans):
    print('this is draw_points',humans)

    image_h, image_w = npimg.shape[:2]
    centers = {}
    index = 0
    for human in humans:

        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = [(int(x * image_w + 0.5), int(y * image_h + 0.5)) for x, y in pose_2d_mpii]
            center = (body_part[0] , body_part[1] )
            print(center)
            centers[i]=center
            cv2.circle(npimg, center, 3, (255,0,0), thickness=3, lineType=8, shift=0)

            cv2.putText(npimg, str(index), center, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        index += 1

        # # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

    return npimg

def estimate_paf(peaks, heat_mat, paf_mat):
    # sess=tf.Session()
    # peaks=peaks.eval(session=sess)
    # heat_mat=heat_mat.eval(session=sess)
    # paf_mat=paf_mat.eval(session=sess)
    pafprocess.process_paf(peaks, heat_mat, paf_mat)
    print('human num is',pafprocess.get_num_humans())
    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human([])
        is_added = False

        for part_idx in range(18):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                pafprocess.get_part_score(c_idx)
            )

        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)

    return humans


if __name__=='__main__':
    model_1, model_2, engine, context=init_multi()
    img_path='./image/11.jpg'
    process_multi(img_path, model_1, model_2, engine, context)
