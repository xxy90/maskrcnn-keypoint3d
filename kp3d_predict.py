import datetime
import os
import time
import sys
sys.path.append('/data/ai/xxy/mask_rcnn')
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from mpl_toolkits.mplot3d import Axes3D




# from coco_utils import get_coco, get_coco_kp
#
# from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# from engine import train_one_epoch, evaluate

#import torch_maskrcnn.utils
import torch_maskrcnn.transforms as T

from PIL import Image
from torch_maskrcnn.plot import plot_poses,Plot_save3d,plot_poses2,plot_poses3
import numpy as np
import cv2
from transforms import functional as F
from maskrcnn_3d.models.detection.keypoint_rcnn_3d2 import keypointrcnn_resnet50_fpn_3d


def load_model():
    model2d=torchvision.models.detection.__dict__['keypointrcnn_resnet50_fpn'](num_classes=2,pretrained=True,box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5)
    model3d = keypointrcnn_resnet50_fpn_3d(num_classes=2, pretrained=True, finetune=False)

    return model2d,model3d

def pad_image(img):
    h,w,c=img.shape
    diff=h-w
    delta = int(abs(diff) / 2)
    if diff==0:
        return img
    elif diff<0:
        pad=np.zeros([delta,w,c])
        #img=np.concatenate([pad,img,pad],axis=0)
        img=torch.nn.functional.interpolate(img[None],[w,w])

        return img
    elif diff>0:
        pad=np.zeros([h,delta,c])
        #img=np.concatenate([pad,img,pad],axis=1)
        img = torch.nn.functional.interpolate(img, [h, h])
        return img


def main(img):
    device = torch.device("cpu:0")
    img_path='./torch_maskrcnn/examples/'+img
    # Data loading code
    model2d,model3d=load_model()
    img_org=cv2.imread(img_path)


    #img_r=cv2.resize(img,(224,224))
    img_o=F.to_tensor(img_org)

    #img_o = pad_image(img_o)
    img_r=F.to_tensor(img_org).to(device)
    imgs=[]
    imgs.append(img_r)

    print("Creating model")



    model2d.eval()
    model3d.eval()
    

    # checkpoint = torch.load(args.resume, map_location='cpu')
    # model_without_ddp.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    

    detect_threshold = 0.7
    keypoint_score_threshold = 2
    edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
             [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
             [6, 8], [8, 9]]
    with torch.no_grad():


        prediction = model2d([img_o.to(device)])
        print(prediction[-1].keys())

        prediction3d = model3d(imgs)
        keypoints = prediction3d[0]['keypoints'].cpu().numpy()
        keypoints3d=prediction3d[0]['keypoints3d'].cpu().numpy()
        
        boxes=prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        keypoints_scores = prediction[0]['keypoints_scores'].cpu().numpy()

        print(keypoints.shape,boxes.shape,scores)
        idx = np.where(scores > detect_threshold)
        keypoints=keypoints[idx]
        idx3d=np.argmax(scores)
        boxes=boxes[idx]

        keypoints3d = np.expand_dims(keypoints3d[idx3d], axis=0)
        print(keypoints.shape, boxes.shape)

        # for j in range(keypoints.shape[0]):
        #     for num in range(16):
        #         if keypoints_scores[j][num] < keypoint_score_threshold:
        #             keypoints[j][num] = [0, 0, 0]
        img_r = img_r.mul(255).permute(1, 2, 0).byte().numpy()

        plot_poses3(img_org, keypoints, boxes,save_name='./torch_maskrcnn/result2/' + img)
        Plot_save3d(edges=edges).add_point_3d(keypoints3d,save_name='./torch_maskrcnn/result/' + img)



if __name__ == "__main__":
    img='01.jpg'
    main(img)
