import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from torchvision import transforms

# from coco_utils import get_coco, get_coco_kp
#
# from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# from engine import train_one_epoch, evaluate

import utils
import transforms as T

from PIL import Image
from plot import plot_poses
import numpy as np
import cv2
from torchvision.transforms import functional as F
from bn_fusion import fuse_bn_recursively

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device("cuda:0")

    # Data loading code
    print("Loading data")
    pth='b.jpg'
    img_path=os.path.join('./examples/'+pth)
    img=cv2.imread(img_path)
    img=F.to_tensor(img)
    print("tensor size is ",img.size())
    num_classes=2
    model = torchvision.models.detection.__dict__['keypointrcnn_resnet50_fpn'](num_classes=num_classes,
                                             pretrained=True,box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5)
    model.to(device)
    im=[img.to(device)]
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # model_without_ddp.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model.eval()

    detect_threshold = 0.7
    keypoint_score_threshold = 2
    start=time.time()
    with torch.no_grad():


        prediction = model(im)
        keypoints = prediction[0]['keypoints'].cpu().numpy()
        boxes=prediction[0]['boxes'].cpu().numpy()
        print('the shape of boxes is ',boxes.shape)
        scores = prediction[0]['scores'].cpu().numpy()
        keypoints_scores = prediction[0]['keypoints_scores'].cpu().numpy()
        idx = np.where(scores > detect_threshold)
        boxes=boxes[idx]
        keypoints = keypoints[idx]
        print('the shape of keypoints is ',keypoints.shape)
        keypoints_scores = keypoints_scores[idx]
        for j in range(keypoints.shape[0]):
            for num in range(17):
                if keypoints_scores[j][num] < keypoint_score_threshold:
                    keypoints[j][num] = [0, 0, 0]
        #img = img.mul(255).permute(1, 2, 0).byte().numpy()
        

        #plot_poses(img, keypoints,boxes, save_name='./result2/' + pth)
    print('processing time is :',time.time()-start)


def main2():
    device = torch.device("cuda:0")

    # Data loading code
    print("Loading data")
    pth = 'b.jpg'
    img_path = os.path.join('./examples/' + pth)
    img = cv2.imread(img_path)
    img = F.to_tensor(img)
    print("tensor size is ", img.size())
    num_classes = 2
    model = torchvision.models.detection.__dict__['keypointrcnn_resnet50_fpn'](num_classes=num_classes,
                                                                               pretrained=True, box_fg_iou_thresh=0.5,
                                                                          box_bg_iou_thresh=0.5)



    model=torch.nn.DataParallel(model,device_ids=[0]).cuda()
    #model=fuse_bn_recursively(model)
    model=model.cuda()
    im = [img.to(device)]




    model.eval()

    detect_threshold = 0.7
    keypoint_score_threshold = 2
    start = time.time()
    with torch.no_grad():

        prediction = model(im)
        keypoints = prediction[0]['keypoints'].cpu().numpy()
        boxes = prediction[0]['boxes'].cpu().numpy()
        print('the shape of boxes is ', boxes.shape)
        scores = prediction[0]['scores'].cpu().numpy()
        keypoints_scores = prediction[0]['keypoints_scores'].cpu().numpy()
        idx = np.where(scores > detect_threshold)
        boxes = boxes[idx]
        keypoints = keypoints[idx]
        print('the shape of keypoints is ', keypoints.shape)
        keypoints_scores = keypoints_scores[idx]
        for j in range(keypoints.shape[0]):
            for num in range(17):
                if keypoints_scores[j][num] < keypoint_score_threshold:
                    keypoints[j][num] = [0, 0, 0]
        #img = img.mul(255).permute(1, 2, 0).byte().numpy()

        #plot_poses(img, keypoints, boxes, save_name='./result2/' + pth)
    print('processing time paralell is :', time.time() - start)


if __name__ == "__main__":
    #main()
    main2()
