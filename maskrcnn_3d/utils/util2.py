
import torch.utils.data as data
import numpy as np
from maskrcnn_3d.utils import ref
import torch
from h5py import File
import json
import cv2
from .image import Crop, DrawGaussian, Transform3D
from .image import get_affine_transform, adjust_aspect_ratio, draw_gaussian, affine_transform
import os
from transforms import functional as F

class H36M(data.Dataset):
    def __init__(self, opt, split):
        print('util2==> initializing 3D {} data.'.format(split))
        annot = {}
        f_dir =os.path.join('/data/ai/xxy/mask_rcnn/maskrcnn_3d/data/annotSampleTest.h5')
        f = File(f_dir ,'r')
        tags = ['action', 'bbox', 'camera', 'id', 'joint_2d', 'joint_3d_mono', 'subaction', 'subject', 'istrain']
        # tags=["category_id", "num_keypoints", "is_crowd",   "keypoints",     "keypoints3D", "bbox","image_id"  ]
        if split=='train':
            label = f['istrain'][:].tolist()
            ids = [i for i, x in enumerate(label) if x == 1][0:100000]


        else:
            label = f['istrain'][:].tolist()
            ids = [i for i, x in enumerate(label) if x == 0]


        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        # ids = np.arange(annot['image_id'].shape[0])
        for tag in tags:
            annot[tag] = annot[tag][ids]

        self.num_joints = 16
        self.num_eval_joints = 16
        self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
        self.mean_bone_length = 4296.99233013
        self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
                      [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
                      [6, 8], [8, 9]]
        self.edges_3d = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
                         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
                         [6, 8], [8, 9]]
        self.shuffle_ref = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        self.aspect_ratio = 1.0 * opt.input_w / opt.input_h  #### 1
        self.root = 7
        self.split = split
        self.opt = opt
        self.annot = annot
        self.nSamples = len(self.annot['id'])
        # self.idxs = np.arange(self.nSamples) if split == 'train' else np.arange(0, self.nSamples, 1 if opt.full_test else 10)
        self.idxs = np.arange(self.nSamples)
        print('Loaded 3D {} {} samples'.format(split, len(self.annot['id'])))

    def LoadImage(self, index):
        folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(self.annot['subject'][self.idxs[index]],
                                                                      self.annot['action'][self.idxs[index]],
                                                                      self.annot['subaction'][self.idxs[index]],
                                                                      self.annot['camera'][self.idxs[index]])
        path = '{}/{}/{}_{:06d}.jpg'.format(ref.h36mImgDir, folder, folder, self.annot['id'][self.idxs[index]])
        img = cv2.imread(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img

    
    def round_list(points):
        ## round the points to 2 decimal and transform to list
        r = []
        for i in range(len(points)):
            r.append(round(points[i][0], 2))
            r.append(round(points[i][1], 2))
            r.append(round(points[i][2], 2))

        return r

    def insert_v_point(self,points):
        # append the visbility to keypoint i.e [30,30] to[30,30,v],
        # v=[0,1,2],0 is not labeled and not visible ,1 is labeled and not visible ,2 is visible

        v = []

        for i in range(len(points)):
            tmp=[]
            if points[i].all() == 0:
                tmp.append(round(points[i][0], 2))
                tmp.append(round(points[i][1], 2))

                tmp.append(0)
            else:
                tmp.append(round(points[i][0], 2))
                tmp.append(round(points[i][1], 2))
                tmp.append(2)

            v.append(tmp)


        return np.array(v)
    def insert_v_point2(self,points):
        # append the visbility to keypoint i.e [30,30] to[30,30,v],
        # v=[0,1,2],0 is not labeled and not visible ,1 is labeled and not visible ,2 is visible
        v = []
        tmp = []
        for i in range(len(points)):

            if points[i].all() == 0:
                tmp.append(round(points[i][0], 2))
                tmp.append(round(points[i][1], 2))

                tmp.append(0)
            else:
                tmp.append(round(points[i][0], 2))
                tmp.append(round(points[i][1], 2))
                tmp.append(2)



        return np.array(tmp)

    def extract_box(self ,points):
        ## get the bbox from keypoints

        min_x = int(min([points[i][0] for i in range(16)]))
        max_x = int(max([points[i][0] for i in range(16)]))
        min_y = int(min([points[i][1] for i in range(16)]))
        max_y = int(max([points[i][1] for i in range(16)]))
        x0=max(0,min_x-10)
        y0=max(0,min_y-10)
        x1=min(224,max_x+10)
        y1=min(224,min_y+2)


        #return np.array([min_x, min_y, max_x, max_y])
        return np.array([x0,y0,x1,y1])

    def GetPartInfo(self, index):
        pts = np.array(self.annot['joint_2d'][self.idxs[index]], np.float32).copy()
        pts_3d_mono = np.array(self.annot['joint_3d_mono'][self.idxs[index]], np.float32).copy()  #### 开始的3d坐标是以第7个点为中心，臀部的点
        pts_3d = np.array(self.annot['joint_3d_mono'][self.idxs[index]], np.float32)
        c = np.array([112 ,112], dtype=np.float32)
        # c = np.ones(2) * ref.h36mImgSize / 2
        s = np.array(ref.h36mImgSize * 1.0)


        min_id=np.argmin(pts_3d[:,2])
        pts_3d = pts_3d - pts_3d[self.root]  #### transform the minus depth to[0,x]
        ######  下面的操作主要是将 以第8个点为中心 转化成 以左上角为中心重新计算每个点的3d坐标
        s2d, s3d = 0, 0
        for e in ref.edges:
            s2d += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
            s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
        scale = s2d / s3d

        for j in range(ref.nJoints):   ##### pts_3d 是处理后，第1和2的维度是，跟2d相似的坐标，第3维度表示深度（参考第7个点，深度为112）
            pts_3d[j, 0] = pts_3d[j, 0] * scale + pts[self.root, 0]
            pts_3d[j, 1] = pts_3d[j, 1] * scale + pts[self.root, 1]
            pts_3d[j, 2] = pts_3d[j, 2] * scale + ref.h36mImgSize / 2
            #if pts_3d[j,2]<0:

                #print('the negative depth is {}_th pts_3d{}'.format(j,pts_3d[j,2]))



        return pts, c, s, pts_3d, pts_3d_mono

    def __getitem__(self, index):
        if index == 0 :
            self.idxs = np.random.choice(self.nSamples, self.nSamples, replace=False)  ### 随机获取索引
        img = self.LoadImage(index)
        pts, c, s, pts_3d, pts_3d_mono = self.GetPartInfo(index)   #### 都是float32的类型



        ##### pts_3d是根据pts_3d_mono转化而来；第1和2维度坐标，其实对应pts的坐标(细微误差)；第3维度设置骨盆点深度为0
        # pts_3d[7] = (pts_3d[12] + pts_3d[13]) / 2
        # pts_3d[:, 2] = pts_3d[:, 2] - pts_3d[6, 2]

        inp = Crop(img, c, s, 0, ref.inputRes)
        inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)



        #### 返回的值
        s = np.array([s, s])


        #pts_3d_mono[:,2]=pts_3d_mono[:,2]/224
        pts_3d[:, 2] = (pts_3d[:, 2] +100)/10
        pts_2d=self.insert_v_point(pts)
        pts_2d=pts_2d[np.newaxis,:]
        pts_3d=pts_3d[np.newaxis,:]
        boxes =self.extract_box(pts)
        #masks =self.convert_keypoints_to_mask(pts_2d)

        targets = {}
        targets['image_id'] = np.array(self.annot['id'])
        targets['is_crowd'] = np.zeros([1])
        targets["boxes"] = boxes
        targets["labels"] = np.ones([1])
        #targets["masks"] = masks
        # targets["outMap"] = outMap
        # targets["reg_target"] = reg_target
        # targets["reg_ind"] = reg_ind
        # targets["reg_mask"] = reg_mask


        targets['keypoints']=pts_2d
        targets['keypoints3d'] = pts_3d






        return img,targets

    def convert_keypoints_to_mask2(self ,pt):
        masks =np.zeros([ref.outputRes ,ref.outputRes] ,dtype=np.float32)

        print('type is ',type(pt))
        masks[np.trunc(pt[: ,1]).tolist() ,np.trunc(pt[: ,0]).tolist() ] =1

        return masks

    def convert_keypoints_to_mask(self ,pt):
        masks =np.zeros([ref.inputRes ,ref.inputRes] ,dtype=np.float32)
        for i in pt[:,1]:
            for j in pt[:,0]:
                
                masks[int(i),int(j)]=1



        return masks


     





    def __len__(self):
        return self.nSamples

    def convert_eval_format(self, pred_h36m):  #### pred是相对于骨盆的单位坐标，【16,3】
        sum_bone_length = self._get_bone_length(pred_h36m)  #### 根据所有骨骼点连线的长度
        mean_bone_length = self.mean_bone_length
        pred_h36m = pred_h36m * mean_bone_length / sum_bone_length
        return pred_h36m

    def _get_bone_length(self, pts):
        sum_bone_length = 0
        for e in ref.edges:  #### ref.edges是骨骼点的链接
            sum_bone_length += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
        return sum_bone_length











