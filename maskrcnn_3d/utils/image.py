# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
from maskrcnn_3d.utils import ref

sigma_inp = ref.hmGaussInp
n = sigma_inp * 6 + 1
g_inp = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma_inp * sigma_inp))

#### human3.6数据集，设计的处理函数
def GetTransform(center, scale, rot, res):
    h = scale
    t = np.eye(3)

    t[0, 0] = res / h
    t[1, 1] = res / h
    t[0, 2] = res * (- center[0] / h + 0.5)
    t[1, 2] = res * (- center[1] / h + 0.5)

    if rot != 0:
        rot = -rot
        r = np.eye(3)
        ang = rot * np.math.pi / 180
        s = np.math.sin(ang)
        c = np.math.cos(ang)
        r[0, 0] = c
        r[0, 1] = - s
        r[1, 0] = s
        r[1, 1] = c
        t_ = np.eye(3)
        t_[0, 2] = - res / 2
        t_[1, 2] = - res / 2
        t_inv = np.eye(3)
        t_inv[0, 2] = res / 2
        t_inv[1, 2] = res / 2
        t = np.dot(np.dot(np.dot(t_inv, r), t_), t)
    return t

def Transform(pt, center, scale, rot, res, invert=False):
    pt_ = np.ones(3)
    pt_[0], pt_[1] = pt[0], pt[1]

    t = GetTransform(center, scale, rot, res)
    if invert:
        t = np.linalg.inv(t)
    new_point = np.dot(t, pt_)[:2]
    new_point = new_point.astype(np.int32)
    return new_point

def Crop(img, center, scale, rot, res):  ## res=256
    ht, wd = img.shape[0], img.shape[1]
    tmpImg, newImg = img.copy(), np.zeros((res, res, 3), dtype=np.uint8)

    scaleFactor = scale / res
    if scaleFactor < 2:
        scaleFactor = 1
    else:
        newSize = int(np.math.floor(max(ht, wd) / scaleFactor))  ### 256*256图中，新人体框的最大边框
        newSize_ht = int(np.math.floor(ht / scaleFactor))  #### 256*256图中，人体框的高
        newSize_wd = int(np.math.floor(wd / scaleFactor))  #### 256*256图中，人体框的宽
        if newSize < 2:
            return newImg
        else:
            tmpImg = cv2.resize(tmpImg, (newSize_wd, newSize_ht))  # TODO
            ht, wd = tmpImg.shape[0], tmpImg.shape[1]

    c, s = 1.0 * center / scaleFactor, scale / scaleFactor
    c[0], c[1] = c[1], c[0]
    ul = Transform((0, 0), c, s, 0, res, invert=True)
    br = Transform((res, res), c, s, 0, res, invert=True)

    if scaleFactor >= 2:
        br = br - (br - ul - res)

    pad = int(np.math.ceil((((ul - br) ** 2).sum() ** 0.5) / 2 - (br[0] - ul[0]) / 2))
    if rot != 0:
        ul = ul - pad
        br = br + pad

    old_ = [max(0, ul[0]), min(br[0], ht), max(0, ul[1]), min(br[1], wd)]
    new_ = [max(0, - ul[0]), min(br[0], ht) - ul[0], max(0, - ul[1]), min(br[1], wd) - ul[1]]

    newImg = np.zeros((br[0] - ul[0], br[1] - ul[1], 3), dtype=np.uint8)
    # print 'new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
    try:
        newImg[new_[0]:new_[1], new_[2]:new_[3], :] = tmpImg[old_[0]:old_[1], old_[2]:old_[3], :]
    except:
        # print 'ERROR: new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
        return np.zeros((3, res, res), np.uint8)
    if rot != 0:
        M = cv2.getRotationMatrix2D((newImg.shape[0] / 2, newImg.shape[1] / 2), rot, 1)
        newImg = cv2.warpAffine(newImg, M, (newImg.shape[0], newImg.shape[1]))
        newImg = newImg[pad + 1:-pad + 1, pad + 1:-pad + 1, :].copy()

    if scaleFactor < 2:
        newImg = cv2.resize(newImg, (res, res))

    return newImg

def Gaussian(sigma):
  if sigma == 7:
    return np.array([0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]).reshape(7, 7)
  elif sigma == n:
    return g_inp
  else:
    raise Exception('Gaussian {} Not Implement'.format(sigma))

def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img
    size = 2 * tmpSize + 1
    g = Gaussian(size)
    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]
    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def getTransform3D(center, scale, rot, res):
    h = 1.0 * scale
    t = np.eye(4)
    t[0][0] = res / h
    t[1][1] = res / h
    t[2][2] = res / h
    t[0][3] = res * (- center[0] / h + 0.5)
    t[1][3] = res * (- center[1] / h + 0.5)
    if rot != 0:
        raise Exception('Not Implement')
    return t

def Transform3D(pt, center, scale, rot, res, invert = False):
  pt_ = np.ones(4)
  pt_[0], pt_[1], pt_[2] = pt[0], pt[1], pt[2]
  #print 'c s r res', center, scale, rot, res
  t = getTransform3D(center, scale, rot, res)
  if invert:
    t = np.linalg.inv(t)
  #print 't', t
  #print 'pt_', pt_
  new_point = np.dot(t, pt_)[:3]
  #print 'new_point', new_point
  #if not invert:
  #  new_point = new_point.astype(np.int32)
  return new_point

####### 关于iccv数据集的处理函数 ########
def flip(img):
  return img[:, :, ::-1].copy()  
  
def shuffle_lr(x, shuffle_ref):
  for e in shuffle_ref:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

######  其实是一个反射变换，通过转换前后的3个点来确定输出的图像
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift   ### 原图的中心点
    src[1, :] = center + src_dir + scale_tmp * shift   #### 原图中的某个点
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]   #### 转换后的中心点
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir   ### 转换后图中的某个点

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)   ### sin0=0，cos0=1

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  try:
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
      heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
      g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  except:
    print('center', center)
    print('gx, gy', g_x, g_y)
    print('img_x, img_y', img_x, img_y)
  return heatmap

def adjust_aspect_ratio(s, aspect_ratio, fit_short_side=False):   ### aspect_ratio=1.0
  w, h = s[0], s[1]
  if w > aspect_ratio * h:
    if fit_short_side:
      w = h * aspect_ratio   ### 适应高度值，适当缩短宽度
    else:
      h = w * 1.0 / aspect_ratio   ### 不必适应短边，适当增加高度值
  elif w < aspect_ratio * h:
    if fit_short_side:
      h = w * 1.0 / aspect_ratio
    else:
      w = h * aspect_ratio
  return np.array([w, h])