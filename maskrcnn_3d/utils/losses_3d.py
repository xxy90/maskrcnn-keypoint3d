import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)  #### feat按ind的行的索引，进行值的获取
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    # print(feat.shape)   #### torch.size([1, 16, 1])
    return feat


'''
def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
'''


def _tranpose_and_gather_scalar(feat, ind):  ### ind是【16,1】
    feat = feat.permute(0, 2, 3, 1).contiguous()  ### 【1,64,64,16】
    # feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = feat.view(feat.size(0), -1, 1)  ### 换成一列
    feat = _gather_feat(feat, ind)
    return feat


def reg_loss(regr, gt_regr, mask):   #### regr: [1,16,1]   gt_regr:[16,1]   mask:[16,1]
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr * mask.float()
    gt_regr = gt_regr * mask.float()

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    # regr_loss = nn.functional.mse_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_scalar(output, ind)   #### torch.size([1,16,1])
        loss = reg_loss(pred, target, mask)
        return loss


class FusionLoss(nn.Module):
    def __init__(self, reg_weight, var_weight):
        super(FusionLoss, self).__init__()
        self.reg_weight = reg_weight
        self.var_weight = var_weight

    ##### (output[-1]['depth'], batch['reg_mask'], batch['reg_ind'], batch['reg_target'], gt_2d)
    #### mask   [1,1,1,....1] (16, 1)
    #### ind 回归的一个值，根据2d骨骼点位置，计算的一个值
    ###  target 也是一个回归值，计算z轴的值与224的比值  [16,1]
    def forward(self, output, mask, ind, target, gt_2d):
        pred = _tranpose_and_gather_scalar(output, ind)
        loss = torch.FloatTensor(1)[0] * 0
        if self.reg_weight > 0:
            loss += self.reg_weight * reg_loss(pred, target, mask)
        if self.var_weight > 0:
            # print('1: ',pred)
            # print('2: ',target)
            # print('3: ',mask)
            # print('4: ',gt_2d)
            loss += VarLoss(self.var_weight)(pred, target, mask, gt_2d)[0]  # target for visibility
        ##### return loss.to(self.device, non_blocking=True)
        return loss.cuda()


class VarLoss(Function):
    def __init__(self, var_weight):
        super(VarLoss, self).__init__()
        ### self.device = device
        self.var_weight = var_weight
        self.skeleton_idx = [[[0, 1], [1, 2],
                              [3, 4], [4, 5]],    #### 下肢：主要包括左腿和右腿
                             [[10, 11], [11, 12],
                              [13, 14], [14, 15]],  #### 上肢：主要包括左臂和右臂
                             [[2, 6], [3, 6]],  ### 骨盆处的骨架
                             [[12, 8], [13, 8]]]  ### 肩膀处的骨架
        #### 小腿与大腿长，权重大；同样小臂比大臂长，权重大
        self.skeleton_weight = [[1.0085885098415446, 1,
                                 1, 1.0085885098415446],
                                [1.1375361376887123, 1,
                                 1, 1.1375361376887123],
                                [1, 1],
                                [1, 1]]

    #### input：torch.size([1,16,1])；visible：[1,16,1]； mask：[1,16,1]  gt_2d：[1,16,2]
    def forward(self, input, visible, mask, gt_2d):
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        batch_size = input.size(0)
        output = torch.FloatTensor(1) * 0
        for t in range(batch_size):
            if mask[t].sum() == 0:  ## mask is the mask for supervised depth
                # xy[t] = 2.0 * xy[t] / ref.outputRes - 1
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                                    (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    loss = 0
                    for j in range(N):
                        if l[j] > 0:
                            loss += (l[j] - E) ** 2 / 2. / num
                    output += loss
        output = self.var_weight * output / batch_size
        self.save_for_backward(input, visible, mask, gt_2d)
        ### output = output.cuda(self.device, non_blocking=True)
        # output = output.cuda()
        output = output
        return output

    def backward(self, grad_output):
        input, visible, mask, gt_2d = self.saved_tensors
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        grad_input = torch.zeros(input.size())
        batch_size = input.size(0)
        for t in range(batch_size):
            if mask[t].sum() == 0:  # mask is the mask for supervised depth
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                                    (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    for j in range(N):
                        if l[j] > 0:
                            id1, id2 = self.skeleton_idx[g][j]
                            grad_input[t][id1] += self.var_weight * \
                                                  self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                                                  / l[j] * (input[t, id1] - input[t, id2]) / batch_size
                            grad_input[t][id2] += self.var_weight * \
                                                  self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                                                  / l[j] * (input[t, id2] - input[t, id1]) / batch_size
        ### grad_input = grad_input.cuda(self.device, non_blocking=True)
        # grad_input = grad_input.cuda()
        grad_input = grad_input
        return grad_input, None, None, None