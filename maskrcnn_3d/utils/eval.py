import numpy as np
import torch
criterion = torch.nn.SmoothL1Loss()

indexs = [[0, 1, 2], [3, 4, 5], [10, 11, 12], [13, 14, 15], [2, 6, 8], [3, 6, 8]]  ### 右腿，左腿，右臂，左臂，右跨，左跨
group = len(indexs)

###### 自定义的损失，主要是计算关节长度的损失
def my_L1_loss(x, y):
    return torch.sum(torch.abs(x-y))


#### 计算夹角（这边其实采用的是弧度值，因为弧度制与关节长度，值域处于相同级别，转化成夹角则两者值域不是同一级别，损失反向更新可能出现倾向问题）
def get_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    if (Lx * Ly) == 0.0 or x.all() == y.all():
        return 0.0
    else:
        cos_angle = x.dot(y) / (Lx * Ly)
        angle = np.arccos(cos_angle)  #### 弧度制（单位pi）
        return angle*np.pi


##### 所选6个关节点组，构成夹角的（关节长度损失）
def get_L1_loss(pre_joints, tru_joints):
    batch = pre_joints.shape[0]
    loss = torch.from_numpy(np.array(0, dtype=np.float32))
    for i in range(batch):
        angle_mean = torch.from_numpy(np.array(0, dtype=np.float32))
        for j in range(group):
            x0 = pre_joints[i, indexs[j][0], 0]
            y0 = pre_joints[i, indexs[j][0], 1]
            x1 = pre_joints[i, indexs[j][1], 0]
            y1 = pre_joints[i, indexs[j][1], 1]
            x2 = pre_joints[i, indexs[j][2], 0]
            y2 = pre_joints[i, indexs[j][2], 1]
            #### 涉及关节长度（并且计算他们的平滑L1损失函数）
            angle1_pre = [np.linalg.norm(np.array([x0, y0]) - np.array([x1, y1])) , np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))]
            angle1_tru = [np.linalg.norm(np.array([tru_joints[i, indexs[j][0], 0] - tru_joints[i, indexs[j][1], 0], \
                                             tru_joints[i, indexs[j][0], 1] - tru_joints[i, indexs[j][1], 1]])), \
                                   np.linalg.norm(np.array([tru_joints[i, indexs[j][2], 0] - tru_joints[i, indexs[j][1], 0], \
                                             tru_joints[i, indexs[j][2], 1] - tru_joints[i, indexs[j][1], 1]]))
                                   ]
            ##### 损失函数
            x = torch.from_numpy(np.array(angle1_pre, dtype=np.float32))
            y = torch.from_numpy(np.array(angle1_tru, dtype=np.float32))
            angle_mean += my_L1_loss(x, y)
        loss += angle_mean / group / 64
    return loss


##### 主要计算6组关节点之间的夹角损失值（之前使用L2平方差，效果差异小，这边更新后使用 L1_Smooth_loss, 并等比缩小64倍【计算损失与heatmap_loss】）
def get_angle_loss(pre_joints, tru_joints):
    batch = pre_joints.shape[0]
    loss = torch.from_numpy(np.array(0, dtype=np.float32))
    for i in range(batch):
        angle_mean = torch.from_numpy(np.array(0, dtype=np.float32))
        for j in range(group):
            x0 = pre_joints[i, indexs[j][0], 0]
            y0 = pre_joints[i, indexs[j][0], 1]
            x1 = pre_joints[i, indexs[j][1], 0]
            y1 = pre_joints[i, indexs[j][1], 1]
            x2 = pre_joints[i, indexs[j][2], 0]
            y2 = pre_joints[i, indexs[j][2], 1]
            angle1_pre = get_angle(np.array([x0, y0]) - np.array([x1, y1]), np.array([x2, y2]) - np.array([x1, y1]))
            angle1_tru = get_angle(np.array([tru_joints[i, indexs[j][0], 0] - tru_joints[i, indexs[j][1], 0], \
                                             tru_joints[i, indexs[j][0], 1] - tru_joints[i, indexs[j][1], 1]]), \
                                   np.array([tru_joints[i, indexs[j][2], 0] - tru_joints[i, indexs[j][1], 0], \
                                             tru_joints[i, indexs[j][2], 1] - tru_joints[i, indexs[j][1], 1]]))
            # ######  自定义的函数 1
            # loss += np.abs(angle1_pre-angle1_tru).astype(np.float32)
            ##### 自定义损失函数2
            x = torch.from_numpy(np.array(angle1_pre, dtype=np.float32))
            y = torch.from_numpy(np.array(angle1_tru, dtype=np.float32))
            angle_mean += criterion(x, y)
        loss += angle_mean / group / 64
    return loss

######  求骨架长度，方便单位化
def _get_bone_length(pts):
    sum_bone_length = 0
    for i in range(len(pts)):
        sum_bone_length += (pts[i] ** 2).sum() ** 0.5
    return sum_bone_length

def get_unit_3d(pts, bone_length):
    for i in range(len(pts)):
        pts[i] = pts[i]*((pts[i] ** 2).sum() ** 0.5/bone_length)
    return pts


###### 关于3d姿态，主要关节点之间的角度损失（之前同样使用L2损失，效果差异小； 这边更新后同样使用： L1_Smooth_loss）
def mpjpe_angle_loss(heatmap, depthmap, gt_3d, convert_func):  #### gt_3d 真实3d相对于中心骨盆点的坐标
    preds_3d = get_preds_3d(heatmap, depthmap)  #### 相对于骨盆点坐标，及深度值
    loss = torch.from_numpy(np.array(0, dtype=np.float32))
    batch = preds_3d.shape[0]
    for i in range(batch):  ### 【batch,16,3】
        angle_mean = torch.from_numpy(np.array(0, dtype=np.float32))
        pred_3d_h36m = convert_func(preds_3d[i])  #### 预测的3d相对于中心骨盆点的坐标  【16,3】

        # #####  真实场景求夹角误差或者夹角边误差，损失值过大(单位mm)，跟特征图的值或者位置点坐标的值不是一个级别；
        # ### 所以这边对骨架进行单位化，转换到单位化的3d空间
        # pre_bone_length = _get_bone_length(pred_3d_h36m)
        # pred_3d_h36m = get_unit_3d(pred_3d_h36m, pre_bone_length)
        # tru_bone_length = _get_bone_length(gt_3d)
        # gt_3d = get_unit_3d(gt_3d, tru_bone_length)

        for j in range(group):
            angle1_pre = get_angle(np.array(pred_3d_h36m[indexs[j][0]] - pred_3d_h36m[indexs[j][1]]),
                                   np.array(pred_3d_h36m[indexs[j][2]] - pred_3d_h36m[indexs[j][1]]))
            angle1_tru = get_angle(np.array(gt_3d[i][indexs[j][0]] - gt_3d[i][indexs[j][1]]),
                                   np.array(gt_3d[i][indexs[j][2]] - gt_3d[i][indexs[j][1]]))
            x = torch.from_numpy(np.array(angle1_pre, dtype=np.float32))
            y = torch.from_numpy(np.array(angle1_tru, dtype=np.float32))
            angle_mean += criterion(x, y)
        loss += angle_mean / group / batch
    return loss


def get_preds(hm, return_conf=False):
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    h = hm.shape[2]
    w = hm.shape[3]
    hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])  ##### (1, 16, 64*64)
    idx = np.argmax(hm, axis=2)

    preds = np.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w  #### 求heatmap图中，峰值的坐标位置
    if return_conf:
        conf = np.amax(hm, axis=2).reshape(hm.shape[0], hm.shape[1], 1)
        return preds, conf
    else:
        return preds


def calc_dists(preds, gt, normalize):  #### normalize = [6.4]
    # print('preds:', preds.shape)   #### (1,16,2)
    # print('gt:', gt.shape)    #### (1,16,2)
    # print('norm:', normalize)   #### [6.4]
    dists = np.zeros((preds.shape[1], preds.shape[0]))  ### (16,1)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
                dists[j][i] = \
                    ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
            else:
                dists[j][i] = -1
    return dists


def dist_accuracy(dist, thr=0.5):  #### dist <---> [2, 1]
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum() / len(dist)
    else:
        return -1


##### 计算准确性
def accuracy(output, target, acc_idxs):  ####  target 【1,16,64,64】  acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    ### 分别找每个热图的峰值作为最大概率的骨骼点位置
    preds = get_preds(output)  ### 【1,16,2】
    gt = get_preds(target)  ### 【1,16,2】
    dists = calc_dists(preds, gt, np.ones(target.shape[0]) * target.shape[2] / 10)
    acc = np.zeros(len(acc_idxs))
    avg_acc = 0
    bad_idx_count = 0

    for i in range(len(acc_idxs)):
        acc[i] = dist_accuracy(dists[acc_idxs[i]])
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
        else:
            bad_idx_count = bad_idx_count + 1

    if bad_idx_count == len(acc_idxs):
        return 0
    else:
        return avg_acc / (len(acc_idxs) - bad_idx_count)


def get_preds_3d(heatmap, depthmap):  #### 【1,16,64,64】
    output_res = max(heatmap.shape[2], heatmap.shape[3])
    preds = get_preds(heatmap).astype(np.int32)  ### 峰值位置  【1,16,2】
    preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)  ### 【1,16,3】
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            idx = min(j, depthmap.shape[1] - 1)
            pt = preds[i, j]
            preds_3d[i, j, 2] = depthmap[i, idx, pt[1], pt[0]]
            preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
        preds_3d[i] = preds_3d[i] - preds_3d[i, 6:7]
    return preds_3d


##### gt_3d是对应真实的3d坐标，convert_func计算预测点产生的真实骨架长度
def mpjpe(heatmap, depthmap, gt_3d, convert_func):
    preds_3d = get_preds_3d(heatmap, depthmap)  #### 相对于骨盆点坐标，及深度值
    # print(preds_3d)   ####  0.15625     0.234375   -0.00229681]
    cnt, pjpe = 0, 0
    for i in range(preds_3d.shape[0]):  ### 【batch_size,16,3】    #### [32,16,3]
        if gt_3d[i].sum() ** 2 > 0:
            cnt += 1
            pred_3d_h36m = convert_func(preds_3d[i])  #### preds_3d[0]的shape大小是【16,3】
            ##### 此时的深度是相对骨盆坐标点的真实深度
            err = (((gt_3d[i] - pred_3d_h36m) ** 2).sum(axis=1) ** 0.5).mean()
            pjpe += err
    if cnt > 0:
        pjpe /= cnt
    return pjpe, cnt

