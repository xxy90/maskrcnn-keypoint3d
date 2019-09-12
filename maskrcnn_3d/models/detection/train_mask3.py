import torch
import numpy as np
from maskrcnn_3d.utils.image import flip, shuffle_lr
from maskrcnn_3d.utils.eval import accuracy, get_preds, mpjpe, get_preds_3d, get_angle_loss, mpjpe_angle_loss
import cv2
from progress.bar import Bar
from maskrcnn_3d.utils.debugger import Debugger
from maskrcnn_3d.utils.losses_3d import RegLoss, FusionLoss
import time
from torch.autograd import Variable

#### 主要采用3d数据集的训练方式

def step(split, epoch, opt, data_loader, model, optimizer=None):
    # if split == 'train':
    #     model.train()
    # else:
    #     model.eval()

    crit = torch.nn.MSELoss()
    crit_3d = FusionLoss(opt.weight_3d, opt.weight_var)

   
    Loss, Loss3D, Angle_Loss, Angle_3d_Loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
   
    preds = []
    

    nIters = len(data_loader)
    print(nIters)
    bar = Bar('{}'.format(opt.exp_id), max=nIters)

    end = time.time()
    #### meta :{'index' : self.idxs[index], 'center' : c, 'scale' : s, 'gt_3d': gt_3d, 'pts_crop': pts_crop}
    #### {is_crowd ,labels,keypoints,reg_ind,boxes,masks,outMap,keypoints3d,reg_mask,image_id,reg_target}
    ###target['boxes'],target['keypoints'],target['labels'],target['keypoints3d']
    for i, data_batchs in enumerate(data_loader):
        
        image, batch = data_batchs  ## batch is targets
        print('the size of image ',image.size())
        image=image.cuda()
        batch_size = image.size(0)

        for k in batch.keys():

            if k != 'image_id':
                batch[k] = batch[k].cuda()
            #print('type data_loader is ',k, batch[k].shape)
        gt_2d = batch['keypoints'].float() / opt.output_h
        batch_post = process_target(batch, batch_size)
        
        try:
            output = model(image, batch_post)
            if split == 'train':
                loss = 0
                preds = {}

                for k in output.keys():
                    # print('output[k] are ',k,'---',output[k])
                    loss += output[k]
                    preds['%s' % k] = output[k]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                preds = output[-1]

            Loss.update(loss.item(), image.size(0))
            print('the current logs epoch: {:02d} iters:{:02d}/{:02d} loss: {:05f}'.format(epoch, i, nIters, loss / i))
            

            Bar.suffix = '{split}: [{0}][{1}/{2}] |Total {total:} |ETA {eta:} '

            bar.finish()
        except:
            continue
       # print('output are ', '---', output)

        


    return  Loss.avg, preds


def process_target(t,s):
    ## org target is [dict,batch,shape] to [batch,dict,shape]
    tt=[]
    for i in range(s):
        tmp={}
        for key in t.keys():
            if key !='image_id':
                tmp['%s'%key]=t[key][i].cuda().float()
            else:
                tmp['%s'%key]=t[key][i]

           
            
        tt.append(tmp)

    return tt



def train_3d(epoch, opt, train_loader, model, optimizer):
    return step('train', epoch, opt, train_loader, model, optimizer)


def val_3d(epoch, opt, val_loader, model):
    return step('val', epoch, opt, val_loader, model)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
