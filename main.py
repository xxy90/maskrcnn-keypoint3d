# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch

if torch.cuda.is_available():
    print('use cuda')
else:
    print('no use')

import torch.utils.data
from opts import opts
from maskrcnn_3d.models.detection.model import create_model, save_model

from maskrcnn_3d.utils.util2 import H36M
from maskrcnn_3d.utils.logger import Logger
#from maskrcnn_3d.models.detection.train import train, val
#from maskrcnn_3d.models.detection.train_3d import train_3d, val_3d
from maskrcnn_3d.models.detection.train_mask2 import train_3d, val_3d
import scipy.io as sio
import os

# os.environ["CUDA_VISIBLE_DEVICES"]='6'



def main(opt):
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print('Cudnn is disabled.')

    logger = Logger(opt)

    # opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))

    Dataset = H36M(opt, 'train')

    Dataset_val = H36M(opt, 'val')


    model, optimizer, start_epoch = create_model(opt)  ### 阶段1：model，优化方法，start_epoch=1

    # if len(opt.gpus)>1:
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda(opt.device)
    # else:
    #     # model = model.cuda(opt.device)  ##### 外部直接设置gpu
    #     model = model.cuda()

    model = model.cuda()





    train_loader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=opt.batch_size ,
        shuffle=True,  # if opt.debug == 0 else False,
        num_workers=opt.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        Dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    best = -1
    for epoch in range(start_epoch, opt.num_epochs):
        mark = epoch if opt.save_all_models else 'last'
        loss, log_dict_train = train_3d(epoch, opt, train_loader, model, optimizer)
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
