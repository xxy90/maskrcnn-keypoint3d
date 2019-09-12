import torch
from torch import nn
from torchvision.transforms import functional as F
import os
import cv2
import torchvision.models as models
import time
import numpy as np
# from bn_fusion import fuse_bn_recursively
# from kmeans import kmeans_cpu,kmeans_gpu
# from huffmancoding import huffman_encode_model
# from prune2 import prune_by_std,prune_by_percentile
from multiprocessing import Pool,Manager, Queue,Process
from numba import jit

os.environ['CUDA_VISIBLE_DEVICES']='5'

#@jit
def main(pth):
    model=models.resnet18(pretrained=True)

    device = torch.device("cuda:0")

    # Data loading code
    print("Loading data")
    #pth = 'b.jpg'
    img_path = os.path.join('./examples/' + pth)
    img = cv2.imread(img_path)
    img=img[np.newaxis,:]
    #img=np.concatenate((img,img,img,img,img,img),axis=0)

    #img = F.to_tensor(img)
    img=torch.Tensor(img)
    img=img.permute(0,3,1,2).cuda()


    #model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    #model=fuse_bn_recursively(model)
    #apply_weight_sharing(model)
    #model=kmeans_gpu(model)
    #huffman_encode_model(model)
    #prune_by_std(model)
    #prune_by_percentile(model)
    #torch.save(model,'./result/quantization_model.pth')
    model = model.cuda()
    #img = [img.to(device)]

    start = time.time()

    prediction=model(img)

    print('processing time paralell is :', time.time() - start)
def write(q,urls):
    for url in urls:
        q.put(url,block=False)
        print('put %s to queue ...' %url)
def read(q):
    while not q.empty():
        url=q.get(block=False)
        print('get %s from queue.'%url)
        main(url)


if __name__ == "__main__":

    pth='b.jpg'
    pth2='a.jpg'
    start1=time.time()
    q=Queue()
    write1=Process(target=write,args=(q,[pth,pth2]))
    write1.start()
    read1=Process(target=read,args=(q,))

    read1.start()
    write1.join()

    #print('processing time all is :', time.time() - start1)
    #read1.join()


