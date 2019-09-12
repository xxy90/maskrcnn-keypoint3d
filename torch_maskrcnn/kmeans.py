import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix,coo_matrix
import sklearn.cluster._k_means
from sklearn.utils.extmath import row_norms,squared_norm
from numpy.random import RandomState
import time
from libKMCUDA import kmeans_cuda



def kmeans_cpu(model, bits=5, seed=int(time.time())):

    for name,module in model.named_children():
        if 'conv' in name :
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            print(name,shape)
            filters_num, filters_channel, filters_size = shape[0], shape[1], shape[2]
            weight_vector = weight.reshape(-1, filters_size).astype('float32')
            if weight_vector.shape[0]<=2**bits:
                weight_vector_compress=weight_vector
            else:
                kmeans_result = KMeans(n_clusters=2**bits, init='k-means++', precompute_distances=True, random_state = seed).fit(weight_vector)
                labels = kmeans_result.labels_
                centers = kmeans_result.cluster_centers_
                weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
                for v in range(weight_vector.shape[0]):
                    weight_vector_compress[v, :] = centers[labels[v], :]
                # weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
            weight_vector_compress=weight_vector_compress.reshape(filters_num,filters_channel,filters_size,-1)
            weight_vector_compress=torch.from_numpy(weight_vector_compress).cuda()  # type: object
            module.weight.data=weight_vector_compress
    return model

def kmeans_gpu(model, bits=5,seed=int(time.time())):
    """
    Applies weight sharing to the given model
    """
    gpu_id=[0,1,2]
    for name,module in model.named_children():
        if 'conv' in name:

            dev = module.weight.device
            weight = module.weight[0].data.cpu().numpy()
            shape = weight.shape
            filters_num,filters_channel,filters_size=shape[0],shape[1],shape[2]
            weight_vector=weight.reshape(-1,filters_size).astype('float32')

            init_centers=sklearn.cluster.k_means_._k_init(X=weight_vector, n_clusters=2**bits,
                        x_squared_norms=row_norms(weight_vector, squared=True), random_state=RandomState(seed))
            if weight_vector.shape[0] <= 2 ** bits:
                weight_vector_compress = weight_vector
            else:

                centers,labels=kmeans_cuda(samples=weight_vector,clusters=2**bits,init=init_centers,seed=seed,device=gpu_id,verbosity=0)
                weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
                for v in range(weight_vector.shape[0]):
                    weight_vector_compress[v, :] = centers[labels[v], :]
            weight_vector_compress = weight_vector_compress.reshape(filters_num, filters_channel, filters_size, -1)
            weight_vector_compress = torch.from_numpy(weight_vector_compress).cuda()  # type: object
            module.weight[0].data = weight_vector_compress
    return model