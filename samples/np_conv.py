import numpy as np
import sys


def conv_(img, conv_filter):
    ##depthwise conv
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    # 循环遍历图像以应用卷积运算
    for r in np.uint16(np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1)):
        for c in np.uint16(np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1)):
            # 卷积的区域
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)),
                          c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
            # 卷积操作
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)
            # 将求和保存到特征图中
            result[r, c] = conv_sum

        # 裁剪结果矩阵的异常值
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0),
                   np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result

def conv_d(z, K,  padding='SAME', strides=(1, 1),depthwise=True):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,H,W,c)，N为batch_size，C为通道数
    :param K: 卷积核,形状(k1,k2,C,D), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    ## pad width ：表示上下左右分别填充的个数


    k1, k2, C, D = K.shape

    top, bottom, left, right=get_pad_num(z,K.shape,strides,padding)

    padding_z = np.lib.pad(z, ((0, 0), (top, bottom), (left, right), (0, 0)), 'constant', constant_values=0)
    N,  height, width ,_= padding_z.shape

    print('K.shape is ', K.shape, padding_z.shape)
    #assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    #assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((N,  1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1],D*C))
    if depthwise:
        for n in np.arange(N):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    for d in np.arange(C):
                        conv_z[n, h // strides[0], w // strides[1],d] = np.sum(padding_z[n, h:h + k1, w:w + k2,:] * K[:,:,d])
    else:
        for n in np.arange(N):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    for d in np.arange(D):
                        conv_z[n, h // strides[0], w // strides[1], d] = np.sum(padding_z[n, h:h + k1, w:w + k2, :] * K[:, :,  d])
    return conv_z

def conv(img, conv_filter,depthwise=False):
    ## pointwise conv img is (batch,h,w,c)
    # 检查图像通道的数量是否与过滤器深度匹配
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print('the channel of image and filters must match')
            sys.exit()

    # 检查过滤器是否是方阵
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('filters must be square')
        sys.exit()

    # 检查过滤器大小是否是奇数
    if conv_filter.shape[1] % 2 == 0:
        print('filter numbers must be odd')
        sys.exit()

    # 定义一个空的特征图，用于保存过滤器与图像的卷积输出
    feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                             img.shape[1] - conv_filter.shape[1] + 1,
                             conv_filter.shape[0]))

    # 卷积操作

    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]
        if depthwise:
            conv_map = conv_(img, curr_filter)
        else:
            # 检查单个过滤器是否有多个通道。如果有，那么每个通道将对图像进行卷积。所有卷积的结果加起来得到一个特征图。
            if len(curr_filter.shape) > 2:
                conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
                for ch_num in range(1, curr_filter.shape[-1]):
                    conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
            else:
                conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map
    return feature_maps
def get_pad_num(z,kernel_shape,strides,padding='SAME'):
    k1, k2,C, D = kernel_shape
    N,  H, W,C = z.shape
    if padding=='SAME':

        oh=np.ceil(H / strides[0])
        ow = np.ceil(W / strides[1])

        padh=max((oh-1)*strides[0]+k1-H,0)
        top=int(np.floor(padh/2))
        bottom=int(padh-top)

        padw = max((ow - 1) * strides[1] + k2 - W, 0)
        left=int(np.floor(padw/2))
        right=int(padw-left)


    else:
        oh = np.ceil((H-k1+1)/ strides[0])
        ow = np.ceil((W-k2+1) / strides[1])

        padh = max((oh - 1) * strides[0] + k1 - H, 0)
        top = int(np.floor(padh / 2))
        bottom = int(padh - top)

        padw = max((ow - 1) * strides[1] + k2 - W, 0)
        left = int(np.floor(padw / 2))
        right = int(padw - left)

    return top,bottom,left,right

def pooling(feature_map, window_size=2, strides=(1,1),padding='SAME'):
    ##feature map is (batch,h,w,c)
    k_shape=(1,1,window_size,window_size)
    top, bottom, left, right=get_pad_num(feature_map,k_shape,strides,padding)
    stride=strides[0]
    feature_map = np.lib.pad(feature_map, ((0, 0), (top, bottom), (left, right), (0, 0)), 'constant', constant_values=0)
    # 定义池化操作的输出
    if padding=='SAME':
        pool_out = np.zeros((feature_map.shape[0],np.uint16(np.ceil(feature_map.shape[1] / stride )),
                             np.uint16(np.ceil(feature_map.shape[2] / stride) ),
                             feature_map.shape[-1]))
    else:
        pool_out = np.zeros((feature_map.shape[0],np.uint16(np.ceil((feature_map.shape[1] - window_size + 1) / stride )),
                         np.uint16(np.ceil((feature_map.shape[2] - window_size + 1) / stride )),
                         feature_map.shape[-1]))
    for n in range(feature_map.shape[0]):
        for map_num in range(feature_map.shape[-1]):
            r2 = 0
            for r in np.arange(0, feature_map.shape[1] - window_size + 1, stride):
                c2 = 0
                for c in np.arange(0, feature_map.shape[2] - window_size + 1, stride):
                    pool_out[n,r2, c2, map_num] = np.max([feature_map[n,r: r+window_size, c: c+window_size, map_num]])
                    c2 = c2 + 1
                r2 = r2 + 1
    return pool_out

