import torch
from torch import nn
from torchvision.transforms import functional as F
import os
import cv2
import torchvision.models as models
import time
import numpy as np
from torch.nn import Parameter




def prune_by_std(model, s=0.25):
    """
    Note that `s` is a quality parameter / sensitivity value according to the paper.
    According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
    'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

    I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
    Note : In the paper, the authors used different sensitivity values for different layers.
    """
    for module in model.children():
        if hasattr(module,'weight'):
            threshold = np.std(module.weight.data.cpu().numpy()) * s
            print('Pruning with threshold : {threshold} for layer {name}')
            prune(module,threshold)

def prune(module, threshold):
    weight_dev = module.weight.device
    mask_dev = weight_dev
    # Convert Tensors to numpy and calculate
    tensor = module.weight.data.cpu().numpy()
    mask = Parameter(torch.from_numpy(np.ones_like(tensor)), requires_grad=False)
    new_mask = np.where(abs(tensor) < threshold, 0, mask)
    # Apply new weight and mask
    module.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
    #module.mask.data = torch.from_numpy(new_mask).to(mask_dev)


def prune_by_percentile(model, q=90.0, **kwargs):
    """
    Note:
         The pruning percentile is based on all layer's parameters concatenated
    Args:
        q (float): percentile in float
        **kwargs: may contain `cuda`
    """
    # Calculate percentile value
    alive_parameters = []
    for name, p in model.named_parameters():
        # We do not prune bias term
        if 'bias' in name or 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
        alive_parameters.append(alive)

    all_alives = np.concatenate(alive_parameters)
    percentile_value = np.percentile(abs(all_alives), q)
    print('Pruning with threshold : {}'.format(percentile_value))

    # Prune the weights and mask
    # Note that module here is the layer
    # ex) fc1, fc2, fc3
    for name, module in model.named_modules():
        if hasattr(module,'weight'):
            prune(module,threshold=percentile_value)