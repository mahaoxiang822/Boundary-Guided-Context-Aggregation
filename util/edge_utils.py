"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from skimage import segmentation as skimage_seg

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def insmask_to_onehot(mask, ins_index):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == ins_index[i] for i in range(len(ins_index))]
    return np.array(_mask).astype(np.uint8)

def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask

def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask
    
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)
        
    return np.array(channels)

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def onehot_to_DTM(mask, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a SDF map (K,H,W)

    """

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape)

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        edgemap[i] = dist
    return edgemap

def onehot_to_SDF(mask, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a SDF map (K,H,W)

    """

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    sdfmap = np.zeros(mask.shape)

    for i in range(num_classes):
        pos_mask = mask_pad[i].astype(np.bool)
        if pos_mask.any():
            posdis = distance_transform_edt(pos_mask)
            negdis = distance_transform_edt(~pos_mask)
            boundary = skimage_seg.find_boundaries(pos_mask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                        np.max(posdis) - np.min(posdis))
            sdf[boundary==1] = 0
            sdf = sdf[1:-1, 1:-1]
            sdfmap[i] = sdf
    return sdfmap

