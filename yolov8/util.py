from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
import pickle as pkl
import pandas as pd
import random


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.

    Parameters:
        img (np.array): Input image.
        new_shape (tuple): Desired output shape as (height, width).
        color (tuple): Border color for padding.
        auto (bool): If True, make the padding a multiple of stride.
        scaleFill (bool): If True, stretch the image to fill new_shape (may distort aspect ratio).
        scaleup (bool): If False, only downscale the image.
        stride (int): Stride for ensuring padded dimensions are multiples.

    Returns:
        img (np.array): The resized and padded image.
        ratio (float): Scaling ratio used.
        (dw, dh) (tuple): Padding added (width, height) on each side.
    """
    # Current shape [height, width]
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate scale ratio (new / old) and ensure it doesn't upscale if scaleup is False.
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute new unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    # Compute padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        # Make sure padding is a multiple of stride
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        # Stretch image to fill new_shape, disregarding aspect ratio
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # Divide padding into 2 sides
    dw /= 2  # width padding (left/right)
    dh /= 2  # height padding (top/bottom)

    # Resize image if necessary
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Compute padding values for top, bottom, left, right
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Apply padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)




import torch
def preprocess_image(image_path):
    # Load the image using OpenCV
    img_bgr = cv2.imread(image_path)

    # Step 1: Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: Letterbox resize to desired dimensions (e.g., 640x640)
    img_resized, ratio, pad = letterbox(img_rgb, new_shape=(640, 640), auto=True)

    # Step 3: Convert image to float and normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Step 4: Change data layout from HWC to CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # Step 5: Add batch dimension
    img_batch = np.expand_dims(img_transposed, 0)

    # Step 6: Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img_batch)
    
    return img_tensor


