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


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def find_predicted_image(pred_folder, original_filename):
    """Finds the predicted image file inside the YOLO output folder."""
    all_files = os.listdir(pred_folder)
    
    # Try exact match first
    if original_filename in all_files:
        return os.path.join(pred_folder, original_filename)
    
    # If not found, find the closest match
    for filename in all_files:
        if filename.startswith(original_filename.split('.')[0]):  # Match without extension
            return os.path.join(pred_folder, filename)
    
    return None  # No match found

def compare_original_and_adversarial(model, image_path, adversarial_image, conf_threshold):
    """
    Compares the original and adversarial images by displaying them side by side
    and also showing the model’s predictions on both.
    """

    # Preprocess the original image
    image = preprocess_image(image_path)

    # Run and save predictions with unique save directories
    original_save_dir = "runs/detect/original_pred"
    adversarial_save_dir = "runs/detect/adversarial_pred"

    print("original_pred: ", end="")
    model(image_path, conf=conf_threshold, save=True, project="runs/detect", name="original_pred", exist_ok=True)

    print("adversarial_pred: ", end="")
    adv_image_path = 'adversarial_image.png'

    # Ensure proper conversion before saving
    adversarial_image = adversarial_image.clone().detach()
    bgr_image = cv2.cvtColor(
        (adversarial_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), 
        cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(adv_image_path, bgr_image)

    model(adv_image_path, conf=conf_threshold, save=True, project="runs/detect", name="adversarial_pred", exist_ok=True)

    # Locate saved prediction images (use dynamic filename search)
    original_pred_path = find_predicted_image(original_save_dir, os.path.basename(image_path))
    adversarial_pred_path = find_predicted_image(adversarial_save_dir, os.path.basename(adv_image_path))

    print(f"Original prediction found: {original_pred_path}")
    print(f"Adversarial prediction found: {adversarial_pred_path}")

    if not (original_pred_path and adversarial_pred_path):
        print("Predicted images not found.")
        return

    # Load prediction images
    original_pred_np = cv2.imread(original_pred_path)
    adversarial_pred_np = cv2.imread(adversarial_pred_path)

    # Convert BGR to RGB for proper display
    original_pred_np = cv2.cvtColor(original_pred_np, cv2.COLOR_BGR2RGB)
    adversarial_pred_np = cv2.cvtColor(adversarial_pred_np, cv2.COLOR_BGR2RGB)

    # Convert tensors to displayable format
    adversarial_image_np = (adversarial_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    original_image_np = (image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Display Original vs Adversarial Image (Raw)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image_np)
    plt.axis('off')
    plt.title("Adversarial Image")

    plt.show()

    # Display Original Prediction vs Adversarial Prediction
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_pred_np)
    plt.axis('off')
    plt.title("Original Prediction")

    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_pred_np)
    plt.axis('off')
    plt.title("Adversarial Prediction")

    plt.show()
