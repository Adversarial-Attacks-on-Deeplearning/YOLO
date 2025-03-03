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
import matplotlib.pyplot as plt
import tifffile
from PIL import Image



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



def save_tensor_image_tiff(tensor, save_path):
    """
    Saves a tensor as a float32 TIFF image.
    This preserves the exact floating point values (assuming tensor values are in [0,1]).
    """
    # Remove batch dimension and convert from (C, H, W) to (H, W, C)
    tensor_np = tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float32)
    # Save as TIFF with float32 data
    tifffile.imwrite(save_path, tensor_np)

def load_tiff_image_as_tensor(save_path):
    """
    Loads a TIFF image saved in float32 format and converts it to a torch tensor.
    """
    # Read the TIFF file with tifffile to preserve float precision
    image_np = tifffile.imread(save_path)
    # If the model expects a tensor in shape (1, C, H, W) with values in [0,1]
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return tensor

def compare_original_and_adversarial(model, image_path, adversarial_image, conf_threshold):
    """
    Compares the original and adversarial images by saving the adversarial tensor to disk
    using a lossless (float32) TIFF and then running predictions on both the original image
    and the saved adversarial image. This approach aims to maintain the subtle adversarial
    perturbations that might be lost during uint8 conversion.
    """
    # Preprocess the original image (this function should return a tensor in the expected format)
    original_tensor = preprocess_image(image_path)
    
    # Run prediction on the original tensor
    print("Running prediction on original tensor...")
    original_results = model(original_tensor, conf=conf_threshold)
    
    # Save the adversarial image as a TIFF (preserving float32 precision)
    adv_image_path = "adversarial_image.tiff"
    adversarial_tensor = adversarial_image.clone().detach()  # ensure no gradient is attached
    save_tensor_image_tiff(adversarial_tensor, adv_image_path)
    
    # Now load the saved TIFF as a tensor to run prediction
    loaded_adv_tensor = load_tiff_image_as_tensor(adv_image_path)
    
    print("Running prediction on loaded adversarial tensor...")
    adversarial_results = model(loaded_adv_tensor, conf=conf_threshold)
    
    # For display purposes, convert tensors to numpy arrays (scaling back to 0-255 for visualization)
    original_np = (original_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    adversarial_np = (adversarial_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Render prediction outputs if the results object supports a plotting method (e.g., .plot())
    try:
        original_pred_img = original_results[0].plot()  # assuming results[0] has a plot() method
    except Exception as e:
        print("Error rendering original predictions:", e)
        original_pred_img = None

    try:
        adversarial_pred_img = adversarial_results[0].plot()
    except Exception as e:
        print("Error rendering adversarial predictions:", e)
        adversarial_pred_img = None

    # Display raw images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_np)
    plt.axis('off')
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_np)
    plt.axis('off')
    plt.title("Adversarial Image")
    plt.show()
    
    # Display predictions (if available)
    if original_pred_img is not None and adversarial_pred_img is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_pred_img)
        plt.axis('off')
        plt.title("Original Prediction")
    
        plt.subplot(1, 2, 2)
        plt.imshow(adversarial_pred_img)
        plt.axis('off')
        plt.title("Adversarial Prediction")
        plt.show()
    else:
        print("Could not render one or both prediction images.")
