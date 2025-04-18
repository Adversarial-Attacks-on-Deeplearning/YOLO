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
import random



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
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
    img_resized, ratio, pad = letterbox(img_rgb, new_shape=(640, 640), auto=False)

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

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms


def compare_original_and_adversarial_png(model, image_path, adversarial_image, conf_threshold):
    """
    Compares the original and adversarial images by:
      1. Preprocessing the original image and running a prediction.
      2. Saving the adversarial tensor as a PNG image.
      3. Loading the saved PNG image as a tensor (which may lose some precision).
      4. Running predictions on both images and displaying the results.
      
    Args:
        model: The YOLO model to run predictions.
        image_path (str): Path to the original image.
        adversarial_image (torch.Tensor): Adversarial image tensor (shape [1, C, H, W], values in [0,1]).
        conf_threshold (float): Confidence threshold for predictions.
    """
    # Preprocess the original image (assuming preprocess_image returns a tensor in the expected format)
    original_tensor = preprocess_image(image_path)
    
    # Run prediction on the original tensor
    print("Running prediction on original tensor...")
    original_results = model(original_tensor, conf=conf_threshold)
    
    # Save the adversarial image as a PNG (this will quantize values to 8-bit)
    adv_image_path = "adversarial_image.png"
    adversarial_tensor = adversarial_image.clone().detach()  # ensure no gradient is attached
    save_image(adversarial_tensor, adv_image_path)  # saves as PNG by default based on the filename extension
    
    # Load the saved PNG as a tensor (using PIL and torchvision transforms)
    adv_img_pil = Image.open(adv_image_path).convert("RGB")
    transform = transforms.ToTensor()  # Converts image to tensor in [0,1]
    loaded_adv_tensor = transform(adv_img_pil).unsqueeze(0)  # shape: [1, C, H, W]
    
    print("Running prediction on loaded adversarial tensor...")
    adversarial_results = model(loaded_adv_tensor, conf=conf_threshold)
    
    # For display, convert tensors to numpy arrays scaled to 0-255 for visualization
    original_np = (original_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    adversarial_np = (loaded_adv_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Render prediction outputs if available (assuming results[0] has a .plot() method)
    try:
        original_pred_img = original_results[0].plot()
    except Exception as e:
        print("Error rendering original predictions:", e)
        original_pred_img = None

    try:
        adversarial_pred_img = adversarial_results[0].plot()
    except Exception as e:
        print("Error rendering adversarial predictions:", e)
        adversarial_pred_img = None

    # Display raw images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_np)
    plt.axis('off')
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_np)
    plt.axis('off')
    plt.title("Adversarial Image (PNG)")
    plt.show()
    
    # Display predictions side by side (if available)
    if original_pred_img is not None and adversarial_pred_img is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_pred_img)
        plt.axis('off')
        plt.title("Original Prediction")
    
        plt.subplot(1, 2, 2)
        plt.imshow(adversarial_pred_img)
        plt.axis('off')
        plt.title("Adversarial Prediction (PNG)")
        plt.show()
    else:
        print("Could not render one or both prediction images.")




def sample_images_by_class(
    images_dir: str,
    labels_dir: str,
    num_per_class: int = 100
):
    """
    Collect up to `num_per_class` images for each class from YOLO-style
    images/labels folders, ensuring a roughly balanced distribution.

    :param images_dir: Path to the folder containing images.
    :param labels_dir: Path to the folder containing corresponding .txt labels.
    :param num_per_class: Maximum number of images to select for each class.
    :return: (final_image_paths, class_distribution)
       - final_image_paths: a list of unique image paths after balancing.
       - class_distribution: a dict: class_id -> count of images in final_image_paths.
    """
    images_by_class = {}

    # 1. Group images by class
    for filename in os.listdir(images_dir):
        # only process valid images
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(filename)[0]

            # Look for the label in the labels folder
            txt_path = os.path.join(labels_dir, base_name + ".txt")
            if not os.path.exists(txt_path):
                # no label => skip
                continue

            # read YOLO label file
            with open(txt_path, 'r') as f:
                lines = f.read().strip().splitlines()

            # gather classes
            classes_in_this_image = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id_str = parts[0]
                class_id = int(class_id_str)
                classes_in_this_image.add(class_id)

            # store the image path under each relevant class
            full_image_path = os.path.join(images_dir, filename)
            for c in classes_in_this_image:
                images_by_class.setdefault(c, []).append(full_image_path)

    # 2. Stratified sampling
    final_image_paths = []
    for c, img_list in images_by_class.items():
        random.shuffle(img_list)
        selected = img_list[:num_per_class]
        final_image_paths.extend(selected)

    # remove duplicates if an image belongs to multiple classes
    final_image_paths = list(set(final_image_paths))
    random.shuffle(final_image_paths)

    # 3. Build a dictionary of final distribution
    class_distribution = {}
    for c, img_list in images_by_class.items():
        count = len(set(img_list).intersection(final_image_paths))
        class_distribution[c] = count

    return final_image_paths, class_distribution