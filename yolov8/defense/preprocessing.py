import os
import glob
import shutil
import torch
from PIL import Image
import io
import random
from torchvision import models, transforms
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import time

def preprocess_image(image_path, output_image_path, quality=30, scale_range=(0.8, 1.2), sigma=1.5):
    """
    Preprocess an adversarial image with JPEG compression, Gaussian blurring, and random resizing.

    Args:
        image_path (str): Path to input image.
        output_image_path (str): Path to save processed image.
        quality (int): JPEG compression quality (0-100).
        scale_range (tuple): Min and max scale factors for random resizing.
        sigma (float): Standard deviation for Gaussian blurring.

    Returns:
        tuple: (success (bool), scale (float), old_w (int), old_h (int), new_w (int), new_h (int))
               success: True if successful, False otherwise.
               scale: Scale factor used for resizing.
               old_w, old_h: Original image dimensions.
               new_w, new_h: New image dimensions.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        old_w, old_h = img.width, img.height

        # JPEG compression
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        comp = Image.open(buf).convert("RGB")

        # Gaussian blurring
        comp_np = np.array(comp)
        blurred = gaussian_filter(comp_np, sigma=sigma)
        comp = Image.fromarray(blurred)

        # Random resize
        scale = random.uniform(*scale_range)
        new_w = int(comp.width * scale)
        new_h = int(comp.height * scale)
        resized = comp.resize((new_w, new_h), Image.Resampling.LANCZOS)

        resized.save(output_image_path)
        return True, scale, old_w, old_h, new_w, new_h
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return False, 1.0, 0, 0, 0, 0

def rescale_yolo_labels(txt_lines, scale, old_w, old_h, new_w, new_h):
    """
    Rescale YOLO labels to match resized image dimensions, ensuring coordinates stay within [0, 1].

    Args:
        txt_lines (list): List of YOLO label lines (cls x_center_rel y_center_rel w_rel h_rel).
        scale (float): Scale factor used in resizing.
        old_w, old_h (int): Original image width and height.
        new_w, new_h (int): New image width and height.

    Returns:
        list: List of rescaled label lines.
    """
    out = []
    for ln in txt_lines:
        try:
            cls, x_rel, y_rel, w_rel, h_rel = ln.strip().split()
            x_rel, y_rel, w_rel, h_rel = map(float, [x_rel, y_rel, w_rel, h_rel])

            # Convert to absolute coordinates
            x_abs = x_rel * old_w
            y_abs = y_rel * old_h
            w_abs = w_rel * old_w
            h_abs = h_rel * old_h

            # Apply scale
            x_abs_new = x_abs * scale
            y_abs_new = y_abs * scale
            w_abs_new = w_abs * scale
            h_abs_new = h_abs * scale

            # Convert back to normalized coordinates with boundary checks
            x_rel_new = max(0, min(1, x_abs_new / new_w)) if new_w > 0 else x_rel
            y_rel_new = max(0, min(1, y_abs_new / new_h)) if new_h > 0 else y_rel
            w_rel_new = max(0, min(1, w_abs_new / new_w)) if new_w > 0 else w_rel
            h_rel_new = max(0, min(1, h_abs_new / new_h)) if new_h > 0 else h_rel

            out.append(f"{cls} {x_rel_new:.6f} {y_rel_new:.6f} {w_rel_new:.6f} {h_rel_new:.6f}\n")
        except Exception as e:
            print(f"Error rescaling label line '{ln.strip()}': {e}")
            out.append(ln)  # Keep original line if rescaling fails
    return out

def load_classifier(model_path, device):
    """
    Load the trained MobileNetV2 classifier.

    Args:
        model_path (str): Path to the model weights.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded classifier model.
    """
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def format_time(seconds):
    """Format time in a human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def classify_and_preprocess(
    input_image_dir, input_label_dir,
    output_image_dir, output_label_dir,
    model_path, quality=30, scale_range=(0.8, 1.2), sigma=1.5, threshold=0.3
):
    """
    Classify images as adversarial or clean, preprocess adversarial ones, and adjust labels.

    Args:
        input_image_dir (str): Directory with input images.
        input_label_dir (str): Directory with input labels.
        output_image_dir (str): Directory for processed images.
        output_label_dir (str): Directory for processed labels.
        model_path (str): Path to the classifier model weights.
        quality (int): JPEG compression quality (0-100).
        scale_range (tuple): Min and max scale factors for resizing.
        sigma (float): Standard deviation for Gaussian blurring.
        threshold (float): Probability threshold for classifying an image as adversarial (0-1).
    """
    total_start_time = time.time()
    
    # Step 1: Setup and initialization
    setup_start_time = time.time()
    adv_count = 0
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    setup_end_time = time.time()
    print(f"⏱️  Setup time: {format_time(setup_end_time - setup_start_time)}")

    # Step 2: Load classifier
    model_load_start_time = time.time()
    classifier = load_classifier(model_path, device)
    model_load_end_time = time.time()
    print(f"⏱️  Model loading time: {format_time(model_load_end_time - model_load_start_time)}")

    # Step 3: Setup transforms
    transform_setup_start_time = time.time()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_setup_end_time = time.time()
    print(f"⏱️  Transform setup time: {format_time(transform_setup_end_time - transform_setup_start_time)}")

    # Step 4: File discovery
    file_discovery_start_time = time.time()
    # Collect all .png, .jpg, .jpeg files (case-insensitive)
    exts = ('png', 'jpg', 'jpeg')
    image_paths = []
    for e in exts:
        image_paths += glob.glob(os.path.join(input_image_dir, f"*.{e}"))
        image_paths += glob.glob(os.path.join(input_image_dir, f"*.{e.upper()}"))

    label_paths = glob.glob(os.path.join(input_label_dir, "*.txt"))
    file_discovery_end_time = time.time()
    print(f"⏱️  File discovery time: {format_time(file_discovery_end_time - file_discovery_start_time)}")

    print(f"Found {len(image_paths)} images in {input_image_dir}")
    print(f"Found {len(label_paths)} labels in {input_label_dir}")
    if not image_paths:
        print("No images to process; exiting.")
        return

    # Step 5: Main processing loop
    processing_start_time = time.time()
    missing_output_images = []
    missing_output_labels = []
    processed_count = 0
    
    classification_total_time = 0
    preprocessing_total_time = 0
    label_processing_total_time = 0
    file_io_total_time = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        base, ext = os.path.splitext(os.path.basename(img_path))
        ext = ext.lower()  # Normalize extension for output
        out_img = os.path.join(output_image_dir, base + ext)
        out_lbl = os.path.join(output_label_dir, base + ".txt")
        lbl_path = os.path.join(input_label_dir, base + ".txt")

        try:
            # Classification step
            classification_start = time.time()
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classifier(img_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()  # Probability of being adversarial
            is_adv = prob > threshold
            classification_end = time.time()
            classification_total_time += (classification_end - classification_start)

            # Process or copy
            if is_adv:
                # Preprocess adversarial image and get resize info
                preprocessing_start = time.time()
                success, scale, old_w, old_h, new_w, new_h = preprocess_image(
                    img_path, out_img, quality, scale_range, sigma
                )
                if not success:
                    shutil.copy(img_path, out_img)
                    scale, old_w, old_h, new_w, new_h = 1.0, img.width, img.height, img.width, img.height
                preprocessing_end = time.time()
                preprocessing_total_time += (preprocessing_end - preprocessing_start)
                adv_count = adv_count +1
            else:
                # Copy clean image without preprocessing
                file_io_start = time.time()
                shutil.copy(img_path, out_img)
                scale, old_w, old_h, new_w, new_h = 1.0, img.width, img.height, img.width, img.height
                file_io_end = time.time()
                file_io_total_time += (file_io_end - file_io_start)

            # Handle label
            label_processing_start = time.time()
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                if is_adv and lines:
                    # Rescale labels for adversarial images
                    new_lines = rescale_yolo_labels(lines, scale, old_w, old_h, new_w, new_h)
                    with open(out_lbl, 'w') as f:
                        f.writelines(new_lines)
                else:
                    # Copy label unchanged for clean images or empty labels
                    shutil.copy(lbl_path, out_lbl)
            else:
                # Create empty label file
                open(out_lbl, 'w').close()
            label_processing_end = time.time()
            label_processing_total_time += (label_processing_end - label_processing_start)

        except Exception as e:
            print(f"Error on {img_path}: {e}")
            # Fallback: Copy image and label
            fallback_start = time.time()
            shutil.copy(img_path, out_img)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, out_lbl)
            else:
                open(out_lbl, 'w').close()
            fallback_end = time.time()
            file_io_total_time += (fallback_end - fallback_start)

        # Record any failures
        if not os.path.exists(out_img):
            missing_output_images.append(base + ext)
        if not os.path.exists(out_lbl):
            missing_output_labels.append(base + ".txt")
        processed_count += 1

    processing_end_time = time.time()
    
    # Step 6: Final counting and verification
    counting_start_time = time.time()
    # Count outputs across all image extensions
    output_image_count = sum(
        len(glob.glob(os.path.join(output_image_dir, f"*.{e}")))
        + len(glob.glob(os.path.join(output_image_dir, f"*.{e.upper()}")))
        for e in exts
    )
    output_label_count = len(glob.glob(os.path.join(output_label_dir, "*.txt")))
    input_label_count = len(label_paths)
    counting_end_time = time.time()
    
    total_end_time = time.time()

    # Print timing summary
    print(f"\n{'='*50}")
    print(f"TIMING SUMMARY")
    print(f"{'='*50}")
    print(f"⏱️  Setup & Initialization:     {format_time(setup_end_time - setup_start_time)}")
    print(f"⏱️  Model Loading:             {format_time(model_load_end_time - model_load_start_time)}")
    print(f"⏱️  Transform Setup:           {format_time(transform_setup_end_time - transform_setup_start_time)}")
    print(f"⏱️  File Discovery:            {format_time(file_discovery_end_time - file_discovery_start_time)}")
    print(f"⏱️  Main Processing:           {format_time(processing_end_time - processing_start_time)}")
    print(f"   ├─ Classification:          {format_time(classification_total_time)}")
    print(f"   ├─ Adversarial Preprocessing: {format_time(preprocessing_total_time)}")
    print(f"   ├─ Label Processing:        {format_time(label_processing_total_time)}")
    print(f"   └─ File I/O (clean images): {format_time(file_io_total_time)}")
    print(f"⏱️  Final Counting:            {format_time(counting_end_time - counting_start_time)}")
    print(f"{'='*50}")
    print(f"⏱️  TOTAL PIPELINE TIME:       {format_time(total_end_time - total_start_time)}")
    print(f"{'='*50}")
    
    # Average times per image - more detailed breakdown
    if processed_count > 0:
        avg_total_per_image = (total_end_time - total_start_time) / processed_count
        avg_classification_per_image = classification_total_time / processed_count
        avg_label_processing_per_image = label_processing_total_time / processed_count
        
        print(f"\n📊 PER-IMAGE TIMING BREAKDOWN:")
        print(f"   ├─ Total time per image:           {format_time(avg_total_per_image)}")
        print(f"   ├─ Classification per image:       {format_time(avg_classification_per_image)}")
        print(f"   ├─ Label processing per image:     {format_time(avg_label_processing_per_image)}")
        
        if adv_count > 0:
            avg_preprocessing_per_adv = preprocessing_total_time / adv_count
            print(f"   ├─ Preprocessing per ADV image:    {format_time(avg_preprocessing_per_adv)}")
            
            # Show what an average adversarial image processing looks like
            avg_adv_total = avg_classification_per_image + avg_preprocessing_per_adv + avg_label_processing_per_image
            print(f"   └─ Total per ADVERSARIAL image:    {format_time(avg_adv_total)} (classification + preprocessing + labels)")
        
        clean_count = processed_count - adv_count
        if clean_count > 0:
            avg_file_io_per_clean = file_io_total_time / clean_count if clean_count > 0 else 0
            avg_clean_total = avg_classification_per_image + avg_file_io_per_clean + avg_label_processing_per_image
            print(f"   └─ Total per CLEAN image:          {format_time(avg_clean_total)} (classification + copy + labels)")
        
        print(f"\n📈 PROCESSING DISTRIBUTION:")
        print(f"   ├─ Adversarial images: {adv_count}/{processed_count} ({(adv_count/processed_count)*100:.1f}%)")
        print(f"   └─ Clean images:       {clean_count}/{processed_count} ({(clean_count/processed_count)*100:.1f}%)")

    print(f"\nInput images:  {len(image_paths)}")
    print(f"Output images: {output_image_count}")
    print(f"Input labels:  {input_label_count}")
    print(f"Output labels: {output_label_count}")
    print(f"Adversarial images: {adv_count}")

    # Detailed mismatch report
    if len(image_paths) != output_image_count:
        print(f"‼ IMAGE COUNT MISMATCH: expected {len(image_paths)}, got {output_image_count}")
    if len(image_paths) != output_label_count:
        print(f"‼ LABEL COUNT MISMATCH: expected {len(image_paths)}, got {output_label_count}")
    if missing_output_images:
        print("Missing processed images:", missing_output_images)
    if missing_output_labels:
        print("Missing processed labels:", missing_output_labels)
    if len(image_paths) == output_image_count and len(image_paths) == output_label_count:
        print("All counts match!")
    else:
        print("Check the mismatches above.")

    print(f"Done. Processed {processed_count} images.")