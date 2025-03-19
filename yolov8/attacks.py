import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torchvision import transforms

def disappearance_dag_attack(
    image_path,
    raw_model_path='yolov8n.pt',
    high_level_model_path='yolov8n.pt',
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    device='cpu'
):
    """
    Demonstration of a DAG-like multi-iteration attack on YOLOv8
    that pushes all predictions (with high confidence) away from
    their currently predicted class.

    Args:
        image_path (str): Path to the input image.
        raw_model_path (str): Path to the YOLOv8 weights for the raw model.
        high_level_model_path (str): Path to the YOLOv8 weights for the high-level model.
        num_iterations (int): Number of attack iterations.
        gamma (float): L_inf step size per iteration.
        conf_threshold (float): Confidence threshold for selecting rows to attack.
        device (str): 'cuda' or 'cpu'.
    """

    # ----------------------------------------------------------------
    # 1. Load the models
    # ----------------------------------------------------------------
    # High-level model (used for convenience to see post-processed predictions)
    high_level_model = YOLO(high_level_model_path)

    # Underlying raw model (used for gradient computations)
    underlying_model = YOLO(raw_model_path).model
    underlying_model.eval()
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # ----------------------------------------------------------------
    # 2. Preprocess and load the image
    # ----------------------------------------------------------------
    image = preprocess_image(image_path)  # should return a torch.Tensor of shape [1, 3, H, W]
    image = image.to(device)
    image.requires_grad_(True)

    # ----------------------------------------------------------------
    # 3. Iterative attack loop
    # ----------------------------------------------------------------
    for iteration in range(num_iterations):
        # ------------------------------------------------------------
        # (a) Forward pass through the raw underlying model
        # ------------------------------------------------------------
        # The raw model can return a tuple (preds, training_outputs), so unpack:
        raw_preds, _ = underlying_model(image)

        # raw_preds might be shape [1, 84, N], so we permute to [1, N, 84]
        raw_preds = raw_preds.permute(0, 2, 1)  # now [1, N, 84]
        # Split out the class logits: first 4 channels are box coords, next 80 are class scores
        class_logits = raw_preds[..., 4:]  # shape [1, N, 80]

        # Apply sigmoid if these are logits
        conf = class_logits.sigmoid()  # shape [1, N, 80]

        # Get max confidence per row and the corresponding predicted class
        pred_conf, pred_class = conf.max(dim=-1)  # both [1, N]

        # ------------------------------------------------------------
        # (b) Identify rows to attack
        # ------------------------------------------------------------
        # Let's see what the *high-level* model is currently predicting
        # (This is optional, but helps you track final bounding boxes, etc.)
        with torch.no_grad():
            high_level_preds = high_level_model(image, conf=conf_threshold)
        results = high_level_preds[0]  # 'Results' object
        pred_class_ids = results.boxes.cls.tolist()  # the classes YOLO is showing after NMS

        # Now, from the raw model side, pick all rows with conf > conf_threshold
        # whose predicted class is in pred_class_ids. (We treat them as "currently predicted")
        rows_to_attack = []
        target_classes = set(int(x) for x in pred_class_ids)
        for i in range(pred_class.shape[1]):
            row_pred_class = int(pred_class[0, i].item())
            row_pred_conf  = float(pred_conf[0, i].item())
            if row_pred_class in target_classes and row_pred_conf > conf_threshold:
                rows_to_attack.append(i)

        # If no rows to attack, we can break early
        if not rows_to_attack:
            print(f"Iteration {iteration}: no rows above threshold => stopping early.")
            break

        # ------------------------------------------------------------
        # (c) Build the loss
        # ------------------------------------------------------------
        # We'll do an untargeted approach: push the logit of the predicted class down.
        loss = torch.zeros(1, device=device)
        for i in rows_to_attack:
            c = pred_class[0, i]  # the predicted class index
            logit_c = class_logits[0, i, c]
            loss += -logit_c  # negative logit => push it down

        # ------------------------------------------------------------
        # (d) Backprop and update the image
        # ------------------------------------------------------------
        # Zero out old gradients
        if image.grad is not None:
            image.grad.zero_()

        # Backprop
        loss.backward()

        # Get the gradient on the image
        grad = image.grad.data

        # L-infinity step
        step = gamma * grad.sign()
        with torch.no_grad():
            image += step
            image.clamp_(0, 1)

        image.grad.zero_()  # reset for next iteration

        print(f"Iteration {iteration}: #rows_to_attack={len(rows_to_attack)}, loss={loss.item():.4f}")

    # ----------------------------------------------------------------
    # 4. Return or save the final adversarial image
    # ----------------------------------------------------------------
    return image.detach()



def targeted_dag_attack(
    image_path,
    raw_model_path='yolov8n.pt',
    high_level_model_path='yolov8n.pt',
    adversarial_class=2,
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    device='cpu'
):
    """
    Demonstration of a DAG-like targeted attack that:
      1) Attacks rows whose predicted class is in 'target_classes' (from the clean image).
      2) If no rows remain, we push the adversarial_class globally (so it still emerges).
      3) We stop early if the adversarial_class appears in high-level predictions
         and none of the original target classes remain.

    Args:
        image_path (str): Path to input image.
        raw_model_path (str): YOLOv8 weights for the raw model (for gradients).
        high_level_model_path (str): YOLOv8 weights for the high-level model (for post-NMS checks).
        adversarial_class (int): Class index to push everything toward.
        num_iterations (int): Max number of iterations.
        gamma (float): L_inf step size.
        conf_threshold (float): Confidence threshold for selecting rows and for post-NMS.
        device (str): 'cuda' or 'cpu'.
    """

    # ---------------------------------------------------------------
    # 1. Load the models
    # ---------------------------------------------------------------
    high_level_model = YOLO(high_level_model_path)
    underlying_model = YOLO(raw_model_path).model
    underlying_model.eval()
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # ---------------------------------------------------------------
    # 2. Load and prepare the image
    # ---------------------------------------------------------------
    image = preprocess_image(image_path)  # [1, 3, H, W]
    image = image.to(device)
    image.requires_grad_(True)

    # ---------------------------------------------------------------
    # 3. Determine original target classes from the clean image
    # ---------------------------------------------------------------
    with torch.no_grad():
        clean_preds = high_level_model(image, conf=conf_threshold)
    clean_results = clean_preds[0]
    original_pred_class_ids = clean_results.boxes.cls.tolist()
    target_classes = set(int(x) for x in original_pred_class_ids)

    print(f"Original classes above conf={conf_threshold}: {target_classes}")
    # If none found, we can either return or proceed with an empty set.

    # ---------------------------------------------------------------
    # 4. Attack loop
    # ---------------------------------------------------------------
    for iteration in range(num_iterations):
        # -----------------------------------------------------------
        # (a) Raw forward pass (for gradients)
        # -----------------------------------------------------------
        raw_preds, _ = underlying_model(image)        # shape [1, 84, N]
        raw_preds = raw_preds.permute(0, 2, 1)        # [1, N, 84]
        class_logits = raw_preds[..., 4:]             # [1, N, 80]
        conf = class_logits.sigmoid()                # [1, N, 80]
        pred_conf, pred_class = conf.max(dim=-1)      # both [1, N]
        num_rows = pred_class.shape[1]

        # -----------------------------------------------------------
        # (b) Identify rows to attack (pred_class in target_classes)
        # -----------------------------------------------------------
        rows_to_attack = []
        for i in range(num_rows):
            row_pred_class = int(pred_class[0, i].item())
            row_pred_conf  = float(pred_conf[0, i].item())
            if row_pred_class in target_classes and row_pred_conf > conf_threshold:
                rows_to_attack.append(i)

        # -----------------------------------------------------------
        # (c) Build the loss
        # -----------------------------------------------------------
        loss = torch.zeros(1, device=device)
        if rows_to_attack:
            # (1) Usual targeted push: (logit_adv - logit_orig)
            for i in rows_to_attack:
                c = pred_class[0, i]
                logit_orig = class_logits[0, i, c]
                logit_adv  = class_logits[0, i, adversarial_class]
                loss += (logit_adv - logit_orig)
        else:
            # (2) If no rows remain, just boost the adversarial class globally
            #     so we force it to appear somewhere in the image.
            print(f"Iteration {iteration}: no rows above threshold => boosting adversarial_class globally.")
            for i in range(num_rows):
                logit_adv = class_logits[0, i, adversarial_class]
                loss += logit_adv

        # -----------------------------------------------------------
        # (d) Backprop + update
        # -----------------------------------------------------------
        if image.grad is not None:
            image.grad.zero_()
        loss.backward()

        grad = image.grad.data
        step = gamma * grad.sign()
        with torch.no_grad():
            image += step
            image.clamp_(0, 1)
        image.grad.zero_()

        # -----------------------------------------------------------
        # (e) Check stop condition
        #     If adversarial_class is in current predictions
        #     AND none of the original target_classes remain
        # -----------------------------------------------------------
        with torch.no_grad():
            current_preds = high_level_model(image, conf=conf_threshold)
        results = current_preds[0]
        current_class_ids = set(int(box.cls.item()) for box in results.boxes)
        # Condition: adversarial_class in current_class_ids AND disjoint from target_classes
        if (adversarial_class in current_class_ids) and current_class_ids.isdisjoint(target_classes):
            print(f"Iteration {iteration}: Stop condition met! {adversarial_class=} present, {target_classes=} gone.")
            break

        print(f"Iteration {iteration}: #rows_to_attack={len(rows_to_attack)}, loss={loss.item():.4f}")

    return image.detach()



def fool_detectors_attack(
    image_path,
    raw_model_path='yolov8n.pt',
    high_level_model_path='yolov8n.pt',
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    lambda_reg=0.01,
    target_classes=[11],
    device='cpu'
):
    """
    Implements a DAG-like multi-iteration attack on YOLOv8 targeting specified classes,
    inspired by the paper "Adversarial Examples that Fool Detectors".
    
    The attack minimizes the mean detection confidence for the target classes in the image by
    iteratively updating the input image using the sign of the gradient (L_inf update). An L2 penalty 
    is added to ensure the adversarial image remains close to the original. The attack stops early if 
    the high-level model no longer detects any objects of the target classes.
    
    Args:
        image_path (str): Path to the input image.
        raw_model_path (str): Path to the YOLOv8 weights for the raw model (used for gradient computation).
        high_level_model_path (str): Path to the YOLOv8 weights for the high-level model (used for detection).
        num_iterations (int): Number of iterations for the attack.
        gamma (float): Step size for the L_inf gradient update.
        conf_threshold (float): Confidence threshold for selecting detections.
        lambda_reg (float): Weight for the L2 penalty to maintain similarity to the original image.
        target_classes (list of int): List of COCO class IDs to target (e.g., [11] for stop signs).
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: The final adversarial image tensor.
    
    Alternative Name:
        FoolDetectorsAttack
    """
    # Use target_classes as provided.
    
    # 1. Load the YOLO models.
    # --------------------------------------------------------------------------
    # The high-level model is used to check detection results (post-processing),
    # while the underlying (raw) model is used for gradient-based updates.
    high_level_model = YOLO(high_level_model_path)
    underlying_model = YOLO(raw_model_path).model
    underlying_model.eval()  # Set to evaluation mode.
    # Disable gradient computations for the underlying model parameters.
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # 2. Preprocess and load the input image.
    # --------------------------------------------------------------------------
    # preprocess_image should return a torch.Tensor of shape [1, 3, H, W]
    image = preprocess_image(image_path)
    image = image.to(device)
    # Save a copy of the original image for the L2 (similarity) penalty.
    original_image = image.clone().detach()
    # Enable gradient computation on the input image.
    image.requires_grad_(True)

    # 3. Iterative attack loop.
    # --------------------------------------------------------------------------
    for iteration in range(num_iterations):
        # Forward pass: compute predictions using the raw model.
        raw_preds, _ = underlying_model(image)

        # Rearrange predictions from shape [1, channels, N_boxes] to [1, N_boxes, channels].
        raw_preds = raw_preds.permute(0, 2, 1)
        # Extract class logits; assuming the first 4 channels are box coordinates.
        class_logits = raw_preds[..., 4:]
        # Convert logits to confidence scores using the sigmoid activation.
        conf_scores = class_logits.sigmoid()  # Shape: [1, N_boxes, num_classes]

        # Extract confidence scores for the target classes.
        # This indexes dimension 2 with the list of target classes.
        target_confidences = conf_scores[..., target_classes]  # Shape: [1, N_boxes, len(target_classes)]
        # Compute the detector loss as the mean confidence score for target classes.
        loss_det = target_confidences.mean()

        # Compute the L2 penalty to ensure the adversarial image remains similar to the original.
        loss_l2 = lambda_reg * F.mse_loss(image, original_image)

        # Total loss: detector loss plus the L2 penalty.
        loss = loss_det + loss_l2

        # Optional: Use the high-level model to check if any target objects are detected.
        with torch.no_grad():
            high_level_preds = high_level_model(image, conf=conf_threshold)
        results = high_level_preds[0]
        # Extract predicted class IDs (assuming results.boxes.cls holds these).
        detected_classes = results.boxes.cls.tolist() if hasattr(results.boxes, "cls") else []
        # Early stopping: if none of the detected classes is in target_classes, break.
        if not set(target_classes).intersection(set(detected_classes)):
            print(f"Iteration {iteration}: No target objects detected. Early stopping.")
            break

        # Backpropagation: Compute gradients of the loss with respect to the image.
        if image.grad is not None:
            image.grad.zero_()  # Reset gradients.
        loss.backward()

        # Get the gradient and update the image in the negative gradient direction to minimize the loss.
        grad = image.grad.data
        with torch.no_grad():
            # Subtract the sign of the gradient scaled by gamma (L_inf update).
            image -= gamma * grad.sign()
            # Ensure the pixel values remain valid (clamped to [0, 1]).
            image.clamp_(0, 1)
        image.grad.zero_()  # Reset gradient for next iteration.

        # Print the current loss values for debugging.
        print(f"Iteration {iteration}: Loss_det={loss_det.item():.4f}, "
              f"Loss_L2={loss_l2.item():.4f}, Total_loss={loss.item():.4f}")

    # Return the final adversarial image tensor.
    return image.detach()

# Adversarial patch attack

# Define the loss function for the Creation Attack
def creation_attack_loss(raw_pred, confidences, patch_location,target_id):
    """
    Compute the Creation Attack Loss.
    Args:
        raw_pred: raw output tensor from YOLO v8 to get box location.
        confidences: classes confidences to optimize for the target class.
        patch_location: Tuple (x, y) representing the grid cell where the patch is placed.
    return: 
    Loss value
    stop flag
    """
    loss = None
    grid_cell_x, grid_cell_y = patch_location
    # loop on cells to get the patch cell 
    for pred, conf in zip(raw_pred[0], confidences):
        x_min,y_min = pred[0].int(), pred[1].int()
        if (x_min == int(grid_cell_x) and y_min == int(grid_cell_y) ): 
            loss = 1 - conf[target_id]   # Minimize this to increase the target class probability
            break
    
    return loss

# Generate an initial random patch
def preprocess_init_patch(size):
    """
    preprocess initial patch.
    Args:
        size: Tuple (width, height) for the patch size.
    return: 
        initial patch as a numpy array.
    """
    patch = Image.open('ss_patch.png').convert('RGB')
    patch = patch.resize(size)
    patch = np.array(patch) / 255.0
    patch = np.transpose(patch, (2,0,1))
    return patch
## Apply the patch to an image at a fixed random location
def apply_patch(image, patch, patch_location, patch_size):
    """
    Apply the adversarial patch to an image at a predetermined location.
    Args:
        image: Input image (tensor).
        patch: Adversarial patch (tensor).
        patch_location: Tuple (x, y) representing the location to place the patch.
    return:
        Tensor image with the patch applied.
    """

    _, _, image_width, image_height= image.shape
    patch_w, patch_h= patch_size

    x, y = patch_location
    # Ensure the patch fits within the image boundaries
    x_start = max(0, int(x) - patch_w // 2)
    y_start = max(0, int(y) - patch_h // 2)
    x_end = min(image_width, x_start + patch_w)
    y_end = min(image_height, y_start + patch_h)

    # Adjust patch slicing if the patch extends beyond the image boundaries
    patch_x_start = 0
    patch_y_start = 0
    patch_x_end = patch_w 
    patch_y_end = patch_h

    # Ensure the target region and patch have the same dimensions
    target_region = image[:, :, y_start:y_end, x_start:x_end]
    patch_slice = patch[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end]

    # Check if dimensions match
    if target_region.shape[2:] != patch_slice.shape[1:]:
        print(f"Warning: Patch dimensions {patch_slice.shape[1:]} do not match target region dimensions {target_region.shape[2:]}. Resizing patch to fit.")
        
        # Resize the patch to match the target region dimensions
        from torch.nn.functional import interpolate
        patch_slice = interpolate(
            patch_slice.unsqueeze(0),  # Add batch dimension
            size=target_region.shape[2:],  # Target size (height, width)
            mode='bilinear',  # Interpolation mode
            align_corners=False
        ).squeeze(0)  # Remove batch dimension

    # Apply the patch to the image
    image[:, :, y_start:y_end, x_start:x_end] = patch_slice
    return image

# Optimize the adversarial patch
def optimize_patch(
    image_path,
    patch, 
    target_id,
    num_iterations=200,
    raw_model_path="yolov8n_road.pt",
    learning_rate=0.06,
    conf_th = 0.5,
    device = "cuda"
):
    """
    Optimize the adversarial patch using gradient descent.
    Args:
        images: List of images for optimization.
        patch: Initial adversarial patch.
        num_iterations: Number of optimization iterations.
        raw_model_path (str): YOLOv8 weights for the raw model (for gradients).
        learning_rate: Learning rate for gradient descent.
    return: 
        Optimized adversarial patch.
    """
    # Move the patch to the model device
    if device == "cuda":
        patch = torch.tensor(patch, dtype=torch.float32, device="cuda", requires_grad=True)
    else:
        patch = torch.tensor(patch, dtype=torch.float32, requires_grad=True)
    
    # Load the yolo model
    high_level_model = YOLO(raw_model_path)
    underlying_model = YOLO(raw_model_path).model
    underlying_model.eval()  # Set to evaluation mode.
    # Disable gradient computations for the underlying model parameters.
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)  
    
    # Preprocess and load the input image.
    image = preprocess_image(image_path)
    image = image.to(device)
    # Save a copy of the original image
    original_image = image.clone().detach()

    
    # Forward pass: compute predictions using the raw model.
    raw_preds, _ = underlying_model(original_image)
    # Rearrange predictions from shape [1, channels, N_boxes] to [1, N_boxes, channels].
    raw_preds = raw_preds.permute(0, 2, 1)   
    class_logits = raw_preds[0][:,4:]            
    conf = class_logits.sigmoid()                
   
    # Select the box with the highest class probability for the target class
    best_pred_idx = torch.argmax(conf[:, target_id]).item()
    best_pred = raw_preds[0][best_pred_idx,:]
    # Get the bounding box coordinates (x, y) for the best prediction
    patch_x, patch_y = best_pred[0].item(), best_pred[1].item()
    patch_location = (patch_x, patch_y)
    patch_size = (best_pred[2].int().item(), best_pred[3].int().item())
    for iteration in range(num_iterations):
        torch.cuda.empty_cache()
        # Original loss calculation
        loss = torch.tensor(0.0, device=image.device,requires_grad=True)
        # apply patch on determined box
        patched_image = apply_patch(original_image.clone(), patch, patch_location, patch_size)
        
        # Forward pass: Compute predictions using the raw model on patched image.
        raw_preds, _ = underlying_model(patched_image)  
        raw_preds = raw_preds.permute(0, 2, 1)
        class_logits = raw_preds[0][:, 4:]             
        conf = class_logits.sigmoid()  

        # Compute the loss
        loss = creation_attack_loss(raw_preds,conf, patch_location,target_id)

        # stop when target object added
        with torch.no_grad():
            current_preds = high_level_model(patched_image, conf=conf_th)
        results = current_preds[0]
        current_class_ids = set(int(box.cls.item()) for box in results.boxes)
        # Condition: adversarial_class in current_class_ids
        if (target_id in current_class_ids):
            print(f"Iteration {iteration}: Stop condition met!")
            break

        # Backpropagate and update the patch
        if loss is not None:
            loss.backward(retain_graph=True)
            # Manually update the patch using gradient descent
            with torch.no_grad():
                grad = patch.grad.data
                grad_norm = torch.norm(grad, p=float('inf'))
                if grad_norm == 0:
                    grad_norm = 1e-8
                patch -= learning_rate * (grad / grad_norm)# Update patch
                patch.grad.zero_()  # Reset gradients
                # Clip the patch values to ensure they are valid pixel values
                patch.data = torch.clamp(patch.data, 0, 1)
        
        

            # Re-enable gradients for the next iteration
            patch.requires_grad = True
        
            # Print the loss for monitoring            
            print(f"Iteration {iteration}, Loss: {loss.item()}")
        
        else:
            print(f"Iteration {iteration}: location not found")
            # change patch location 
            # Select the prediction with the highest class probability for the target class
            best_pred_idx = torch.argmax(conf[:, target_id]).item()
            best_pred = raw_preds[0][best_pred_idx,:]
            # Get the bounding box coordinates (x, y) for the best prediction
            patch_x, patch_y= best_pred[0].item(), best_pred[1].item()
            patch_location = (patch_x, patch_y)
            
            
    return patched_image.detach().cpu(),patch.detach().cpu()




import torch
from tqdm import tqdm
from ultralytics import YOLO

def generate_yolov8_uap(
    model: YOLO,
    dataloader: torch.utils.data.DataLoader,
    epsilon: float = 8/255,
    lr: float = 0.01,
    num_epochs: int = 10,
    device: str = "cuda",
    verbose: bool = True
) -> torch.Tensor:
    """
    Generates a Universal Adversarial Perturbation (UAP) for YOLOv8.
    
    Args:
        model (YOLO): Pretrained YOLOv8 model
        dataloader (DataLoader): Training data loader with (images, targets)
        epsilon (float): Maximum perturbation magnitude (L∞ norm)
        lr (float): Learning rate for optimization
        num_epochs (int): Number of attack epochs
        device (str): Device for computation ('cuda' or 'cpu')
        verbose (bool): Print progress
    
    Returns:
        torch.Tensor: Universal adversarial perturbation tensor (C, H, W)
    """
    
    # Initialize perturbation
    img_sample, _ = next(iter(dataloader))
    _, C, H, W = img_sample.shape
    delta = torch.zeros((1, C, H, W), requires_grad=True, device=device)
    
    # Optimizer (SGD with momentum works better for UAP)
    optimizer = torch.optim.SGD([delta], lr=lr, momentum=0.9)
    
    # Training loop
    model.to(device)
    model.train()  # Required to access loss components
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for images, targets in tqdm(dataloader, disable=not verbose):
            images = images.to(device)
            targets = [{"bbox": t[0].to(device), "cls": t[1].to(device)} for t in targets]
            
            # Apply perturbation (before model preprocessing)
            perturbed_images = torch.clamp(images + delta, 0, 1)
            
            # Forward pass (YOLOv8 returns loss dict in training mode)
            outputs = model(perturbed_images)
            
            # Extract YOLOv8's composite loss
            loss = sum([
                outputs.loss_box,  # Bounding box regression loss
                outputs.loss_cls,  # Classification loss
                outputs.loss_obj   # Objectness loss
            ])
            
            # Maximize loss to degrade detection
            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent
            optimizer.step()
            
            # Project perturbation to epsilon L∞-ball
            with torch.no_grad():
                delta.data = torch.clamp(delta, -epsilon, epsilon)
            
            epoch_loss += loss.item()
        
        if verbose:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    return delta.data.squeeze(0)  # Return (C, H, W) perturbation