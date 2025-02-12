import torch
from util import write_results
from yolo_detector import prepare_image


import os
import torch
from torchvision.utils import save_image

def dag_attack(
    model, 
    images, 
    target_objects, 
    adversarial_classes, 
    max_iter=20, 
    gamma=0.1, 
    CUDA=True
):
    # Prepare images. (Assumes prepare_image returns batches of tensors, image dimensions, etc.)
    im_batches, _, _, _ = prepare_image(images, model.net_info["height"])
    adversarial_images = []
    perturbations = []

    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        
        original_batch = batch.clone().detach()
        # Initialize perturbation with requires_grad so we can compute gradients on it.
        perturbation = torch.zeros_like(original_batch, requires_grad=True)
        attack_successful = False
        
        for m in range(max_iter):
            model.zero_grad()
            adv_batch = original_batch + perturbation
            adv_batch = torch.clamp(adv_batch, 0, 1)
            
            # Ensure gradients are enabled for the attack.
            with torch.enable_grad():
                prediction = model(adv_batch, CUDA)
            
            filtered_preds = write_results(prediction, confidence=0.5, num_classes=80, nms_conf=0.4)
            
            # Early stopping: if no detections match our target objects, stop the attack.
            remaining_targets = 0
            if not isinstance(filtered_preds, int):
                # (Optional) Print out detected class IDs for debugging.
                for det in filtered_preds:
                    class_id = int(det[-1].item())
                    print(f"Class ID: {class_id}")
                remaining_targets = sum(1 for det in filtered_preds if int(det[-1].item()) in target_objects)
            
            if remaining_targets == 0:
                print(f"Iteration {m+1}: Attack succeeded. Stopping early.")
                attack_successful = True
                break
            
            # Compute loss. We initialize a tensor with gradients enabled.
            loss = torch.tensor(0.0, device=adv_batch.device, requires_grad=True)
            
            # Skip loss computation if there are no detections.
            if isinstance(filtered_preds, int):
                continue  
            
            for det in filtered_preds:
                class_id = int(det[-1].item())
                if class_id in target_objects:
                    idx = target_objects.index(class_id)
                    adv_class = adversarial_classes[idx]
                    # Increase the logit for the adversarial class and reduce for the original class.
                    loss = loss + prediction[..., 5 + adv_class].sum()
                    loss = loss - prediction[..., 5 + class_id].sum()
            
            # If loss is valid, backpropagate to update the perturbation.
            if loss.grad_fn is not None:
                loss.backward(retain_graph=True)
                print(f"Step {m+1}/{max_iter}: Loss = {loss.item()}")
                
                grad = perturbation.grad.data
                grad_norm = torch.norm(grad, p=float('inf'))
                # Avoid division by zero.
                if grad_norm == 0:
                    grad_norm = 1e-8
                perturbation_step = (gamma / grad_norm) * grad
                perturbation.data += perturbation_step
                perturbation.grad.zero_()
            else:
                continue
        
        adversarial_images.append(adv_batch.detach())
        perturbations.append(perturbation.detach())
    
    # ===== SAVE ADVERSARIAL IMAGES AT THE END =====
    # Create the output directory if it doesn't exist.
    output_dir = "dag_adversarial_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # If adversarial_images contains batches (tensors with shape [B, 3, H, W]), flatten them.
    flat_adv_images = []
    for batch in adversarial_images:
        if batch.dim() == 4:  # If it's a batch, iterate over images in the batch.
            for img in batch:
                flat_adv_images.append(img)
        else:
            flat_adv_images.append(batch)
    
    # Save each adversarial image with a 12-digit zero-padded index.
    for i, adv_img in enumerate(flat_adv_images):
        # Format the file name. For example, the first image (i==0) becomes:
        # "000000000001_adv_dag.jpg"
        file_name = os.path.join(output_dir, f"{i+1:012d}_adv_dag.png")
        detach_adv_img = adv_img.detach().cpu()
        detach_adv_img = torch.clamp(detach_adv_img, 0.0, 1.0)
        save_image(detach_adv_img, file_name)
        print(f"Saved adversarial image to: {file_name}")
    # =============================================
    
    return adversarial_images, perturbations


def dag_disappearance_attack(
    model, 
    images, 
    target_objects, 
    max_iter=20, 
    gamma=0.1, 
    CUDA=True
):
    im_batches, _, _, _ = prepare_image(images, model.net_info["height"])
    adversarial_images = []
    perturbations = []

    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        
        original_batch = batch.clone().detach()
        perturbation = torch.zeros_like(original_batch, requires_grad=True)
        attack_successful = False
        
        for m in range(max_iter):
            model.zero_grad()
            adv_batch = original_batch + perturbation
            adv_batch = torch.clamp(adv_batch, 0, 1)
            
            with torch.enable_grad():
                prediction = model(adv_batch, CUDA)
            
            filtered_preds = write_results(prediction, confidence=0.5, num_classes=80, nms_conf=0.4)
            
            remaining_targets = 0
            if not isinstance(filtered_preds, int):
                for det in filtered_preds:
                    class_id = int(det[-1].item())
                    print(f"Class ID: {class_id}")
                remaining_targets = sum(1 for det in filtered_preds if int(det[-1].item()) in target_objects)
            
            if remaining_targets == 0:
                print(f"Iteration {m+1}: Attack succeeded. Stopping early.")
                attack_successful = True
                break
            
            loss = torch.tensor(0.0, device=adv_batch.device, requires_grad=True)
            
            if remaining_targets == 1:
                for det in filtered_preds:
                    class_id = int(det[-1].item())
                    if class_id in target_objects:
                        loss = loss - det[4]
            else:
                for det in filtered_preds:
                    class_id = int(det[-1].item())
                    if class_id in target_objects:
                        loss = loss - det[4]
            
            if loss.grad_fn is not None:
                loss.backward(retain_graph=True)
                print(f"Step {m+1}/{max_iter}: Loss = {loss.item()}")
                
                grad = perturbation.grad.data
                grad_norm = torch.norm(grad, p=float('inf'))
                perturbation_step = (gamma / grad_norm) * grad
                perturbation.data += perturbation_step
                perturbation.grad.zero_()
            else:
                continue
        
        adversarial_images.append(adv_batch.detach())
        perturbations.append(perturbation.detach())
    
    # =====  SAVING  =====
    output_dir = "disappearance_adversarial_images"
    os.makedirs(output_dir, exist_ok=True)
    
    flat_adv_images = []
    for batch in adversarial_images:
        if batch.dim() == 4:
            for img in batch:
                flat_adv_images.append(img)
        else:
            flat_adv_images.append(batch)
    
    for i, adv_img in enumerate(flat_adv_images):
        file_name = os.path.join(output_dir, f"{i+1:012d}_adv_disappear.png")
        detach_adv_img = adv_img.detach().cpu()
        detach_adv_img = torch.clamp(detach_adv_img, 0.0, 1.0)
        save_image(detach_adv_img, file_name)
        print(f"Saved adversarial image to: {file_name}")
    # =============================
    
    return adversarial_images, perturbations
