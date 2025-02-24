import torch
from util import write_results
from yolo_detector import prepare_image


import os
import torch
from torchvision.utils import save_image

def targeted_dag_attack(
    model, 
    images, 
    target_objects, 
    adversarial_classes, 
    max_iter=20, 
    gamma=0.1, 
    output_dir="targeted_dag_attack_adversarial_images",
    im_name_prefix="",
    CUDA=True
):
    # Prepare images
    im_batches, _, _, _ = prepare_image(images, model.net_info["height"])
    adversarial_images = []
    perturbations = []

    for batch_idx, batch in enumerate(im_batches):
        if CUDA:
            batch = batch.cuda()
        
        original_batch = batch.clone().detach()
        perturbation = torch.zeros_like(original_batch, requires_grad=True)
        attack_successful = False
        
        for m in range(max_iter):
            model.zero_grad()
            adv_batch = original_batch + perturbation
            adv_batch = torch.clamp(adv_batch, 0, 1)
            
            # ====== Always print iteration start ======
            print(f"\nBatch {batch_idx+1} | Iter {m+1}/{max_iter}:")

            # Forward pass with gradients
            with torch.enable_grad():
                prediction = model(adv_batch, CUDA)
            
            filtered_preds = write_results(prediction, confidence=0.5, num_classes=80, nms_conf=0.4)
            
            # ====== Early Stopping Condition ======
            stop_attack = False
            if not isinstance(filtered_preds, int):
                detected_classes = [int(det[-1].item()) for det in filtered_preds]
                detected_set = set(detected_classes)
                adv_presence = [adv_cls in detected_set for adv_cls in adversarial_classes]
                all_adv_present = all(adv_cls in detected_set for adv_cls in adversarial_classes)
                no_targets_left = all(target_cls not in detected_set for target_cls in target_objects)
                
                if all_adv_present and no_targets_left:
                    print(f"  Early stopping condition met!")
                    stop_attack = True
            else:
                print(f"  No detections")
                no_targets_left = True if target_objects else False
                all_adv_present = False
                stop_attack = no_targets_left and (not adversarial_classes)

            if stop_attack:
                attack_successful = True
                break
            # ======================================

            # ====== Always show detection status ======
            if not isinstance(filtered_preds, int):
                print(f"  Detected classes: {[int(d[-1].item()) for d in filtered_preds]}")
            # ==========================================

            # Original loss calculation
            loss = torch.tensor(0.0, device=adv_batch.device, requires_grad=True)
            
            if isinstance(filtered_preds, int):
                print(f"  No target detections - skipping loss calculation")
                continue

            # Calculate loss only if targets exist in detections
            target_found = False
            for det in filtered_preds:
                class_id = int(det[-1].item())
                if class_id in target_objects:
                    target_found = True
                    idx = target_objects.index(class_id)
                    adv_class = adversarial_classes[idx]
                    loss = loss + prediction[..., 5 + adv_class].sum() - prediction[..., 5 + class_id].sum()
            
            if not target_found:
                # Get currently detected classes
                detected_set = set()
                if not isinstance(filtered_preds, int):
                    detected_set = {int(det[-1].item()) for det in filtered_preds}
                
                # Identify undetected adversarial classes
                undetected_adv = [cls for cls in adversarial_classes if cls not in detected_set]
                
                # Boost confidence for undetected classes
                for cls in undetected_adv:
                    # Negative sign = maximize confidence (lower loss → higher confidence)
                    loss = loss + prediction[..., 5 + cls].sum()  
                
                print(f"  Boosting undetected classes: {undetected_adv}")

            if loss.grad_fn is not None:
                loss.backward(retain_graph=True)
                print(f"  Current loss: {loss.item():.4f}")
                
                # Update perturbation
                grad = perturbation.grad.data
                grad_norm = torch.norm(grad, p=float('inf'))
                if grad_norm == 0:
                    grad_norm = 1e-8
                perturbation.data += (gamma / grad_norm) * grad
                perturbation.grad.zero_()

        # Final status
        status = "SUCCESS" if attack_successful else f"REACHED MAX ITERS ({max_iter})"
        print(f"\nBatch {batch_idx+1} Final Status: {status}")
        print("="*50)

        # Store results
        final_adv = torch.clamp(original_batch + perturbation, 0, 1)
        adversarial_images.append(final_adv.detach())
        perturbations.append(perturbation.detach())

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(torch.cat(adversarial_images)):
        save_image(img, os.path.join(output_dir, f"{im_name_prefix}_adv_{i:04d}.png"))
    
    return adversarial_images, perturbations


def disappearance_dag_attack(
    model, 
    images, 
    target_objects, 
    max_iter=20, 
    gamma=0.1, 
    output_dir="disappearance_adversarial_images",
    im_name_prefix="",
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
            #print the detected classes
            if not isinstance(filtered_preds, int):
                print(f"  Detected classes: {[int(d[-1].item()) for d in filtered_preds]}")
            if not isinstance(filtered_preds, int):
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
    os.makedirs(output_dir, exist_ok=True)
    
    flat_adv_images = []
    for batch in adversarial_images:
        if batch.dim() == 4:
            for img in batch:
                flat_adv_images.append(img)
        else:
            flat_adv_images.append(batch)
    
    for i, adv_img in enumerate(flat_adv_images):
        file_name = os.path.join(output_dir, f"{im_name_prefix}_{i+1:012d}_adv_disappear.png")
        detach_adv_img = adv_img.detach().cpu()
        detach_adv_img = torch.clamp(detach_adv_img, 0.0, 1.0)
        save_image(detach_adv_img, file_name)
        print(f"Saved adversarial image to: {file_name}")
    # =============================
    
    return adversarial_images, perturbations



