import os
import cv2
import torch
import numpy as np
from util import write_results
from yolo_detector import prepare_image


def reverse_prepare_image(im_batches, im_dim_list, save_dir, inp_dim):
    """
    Reverse the operations performed by prepare_image and save images to a directory.
    
    Args:
        im_batches (list of torch.Tensor): List of image tensors.
        im_dim_list (torch.Tensor): Original dimensions of images.
        save_dir (str): Directory to save restored images.
        inp_dim (int): Input dimension used during preprocessing.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    index = 0
    for batch in im_batches:
        for img_tensor in batch:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)  # Convert back to uint8 image
            
            # Convert RGB to BGR (OpenCV format)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Extract original width and height correctly
            orig_w, orig_h = im_dim_list[index][:2].int().tolist()
            
            # Compute scaling factor (same used during preprocessing)
            scale = min(inp_dim / orig_w, inp_dim / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            
            # Compute padding added
            pad_w = (inp_dim - new_w) // 2
            pad_h = (inp_dim - new_h) // 2
            
            # Crop out the padding
            img_np = img_np[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
            
            # Resize back to original dimensions
            img_np = cv2.resize(img_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            save_path = os.path.join(save_dir, f"restored_{index}.png")
            cv2.imwrite(save_path, img_np)
            index += 1
    
    print(f"Restored images saved in {save_dir}")





def dag_attack(
model, 
images, 
target_objects,  # List of target classes to attack
adversarial_classes,  # List of corresponding adversarial classes
max_iter=10, 
gamma=0.1, 
CUDA=True
):
    """
    Perform the Decision-based Adversarial Generation (DAG) attack on the given images.
    """
    im_batches, im_dim_list, imlist, loaded_ims = prepare_image(images, model.net_info["height"])
    adversarial_images = []
    perturbations = []

    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        
        original_batch = batch.clone().detach()
        perturbation = torch.zeros_like(original_batch, requires_grad=True)
        
        for m in range(max_iter):
            model.zero_grad()
            adv_batch = original_batch + perturbation
            adv_batch = torch.clamp(adv_batch, 0, 1)
            
            with torch.enable_grad():
                prediction = model(adv_batch, CUDA)
            
            filtered_preds = write_results(prediction, confidence=0.0, num_classes=80, nms_conf=0.4)
            
            if isinstance(filtered_preds, int):
                break
            
            loss = 0
            for det in filtered_preds:
                class_id = int(det[-1].item())
                if class_id in target_objects:
                    idx = target_objects.index(class_id)  # Find corresponding adversarial class
                    adv_class = adversarial_classes[idx]
                    
                    loss += prediction[..., 5 + adv_class].sum()
                    loss -= prediction[..., 5 + class_id].sum()
            
            loss.backward(retain_graph=True)
            print(f"Step {m+1}/{max_iter}: Loss = {loss.item()}")
            
            grad = perturbation.grad.data
            grad_norm = torch.norm(grad, p=float('inf'))
            perturbation_step = (gamma / grad_norm) * grad
            
            perturbation.data += perturbation_step
            perturbation.grad.zero_()
        
        adversarial_images.append(adv_batch.detach())
        perturbations.append(perturbation.detach())
    
    return adversarial_images, perturbations , im_dim_list
