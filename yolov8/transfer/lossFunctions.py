import torch
import torch.nn.functional as F
from ultralytics import YOLO

def discriminator_gan_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """
    Computes the GAN loss for the discriminator.
    
    Args:
        d_real (torch.Tensor): Discriminator predictions on real images (shape: [batch_size, 1]).
        d_fake (torch.Tensor): Discriminator predictions on fake images (shape: [batch_size, 1]).
        
    Returns:
        torch.Tensor: Total discriminator loss.
    """
    real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))  # Real images labeled as 1
    fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))  # Fake images labeled as 0
    return (real_loss + fake_loss) / 2  # Average the two losses

def generator_gan_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Computes the GAN loss for the generator.
    
    Args:
        d_fake (torch.Tensor): Discriminator predictions on fake images (shape: [batch_size, 1]).
        
    Returns:
        torch.Tensor: Generator loss (tries to fool the discriminator).
    """
    return F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))  # Fake images labeled as 1


import torch
import torch.nn.functional as F

def discriminator_gan_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    # (Previous implementation)
    ...

def generator_gan_loss(d_fake: torch.Tensor) -> torch.Tensor:
    # (Previous implementation)
    ...

def l2_loss(clean_image: torch.Tensor, adversarial_image: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 loss (Mean Squared Error) between clean and adversarial images.
    
    Args:
        clean_image (torch.Tensor): Original unperturbed image (shape: [B, C, H, W]).
        adversarial_image (torch.Tensor): Generated adversarial image (shape: [B, C, H, W]).
        
    Returns:
        torch.Tensor: L2 loss (scalar).
    """
    return F.mse_loss(clean_image, adversarial_image)

def yolo_attack_loss(
    yolo_results: ultralytics.engine.results.Results,
    target_class: int = None,
    true_class: int = None,
    mode: str = "untargeted"
) -> torch.Tensor:
    """
    Compute adversarial loss based on YOLOv8 predictions.
    
    Args:
        yolo_results: YOLOv8 output for the adversarial image.
        target_class: Desired wrong class (for targeted attacks).
        true_class: Original class to attack (for targeted attacks).
        mode: 'untargeted' (suppress detections) or 'targeted' (misclassify).
        
    Returns:
        Loss value to minimize.
    """
    loss = torch.tensor(0.0, device=yolo_results.boxes.conf.device)
    
    if yolo_results.boxes.shape[0] == 0:
        return loss  # No detections
    
    # Extract predictions: [x1, y1, x2, y2, conf, class]
    boxes = yolo_results.boxes
    confs = boxes.conf  # Objectness scores [N]
    class_probs = boxes.cls  # Class probabilities [N, num_classes]
    
    if mode == "untargeted":
        # Penalize high confidence & class certainty for traffic signs
        loss = torch.sum(confs * torch.max(class_probs, dim=1)[0])
        
    elif mode == "targeted":
        # Ensure correct args are provided
        assert target_class is not None and true_class is not None
        
        # Get probabilities for true and target classes
        true_probs = class_probs[:, true_class]  # [N]
        target_probs = class_probs[:, target_class]  # [N]
        
        # Encourage low confidence for true class, high for target
        loss = torch.sum(confs * (true_probs - target_probs))
        
    return loss