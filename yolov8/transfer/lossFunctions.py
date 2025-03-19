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

def yolo_class_loss(yolo_results, mode="untargeted", target_class=None, true_class=None):
    loss = torch.tensor(0.0, device=yolo_results.boxes.conf.device)
    
    if yolo_results.boxes.shape[0] == 0:
        return loss  # No detections to attack
    
    # Extract predictions
    confs = yolo_results.boxes.conf  # Objectness scores [N]
    cls_probs = yolo_results.boxes.cls  # Class probabilities [N, C]
    
    if mode == "untargeted":
        # Suppress all detections: penalize high confidence + class certainty
        loss = torch.sum(confs * torch.max(cls_probs, dim=1)[0])
        
    elif mode == "targeted":
        # Force misclassification (e.g., "car" → "dog")
        true_probs = cls_probs[:, true_class]  # [N]
        target_probs = cls_probs[:, target_class]  # [N]
        loss = torch.sum(confs * (true_probs - target_probs))
        
    return loss