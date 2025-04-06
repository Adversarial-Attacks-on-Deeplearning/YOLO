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




def compute_yolo_loss(adv_image, model, conf_threshold=0.2, device="cuda"):
    """
    Computes YOLO classification loss for adversarial images.
    
    Args:
        adv_image (torch.Tensor): Adversarial image tensor [B, C, H, W]
        model: YOLO model (with .model as PyTorch module)
        conf_threshold: Confidence threshold for valid detections
        device: Torch device
    
    Returns:
        torch.Tensor: Classification loss (scalar)
    """
    # Forward pass through YOLO
    with torch.no_grad():  # No need to track gradients for model parameters
        raw_outputs = model.model(adv_image)
    
    if raw_outputs[0].shape[-1] < 5:
        raise RuntimeError(f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}")

    raw_preds = raw_outputs[0]
    
    # Extract components from raw predictions
    obj_scores = raw_preds[..., 4:5]  # Objectness scores
    cls_logits = raw_preds[..., 5:]   # Class logits
    
    # Calculate confidence scores
    obj_probs = obj_scores.sigmoid()
    cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
    conf_scores = obj_probs * cls_probs
    
    # Create detection mask
    mask = conf_scores.squeeze(-1) > conf_threshold
    
    if not mask.any():
        return torch.tensor(0.0, device=device)
    
    # Compute classification loss
    target_classes = cls_logits.argmax(dim=-1)[mask].detach()
    current_logits = cls_logits[mask]
    
    return torch.nn.functional.cross_entropy(current_logits, target_classes)