"""
Contrastive loss implementation for ICAR model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ContrastiveLoss(nn.Module):
    """
    Symmetric contrastive loss for vision-language model training.
    
    This implements the InfoNCE loss used in CLIP and similar models,
    with symmetric computation for both image-to-text and text-to-image losses.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute symmetric contrastive loss.
        
        Args:
            image_features: L2-normalized image features [batch_size, embed_dim]
            text_features: L2-normalized text features [batch_size, embed_dim]
            temperature: Temperature parameter for scaling logits
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary containing intermediate values for logging
        """
        batch_size = image_features.shape[0]
        
        # Compute similarity matrix: [batch_size, batch_size]
        # Each element (i,j) is the cosine similarity between image_i and text_j
        logits = image_features @ text_features.T
        
        # Scale by temperature (learned parameter)
        logits = logits * temperature.exp()
        
        # Create targets (diagonal elements are positive pairs)
        targets = torch.arange(batch_size, device=logits.device)
        
        # Compute image-to-text loss
        loss_i2t = F.cross_entropy(logits, targets)
        
        # Compute text-to-image loss (transpose logits)
        loss_t2i = F.cross_entropy(logits.T, targets)
        
        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2.0
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            # Image-to-text accuracy
            pred_i2t = logits.argmax(dim=1)
            acc_i2t = (pred_i2t == targets).float().mean()
            
            # Text-to-image accuracy
            pred_t2i = logits.T.argmax(dim=1)
            acc_t2i = (pred_t2i == targets).float().mean()
        
        metrics = {
            'loss_i2t': loss_i2t.detach(),
            'loss_t2i': loss_t2i.detach(),
            'acc_i2t': acc_i2t,
            'acc_t2i': acc_t2i,
            'temperature': temperature.detach()
        }
        
        return loss, metrics


class DualPathContrastiveLoss(nn.Module):
    """
    Contrastive loss for dual-path ICAR model.
    
    Combines losses from early exit and full model paths with configurable weighting.
    Both paths are trained on ALL images to ensure robustness and alignment.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize dual-path loss.
        
        Args:
            alpha: Weight for early path loss. Final loss = alpha * early + (1-alpha) * full
        """
        super().__init__()
        self.alpha = alpha
        self.contrastive_loss = ContrastiveLoss()
    
    def forward(
        self,
        early_image_features: torch.Tensor,
        early_text_features: torch.Tensor,
        early_temperature: torch.Tensor,
        full_image_features: torch.Tensor,
        full_text_features: torch.Tensor,
        full_temperature: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss for dual-path model.
        
        Both paths compute loss on ALL samples to ensure:
        1. The early path learns to handle complex images
        2. Both paths maintain aligned embedding spaces
        3. Robustness to misrouted images during inference
        
        Args:
            early_image_features: Early exit image features [batch_size, embed_dim]
            early_text_features: Early exit text features [batch_size, embed_dim]
            early_temperature: Early exit temperature
            full_image_features: Full model image features [batch_size, embed_dim]
            full_text_features: Full model text features [batch_size, embed_dim]
            full_temperature: Full model temperature
            
        Returns:
            loss: Combined scalar loss value
            metrics: Dictionary containing metrics from both paths
        """
        # Compute early path loss on all samples
        early_loss, early_metrics = self.contrastive_loss(
            early_image_features,
            early_text_features,
            early_temperature
        )
        
        # Compute full path loss on all samples
        full_loss, full_metrics = self.contrastive_loss(
            full_image_features,
            full_text_features,
            full_temperature
        )
        
        # Combine losses with alpha weighting
        combined_loss = self.alpha * early_loss + (1 - self.alpha) * full_loss
        
        # Combine metrics
        metrics = {
            'loss': combined_loss.detach(),
            'early_loss': early_loss.detach(),
            'full_loss': full_loss.detach(),
            **{f'early_{k}': v for k, v in early_metrics.items()},
            **{f'full_{k}': v for k, v in full_metrics.items()}
        }
        
        return combined_loss, metrics


def create_loss_fn(loss_type: str = 'contrastive', **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('contrastive' or 'dual_path')
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    if loss_type == 'contrastive':
        return ContrastiveLoss()
    elif loss_type == 'dual_path':
        alpha = kwargs.get('alpha', 0.5)
        return DualPathContrastiveLoss(alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")