"""Image Complexity Classifier integration for ICAR.

This module provides a minimal inference-only wrapper for the pre-trained
Image Complexity Classifier. Pure PyTorch implementation without Lightning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional


class ConvNeXtICC(nn.Module):
    """
    Image Complexity Classifier based on ConvNeXt-Tiny architecture.
    
    Pure PyTorch implementation for inference with pre-trained weights.
    Architecture: ConvNeXt-Tiny backbone -> LayerNorm -> Linear(768, 2)
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize ICC model.
        
        Args:
            checkpoint_path: Path to the pre-trained checkpoint.
                           If None, creates architecture only (for testing).
        """
        super().__init__()
        
        # Default hyperparameters
        self.model_name = 'convnextv2_tiny.fcmae_ft_in22k_in1k'
        self.num_classes = 2
        self.feature_dim = 768  # ConvNeXt-Tiny feature dimension
        self.default_threshold = 0.5
        
        # Create the backbone (without pretrained weights if loading from checkpoint)
        self.backbone = timm.create_model(
            self.model_name, 
            pretrained=(checkpoint_path is None),  # Only use pretrained if no checkpoint
            num_classes=0,  # Remove the classification head
        )
        
        # Create the classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.num_classes)
        )
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load weights from a pure PyTorch checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle our converted checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Update hyperparameters if available
            if 'hparams' in checkpoint:
                self.model_name = checkpoint['hparams'].get('model_name', self.model_name)
                self.num_classes = checkpoint['hparams'].get('num_classes', self.num_classes)
                self.feature_dim = checkpoint['hparams'].get('feature_dim', self.feature_dim)
                if 'threshold' in checkpoint['hparams']:
                    try:
                        self.default_threshold = float(checkpoint['hparams']['threshold'])
                    except Exception:
                        self.default_threshold = 0.5
        else:
            # Direct state dict format
            state_dict = checkpoint
        
        # Load the state dict
        self.load_state_dict(state_dict, strict=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Logits tensor [B, 2]
        """
        # Extract features from the backbone
        features = self.backbone(x)
        
        # Apply the classifier head
        return self.classifier(features)
    
    def predict_complexity(
        self, 
        x: torch.Tensor, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict complexity with binary decision and probability scores.
        
        Args:
            x: Input images [B, 3, H, W]
            threshold: Threshold for complexity decision
            
        Returns:
            Tuple of:
                - is_complex: Binary tensor [B] (True = complex)
                - complex_prob: Probability of being complex [B]
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            complex_prob = probs[:, 1]  # Probability of complex class
            t = self.default_threshold if threshold is None else float(threshold)
            is_complex = complex_prob > t
            
        return is_complex, complex_prob
