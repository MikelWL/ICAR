"""ICAR Model Implementation with dual-path vision encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision import transforms
from typing import Dict, Optional, Tuple


class ICARModel(nn.Module):
    """Image Complexity-Aware Retrieval model with early exit capability.
    
    This model extends OpenCLIP with an early exit path at an intermediate
    transformer layer, allowing adaptive computation based on image complexity.
    """
    
    def __init__(
        self, 
        clip_model_name: str = 'ViT-L-14', 
        pretrained: str = 'laion2b_s32b_b82k',
        early_exit_layer: int = 12
    ):
        """Initialize ICAR model.
        
        Args:
            clip_model_name: Name of the CLIP model architecture
            pretrained: Pretrained weights to load
            early_exit_layer: Which transformer block to exit early from (0-indexed)
        """
        super().__init__()
        
        # Store model configuration
        self.clip_model_name = clip_model_name
        self.pretrained = pretrained
        self.early_exit_layer = early_exit_layer
        
        # Load base CLIP model (but we'll override preprocessing)
        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        
        # Store architecture info
        self.hidden_dim = self.clip.visual.transformer.width  # 1024 for ViT-L
        self.output_dim = self.clip.visual.output_dim  # 768 for ViT-L
        
        # Early exit components
        self.early_ln = nn.LayerNorm(self.hidden_dim)
        self.early_proj = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        
        # Temperature parameters (CRITICAL: proper initialization)
        # Initialize as log(1/0.07) H 2.659
        self.early_temp = nn.Parameter(torch.log(torch.tensor(1.0/0.07)))
        self.register_buffer('temp_clamp', torch.tensor(100.0))
        
        # Define model-specific preprocessing
        # Base preprocessing shared by both models
        self.base_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # CLIP-specific normalization (range [-1, 1])
        self.clip_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
        
        # ICC-specific normalization (ImageNet)
        self.icc_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def encode_image(self, images: torch.Tensor, use_early_exit: bool = False, normalize_for_clip: bool = True) -> torch.Tensor:
        """Encode images using either early exit or full path.
        
        Args:
            images: Batch of images [B, 3, H, W]
            use_early_exit: Whether to use early exit path
            normalize_for_clip: Whether to apply CLIP normalization (set False if already normalized)
            
        Returns:
            Normalized image features [B, output_dim]
        """
        # Apply CLIP normalization if needed
        if normalize_for_clip:
            images = self.clip_normalize(images)
        
        if use_early_exit:
            # Early exit path using native API
            # Call forward_intermediates on the main model, not visual encoder
            # CRITICAL: stop_early=True to actually stop computation at the early exit layer
            intermediates = self.clip.forward_intermediates(
                image=images,
                image_indices=[self.early_exit_layer],
                normalize_intermediates=False,
                image_output_fmt='NLC',  # Returns [batch, seq_len, hidden_dim]
                stop_early=True  # Stop computation after extracting the intermediate
            )
            
            # Extract features from the returned dictionary
            # Should contain 'image_intermediates' which is a list
            features = intermediates['image_intermediates'][0][:, 0, :]  # [B, hidden_dim] - first token
            
            # Apply early exit projection
            features = self.early_ln(features)
            features = self.early_proj(features)
            
            # L2 normalize
            return F.normalize(features, dim=-1)
        else:
            # Full path - use standard CLIP encoding with normalization
            return self.clip.encode_image(images, normalize=True)
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Encode text using CLIP text encoder.
        
        Args:
            texts: Tokenized text inputs
            
        Returns:
            Normalized text features [B, output_dim]
        """
        return self.clip.encode_text(texts, normalize=True)
    
    def forward(
        self, 
        images: torch.Tensor, 
        texts: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with dual-path image encoding.
        
        This is used during training to compute losses for both paths.
        
        Args:
            images: Batch of images [B, 3, H, W] (assumed to be base preprocessed)
            texts: Tokenized text inputs [B, seq_len]
            return_dict: Whether to return a dictionary (vs tuple)
            
        Returns:
            Dictionary containing:
                - early_logits: Similarity logits from early exit path
                - final_logits: Similarity logits from full path
                - early_scale: Clamped temperature for early path
                - final_scale: Clamped temperature for final path
        """
        # Get text features (already normalized by encode_text)
        text_features = self.encode_text(texts)
        
        # Get both image paths (images need CLIP normalization)
        early_image_features = self.encode_image(images, use_early_exit=True, normalize_for_clip=True)
        final_image_features = self.encode_image(images, use_early_exit=False, normalize_for_clip=True)
        
        # Clamp temperatures (CRITICAL for training stability)
        early_scale = torch.clamp(self.early_temp.exp(), max=self.temp_clamp)
        final_scale = torch.clamp(self.clip.logit_scale.exp(), max=self.temp_clamp)
        
        # Compute similarity logits
        early_logits = early_scale * early_image_features @ text_features.T
        final_logits = final_scale * final_image_features @ text_features.T
        
        if return_dict:
            return {
                'early_logits': early_logits,
                'final_logits': final_logits,
                'early_scale': early_scale,
                'final_scale': final_scale,
                'early_image_features': early_image_features,
                'final_image_features': final_image_features,
                'text_features': text_features
            }
        else:
            return early_logits, final_logits
    
    def get_tokenizer(self):
        """Get the CLIP tokenizer for this model."""
        return open_clip.get_tokenizer(self.clip.model_name)