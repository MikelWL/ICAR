"""ICAR model evaluator."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
from tqdm import tqdm

from ..models.icar_model import ICARModel
from ..models.icc import ConvNeXtICC
from .metrics import (
    compute_image_text_retrieval_metrics,
    compute_retrieval_metrics_with_mappings,
    compute_efficiency_metrics,
    aggregate_metrics
)

logger = logging.getLogger(__name__)


class ICARInferenceWrapper(nn.Module):
    """Wrapper for ICAR model with complexity-aware routing during inference."""
    
    def __init__(
        self,
        icar_model: ICARModel,
        icc_model: ConvNeXtICC,
        icc_threshold: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize inference wrapper.
        
        Args:
            icar_model: ICAR model with dual paths
            icc_model: Image Complexity Classifier
            icc_threshold: Threshold for complexity classification
            device: Device to run on
        """
        super().__init__()
        self.icar_model = icar_model.to(device)
        self.icc_model = icc_model.to(device).eval()
        self.icc_threshold = (
            float(icc_threshold)
            if icc_threshold is not None
            else float(getattr(icc_model, "default_threshold", 0.5))
        )
        self.device = device
        
        # Initialize routing statistics
        self.routing_stats = {'early': 0, 'full': 0}
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images with complexity-aware routing.
        
        Args:
            images: Batch of images [B, 3, H, W] (assumed to be base preprocessed)
            
        Returns:
            Image embeddings [B, D]
        """
        batch_size = images.size(0)
        images = images.to(self.device)
        
        # Get complexity predictions
        with torch.no_grad():
            # ICC expects ImageNet normalized images
            icc_images = self.icar_model.icc_normalize(images)
            complexity_logits = self.icc_model(icc_images)
            complexity_probs = torch.softmax(complexity_logits, dim=1)
            is_complex = complexity_probs[:, 1] > self.icc_threshold
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, self.icar_model.output_dim, device=self.device)
        
        # Route to early exit
        early_mask = ~is_complex
        if early_mask.any():
            embeddings[early_mask] = self.icar_model.encode_image(
                images[early_mask], use_early_exit=True, normalize_for_clip=True
            )
            self.routing_stats['early'] += early_mask.sum().item()
        
        # Route to full model
        if is_complex.any():
            embeddings[is_complex] = self.icar_model.encode_image(
                images[is_complex], use_early_exit=False, normalize_for_clip=True
            )
            self.routing_stats['full'] += is_complex.sum().item()
        
        return embeddings
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Encode text using the model."""
        return self.icar_model.encode_text(texts.to(self.device))
    
    def reset_routing_stats(self):
        """Reset routing statistics."""
        self.routing_stats = {'early': 0, 'full': 0}


class ICARBaseline(nn.Module):
    """Baseline model that always uses full path (no adaptive routing)."""
    
    def __init__(self, icar_model: ICARModel, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.icar_model = icar_model.to(device)
        self.device = device
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using full path only."""
        images = images.to(self.device)
        return self.icar_model.encode_image(images, use_early_exit=False, normalize_for_clip=True)
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Encode text using the model."""
        return self.icar_model.encode_text(texts.to(self.device))


class ICARForceEarlyExit(nn.Module):
    """Model wrapper that forces all images through early exit path."""
    
    def __init__(self, icar_model: ICARModel, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.icar_model = icar_model.to(device)
        self.device = device
        self.early_exit_layer = icar_model.early_exit_layer
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using early exit path only."""
        images = images.to(self.device)
        return self.icar_model.encode_image(images, use_early_exit=True, normalize_for_clip=True)
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Encode text using the model."""
        return self.icar_model.encode_text(texts.to(self.device))


class ICAREvaluator:
    """Evaluator for ICAR model.
    
    IMPORTANT: This evaluator handles both training and evaluation mode dataloaders:
    - Training mode: Returns (images, texts) with 1:1 mapping
    - Evaluation mode: Returns (images, captions_list, img_ids) with many-to-many mapping
    
    The evaluator automatically detects the mode and applies the correct metrics computation.
    For evaluation mode, it properly handles the fact that each image has multiple valid captions.
    """
    
    def __init__(
        self,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        similarity_on_cpu: bool = False,
        temperature: Optional[float] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            device: Device to run evaluation on
            similarity_on_cpu: Whether to compute similarities on CPU (for very large datasets)
            temperature: Temperature scaling factor for similarity scores. If None, no scaling is applied.
                        This should match the temperature used during training.
        """
        self.device = device
        self.similarity_on_cpu = similarity_on_cpu
        self.temperature = temperature
        if temperature is not None:
            logger.info(f"Evaluator initialized with temperature scaling: {temperature:.2f}")
        else:
            logger.info("Evaluator initialized without temperature scaling")
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate (ICARInferenceWrapper or ICARBaseline)
            dataloader: DataLoader for evaluation dataset
            desc: Description for progress bar
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        # Check if this is evaluation mode dataloader
        # First, let's do a test batch to see the format
        test_batch = next(iter(dataloader))
        
        # Check if batch is a dictionary (new format) or tuple (old format)
        if isinstance(test_batch, dict):
            # Dictionary format - check if it has eval-specific keys
            dataloader_is_eval_mode = 'caption_indices' in test_batch or 'imgids' in test_batch
        else:
            # Tuple format - check length
            dataloader_is_eval_mode = len(test_batch) == 3  # (images, captions_list, img_ids)
        
        if dataloader_is_eval_mode:
            # Evaluation mode: properly handle many-to-many mappings
            return self._evaluate_with_mappings(model, dataloader, desc)
        else:
            # Training mode dataloader (1:1 mapping)
            return self._evaluate_simple(model, dataloader, desc)
    
    def _evaluate_simple(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate with simple 1:1 image-text mapping (training mode dataloader).
        """
        all_image_embeddings = []
        all_text_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Handle both dictionary and tuple formats
                if isinstance(batch, dict):
                    images = batch['images']
                    texts = batch['text']
                else:
                    images, texts = batch
                
                # Encode images and texts
                image_embeddings = model.encode_image(images)
                text_embeddings = model.encode_text(texts)
                
                if self.similarity_on_cpu:
                    all_image_embeddings.append(image_embeddings.cpu())
                    all_text_embeddings.append(text_embeddings.cpu())
                else:
                    all_image_embeddings.append(image_embeddings)
                    all_text_embeddings.append(text_embeddings)
        
        # Concatenate all embeddings
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        
        # Compute retrieval metrics
        if not self.similarity_on_cpu and all_image_embeddings.device.type == 'cpu':
            # Move to GPU for faster computation
            logger.info("Moving embeddings to GPU for similarity computation...")
            all_image_embeddings = all_image_embeddings.to(self.device)
            all_text_embeddings = all_text_embeddings.to(self.device)
        
        retrieval_metrics = compute_image_text_retrieval_metrics(
            all_image_embeddings,
            all_text_embeddings,
            temperature=self.temperature
        )
        
        # Flatten retrieval metrics
        metrics = {}
        for task, task_metrics in retrieval_metrics.items():
            for metric_name, value in task_metrics.items():
                metrics[f'{task}_{metric_name}'] = value
        
        # Add routing statistics if available
        if hasattr(model, 'routing_stats'):
            routing_metrics = compute_efficiency_metrics(
                model.routing_stats,
                flops_early=1.0,  # Placeholder - should be computed
                flops_full=2.0,   # Placeholder - should be computed
                flops_icc=0.1     # Placeholder - should be computed
            )
            metrics.update(routing_metrics)
        
        return metrics
    
    def _evaluate_with_mappings(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate with proper many-to-many image-text mappings (eval mode dataloader).
        """
        import open_clip
        
        # Collect all unique images and all captions with mappings
        all_images = []
        all_captions = []
        img2txt = {}
        txt2img = {}
        
        caption_idx = 0
        img_idx = 0
        
        # First pass: collect all data and build mappings
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc} - collecting data"):
                # Handle both dictionary and tuple formats
                if isinstance(batch, dict):
                    batch_images = batch['images']
                    batch_captions_lists = batch['captions']
                    batch_img_ids = batch.get('imgids', None)
                else:
                    batch_images, batch_captions_lists, batch_img_ids = batch
                
                # Process each image in the batch
                for image, captions_list in zip(batch_images, batch_captions_lists):
                    # Store image
                    all_images.append(image)
                    img2txt[img_idx] = []
                    
                    # Store all captions for this image
                    for caption in captions_list:
                        all_captions.append(caption)
                        img2txt[img_idx].append(caption_idx)
                        txt2img[caption_idx] = img_idx
                        caption_idx += 1
                    
                    img_idx += 1
        
        logger.info(f"Collected {len(all_images)} unique images and {len(all_captions)} captions")
        
        # Get the model name for tokenization
        if hasattr(model, 'icar_model') and hasattr(model.icar_model, 'clip_model_name'):
            clip_model_name = model.icar_model.clip_model_name
        elif hasattr(model, 'clip_model_name'):
            clip_model_name = model.clip_model_name
        else:
            clip_model_name = 'ViT-L-14'
        
        # Tokenize all captions
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_image_embeddings = []
        all_text_embeddings = []
        
        # Encode images
        with torch.no_grad():
            for i in tqdm(range(0, len(all_images), batch_size), desc=f"{desc} - encoding images"):
                batch_images = torch.stack(all_images[i:i+batch_size])
                image_embeddings = model.encode_image(batch_images)
                all_image_embeddings.append(image_embeddings.cpu())
        
        # Encode texts
        with torch.no_grad():
            for i in tqdm(range(0, len(all_captions), batch_size), desc=f"{desc} - encoding texts"):
                batch_captions = all_captions[i:i+batch_size]
                batch_tokens = tokenizer(batch_captions).to(model.device)
                text_embeddings = model.encode_text(batch_tokens)
                all_text_embeddings.append(text_embeddings.cpu())
        
        # Concatenate all embeddings
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        
        # Normalize embeddings
        all_image_embeddings = all_image_embeddings / all_image_embeddings.norm(dim=-1, keepdim=True)
        all_text_embeddings = all_text_embeddings / all_text_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarity scores
        logger.info(f"Computing similarity matrix ({len(all_image_embeddings)} images x {len(all_text_embeddings)} texts)")
        if self.similarity_on_cpu:
            # Keep on CPU for very large datasets
            logger.info("Computing similarities on CPU")
            scores_i2t = all_image_embeddings @ all_text_embeddings.T
            scores_t2i = scores_i2t.T  # Transpose instead of recomputing
        else:
            # Move to GPU for faster computation
            logger.info(f"Moving embeddings to {self.device} for similarity computation")
            all_image_embeddings = all_image_embeddings.to(self.device)
            all_text_embeddings = all_text_embeddings.to(self.device)
            logger.info(f"Computing similarity matrix on {self.device}")
            scores_i2t = all_image_embeddings @ all_text_embeddings.T
            scores_t2i = scores_i2t.T  # Transpose instead of recomputing
            logger.info(f"Similarity computation complete, matrix shape: {scores_i2t.shape}")
        
        # Compute retrieval metrics with mappings
        logger.info("Computing retrieval metrics...")
        # Pass embeddings (on GPU) instead of pre-computed scores
        # The metrics function will compute similarities on GPU
        retrieval_metrics = compute_retrieval_metrics_with_mappings(
            all_image_embeddings,
            all_text_embeddings,
            img2txt,
            txt2img,
            temperature=self.temperature if self.temperature is not None else 100.0
        )
        logger.info("Retrieval metrics computation complete")
        
        # Retrieval metrics are already flattened from compute_retrieval_metrics_with_mappings
        metrics = retrieval_metrics
        
        # Add routing statistics if available
        if hasattr(model, 'routing_stats'):
            routing_metrics = compute_efficiency_metrics(
                model.routing_stats,
                flops_early=1.0,  # Placeholder - should be computed
                flops_full=2.0,   # Placeholder - should be computed
                flops_icc=0.1     # Placeholder - should be computed
            )
            metrics.update(routing_metrics)
        
        return metrics
    
    def evaluate_adaptive_vs_baseline(
        self,
        icar_model: ICARModel,
        icc_model: ConvNeXtICC,
        dataloader: DataLoader,
        icc_threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both adaptive and baseline models.
        
        Args:
            icar_model: ICAR model
            icc_model: Image Complexity Classifier
            dataloader: Evaluation dataloader
            icc_threshold: Threshold for complexity classification
            
        Returns:
            Dictionary with 'adaptive' and 'baseline' results
        """
        results = {}
        
        # Evaluate adaptive model
        adaptive_model = ICARInferenceWrapper(
            icar_model, icc_model, icc_threshold, self.device
        )
        logger.info("Evaluating adaptive model...")
        results['adaptive'] = self.evaluate(
            adaptive_model, dataloader, desc="Adaptive evaluation"
        )
        
        # Evaluate baseline model
        baseline_model = ICARBaseline(icar_model, self.device)
        logger.info("Evaluating baseline model...")
        results['baseline'] = self.evaluate(
            baseline_model, dataloader, desc="Baseline evaluation"
        )
        
        # Compute relative performance
        results['comparison'] = {}
        for metric in ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']:
            if metric in results['adaptive'] and metric in results['baseline']:
                baseline_val = results['baseline'][metric]
                adaptive_val = results['adaptive'][metric]
                if baseline_val > 0:
                    relative = (adaptive_val / baseline_val) * 100
                    results['comparison'][f'{metric}_relative'] = relative
        
        return results
