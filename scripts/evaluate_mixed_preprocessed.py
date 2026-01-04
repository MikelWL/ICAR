#!/usr/bin/env python3
"""
Evaluation script for ICAR model on mixed datasets with preprocessed LAION-COCO.

This script evaluates models on augmented test sets where preprocessed LAION-COCO 
samples are added as distractors to the retrieval gallery. Only base dataset 
samples are used as queries. Supports ICC-based routing for efficiency analysis.

Usage:
    # Evaluate without ICC routing
    python scripts/evaluate_mixed_preprocessed.py \
        --config icar/configs/coco.yaml \
        --checkpoint checkpoints/icar_coco/layer_12/latest_checkpoint.pt \
        --base-dataset mscoco \
        --early-exit-layer 12
    
    # Evaluate with ICC routing
    python scripts/evaluate_mixed_preprocessed.py \
        --config icar/configs/coco.yaml \
        --checkpoint checkpoints/icar_coco/layer_12/latest_checkpoint.pt \
        --base-dataset mscoco \
        --early-exit-layer 12 \
        --use-icc-routing \
        --icc-checkpoint data/ICC.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from icar.models.icar_model import ICARModel
from icar.data.mixed_dataset_preprocessed import create_preprocessed_mixed_dataset
from icar.data.coco_dataset import create_collate_fn
from icar.evaluation.evaluator import ICARBaseline, ICARForceEarlyExit
from icar.evaluation.metrics import compute_retrieval_metrics_with_mappings
import open_clip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collate_fn_mixed_eval(batch, tokenizer):
    """Custom collate function for mixed dataset evaluation that handles string image IDs."""
    images = []
    all_captions = []
    imgids = []  # Keep as strings
    
    for img, captions, imgid in batch:
        images.append(img)
        all_captions.append(captions)
        imgids.append(imgid)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # For evaluation, we need to tokenize all captions
    # Flatten the list of caption lists
    flat_captions = []
    caption_indices = []  # To track which captions belong to which image
    
    for i, captions in enumerate(all_captions):
        caption_indices.extend([i] * len(captions))
        flat_captions.extend(captions)
    
    # Tokenize all captions at once
    text_tokens = tokenizer(flat_captions) if flat_captions else None
    
    return {
        'images': images,
        'text': text_tokens,
        'captions': all_captions,  # List of caption lists
        'caption_indices': torch.tensor(caption_indices) if caption_indices else None,
        'imgids': imgids  # Keep as list of strings
    }


class ConvNeXtICC(nn.Module):
    """Image Complexity Classifier for routing decisions."""
    def __init__(self, checkpoint_path: str):
        super().__init__()
        import timm
        
        # Create model architecture
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=False,
            num_classes=0
        )
        
        # Classifier head
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2)
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Handle PyTorch Lightning state dict format
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_k] = v
        
        self.load_state_dict(new_state_dict, strict=True)
        self.eval()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    @torch.no_grad()
    def predict_complexity(self, x, threshold=0.5):
        """Returns both binary prediction and probability score."""
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        complex_prob = probs[:, 1]
        is_complex = complex_prob > threshold
        return is_complex, complex_prob


class ICCRoutedModel(nn.Module):
    """Wrapper that adds ICC routing to a ICAR model."""
    def __init__(self, icar_model: ICARModel, icc_model: ConvNeXtICC, 
                 icc_threshold: float = 0.5, device: str = 'cuda'):
        super().__init__()
        self.icar_model = icar_model
        self.icc_model = icc_model.to(device)
        self.icc_threshold = icc_threshold
        self.device = device
        
        # Normalization transforms
        from icar.data.transforms import normalize_clip, normalize_icc
        self.normalize_clip = normalize_clip
        self.normalize_icc = normalize_icc
        
        # Routing statistics
        self.routing_stats = {'early': 0, 'full': 0}
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Route images based on complexity."""
        batch_size = images.shape[0]
        device = images.device
        
        # Clone for ICC normalization (images are already base-preprocessed)
        # Create normalization constants on the same device as images
        icc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        icc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        icc_images = (images.clone() - icc_mean) / icc_std
        
        # Get complexity predictions
        with torch.no_grad():
            _, complex_probs = self.icc_model.predict_complexity(icc_images, self.icc_threshold)
            is_complex = complex_probs > self.icc_threshold
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, 768, device=self.device)
        
        # Route to appropriate paths
        early_mask = ~is_complex
        full_mask = is_complex
        
        # Update routing statistics
        self.routing_stats['early'] += early_mask.sum().item()
        self.routing_stats['full'] += full_mask.sum().item()
        
        # Process early exit batch
        if early_mask.any():
            # CLIP normalization on same device
            clip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            clip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            clip_images_early = (images[early_mask] - clip_mean) / clip_std
            embeddings[early_mask] = self.icar_model.encode_image(
                clip_images_early, use_early_exit=True, normalize_for_clip=False
            )
        
        # Process full path batch
        if full_mask.any():
            # CLIP normalization on same device
            clip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            clip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            clip_images_full = (images[full_mask] - clip_mean) / clip_std
            embeddings[full_mask] = self.icar_model.encode_image(
                clip_images_full, use_early_exit=False, normalize_for_clip=False
            )
        
        return embeddings
    
    def encode_text(self, text):
        """Forward text encoding to ICAR model."""
        return self.icar_model.encode_text(text)
    
    def get_routing_stats(self):
        """Get and reset routing statistics."""
        stats = self.routing_stats.copy()
        total = stats['early'] + stats['full']
        if total > 0:
            stats['early_percentage'] = stats['early'] / total * 100
            stats['full_percentage'] = stats['full'] / total * 100
        self.routing_stats = {'early': 0, 'full': 0}
        return stats


def load_config(config_path: str, model_config_path: Optional[str] = None) -> Dict:
    """Load and merge configuration files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_config_path:
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        # Merge configs
        for key, value in model_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    return config


def evaluate_mixed_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: str = 'cuda',
    base_img_count: int = 5000,
    return_embeddings: bool = False,
    save_embeddings: bool = True,
    model_info: Dict = None
) -> Dict:
    """Evaluate model on mixed dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        tokenizer: Tokenizer for text encoding
        device: Device to run on
        base_img_count: Number of base dataset images
        return_embeddings: If True, return embeddings in addition to metrics
        save_embeddings: If True, save embeddings to disk
        model_info: Dictionary with model info for naming the saved files
        
    Returns:
        Dictionary with metrics and optionally embeddings
    """
    model.eval()
    
    # Storage for embeddings
    image_embeddings = []
    text_embeddings = []
    
    # Process images and texts
    logger.info("Computing embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = batch['images'].to(device)
            
            # Encode images
            image_emb = model.encode_image(images)
            image_embeddings.append(image_emb)  # Keep on device
            
            # Encode texts (if available in batch)
            if batch['text'] is not None:
                text = batch['text'].to(device)
                text_emb = model.encode_text(text)
                text_embeddings.append(text_emb)  # Keep on device
    
    # Concatenate embeddings (still on device)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0) if text_embeddings else None
    
    # Get mappings from dataset
    dataset = dataloader.dataset
    img2txt, txt2img = dataset.get_mappings()
    
    # Create query indices - only base dataset images/texts are queries
    query_image_indices = list(range(base_img_count))  # First base_img_count images
    base_caption_count = dataset.base_caption_count
    query_text_indices = list(range(base_caption_count))  # First base_caption_count texts
    
    # Save embeddings if requested
    if save_embeddings and image_embeddings is not None and text_embeddings is not None:
        # Generate run name based on model info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_info:
            dataset_name = model_info.get('dataset', 'unknown')
            layer = model_info.get('layer', 'unknown')
            routing = model_info.get('routing', 'unknown')
            run_name = f"{dataset_name}_layer{layer}_{routing}_{timestamp}"
        else:
            run_name = f"run_{timestamp}"
        
        # Create embeddings directory for this run
        embeds_dir = Path(__file__).parent.parent / "embeds" / run_name
        embeds_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image embeddings
        image_embeds_path = embeds_dir / f"image_embeds_{run_name}.npz"
        logger.info(f"Saving image embeddings to {image_embeds_path}")
        np.savez_compressed(
            image_embeds_path,
            embeddings=image_embeddings.cpu().numpy(),
            n_images=len(image_embeddings),
            base_img_count=base_img_count,
            query_image_indices=query_image_indices,
            model_info=model_info or {},
            timestamp=timestamp
        )
        
        # Save text embeddings
        text_embeds_path = embeds_dir / f"text_embeds_{run_name}.npz"
        logger.info(f"Saving text embeddings to {text_embeds_path}")
        np.savez_compressed(
            text_embeds_path,
            embeddings=text_embeddings.cpu().numpy(),
            n_texts=len(text_embeddings),
            base_caption_count=base_caption_count,
            query_text_indices=query_text_indices,
            img2txt=img2txt,
            txt2img=txt2img,
            model_info=model_info or {},
            timestamp=timestamp
        )
        
        logger.info(f"Embeddings saved to: {embeds_dir.name}/")
    
    # Compute retrieval metrics using only base dataset queries
    # This ensures LAION samples are only used as distractors
    logger.info(f"Computing retrieval metrics (base queries: {base_img_count})...")
    logger.info(f"Embeddings device: {image_embeddings.device}")
    
    # Create query indices - only base dataset images/texts are queries
    query_image_indices = list(range(base_img_count))  # First base_img_count images
    
    base_caption_count = dataset.base_caption_count
    query_text_indices = list(range(base_caption_count))  # First base_caption_count texts
    
    # Compute metrics with query indices
    metrics = compute_retrieval_metrics_with_mappings(
        image_embeddings,  # Now on GPU
        text_embeddings,
        img2txt,
        txt2img,
        query_image_indices=query_image_indices,
        query_text_indices=query_text_indices
    )
    
    result = {'metrics': metrics}
    
    if return_embeddings:
        result['embeddings'] = {
            'image': image_embeddings,
            'text': text_embeddings,
            'query_image_indices': query_image_indices,
            'query_text_indices': query_text_indices,
            'img2txt': img2txt,
            'txt2img': txt2img
        }
    
    return result


def evaluate_category_retrieval_from_embeddings(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    category_labels: Dict[str, Dict],
    query_image_indices: List[int],
    query_text_indices: List[int] = None,
    img2txt: Dict[int, List[int]] = None,
    txt2img: Dict[int, int] = None,
    dataset_type: str = 'mixed',
    temperature: float = 100.0
) -> Dict:
    """
    Compute category-level retrieval metrics from pre-computed embeddings.
    
    Args:
        image_embeddings: Pre-computed image embeddings
        text_embeddings: Pre-computed text embeddings
        category_labels: Dictionary mapping image IDs to categories
        query_image_indices: Indices of image query samples
        query_text_indices: Indices of text query samples (optional)
        txt2img: Mapping from text index to image index (optional)
        dataset_type: Type of dataset ('mixed', 'coco', 'flickr30k')
        temperature: Temperature for similarity computation
        
    Returns:
        Dictionary with category-level retrieval metrics
    """
    # Compute similarity matrix with temperature (CPU, matches SCAR behavior).
    logger.info(f"Computing similarity matrix with temperature {temperature:.2f}")
    if isinstance(image_embeddings, torch.Tensor):
        image_embeddings = image_embeddings.cpu().numpy()
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.cpu().numpy()
    sim_matrix = temperature * (image_embeddings @ text_embeddings.T)
    
    # Create category label matrices
    # Get all unique categories
    all_categories = set()
    for img_data in category_labels.values():
        if 'categories' in img_data:
            all_categories.update(img_data['categories'])
    
    category_list = sorted(list(all_categories))
    category_to_idx = {cat: i for i, cat in enumerate(category_list)}
    
    # Create multi-hot encoding for all image samples
    n_images = len(image_embeddings)
    n_categories = len(category_list)
    labels_images = np.zeros((n_images, n_categories), dtype=np.float32)
    
    # Map image indices to category labels
    # This assumes the order of embeddings matches the dataset order
    for idx in range(n_images):
        # Get image ID for this index - implementation depends on dataset
        # For now, assume direct index mapping
        img_id = str(idx)
        if img_id in category_labels:
            categories = category_labels[img_id].get('categories', [])
            for cat in categories:
                if cat in category_to_idx:
                    labels_images[idx, category_to_idx[cat]] = 1.0
    
    # Build text labels from txt2img mapping when available
    n_texts = len(text_embeddings)
    if txt2img is not None:
        labels_texts = np.zeros((n_texts, n_categories), dtype=np.float32)
        for text_idx in range(n_texts):
            img_idx = txt2img.get(text_idx)
            if img_idx is None or img_idx >= n_images:
                continue
            labels_texts[text_idx] = labels_images[img_idx]
    else:
        if n_texts != n_images:
            raise ValueError("txt2img mapping is required when text and image counts differ")
        labels_texts = labels_images

    if query_text_indices is None:
        # Match SCAR's embeddings-only evaluation: one text query per image query.
        if img2txt is not None:
            query_text_indices = []
            for img_idx in query_image_indices:
                captions = img2txt.get(img_idx, [])
                if captions:
                    query_text_indices.append(captions[0])
        elif txt2img is not None:
            query_text_indices = []
            for img_idx in query_image_indices:
                # Pick the first caption mapped to this image.
                for text_idx, mapped_img_idx in txt2img.items():
                    if mapped_img_idx == img_idx:
                        query_text_indices.append(text_idx)
                        break
        else:
            query_text_indices = list(range(n_texts))

    def _compute_category_map(sim_matrix_qr, query_labels, retrieval_labels, k):
        n_retrievals = sim_matrix_qr.shape[1]
        k_val = min(k or n_retrievals, n_retrievals)
        ap_scores = []
        for i in tqdm(range(sim_matrix_qr.shape[0]), desc="Category mAP", leave=False):
            scores = sim_matrix_qr[i]
            if k_val == n_retrievals:
                indices = np.argsort(-scores)
            else:
                topk = np.argpartition(-scores, k_val - 1)[:k_val]
                indices = topk[np.argsort(-scores[topk])]
            current_query_labels = query_labels[i]
            retrieved_labels = retrieval_labels[indices]
            if query_labels.ndim == 1:
                relevant = retrieved_labels == current_query_labels
            else:
                relevant = np.any(np.logical_and(retrieved_labels, current_query_labels[np.newaxis, :]), axis=1)
            if np.any(relevant):
                precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
                ap = np.mean(precision_at_k[relevant])
            else:
                ap = 0.0
            ap_scores.append(ap)
        return float(np.mean(ap_scores))
    
    # Compute category-level retrieval metrics
    logger.info(f"Computing category retrieval metrics for {len(query_image_indices)} image queries")
    k_values = [10, 50, 100]

    # Image-to-text (queries are images, retrievals are texts)
    sim_i2t = sim_matrix[query_image_indices, :]
    query_image_labels = labels_images[query_image_indices]

    # Text-to-image (queries are texts, retrievals are images)
    sim_t2i = sim_matrix.T[query_text_indices, :]
    query_text_labels = labels_texts[query_text_indices]

    mAP_results = {}
    for k in k_values:
        map_i2t = _compute_category_map(sim_i2t, query_image_labels, labels_texts, k)
        map_t2i = _compute_category_map(sim_t2i, query_text_labels, labels_images, k)
        mAP_results[k] = (map_i2t, map_t2i)
    
    # Format results
    results = {
        'mAP_i2t': {},
        'mAP_t2i': {},
        'n_queries': len(query_image_indices),
        'n_total_samples': n_images,
        'n_categories': n_categories,
        'category_list': category_list
    }
    
    for k, (map_i2t, map_t2i) in mAP_results.items():
        k_str = f'mAP@{k}'
        results['mAP_i2t'][k_str] = map_i2t * 100
        results['mAP_t2i'][k_str] = map_t2i * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICAR on mixed datasets")
    
    # Configs
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model-config", type=str, help="Path to model-specific config")
    
    # Model
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--early-exit-layer", type=int, help="Early exit layer")
    
    # Dataset
    parser.add_argument("--base-dataset", type=str, default="mscoco",
                      choices=["mscoco", "flickr30k"], help="Base dataset name")
    parser.add_argument("--base-data-root", type=str, default="./data/caption_datasets",
                      help="Root directory for base dataset")
    parser.add_argument("--laion-data-root", type=str, default="./data/laion_coco_100k",
                      help="Root directory for preprocessed LAION-COCO")
    parser.add_argument("--complexity-scores", type=str,
                      default="./data/laion_coco_100k_metadata/complexity_scores.json",
                      help="Path to complexity scores")
    
    # ICC routing
    parser.add_argument("--use-icc-routing", action="store_true",
                      help="Use ICC for complexity-based routing")
    parser.add_argument("--icc-checkpoint", type=str, default="./data/ICC.pt",
                      help="Path to ICC checkpoint")
    parser.add_argument("--icc-threshold", type=float, default=0.5,
                      help="ICC complexity threshold")
    
    # Forced early exit
    parser.add_argument("--force-early-exit", action="store_true",
                      help="Force all samples through early exit path")
    
    # Evaluation
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./results/icc_routed",
                      help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output filename")

    # Embeddings-only evaluation
    parser.add_argument("--image-embeds", type=str,
                      help="Path to saved image embeddings (.npz)")
    parser.add_argument("--text-embeds", type=str,
                      help="Path to saved text embeddings (.npz)")
    
    # Category evaluation
    parser.add_argument("--eval-category", action="store_true",
                      help="Also evaluate category-level retrieval")
    parser.add_argument("--coco-categories", type=str,
                      default="./data/coco_test_categories.json",
                      help="Path to COCO category labels JSON")
    parser.add_argument("--flickr-categories", type=str,
                      default="./data/flickr30k_test_categories.json",
                      help="Path to Flickr30k category labels JSON")
    parser.add_argument("--laion-categories", type=str,
                      default="./data/laion_coco_100k/category_labels_laion_coco_100k.json",
                      help="Path to LAION category labels JSON")
    
    args = parser.parse_args()
    
    # Embeddings-only path
    if args.image_embeds or args.text_embeds:
        if not (args.image_embeds and args.text_embeds):
            raise ValueError("Provide both --image-embeds and --text-embeds")
        if not Path(args.image_embeds).exists():
            raise FileNotFoundError(f"Image embeddings not found: {args.image_embeds}")
        if not Path(args.text_embeds).exists():
            raise FileNotFoundError(f"Text embeddings not found: {args.text_embeds}")
    else:
        if not args.config:
            raise ValueError("--config is required when not using embeddings")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when not using embeddings")

    # Load config when running full eval
    config = load_config(args.config, args.model_config) if args.config else None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image_embeds and args.text_embeds:
        logger.info("Loading saved embeddings (skipping model evaluation)")
        image_npz = np.load(args.image_embeds, allow_pickle=True)
        text_npz = np.load(args.text_embeds, allow_pickle=True)
        
        image_embeddings = torch.from_numpy(image_npz['embeddings']).to(args.device)
        text_embeddings = torch.from_numpy(text_npz['embeddings']).to(args.device)
        
        base_img_count = int(image_npz['base_img_count']) if 'base_img_count' in image_npz else None
        base_caption_count = int(text_npz['base_caption_count']) if 'base_caption_count' in text_npz else None
        query_image_indices = image_npz['query_image_indices'].tolist() if 'query_image_indices' in image_npz else None
        query_text_indices = text_npz['query_text_indices'].tolist() if 'query_text_indices' in text_npz else None
        img2txt = text_npz['img2txt'].item() if 'img2txt' in text_npz else None
        txt2img = text_npz['txt2img'].item() if 'txt2img' in text_npz else None
        
        metrics = compute_retrieval_metrics_with_mappings(
            image_embeddings,
            text_embeddings,
            img2txt,
            txt2img,
            query_image_indices=query_image_indices,
            query_text_indices=query_text_indices,
            temperature=100.0
        )
        
        eval_result = {
            'metrics': metrics,
            'embeddings': {
                'image': image_embeddings.cpu(),
                'text': text_embeddings.cpu(),
                'query_image_indices': query_image_indices,
                'query_text_indices': query_text_indices,
                'img2txt': img2txt,
                'txt2img': txt2img
            }
        }
        
        class _EmbedStats:
            pass
        
        dataset = _EmbedStats()
        dataset.base_img_count = base_img_count if base_img_count is not None else 0
        dataset.base_caption_count = base_caption_count if base_caption_count is not None else 0
        dataset.images = range(image_embeddings.shape[0])
        dataset.captions = range(text_embeddings.shape[0])
        dataset.complexity_scores = None
    else:
        # Load model
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        if Path(args.checkpoint).suffix == '.pt':
            # PyTorch checkpoint
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            
            # For baseline models, we create a ICARModel with a very high early exit layer
            # This effectively makes it always use the full path
            if args.early_exit_layer:
                early_exit_layer = args.early_exit_layer
            else:
                # Baseline model - set early exit to layer 24 (after all layers)
                early_exit_layer = 24
                
            model = ICARModel(
                clip_model_name=config['model']['clip_model_name'],
                pretrained=None,  # Don't load pretrained weights, we'll load from checkpoint
                early_exit_layer=early_exit_layer
            )
            
            # Load state dict
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Handle DDP state dict
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_k] = v
            
            model.load_state_dict(new_state_dict)
            
            # Extract temperature if available
            # Temperature might be in the state dict as early_temp or logit_scale
            if 'temperature' in checkpoint:
                model.temperature = checkpoint['temperature']
            elif 'early_temp' in new_state_dict:
                # Temperature is part of the model state dict
                pass  # Already loaded with state dict
        
        model = model.to(args.device)
        model.eval()
        
        # Wrap with appropriate wrapper based on mode
        if args.force_early_exit:
            # Force all samples through early exit path
            if not args.early_exit_layer or args.early_exit_layer == 24:
                logger.warning("Force early exit requested but no early exit layer specified or layer=24")
            logger.info("Forcing all samples through early exit path")
            model = ICARForceEarlyExit(model, args.device)
        elif args.use_icc_routing:
            logger.info(f"Loading ICC from {args.icc_checkpoint}")
            icc_model = ConvNeXtICC(args.icc_checkpoint)
            model = ICCRoutedModel(model, icc_model, args.icc_threshold, args.device)
        elif not args.early_exit_layer:
            # For baseline models without ICC routing, wrap with ICARBaseline
            # This ensures the model always uses the full path
            model = ICARBaseline(model, args.device)
        
        # Create mixed dataset
        logger.info(f"Creating mixed dataset: {args.base_dataset} + LAION-COCO")
        dataset = create_preprocessed_mixed_dataset(
            base_dataset_name=args.base_dataset,
            base_data_root=args.base_data_root,
            laion_data_root=args.laion_data_root,
            complexity_scores_path=args.complexity_scores if Path(args.complexity_scores).exists() else None,
            image_size=config['data']['image_size']
        )
        
        # Create dataloader
        tokenizer = open_clip.get_tokenizer(config['model']['clip_model_name'])
        
        # Use custom collate function for mixed dataset
        collate_fn = lambda batch: collate_fn_mixed_eval(batch, tokenizer)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Prepare model info for similarity matrix naming
        model_info = {
            'dataset': args.base_dataset,
            'layer': args.early_exit_layer if args.early_exit_layer else 'baseline',
            'routing': 'forced_early' if args.force_early_exit else ('icc' if args.use_icc_routing else 'none')
        }
        
        # Evaluate
        logger.info("Starting evaluation...")
        eval_result = evaluate_mixed_dataset(
            model,
            dataloader,
            tokenizer,
            device=args.device,
            base_img_count=dataset.base_img_count,
            return_embeddings=args.eval_category,  # Return embeddings if category eval requested
            save_embeddings=True,  # Always save embeddings
            model_info=model_info
        )
    
    # Extract metrics
    metrics = eval_result['metrics'] if isinstance(eval_result, dict) and 'metrics' in eval_result else eval_result
    
    # Prepare results
    results = {
        "dataset": f"{args.base_dataset}_mixed",
        "base_images": dataset.base_img_count,
        "base_captions": dataset.base_caption_count,
        "laion_distractors": len(dataset.images) - dataset.base_img_count,
        "total_images": len(dataset.images),
        "total_captions": len(dataset.captions),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "early_exit_layer": args.early_exit_layer,
        "instance_metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add category-level evaluation if requested
    if args.eval_category and 'embeddings' in eval_result:
        logger.info("Starting category-level evaluation...")
        
        # Load category labels
        category_labels = {}
        
        # Load base dataset categories
        if args.base_dataset == "mscoco":
            if Path(args.coco_categories).exists():
                with open(args.coco_categories, 'r') as f:
                    base_categories = json.load(f)
                    category_labels.update(base_categories)
                logger.info(f"Loaded {len(base_categories)} COCO category labels")
            else:
                logger.warning(f"COCO categories file not found: {args.coco_categories}")
        elif args.base_dataset == "flickr30k":
            if Path(args.flickr_categories).exists():
                with open(args.flickr_categories, 'r') as f:
                    base_categories = json.load(f)
                    category_labels.update(base_categories)
                logger.info(f"Loaded {len(base_categories)} Flickr30k category labels")
            else:
                logger.warning(f"Flickr30k categories file not found: {args.flickr_categories}")
        
        # Load LAION categories
        if Path(args.laion_categories).exists():
            with open(args.laion_categories, 'r') as f:
                laion_categories = json.load(f)
                # We need to load the index->imgid mapping for COCO
                index_to_imgid = None
                if args.base_dataset == "mscoco":
                    # Check if we have the mapping file
                    mapping_path = Path("data/coco_test_index_to_imgid.json")
                    if mapping_path.exists():
                        with open(mapping_path, 'r') as f:
                            index_to_imgid = json.load(f)
                        logger.info("Loaded COCO index->imgid mapping")
                    else:
                        logger.warning("COCO index->imgid mapping not found, will create it")
                        # Create mapping from Karpathy dataset
                        karpathy_path = Path("data/caption_datasets/dataset_coco.json")
                        if karpathy_path.exists():
                            with open(karpathy_path, 'r') as f:
                                karpathy_data = json.load(f)
                            test_images = [img for img in karpathy_data['images'] if img['split'] == 'test']
                            index_to_imgid = {str(i): img['imgid'] for i, img in enumerate(test_images)}
                            # Save for future use
                            with open(mapping_path, 'w') as f:
                                json.dump(index_to_imgid, f)
                            logger.info("Created and saved COCO index->imgid mapping")

                # Remap base dataset categories using proper IDs
                if index_to_imgid:
                    remapped_categories = {}
                    for idx_str, imgid in index_to_imgid.items():
                        if str(imgid) in base_categories:
                            remapped_categories[idx_str] = base_categories[str(imgid)]
                    logger.info(f"Remapped {len(remapped_categories)} COCO categories using imgid mapping")
                    category_labels = remapped_categories
                else:
                    category_labels = base_categories

                # Now add LAION categories with offset
                for img_id, cat_data in laion_categories.items():
                    # Handle both numeric and 'laion_XXXXXXXX' format
                    if img_id.startswith('laion_'):
                        laion_idx = int(img_id.split('_')[1])
                    else:
                        laion_idx = int(img_id)
                    mixed_idx = dataset.base_img_count + laion_idx
                    category_labels[str(mixed_idx)] = cat_data
            logger.info(f"Loaded {len(laion_categories)} LAION category labels")
        else:
            logger.warning(f"LAION categories file not found: {args.laion_categories}")
        
        # Get embeddings and query indices
        embeddings = eval_result['embeddings']
        
        # Compute category-level metrics
        category_metrics = evaluate_category_retrieval_from_embeddings(
            image_embeddings=embeddings['image'],
            text_embeddings=embeddings['text'],
            category_labels=category_labels,
            query_image_indices=embeddings['query_image_indices'],
            query_text_indices=None,
            img2txt=embeddings.get('img2txt'),
            dataset_type=args.base_dataset,
            txt2img=embeddings['txt2img'],
            temperature=100.0  # Default temperature
        )
        
        results['category_metrics'] = category_metrics
        logger.info("Category-level evaluation complete")
    
    # Add routing statistics if using ICC
    if args.use_icc_routing:
        routing_stats = model.get_routing_stats()
        results["routing_stats"] = routing_stats
        logger.info(f"Routing: {routing_stats['early_percentage']:.1f}% early exit, "
                   f"{routing_stats['full_percentage']:.1f}% full path")
    
    # Add complexity statistics if available
    if dataset.complexity_scores:
        complexity_stats = getattr(dataset, 'complexity_stats', {})
        if complexity_stats:
            results["laion_complexity_stats"] = complexity_stats
    
    # Save results
    if args.output_file:
        output_path = output_dir / args.output_file
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_icc" if args.use_icc_routing else ""
        output_path = output_dir / f"mixed_{args.base_dataset}{suffix}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Dataset: {args.base_dataset} + {results['laion_distractors']} LAION distractors")
    print(f"Model: Layer {args.early_exit_layer}" if args.early_exit_layer else "Baseline")
    print(f"Using ICC routing: {args.use_icc_routing}")
    print("\nInstance-Level Retrieval Metrics:")
    # Handle flattened metrics format
    if 'i2t_r1' in metrics:
        print(f"  Image→Text R@1: {metrics['i2t_r1']:.2f}")
        print(f"  Image→Text R@5: {metrics['i2t_r5']:.2f}")
        print(f"  Text→Image R@1: {metrics['t2i_r1']:.2f}")
        print(f"  Text→Image R@5: {metrics['t2i_r5']:.2f}")
        avg_r1 = (metrics['i2t_r1'] + metrics['t2i_r1']) / 2
        print(f"  Average R@1: {avg_r1:.2f}")
    else:
        # Fallback for structured format
        print(f"  Image→Text R@1: {metrics.get('image_retrieval', {}).get('R@1', 0):.2f}")
        print(f"  Image→Text R@5: {metrics.get('image_retrieval', {}).get('R@5', 0):.2f}")
        print(f"  Text→Image R@1: {metrics.get('text_retrieval', {}).get('R@1', 0):.2f}")
        print(f"  Text→Image R@5: {metrics.get('text_retrieval', {}).get('R@5', 0):.2f}")
        print(f"  Average R@1: {metrics.get('average', {}).get('R@1', 0):.2f}")
    
    # Print category metrics if available
    if 'category_metrics' in results:
        cat_metrics = results['category_metrics']
        print("\nCategory-Level Retrieval Metrics:")
        print(f"  Queries: {cat_metrics['n_queries']}, Categories: {cat_metrics['n_categories']}")
        print(f"  mAP@10:")
        print(f"    Image→Text: {cat_metrics['mAP_i2t'].get('mAP@10', 0):.2f}%")
        print(f"    Text→Image: {cat_metrics['mAP_t2i'].get('mAP@10', 0):.2f}%")
        print(f"  mAP@100:")
        print(f"    Image→Text: {cat_metrics['mAP_i2t'].get('mAP@100', 0):.2f}%")
        print(f"    Text→Image: {cat_metrics['mAP_t2i'].get('mAP@100', 0):.2f}%")
    
    if args.use_icc_routing:
        print(f"\nRouting Distribution:")
        print(f"  Early exit: {routing_stats['early_percentage']:.1f}%")
        print(f"  Full path: {routing_stats['full_percentage']:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
