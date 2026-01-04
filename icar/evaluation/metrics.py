"""Evaluation metrics for ICAR model."""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""
    r1: float
    r5: float
    r10: float
    r50: float
    median_rank: float
    mean_rank: float


def compute_retrieval_metrics(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
    temperature: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute retrieval metrics between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [N, D]
        k_values: Values of k for R@k metrics
        temperature: Temperature scaling factor. If None, no scaling is applied.
        
    Returns:
        Dictionary with retrieval metrics
    """
    # Ensure embeddings are normalized
    embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
    embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    similarity = embeddings1 @ embeddings2.T
    
    # Apply temperature scaling if specified
    if temperature is not None:
        similarity = temperature * similarity
    
    # Get ranks
    ranks = get_ranks(similarity)
    
    # Compute metrics
    metrics = {}
    for k in k_values:
        metrics[f'r{k}'] = (ranks < k).float().mean().item() * 100
    
    metrics['median_rank'] = ranks.median().item()
    metrics['mean_rank'] = ranks.float().mean().item()
    
    return metrics


def get_ranks(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """
    Get ranking of ground truth matches.
    
    Args:
        similarity_matrix: Similarity scores [N, N]
        
    Returns:
        Ranks of diagonal elements (ground truth matches)
    """
    # Sort similarities in descending order
    sorted_indices = similarity_matrix.argsort(dim=1, descending=True)
    
    # Find position of ground truth (diagonal elements)
    batch_size = similarity_matrix.size(0)
    gt_indices = torch.arange(batch_size, device=similarity_matrix.device)
    
    # Find rank of each ground truth
    ranks = torch.zeros(batch_size, device=similarity_matrix.device)
    for i in range(batch_size):
        positions = (sorted_indices[i] == gt_indices[i]).nonzero()
        if positions.numel() > 0:
            ranks[i] = positions[0]
        else:
            # If not found (shouldn't happen with proper data), assign last rank
            ranks[i] = batch_size - 1
    
    return ranks


def compute_image_text_retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
    temperature: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute both image-to-text and text-to-image retrieval metrics.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        k_values: Values of k for R@k metrics
        temperature: Temperature scaling factor. If None, no scaling is applied.
        
    Returns:
        Dictionary with i2t and t2i metrics
    """
    # Image-to-text retrieval
    i2t_metrics = compute_retrieval_metrics(
        image_embeddings, text_embeddings, k_values, temperature=temperature
    )
    
    # Text-to-image retrieval
    t2i_metrics = compute_retrieval_metrics(
        text_embeddings, image_embeddings, k_values, temperature=temperature
    )
    
    return {
        'i2t': i2t_metrics,
        't2i': t2i_metrics
    }


def compute_retrieval_metrics_with_mappings(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    img2txt: Dict[int, List[int]],
    txt2img: Dict[int, int],
    k_values: Tuple[int, ...] = (1, 5, 10),
    query_image_indices: List[int] = None,
    query_text_indices: List[int] = None,
    temperature: float = 100.0
) -> Dict[str, float]:
    """
    Compute retrieval metrics with proper image-text mappings.
    
    This handles the many-to-many relationship where each image 
    has multiple valid captions. Supports specifying query indices
    for mixed dataset evaluation.
    
    Args:
        image_features: Image embeddings [N_images, D]
        text_features: Text embeddings [N_texts, D]
        img2txt: Mapping from image index to list of valid text indices
        txt2img: Mapping from text index to its source image index
        k_values: Values of k for R@k metrics
        query_image_indices: Indices of images to use as queries (default: all)
        query_text_indices: Indices of texts to use as queries (default: all)
        temperature: Temperature scaling factor
        
    Returns:
        Dictionary with flattened i2t and t2i metrics
    """
    import logging
    logger = logging.getLogger(__name__)
    
    device = image_features.device
    
    # Compute similarity scores with temperature scaling
    scores_i2t = temperature * image_features @ text_features.T
    scores_t2i = scores_i2t.T
    
    logger.info(f"Computing metrics on {device}, scores shape: {scores_i2t.shape}")
    
    # If no query indices specified, use all samples
    if query_image_indices is None:
        query_image_indices = list(range(image_features.shape[0]))
    if query_text_indices is None:
        query_text_indices = list(range(text_features.shape[0]))
    
    logger.info(f"Using {len(query_image_indices)} image queries and {len(query_text_indices)} text queries")
    
    # Image-to-text retrieval
    import time
    start_time = time.time()
    ranks_i2t = []
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for batch_start in range(0, len(query_image_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(query_image_indices))
        batch_query_indices = query_image_indices[batch_start:batch_end]
        
        for img_idx in batch_query_indices:
            img_scores = scores_i2t[img_idx]
            
            # Sort in descending order
            sorted_indices = img_scores.argsort(descending=True)
            
            # Find the highest ranking valid caption
            valid_captions = img2txt[img_idx]
            best_rank = scores_i2t.shape[1]  # worst case
            
            # Create a mask for valid captions on GPU
            valid_mask = torch.zeros(scores_i2t.shape[1], dtype=torch.bool, device=device)
            valid_mask[valid_captions] = True
            
            # Find first True in the sorted order
            sorted_mask = valid_mask[sorted_indices]
            first_valid = sorted_mask.nonzero(as_tuple=True)[0]
            if first_valid.numel() > 0:
                best_rank = first_valid[0].item()
            
            ranks_i2t.append(best_rank)
    
    ranks_i2t = torch.tensor(ranks_i2t, device=device)
    logger.info(f"I2T ranking completed in {time.time() - start_time:.2f} seconds")
    
    # Text-to-image retrieval
    start_time = time.time()
    ranks_t2i = []
    
    # Process in batches to avoid memory issues
    for batch_start in range(0, len(query_text_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(query_text_indices))
        batch_query_indices = query_text_indices[batch_start:batch_end]
        
        for txt_idx in batch_query_indices:
            txt_scores = scores_t2i[txt_idx]
            
            # Sort in descending order
            sorted_indices = txt_scores.argsort(descending=True)
            
            # Find the rank of the correct image
            correct_img = txt2img[txt_idx]
            rank = (sorted_indices == correct_img).nonzero(as_tuple=True)[0]
            if rank.numel() > 0:
                ranks_t2i.append(rank[0].item())
            else:
                ranks_t2i.append(scores_t2i.shape[1] - 1)  # worst case
    
    ranks_t2i = torch.tensor(ranks_t2i, device=device)
    logger.info(f"T2I ranking completed in {time.time() - start_time:.2f} seconds")
    
    # Compute metrics - return flattened format for consistency
    metrics = {}
    
    for k in k_values:
        metrics[f'i2t_r{k}'] = (ranks_i2t < k).float().mean().item() * 100
        metrics[f't2i_r{k}'] = (ranks_t2i < k).float().mean().item() * 100
    
    metrics['i2t_median_rank'] = ranks_i2t.median().item()
    metrics['i2t_mean_rank'] = ranks_i2t.float().mean().item()
    
    metrics['t2i_median_rank'] = ranks_t2i.median().item()
    metrics['t2i_mean_rank'] = ranks_t2i.float().mean().item()
    
    return metrics


def compute_efficiency_metrics(
    routing_stats: Dict[str, int],
    flops_early: float,
    flops_full: float,
    flops_icc: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute efficiency metrics for adaptive routing.
    
    Args:
        routing_stats: Dictionary with 'early' and 'full' counts
        flops_early: FLOPs for early exit path
        flops_full: FLOPs for full model path
        flops_icc: FLOPs for Image Complexity Classifier (optional)
        
    Returns:
        Dictionary with efficiency metrics
    """
    total_samples = routing_stats['early'] + routing_stats['full']
    if total_samples == 0:
        return {}
    
    # Routing distribution
    early_ratio = routing_stats['early'] / total_samples
    full_ratio = routing_stats['full'] / total_samples
    
    # Average FLOPs without ICC
    avg_flops_baseline = flops_full
    avg_flops_adaptive = early_ratio * flops_early + full_ratio * flops_full
    
    # Add ICC overhead if provided
    if flops_icc is not None:
        avg_flops_adaptive += flops_icc
    
    # Compute savings
    flops_reduction = (avg_flops_baseline - avg_flops_adaptive) / avg_flops_baseline
    speedup_factor = avg_flops_baseline / avg_flops_adaptive
    
    return {
        'early_exit_ratio': early_ratio * 100,
        'full_path_ratio': full_ratio * 100,
        'avg_flops': avg_flops_adaptive,
        'flops_reduction': flops_reduction * 100,
        'speedup_factor': speedup_factor,
        'total_samples': total_samples
    }


def aggregate_metrics(
    metrics_list: list,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Aggregate metrics across batches.
    
    Args:
        metrics_list: List of metric dictionaries
        prefix: Prefix to add to metric names
        
    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}
    
    # Get all metric names from first entry
    metric_names = list(metrics_list[0].keys())
    
    # Compute mean for each metric
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list if name in m]
        if values:
            key = f"{prefix}{name}" if prefix else name
            aggregated[key] = np.mean(values)
    
    return aggregated