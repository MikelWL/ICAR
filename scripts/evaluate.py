#!/usr/bin/env python3
"""
Evaluation script for ICAR model checkpoints.

This script evaluates trained checkpoints on the test split of the dataset.
Currently supports full path evaluation only, with infrastructure for future
early exit variant studies.

Usage:
    # Evaluate a checkpoint
    python scripts/evaluate.py --config path/to/config.yaml --checkpoint path/to/checkpoint.pt
    
    # With model-specific config
    python scripts/evaluate.py --config coco.yaml --model-config vit_l_14.yaml --checkpoint checkpoint.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# Add parent directory to path to import icar modules
sys.path.append(str(Path(__file__).parent.parent))

from icar.models.icar_model import ICARModel
from icar.evaluation.evaluator import ICAREvaluator, ICARBaseline
from icar.data.coco_dataset import COCODataset, create_collate_fn
from icar.data.flickr30k_dataset import Flickr30kDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path, model_config_path: Path = None) -> dict:
    """Load and merge configuration files."""
    # Load base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and merge model-specific config if provided
    if model_config_path and model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Deep merge configs
        for key, value in model_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return config


def create_test_dataset(config: dict, split: str = 'test'):
    """Create test dataset based on config."""
    dataset_name = config['data']['dataset_name'].lower()
    data_root = Path(config['data']['data_root'])
    image_size = config['data']['image_size']
    
    if dataset_name == 'mscoco':
        test_dataset = COCODataset(
            root=data_root,
            split=split,
            image_size=image_size
        )
    elif dataset_name == 'flickr30k':
        test_dataset = Flickr30kDataset(
            root=data_root,
            split=split,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Created {dataset_name} {split} dataset with {len(test_dataset)} samples")
    return test_dataset


def create_test_dataloader(test_dataset, config: dict):
    """Create dataloader for test dataset."""
    batch_size = config['evaluation']['eval_batch_size']
    num_workers = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    
    # Get model name for tokenizer
    model_name = config['model']['clip_model_name']
    
    # Create collate function for evaluation mode
    test_collate_fn = create_collate_fn(model_name=model_name, is_train=False)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=test_collate_fn,
        persistent_workers=(num_workers > 0)
    )
    
    return test_loader


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load checkpoint, handling both single-GPU and DDP formats.
    
    Returns:
        dict: Checkpoint data with at least 'model_state_dict' key
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP state dict (remove 'module.' prefix if present)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Remove 'module.' prefix from DDP checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    checkpoint['model_state_dict'] = new_state_dict
    
    # Log checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICAR model checkpoint")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('icar/configs/coco.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model-config',
        type=Path,
        default=None,
        help='Path to model-specific configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to checkpoint to evaluate'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Override data root from config'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path to save evaluation results (JSON format)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mscoco', 'flickr30k'],
        default=None,
        help='Override dataset from config'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split to evaluate (default: test)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override evaluation batch size from config'
    )
    parser.add_argument(
        '--similarity-on-cpu',
        action='store_true',
        help='Compute similarities on CPU to save GPU memory'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        # Use first available GPU
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config, args.model_config)
    
    # Override config with command line arguments
    if args.data_root:
        config['data']['data_root'] = str(args.data_root)
    if args.dataset:
        config['data']['dataset_name'] = args.dataset
    if args.batch_size:
        config['evaluation']['eval_batch_size'] = args.batch_size
    
    # Validate checkpoint exists
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Create test dataset and dataloader
    test_dataset = create_test_dataset(config, split=args.split)
    test_loader = create_test_dataloader(test_dataset, config)
    
    # Create ICAR model
    icar_model = ICARModel(
        clip_model_name=config['model']['clip_model_name'],
        pretrained=config['model']['pretrained'],
        early_exit_layer=config['model']['early_exit_layer']
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device)
    icar_model.load_state_dict(checkpoint['model_state_dict'])
    icar_model = icar_model.to(device)
    icar_model.eval()
    
    logger.info("Model loaded successfully")
    
    # Extract temperature from model
    temperature = icar_model.clip.logit_scale.exp().item()
    logger.info(f"Model temperature: {temperature:.2f}")
    
    # Create evaluator with similarity computation preference and temperature
    evaluator = ICAREvaluator(device, similarity_on_cpu=args.similarity_on_cpu, temperature=temperature)
    
    # Currently: Full path evaluation only
    # Create baseline evaluator (uses full path for all images)
    baseline_model = ICARBaseline(icar_model, device)
    
    logger.info("=" * 70)
    logger.info("Starting evaluation on test set (full path only)")
    logger.info("=" * 70)
    
    # Run evaluation
    metrics = evaluator.evaluate(
        baseline_model,
        test_loader,
        desc="Evaluating test set"
    )
    
    # This script intentionally evaluates the full-path baseline only.
    
    # Print results
    print("\n" + "=" * 70)
    print(f"Evaluation Results - {config['data']['dataset_name']} Test Set")
    print("=" * 70)
    
    # Print retrieval metrics
    if 'i2t_r1' in metrics:
        print("\nImage-to-Text Retrieval:")
        print(f"  R@1:  {metrics['i2t_r1']:.1f}%")
        print(f"  R@5:  {metrics['i2t_r5']:.1f}%")
        print(f"  R@10: {metrics['i2t_r10']:.1f}%")
        print(f"  Median Rank: {metrics['i2t_median_rank']:.1f}")
        print(f"  Mean Rank: {metrics['i2t_mean_rank']:.1f}")
    
    if 't2i_r1' in metrics:
        print("\nText-to-Image Retrieval:")
        print(f"  R@1:  {metrics['t2i_r1']:.1f}%")
        print(f"  R@5:  {metrics['t2i_r5']:.1f}%")
        print(f"  R@10: {metrics['t2i_r10']:.1f}%")
        print(f"  Median Rank: {metrics['t2i_median_rank']:.1f}")
        print(f"  Mean Rank: {metrics['t2i_mean_rank']:.1f}")
    
    # Print average of i2t and t2i R@1
    if 'i2t_r1' in metrics and 't2i_r1' in metrics:
        avg_r1 = (metrics['i2t_r1'] + metrics['t2i_r1']) / 2
        print(f"\nAverage R@1: {avg_r1:.1f}%")
    
    print("\n" + "=" * 70)
    
    # Log checkpoint info
    print(f"\nCheckpoint: {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    logger.info("Evaluation completed successfully")
    
    # Save results if output path provided
    if args.output:
        import json
        import datetime
        
        # Prepare results dictionary
        results = {
            'experiment_info': {
                'checkpoint': str(args.checkpoint),
                'dataset': config['data']['dataset_name'],
                'split': args.split,
                'evaluation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(device),
                'similarity_on_cpu': args.similarity_on_cpu
            },
            'config': {
                'model': config['model'],
                'evaluation': config['evaluation']
            },
            'metrics': metrics
        }
        
        # Add checkpoint info if available
        if 'epoch' in checkpoint:
            results['experiment_info']['epoch'] = checkpoint['epoch']
        if 'metrics' in checkpoint:
            results['experiment_info']['training_metrics'] = checkpoint['metrics']
        
        # Save to JSON
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
