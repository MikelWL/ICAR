#!/usr/bin/env python3
"""
Training script for ICAR model.

This script handles the full training pipeline including:
- Configuration loading and merging
- Model and dataset initialization
- Training loop with checkpointing
- Logging and monitoring
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from datetime import datetime
import open_clip

# Add parent directory to path to import icar modules
sys.path.append(str(Path(__file__).parent.parent))

from icar.models.icar_model import ICARModel
from icar.models.icc import ConvNeXtICC
from icar.training.trainer_amp import ICARTrainerAMP
from icar.training.trainer_ddp import ICARTrainerDDP
from icar.data.coco_dataset import COCODataset, create_collate_fn
from icar.data.flickr30k_dataset import Flickr30kDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_ddp(gpu_ids=None):
    """
    Initialize DDP environment if torchrun variables are present.
    
    Args:
        gpu_ids: Optional list of GPU IDs to map local ranks to.
        
    Returns:
        Tuple of (rank, world_size, device, is_distributed)
    """
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        if gpu_ids is not None:
            if local_rank >= len(gpu_ids):
                raise ValueError(
                    f"Local rank {local_rank} exceeds number of specified GPUs {len(gpu_ids)}"
                )
            gpu_id = gpu_ids[local_rank]
        else:
            gpu_id = local_rank
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    
    if is_distributed:
        logger.info(f"Initialized process group: rank {rank}/{world_size}, device={device}")
    
    return rank, world_size, device, is_distributed


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: Path, model_config_path: Path = None) -> dict:
    """Load and merge configuration files.
    
    Args:
        config_path: Path to main config file
        model_config_path: Optional path to model-specific config
        
    Returns:
        Merged configuration dictionary
    """
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


def set_seed(seed: int, rank: int = 0, deterministic: bool = True):
    """Set random seed for reproducibility."""
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def create_datasets(config: dict, rank: int = 0):
    """Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset_name = config['data']['dataset_name'].lower()
    data_root = Path(config['data']['data_root'])
    image_size = config['data']['image_size']
    
    if dataset_name == 'mscoco':
        train_dataset = COCODataset(
            root=data_root,
            split='train',
            image_size=image_size
        )
        val_dataset = COCODataset(
            root=data_root,
            split='val',
            image_size=image_size
        )
    elif dataset_name == 'flickr30k':
        train_dataset = Flickr30kDataset(
            root=data_root,
            split='train',
            image_size=image_size
        )
        val_dataset = Flickr30kDataset(
            root=data_root,
            split='val',
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if rank == 0:
        logger.info(f"Created {dataset_name} datasets:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Val: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: dict, rank: int = 0, world_size: int = 1):
    """Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    
    # Get model name for tokenizer
    model_name = config['model']['clip_model_name']
    
    # Create collate functions
    train_collate_fn = create_collate_fn(model_name=model_name, is_train=True)
    val_collate_fn = create_collate_fn(model_name=model_name, is_train=False)
    
    # Create distributed samplers when using DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=train_collate_fn,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['eval_batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate_fn,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader


def create_models(config: dict, device: torch.device, rank: int = 0):
    """Create ICAR and ICC models.
    
    Args:
        config: Configuration dictionary
        device: Device to place models on
        
    Returns:
        Tuple of (icar_model, icc_model)
    """
    # Create ICAR model
    icar_model = ICARModel(
        clip_model_name=config['model']['clip_model_name'],
        pretrained=config['model']['pretrained'],
        early_exit_layer=config['model']['early_exit_layer']
    ).to(device)
    
    if rank == 0:
        logger.info(f"Created ICAR model with {config['model']['clip_model_name']}")
    
    # Load ICC model if checkpoint provided
    icc_model = None
    if config['model']['icc_checkpoint']:
        icc_checkpoint_path = Path(config['model']['icc_checkpoint'])
        if icc_checkpoint_path.exists():
            icc_model = ConvNeXtICC().to(device)
            checkpoint = torch.load(icc_checkpoint_path, map_location=device)
            icc_model.load_state_dict(checkpoint['state_dict'])
            icc_model.eval()
            if rank == 0:
                logger.info(f"Loaded ICC model from {icc_checkpoint_path}")
        else:
            if rank == 0:
                logger.warning(f"ICC checkpoint not found: {icc_checkpoint_path}")
    
    return icar_model, icc_model


def train_epoch(trainer, train_loader, optimizer, scheduler, epoch, config):
    """Train for one epoch.
    
    Args:
        trainer: Trainer instance
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        config: Configuration dictionary
        
    Returns:
        Average metrics for the epoch
    """
    avg_metrics = trainer.train_epoch(train_loader, optimizer, epoch)
    
    # Step scheduler
    if scheduler:
        scheduler.step()
    
    return avg_metrics


def validate(evaluator, model, val_loader, device):
    """Run validation.
    
    Args:
        evaluator: Evaluator instance
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device
        
    Returns:
        Validation metrics
    """
    from icar.evaluation.evaluator import ICAREvaluator, ICARBaseline
    
    evaluator = ICAREvaluator(device)
    baseline = ICARBaseline(model, device)
    
    # The evaluator now handles both training and eval mode dataloaders automatically
    metrics = evaluator.evaluate(baseline, val_loader, desc="Validation")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ICAR model")
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
        '--gpu-ids',
        type=str,
        default=None,
        help='Comma-separated GPU IDs to use for DDP (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Override data root from config'
    )
    parser.add_argument(
        '--icc-checkpoint',
        type=Path,
        default=None,
        help='Path to ICC checkpoint'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=None,
        help='Override checkpoint directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run one batch for testing'
    )
    parser.add_argument(
        '--resume',
        type=Path,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--full-precision',
        action='store_true',
        help='Use full precision training (disables AMP which is on by default)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=None,
        help='Specific GPU ID to use for single-GPU runs'
    )
    parser.add_argument(
        '--early-exit-layer',
        type=int,
        default=None,
        help='Early exit layer (e.g., 8, 12, 16, 20)'
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Train baseline CLIP model only (no early exit)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.model_config)
    
    if args.gpu_id is not None and args.gpu_ids is not None:
        raise ValueError("Use either --gpu-id or --gpu-ids, not both.")
    
    # Override config with command line arguments
    if args.data_root:
        config['data']['data_root'] = str(args.data_root)
    if args.icc_checkpoint:
        config['model']['icc_checkpoint'] = str(args.icc_checkpoint)
    if args.baseline_only:
        if not args.checkpoint_dir:
            base_checkpoint_dir = Path(config['training']['checkpoint_dir'])
            config['training']['checkpoint_dir'] = str(base_checkpoint_dir / 'baseline')
        config['training']['baseline_only'] = True
    else:
        config['training']['baseline_only'] = False
        
    if args.early_exit_layer:
        config['model']['early_exit_layer'] = args.early_exit_layer
        if not args.checkpoint_dir:
            base_checkpoint_dir = Path(config['training']['checkpoint_dir'])
            config['training']['checkpoint_dir'] = str(base_checkpoint_dir / f'layer_{args.early_exit_layer}')
    if args.checkpoint_dir:
        config['training']['checkpoint_dir'] = str(args.checkpoint_dir)
    
    # Validate required paths
    if not config['data']['data_root']:
        raise ValueError("Data root must be specified in config or via --data-root")
    
    requested_device = str(config.get('device', 'cuda')).lower()
    use_cpu = requested_device.startswith('cpu')
    use_distributed_env = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    elif not use_distributed_env:
        if args.gpu_id is not None:
            gpu_ids = [args.gpu_id]
        elif 'cuda_id' in config:
            gpu_ids = [config['cuda_id']]
    
    if use_cpu:
        rank = 0
        world_size = 1
        device = torch.device('cpu')
        is_distributed = False
        if use_distributed_env:
            logger.warning("Ignoring torchrun environment because device is set to CPU.")
    else:
        rank, world_size, device, is_distributed = setup_ddp(gpu_ids)
    
    is_rank_zero = rank == 0
    if is_rank_zero and world_size == 1 and gpu_ids is not None and len(gpu_ids) > 1:
        logger.warning("Multiple --gpu-ids provided but no torchrun detected; using the first GPU only.")
    
    try:
        # Set random seed
        set_seed(config['seed'], rank=rank, deterministic=(world_size == 1))
        
        if is_rank_zero:
            logger.info(f"Using device: {device}")
            logger.info(f"World size: {world_size}")
        
        # Create datasets and dataloaders
        train_dataset, val_dataset = create_datasets(config, rank=rank)
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, config, rank=rank, world_size=world_size
        )
        
        # Create models
        icar_model, icc_model = create_models(config, device, rank=rank)
        
        # Create trainer
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        if is_rank_zero:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if config['training'].get('baseline_only', False) and is_rank_zero:
            logger.info("=" * 70)
            logger.info("BASELINE TRAINING MODE - Training CLIP only (no early exit)")
            logger.info("Early exit layers will NOT be trained")
            logger.info("=" * 70)
        
        use_amp = not args.full_precision
        if world_size > 1:
            if is_rank_zero:
                logger.info(f"Using {'Mixed' if use_amp else 'Full'} Precision Training (DDP)")
            trainer = ICARTrainerDDP(
                model=icar_model,
                device=device,
                checkpoint_dir=checkpoint_dir if is_rank_zero else None,
                log_interval=config['logging']['log_every_n_steps'],
                use_amp=use_amp,
                rank=rank,
                world_size=world_size,
                baseline_only=config['training'].get('baseline_only', False)
            )
        else:
            if is_rank_zero:
                logger.info("Using Mixed Precision Training (AMP) - default" if use_amp else "Using Full Precision Training (FP32)")
            trainer = ICARTrainerAMP(
                model=icar_model,
                device=device,
                checkpoint_dir=checkpoint_dir,
                log_interval=config['logging']['log_every_n_steps'],
                use_amp=use_amp,
                baseline_only=config['training'].get('baseline_only', False)
            )
        
        # Create optimizer
        optimizer = trainer.create_optimizer(
            lr_backbone=config['training']['lr_backbone'],
            lr_early_proj=config['training']['lr_early_proj'],
            lr_temperature=config['training']['lr_temperature'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Create scheduler
        scheduler = None
        if config['training']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(train_loader) * config['training']['num_epochs']
            )
        
        # Resume from checkpoint if specified
        if args.resume:
            checkpoint_data = trainer.load_checkpoint(args.resume, optimizer)
            if is_rank_zero:
                logger.info(f"Resumed from epoch {checkpoint_data['epoch']}")
        
        # Training loop
        if args.dry_run:
            if is_rank_zero:
                logger.info("Running dry run (one batch only)")
            try:
                batch = next(iter(train_loader))
            except StopIteration:
                if is_rank_zero:
                    logger.warning(f"Dataset too small for batch size {config['training']['batch_size']}")
                temp_collate_fn = create_collate_fn(
                    model_name=config['model']['clip_model_name'], is_train=True
                )
                temp_loader = DataLoader(
                    train_dataset,
                    batch_size=min(len(train_dataset), 4),
                    shuffle=True,
                    collate_fn=temp_collate_fn
                )
                batch = next(iter(temp_loader))
            
            images = batch['images']
            texts = batch['text']
            if world_size == 1:
                images = images.to(device)
                texts = texts.to(device)
            
            metrics = trainer.training_step(images, texts, optimizer)
            
            if is_rank_zero:
                logger.info("Dry run completed successfully!")
                logger.info(f"Metrics: {metrics}")
            return
        
        if is_rank_zero:
            logger.info(f"Starting training for {config['training']['num_epochs']} epochs")
        
        for epoch in range(config['training']['num_epochs']):
            if is_rank_zero:
                logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
            
            train_metrics = train_epoch(
                trainer, train_loader, optimizer, scheduler, epoch, config
            )
            
            if is_rank_zero:
                logger.info(f"Train metrics: {train_metrics}")
            
            if is_rank_zero and (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
                logger.info("Running validation...")
                eval_model = trainer.model.module if hasattr(trainer.model, 'module') else icar_model
                val_metrics = validate(None, eval_model, val_loader, device)
                logger.info(f"Val metrics: {val_metrics}")
                
                checkpoint_name = f"checkpoint_epoch{epoch+1}.pt"
                trainer.save_checkpoint(optimizer, train_metrics, checkpoint_name)
        
        if is_rank_zero:
            logger.info("Training completed!")
    finally:
        if is_distributed:
            cleanup_ddp()


if __name__ == "__main__":
    main()
