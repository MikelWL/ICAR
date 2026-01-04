"""ICAR training components with DistributedDataParallel (DDP) support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
import json
from datetime import datetime
import os


logger = logging.getLogger(__name__)


class ICARTrainerDDP:
    """Trainer for ICAR model with DDP and AMP support."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        use_amp: bool = True,
        rank: int = 0,
        world_size: int = 1,
        baseline_only: bool = False,
    ):
        """
        Initialize ICAR trainer with DDP support.
        
        Args:
            model: ICAR model to train
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Steps between logging
            checkpoint_interval: Steps between checkpoints
            use_amp: Whether to use Automatic Mixed Precision
            rank: Process rank for DDP
            world_size: Total number of processes
            baseline_only: Whether to train baseline CLIP only (no early exit)
        """
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.baseline_only = baseline_only
        
        # Move model to device first
        self.model = model.to(device)
        
        # Wrap in DDP if using multiple GPUs
        if world_size > 1:
            self.model = DDP(
                self.model, 
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False
            )
            logger.info(f"Initialized DDP on rank {rank}/{world_size}")
        
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Metrics tracking (only on main process)
        self.metrics_history = [] if self.is_main_process else None
        
        # AMP: Initialize gradient scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        if checkpoint_dir and self.is_main_process:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_main_process:
            logger.info(f"Initialized trainer with AMP: {self.use_amp}, World Size: {world_size}")
    
    def create_optimizer(
        self,
        lr_backbone: float = 1e-6,
        lr_early_proj: float = 1e-4,
        lr_temperature: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with parameter groups for different learning rates.
        
        Args:
            lr_backbone: Learning rate for CLIP backbone
            lr_early_proj: Learning rate for early projection head
            lr_temperature: Learning rate for temperature parameters
            weight_decay: Weight decay for all parameters
            
        Returns:
            Configured optimizer
        """
        # Handle DDP wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Group parameters
        param_groups = []
        
        if not self.baseline_only:
            # Early projection parameters (highest LR) - only for ICAR training
            early_proj_params = []
            early_proj_params.extend(model.early_proj.parameters())
            if hasattr(model, 'early_ln'):
                early_proj_params.extend(model.early_ln.parameters())
            
            param_groups.append({
                'params': early_proj_params,
                'lr': lr_early_proj,
                'weight_decay': weight_decay,
                'name': 'early_projection'
            })
        
        # Temperature parameters (high LR, no weight decay)
        temp_params = []
        if not self.baseline_only and hasattr(model, 'early_temp'):
            temp_params.append(model.early_temp)
        if hasattr(model.clip, 'logit_scale'):
            temp_params.append(model.clip.logit_scale)
        
        if temp_params:
            param_groups.append({
                'params': temp_params,
                'lr': lr_temperature,
                'weight_decay': 0.0,  # No weight decay for temperature
                'name': 'temperature'
            })
        
        # CLIP backbone parameters (lowest LR)
        # Exclude parameters already added
        added_params = set()
        for group in param_groups:
            added_params.update(id(p) for p in group['params'])
        
        backbone_params = [
            p for p in model.parameters() 
            if id(p) not in added_params
        ]
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': lr_backbone,
                'weight_decay': weight_decay,
                'name': 'backbone'
            })
        
        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
        
        # Log parameter groups (only on main process)
        if self.is_main_process:
            for group in param_groups:
                logger.info(
                    f"Parameter group '{group['name']}': "
                    f"{len(group['params'])} params, lr={group['lr']}"
                )
        
        return optimizer
    
    def training_step(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Execute one training step with AMP and DDP support.
        
        Args:
            images: Batch of images [batch_size, 3, 224, 224]
            texts: Batch of tokenized texts [batch_size, context_length]
            optimizer: Optimizer to use
            
        Returns:
            Dictionary of metrics from this step
        """
        self.model.train()
        
        # Move data to device
        images = images.to(self.device)
        texts = texts.to(self.device)
        
        # Get actual model (unwrap DDP if needed)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # AMP: Use autocast for forward pass
        if self.use_amp:
            with autocast():
                if self.baseline_only:
                    # Baseline training: only use CLIP's encode functions directly
                    image_features = model.clip.encode_image(images)
                    text_features = model.clip.encode_text(texts)
                    
                    # Normalize features
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    # Get temperature from CLIP model
                    logit_scale = model.clip.logit_scale.exp()
                    logit_scale = torch.clamp(logit_scale, max=100.0)
                    
                    # Compute similarity and loss
                    logits = logit_scale * image_features @ text_features.T
                    labels = torch.arange(len(images), device=images.device)
                    loss = F.cross_entropy(logits, labels)
                    
                    # For metrics compatibility
                    early_loss = loss
                    final_loss = loss
                    outputs = {'early_scale': logit_scale, 'final_scale': logit_scale}
                else:
                    # Standard ICAR dual-path training
                    outputs = self.model(images, texts)
                    
                    # Extract logits and temperature scales
                    early_logits = outputs['early_logits']
                    final_logits = outputs['final_logits']
                    
                    # Compute losses directly using cross-entropy
                    labels = torch.arange(len(images), device=images.device)
                    early_loss = F.cross_entropy(early_logits, labels)
                    final_loss = F.cross_entropy(final_logits, labels)
                    
                    # Combined loss (equal weighting for balanced optimization)
                    loss = 0.5 * early_loss + 0.5 * final_loss
        else:
            # Standard forward pass without AMP
            if self.baseline_only:
                # Baseline training without AMP
                image_features = model.clip.encode_image(images)
                text_features = model.clip.encode_text(texts)
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logit_scale = model.clip.logit_scale.exp()
                logit_scale = torch.clamp(logit_scale, max=100.0)
                
                logits = logit_scale * image_features @ text_features.T
                labels = torch.arange(len(images), device=images.device)
                loss = F.cross_entropy(logits, labels)
                
                early_loss = loss
                final_loss = loss
                outputs = {'early_scale': logit_scale, 'final_scale': logit_scale}
            else:
                outputs = self.model(images, texts)
                early_logits = outputs['early_logits']
                final_logits = outputs['final_logits']
                
                labels = torch.arange(len(images), device=images.device)
                early_loss = F.cross_entropy(early_logits, labels)
                final_loss = F.cross_entropy(final_logits, labels)
                loss = 0.5 * early_loss + 0.5 * final_loss
        
        # Extract temperature values (outside autocast)
        early_temp = outputs['early_scale']
        final_temp = outputs['final_scale']
        
        # Backward pass
        optimizer.zero_grad()
        
        if self.use_amp:
            # AMP: Scale loss and backward
            self.scaler.scale(loss).backward()
            
            # AMP: Unscale gradients before clipping
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping (MANDATORY for stability)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0
            )
            
            # AMP: Step optimizer and update scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0
            )
            
            # Optimizer step
            optimizer.step()
        
        # Prepare metrics
        metrics = {
            'loss': loss.item(),
            'loss_early': early_loss.item(),
            'loss_full': final_loss.item(),
            'early_temp': early_temp.item(),
            'final_temp': final_temp.item(),
            'grad_norm': grad_norm.item(),
            'lr_backbone': optimizer.param_groups[-1]['lr'],
            'lr_early_proj': optimizer.param_groups[0]['lr'],
        }
        
        # AMP: Add scaler state to metrics
        if self.use_amp:
            scale = self.scaler.get_scale()
            # Ensure scale is a valid float (handle inf/nan)
            if not torch.isfinite(torch.tensor(scale)):
                scale = float('inf') if scale > 0 else 0.0
            metrics['amp_scale'] = float(scale)
        
        # Update global step
        self.global_step += 1
        
        # Log if needed (only on main process)
        if self.is_main_process and self.global_step % self.log_interval == 0:
            self._log_metrics(metrics)
        
        # Save checkpoint if needed (only on main process)
        if self.is_main_process and self.checkpoint_dir and self.global_step % self.checkpoint_interval == 0:
            self.save_checkpoint(optimizer, metrics)
        
        return metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer to use
            epoch: Current epoch number
            
        Returns:
            Average metrics for the epoch
        """
        self.epoch = epoch
        epoch_metrics = []
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data from batch
            if isinstance(batch, dict):
                images = batch['images']
                texts = batch['text']
            else:
                images, texts = batch
            
            # Training step
            metrics = self.training_step(images, texts, optimizer)
            epoch_metrics.append(metrics)
            
            # Progress logging (only on main process)
            if self.is_main_process and batch_idx % self.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {metrics['loss']:.4f} "
                    f"(Early: {metrics['loss_early']:.4f}, "
                    f"Full: {metrics['loss_full']:.4f})"
                )
        
        # Compute epoch averages
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            try:
                # Filter out non-finite values and compute average
                values = [m[key] for m in epoch_metrics if isinstance(m[key], (int, float)) and torch.isfinite(torch.tensor(m[key]))]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
                else:
                    # If all values are non-finite, use 0 or skip
                    avg_metrics[key] = 0.0
                    if self.is_main_process:
                        logger.warning(f"All values for metric '{key}' were non-finite in epoch {epoch}")
            except (TypeError, ValueError) as e:
                # Handle any type conversion issues
                if self.is_main_process:
                    logger.error(f"Error averaging metric '{key}': {e}")
                avg_metrics[key] = 0.0
        
        if self.is_main_process:
            logger.info(
                f"Epoch {epoch} complete. "
                f"Avg Loss: {avg_metrics['loss']:.4f} "
                f"(Early: {avg_metrics['loss_early']:.4f}, "
                f"Full: {avg_metrics['loss_full']:.4f})"
            )
        
        return avg_metrics
    
    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> Path:
        """
        Save training checkpoint with AMP and DDP state.
        
        Args:
            optimizer: Current optimizer state to save
            metrics: Current metrics to save
            checkpoint_name: Optional checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        if not self.is_main_process:
            return  # Only save on main process
            
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")
        
        # Generate checkpoint name if not provided
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch{self.epoch}_step{self.global_step}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Get the actual model (unwrap DDP if needed)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'metrics_history': self.metrics_history[-1000:] if self.metrics_history else [],
            'config': {
                'model_name': model_to_save.clip_model_name,
                'pretrained': model_to_save.pretrained,
                'early_exit_layer': model_to_save.early_exit_layer,
                'use_amp': self.use_amp,
                'world_size': self.world_size,
            }
        }
        
        # AMP: Save scaler state
        if self.use_amp and self.scaler is not None:
            checkpoint['amp_scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint link
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict:
        """
        Load training checkpoint with AMP and DDP state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to restore state
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get the actual model (unwrap DDP if needed)
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Load model state
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # AMP: Load scaler state
        if self.use_amp and self.scaler is not None and 'amp_scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
            logger.info("Loaded AMP scaler state from checkpoint")
        
        # Restore training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        if self.is_main_process:
            self.metrics_history = checkpoint.get('metrics_history', [])
        
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {self.epoch}, step {self.global_step})"
        )
        
        return checkpoint
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and history."""
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['step'] = self.global_step
        metrics['epoch'] = self.epoch
        metrics['rank'] = self.rank
        
        # Add to history
        if self.metrics_history is not None:
            # Create a copy to avoid reference issues
            metrics_copy = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    # Ensure numeric values are finite
                    if torch.isfinite(torch.tensor(v)):
                        metrics_copy[k] = v
                    else:
                        metrics_copy[k] = float('nan')
                else:
                    # Keep non-numeric values as is
                    metrics_copy[k] = v
            self.metrics_history.append(metrics_copy)
        
        # Log key metrics
        log_msg = (
            f"[Rank {self.rank}] Step {self.global_step}: "
            f"Loss={metrics['loss']:.4f} "
            f"(E={metrics['loss_early']:.4f}, F={metrics['loss_full']:.4f}) "
            f"Temps=(E={metrics['early_temp']:.1f}, F={metrics['final_temp']:.1f}) "
            f"Grad={metrics['grad_norm']:.3f}"
        )
        
        if 'amp_scale' in metrics:
            # Handle potential inf/nan values in AMP scale
            scale_val = metrics['amp_scale']
            if isinstance(scale_val, (int, float)) and torch.isfinite(torch.tensor(scale_val)):
                log_msg += f" AMP={scale_val:.1f}"
            else:
                log_msg += f" AMP=UNSTABLE"
        
        logger.info(log_msg)