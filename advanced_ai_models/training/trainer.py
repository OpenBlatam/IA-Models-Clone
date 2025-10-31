from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Callable, Union
import numpy as np
import time
import logging
import os
import json
from pathlib import Path
import wandb
from tqdm import tqdm
import math
                from lion_pytorch import Lion
from typing import Any, List, Dict, Optional
import asyncio
"""
Advanced Training Utilities - Mixed Precision, Distributed Training & Optimization
Featuring advanced training techniques and optimizations.
"""


logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Advanced trainer with mixed precision, distributed training, and optimizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        config: Dict[str, Any] = None
    ):
        
    """__init__ function."""
self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        
        # Training configuration
        self.batch_size = self.config.get("batch_size", 32)
        self.num_epochs = self.config.get("num_epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.gradient_clip_val = self.config.get("gradient_clip_val", 1.0)
        self.accumulate_grad_batches = self.config.get("accumulate_grad_batches", 1)
        
        # Mixed precision
        self.use_mixed_precision = self.config.get("use_mixed_precision", True)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Distributed training
        self.use_distributed = self.config.get("use_distributed", False)
        self.local_rank = self.config.get("local_rank", 0)
        self.world_size = self.config.get("world_size", 1)
        
        # Logging and monitoring
        self.log_every_n_steps = self.config.get("log_every_n_steps", 100)
        self.save_every_n_epochs = self.config.get("save_every_n_epochs", 5)
        self.output_dir = self.config.get("output_dir", "./outputs")
        
        # Initialize components
        self._setup_device()
        self._setup_data_loaders()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Advanced Trainer initialized on device: {self.device}")
    
    def _setup_device(self) -> Any:
        """Setup device for training."""
        if self.use_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        
        # Wrap model for distributed training
        if self.use_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _setup_data_loaders(self) -> Any:
        """Setup data loaders with distributed sampling if needed."""
        if self.use_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            ) if self.val_dataset else None
        else:
            train_sampler = None
            val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=True,
                drop_last=False
            )
        else:
            self.val_loader = None
    
    def _setup_optimizer(self) -> Any:
        """Setup optimizer with advanced configurations."""
        optimizer_name = self.config.get("optimizer", "adamw")
        
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        elif optimizer_name == "lion":
            # Lion optimizer (if available)
            try:
                self.optimizer = Lion(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Any:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "cosine")
        warmup_steps = self.config.get("warmup_steps", 1000)
        total_steps = len(self.train_loader) * self.num_epochs
        
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_name == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif scheduler_name == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif scheduler_name == "one_cycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos"
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        if warmup_steps > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        else:
            self.warmup_scheduler = None
    
    def _setup_logging(self) -> Any:
        """Setup logging and monitoring."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TensorBoard
        if not self.use_distributed or self.local_rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "logs"))
        else:
            self.writer = None
        
        # Weights & Biases
        if self.config.get("use_wandb", False) and (not self.use_distributed or self.local_rank == 0):
            wandb.init(
                project=self.config.get("wandb_project", "advanced-training"),
                config=self.config
            )
        else:
            wandb.init(mode="disabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        if self.use_distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            disable=self.use_distributed and self.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            loss = self._training_step(batch)
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.warmup_scheduler and self.global_step < self.config.get("warmup_steps", 1000):
                    self.warmup_scheduler.step()
                elif self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                        self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                self._log_metrics({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'global_step': self.global_step
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=self.use_distributed and self.local_rank != 0):
                batch = self._move_batch_to_device(batch)
                loss = self._validation_step(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step."""
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        return loss
    
    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single validation step."""
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        return loss
    
    def _move_batch_to_device(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:
            return batch
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard and W&B."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint."""
        if self.use_distributed and self.local_rank != 0:
            return
        
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt"
        
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.use_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.use_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self) -> Any:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log epoch metrics
            if not self.use_distributed or self.local_rank == 0:
                logger.info(f"Epoch {epoch + 1}: {metrics}")
                
                # Save best model
                if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint("best_model.pt")
                
                # Save checkpoint periodically
                if (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_checkpoint()
        
        logger.info("Training completed!")
        
        # Save final model
        self.save_checkpoint("final_model.pt")


class MixedPrecisionTrainer(AdvancedTrainer):
    """
    Specialized trainer with advanced mixed precision features.
    """
    
    def __init__(self, *args, **kwargs) -> Any:
        kwargs['config'] = kwargs.get('config', {})
        kwargs['config']['use_mixed_precision'] = True
        super().__init__(*args, **kwargs)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training step with advanced mixed precision."""
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Add gradient scaling for stability
            if self.scaler:
                loss = loss * self.scaler.get_scale()
        
        return loss


class DistributedTrainer(AdvancedTrainer):
    """
    Specialized trainer for distributed training.
    """
    
    def __init__(self, *args, **kwargs) -> Any:
        kwargs['config'] = kwargs.get('config', {})
        kwargs['config']['use_distributed'] = True
        super().__init__(*args, **kwargs)
    
    def _setup_distributed(self) -> Any:
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    def train(self) -> Any:
        """Distributed training loop."""
        self._setup_distributed()
        super().train()
        
        if dist.is_initialized():
            dist.destroy_process_group()


class CustomLossFunctions:
    """
    Collection of custom loss functions for advanced training.
    """
    
    @staticmethod
    def focal_loss(predictions: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            alpha: Weighting factor
            gamma: Focusing parameter
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def label_smoothing_loss(predictions: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        """
        Label smoothing loss for regularization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            smoothing: Smoothing factor
            
        Returns:
            Label smoothed loss
        """
        num_classes = predictions.size(-1)
        targets_one_hot = torch.zeros_like(predictions).scatter_(
            1, targets.unsqueeze(1), 1
        )
        targets_smooth = targets_one_hot * (1 - smoothing) + smoothing / num_classes
        return F.cross_entropy(predictions, targets_smooth)
    
    @staticmethod
    def dice_loss(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Dice loss for segmentation tasks.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            smooth: Smoothing factor
            
        Returns:
            Dice loss
        """
        predictions = torch.sigmoid(predictions)
        intersection = (predictions * targets).sum()
        dice = (2 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        return 1 - dice
    
    @staticmethod
    def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        Contrastive loss for representation learning.
        
        Args:
            embeddings: Feature embeddings
            labels: Class labels
            temperature: Temperature parameter
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create labels for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-contrast cases
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))
        
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob.mean()
        
        return loss


class TrainingUtils:
    """
    Utility functions for advanced training.
    """
    
    @staticmethod
    def get_learning_rate(optimizer: optim.Optimizer) -> float:
        """Get current learning rate."""
        return optimizer.param_groups[0]['lr']
    
    @staticmethod
    def set_learning_rate(optimizer: optim.Optimizer, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]):
        """Freeze specific layers of the model."""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str]):
        """Unfreeze specific layers of the model."""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb 