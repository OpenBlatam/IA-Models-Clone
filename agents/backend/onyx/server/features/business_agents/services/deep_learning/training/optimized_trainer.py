"""
Optimized Training Manager - Performance Optimized
==================================================

Optimized version of TrainingManager with additional performance improvements.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, List, Any
import logging
from tqdm import tqdm

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from .trainer import TrainingManager
from ..config.config_loader import TrainingConfig
from ..utils.optimization import ModelOptimizer, MemoryOptimizer


class OptimizedTrainingManager(TrainingManager):
    """
    Optimized training manager with additional performance improvements.
    
    Additional optimizations:
    - Model compilation (torch.compile)
    - Optimized data loading
    - Memory management
    - Progress bars with tqdm
    - Better GPU utilization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        experiment_tracker: Optional[Any] = None,
        compile_model: bool = True,
        optimize_memory: bool = True
    ):
        """
        Initialize optimized training manager.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Target device
            criterion: Loss function
            experiment_tracker: TensorBoard writer or wandb
            compile_model: Whether to compile model with torch.compile
            optimize_memory: Whether to optimize memory usage
        """
        super().__init__(
            model, train_loader, val_loader, config, device, criterion, experiment_tracker
        )
        
        # Additional optimizations
        self.compile_model = compile_model
        self.optimize_memory = optimize_memory
        
        # Compile model if requested
        if compile_model and hasattr(torch, 'compile'):
            self.model = ModelOptimizer.compile_model(
                self.model,
                mode="reduce-overhead"
            )
            logger.info("✅ Model compiled for training")
        
        # Enable TF32
        ModelOptimizer.enable_tf32(self.model)
        
        # Memory optimization
        if optimize_memory:
            MemoryOptimizer.set_memory_fraction(0.95)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            ncols=100
        )
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Gradient accumulation
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast(dtype=getattr(torch, self.config.mixed_precision_dtype)):
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.config.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("❌ NaN or Inf loss detected!")
                raise ValueError("Training diverged: NaN/Inf loss")
            
            # Memory cleanup every N batches
            if self.optimize_memory and (batch_idx + 1) % 100 == 0:
                MemoryOptimizer.clear_cache()
        
        pbar.close()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate model with optimizations."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast(dtype=getattr(torch, self.config.mixed_precision_dtype)):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        pbar.close()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}



