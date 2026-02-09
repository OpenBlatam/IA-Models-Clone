"""
Advanced Optimizers - Specialized Optimizers
============================================

Advanced optimizer configurations and specialized optimizers:
- Learning rate finder
- Optimizer state management
- Custom optimizer configurations
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

logger = logging.getLogger(__name__)


class LearningRateFinder:
    """
    Learning rate finder using exponential range test.
    
    Implements the learning rate range test to find optimal learning rate.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        """
        Initialize learning rate finder.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.losses = []
        self.lrs = []
    
    def find(
        self,
        train_loader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05
    ) -> Tuple[List[float], List[float], float]:
        """
        Find optimal learning rate.
        
        Args:
            train_loader: Training DataLoader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor
            
        Returns:
            Tuple of (learning_rates, losses, optimal_lr)
        """
        self.model.train()
        
        # Save initial state
        initial_state = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': self.optimizer.state_dict()
        }
        
        # Exponential range
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        lr = start_lr
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.losses = []
        self.lrs = []
        
        iterator = iter(train_loader)
        best_loss = float('inf')
        optimal_lr = start_lr
        
        try:
            for i in range(num_iter):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (tuple, list)):
                    batch = tuple(v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for v in batch)
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                    if 'labels' in batch:
                        loss = self.criterion(outputs, batch['labels'])
                    else:
                        loss = self.criterion(outputs, batch.get('target'))
                elif isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = outputs.mean()
                else:
                    outputs = self.model(batch)
                    loss = outputs.mean()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Smooth loss
                if len(self.losses) == 0:
                    smooth_loss = loss.item()
                else:
                    smooth_loss = smooth_f * loss.item() + (1 - smooth_f) * self.losses[-1]
                
                self.losses.append(smooth_loss)
                self.lrs.append(lr)
                
                # Track best loss
                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                    optimal_lr = lr
                
                # Update learning rate
                lr *= lr_mult
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Stop if loss explodes
                if smooth_loss > 4 * best_loss:
                    logger.warning(f"Loss exploded at lr={lr:.2e}, stopping")
                    break
        
        finally:
            # Restore initial state
            self.model.load_state_dict(initial_state['model'])
            self.optimizer.load_state_dict(initial_state['optimizer'])
        
        logger.info(f"Optimal learning rate: {optimal_lr:.2e}")
        return self.lrs, self.losses, optimal_lr


def create_optimizer_with_warmup(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    **kwargs
) -> Tuple[optim.Optimizer, Any]:
    """
    Create optimizer with warmup scheduler.
    
    Args:
        model: PyTorch model
        optimizer_type: Optimizer type
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        **kwargs: Additional optimizer arguments
        
    Returns:
        Tuple of (optimizer, warmup_scheduler)
    """
    from ..training.optimizers import create_optimizer, create_scheduler
    
    optimizer = create_optimizer(
        model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        **kwargs
    )
    
    # Create warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)
    )
    
    return optimizer, warmup_scheduler



