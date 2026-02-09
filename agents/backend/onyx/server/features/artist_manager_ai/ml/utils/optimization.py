"""
Optimization Utilities
======================

Utilities for model optimization and compression.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Model optimization utilities."""
    
    @staticmethod
    def prune_model(
        model: nn.Module,
        pruning_ratio: float = 0.2,
        method: str = "magnitude"
    ) -> nn.Module:
        """
        Prune model weights.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of weights to prune
            method: Pruning method ("magnitude", "random")
        
        Returns:
            Pruned model
        """
        if method == "magnitude":
            # Magnitude-based pruning
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    weights = module.weight.data
                    threshold = torch.quantile(
                        torch.abs(weights),
                        pruning_ratio
                    )
                    mask = torch.abs(weights) > threshold
                    module.weight.data *= mask.float()
        
        return model
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        dtype: torch.dtype = torch.int8
    ) -> nn.Module:
        """
        Quantize model.
        
        Args:
            model: Model to quantize
            dtype: Target dtype
        
        Returns:
            Quantized model
        """
        try:
            # Use PyTorch quantization
            model_quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=dtype
            )
            return model_quantized
        except Exception as e:
            logger.warning(f"Quantization failed: {str(e)}")
            return model
    
    @staticmethod
    def fuse_bn(model: nn.Module) -> nn.Module:
        """
        Fuse batch normalization layers.
        
        Args:
            model: Model to fuse
        
        Returns:
            Fused model
        """
        # This is a simplified version
        # In practice, use torch.quantization.fuse_modules
        return model


class GradientAccumulator:
    """Gradient accumulation utility."""
    
    def __init__(self, accumulation_steps: int = 4):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if should perform optimizer step."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False


class LearningRateFinder:
    """Learning rate finder utility."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        """
        Initialize LR finder.
        
        Args:
            model: Model
            optimizer: Optimizer
            criterion: Loss function
            device: Device
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lrs = []
        self.losses = []
    
    def find_lr(
        self,
        train_loader: torch.utils.data.DataLoader,
        init_lr: float = 1e-8,
        final_lr: float = 10.0,
        num_iterations: int = 100
    ) -> tuple:
        """
        Find optimal learning rate.
        
        Args:
            train_loader: Training dataloader
            init_lr: Initial learning rate
            final_lr: Final learning rate
            num_iterations: Number of iterations
        
        Returns:
            (learning_rates, losses) tuple
        """
        # Exponential range
        lr_mult = (final_lr / init_lr) ** (1.0 / num_iterations)
        
        lr = init_lr
        self.model.train()
        
        for i, (features, targets) in enumerate(train_loader):
            if i >= num_iterations:
                break
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record
            self.lrs.append(lr)
            self.losses.append(loss.item())
            
            # Update LR
            lr *= lr_mult
        
        return self.lrs, self.losses




