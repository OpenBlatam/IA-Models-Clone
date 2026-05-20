"""
Learning Rate Finder
====================

Learning rate range test following best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class LearningRateFinder:
    """
    Learning rate finder using exponential range test.
    
    Based on: "Cyclical Learning Rates for Training Neural Networks"
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        """
        Initialize LR finder.
        
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
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self._logger = logger
    
    def find_lr(
        self,
        train_loader: torch.utils.data.DataLoader,
        init_lr: float = 1e-8,
        final_lr: float = 10.0,
        num_iterations: int = 100,
        smooth_factor: float = 0.98
    ) -> Tuple[List[float], List[float]]:
        """
        Find optimal learning rate.
        
        Args:
            train_loader: Training dataloader
            init_lr: Initial learning rate
            final_lr: Final learning rate
            num_iterations: Number of iterations
            smooth_factor: Smoothing factor for loss
        
        Returns:
            (learning_rates, losses) tuple
        """
        # Exponential range
        lr_mult = (final_lr / init_lr) ** (1.0 / num_iterations)
        
        lr = init_lr
        self.model.train()
        
        best_loss = float('inf')
        smoothed_loss = 0.0
        
        for i, (features, targets) in enumerate(train_loader):
            if i >= num_iterations:
                break
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Ensure targets have correct shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            # Forward
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Smooth loss
            smoothed_loss = smooth_factor * smoothed_loss + (1 - smooth_factor) * loss.item()
            
            # Record
            self.lrs.append(lr)
            self.losses.append(smoothed_loss)
            
            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss:
                self._logger.warning(f"Loss exploded at LR={lr:.2e}, stopping")
                break
            
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Update LR
            lr *= lr_mult
        
        return self.lrs, self.losses
    
    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Plot learning rate vs loss.
        
        Args:
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            self._logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Suggest optimal learning rate.
        
        Args:
            skip_start: Skip initial iterations
            skip_end: Skip final iterations
        
        Returns:
            Suggested learning rate
        """
        if len(self.losses) < skip_start + skip_end:
            return self.lrs[len(self.losses) // 2]
        
        # Find steepest descent point
        losses_to_check = self.losses[skip_start:-skip_end]
        lrs_to_check = self.lrs[skip_start:-skip_end]
        
        # Calculate gradients
        gradients = np.gradient(losses_to_check)
        
        # Find minimum gradient (steepest descent)
        min_grad_idx = np.argmin(gradients)
        
        return lrs_to_check[min_grad_idx]




