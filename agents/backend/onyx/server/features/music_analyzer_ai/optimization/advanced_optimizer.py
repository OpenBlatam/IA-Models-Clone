"""
Advanced Optimization Techniques
Gradient accumulation, learning rate finder, and more
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GradientAccumulator:
    """
    Gradient accumulation for large batch training
    """
    
    def __init__(self, accumulation_steps: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def reset(self):
        """Reset accumulator"""
        self.current_step = 0


class LearningRateFinder:
    """
    Find optimal learning rate using learning rate range test
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.learning_rates: List[float] = []
        self.losses: List[float] = []
    
    def find_lr(
        self,
        train_loader,
        init_lr: float = 1e-8,
        final_lr: float = 10.0,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Find optimal learning rate"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        # Save initial state
        initial_state = {
            "model": self.model.state_dict().copy(),
            "optimizer": self.optimizer.state_dict().copy()
        }
        
        # Exponential range
        lr_mult = (final_lr / init_lr) ** (1.0 / num_iterations)
        current_lr = init_lr
        
        self.learning_rates = []
        self.losses = []
        
        iteration = 0
        for batch in train_loader:
            if iteration >= num_iterations:
                break
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Forward pass
            inputs = batch["features"].to(self.device)
            labels = batch["label"].squeeze().to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record
            self.learning_rates.append(current_lr)
            self.losses.append(loss.item())
            
            # Update LR
            current_lr *= lr_mult
            iteration += 1
        
        # Restore initial state
        self.model.load_state_dict(initial_state["model"])
        self.optimizer.load_state_dict(initial_state["optimizer"])
        
        # Find optimal LR (steepest descent)
        if len(self.losses) > 10:
            # Find point of steepest descent
            gradients = np.gradient(self.losses)
            min_grad_idx = np.argmin(gradients)
            optimal_lr = self.learning_rates[min_grad_idx]
        else:
            optimal_lr = self.learning_rates[len(self.learning_rates) // 2]
        
        return {
            "optimal_lr": optimal_lr,
            "learning_rates": self.learning_rates,
            "losses": self.losses,
            "min_loss": min(self.losses),
            "max_loss": max(self.losses)
        }


class OptimizerScheduler:
    """
    Advanced optimizer scheduling
    """
    
    @staticmethod
    def create_warmup_scheduler(
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        target_lr: float
    ):
        """Create warmup scheduler"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup
                return base_lr + (target_lr - base_lr) * (step / warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return target_lr * (0.5 * (1 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def create_one_cycle_scheduler(
        optimizer: optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3
    ):
        """Create one-cycle learning rate scheduler"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        # Simplified one-cycle
        def lr_lambda(step):
            if step < total_steps * pct_start:
                # Increasing phase
                progress = step / (total_steps * pct_start)
                return max_lr * progress
            else:
                # Decreasing phase
                progress = (step - total_steps * pct_start) / (total_steps * (1 - pct_start))
                return max_lr * (1 - progress)
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

