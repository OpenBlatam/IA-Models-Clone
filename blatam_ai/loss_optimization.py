"""
Blatam AI - Advanced Loss Functions and Optimization Engine v6.0.0
Ultra-optimized PyTorch-based loss functions and optimization algorithms
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ADVANCED LOSS FUNCTIONS
# ============================================================================

class AdvancedLossFunction(nn.Module, ABC):
    """Base class for advanced loss functions."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        pass
        
    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction method to loss tensor."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

class FocalLoss(AdvancedLossFunction):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_loss = -self.alpha * (1 - target_probs) ** self.gamma * torch.log(target_probs)
        
        return self._apply_reduction(focal_loss)

class DiceLoss(AdvancedLossFunction):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__(reduction)
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss."""
        # Apply sigmoid for binary segmentation
        if predictions.size(1) == 1:
            predictions = torch.sigmoid(predictions)
        else:
            predictions = F.softmax(predictions, dim=1)
            
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Compute dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return dice loss
        return self._apply_reduction(1 - dice)

class IoULoss(AdvancedLossFunction):
    """Intersection over Union Loss for object detection."""
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__(reduction)
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss."""
        # Apply sigmoid for binary detection
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        # Compute IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU loss
        return self._apply_reduction(1 - iou)

class TripletLoss(AdvancedLossFunction):
    """Triplet Loss for metric learning."""
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss."""
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Compute triplet loss
        triplet_loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return self._apply_reduction(triplet_loss)

class ContrastiveLoss(AdvancedLossFunction):
    """Contrastive Loss for similarity learning."""
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.margin = margin
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss."""
        # Compute distance between pairs
        dist = F.pairwise_distance(x1, x2, p=2)
        
        # Compute loss for similar pairs (y=1) and dissimilar pairs (y=0)
        loss_similar = y * dist.pow(2)
        loss_dissimilar = (1 - y) * torch.clamp(self.margin - dist, min=0.0).pow(2)
        
        # Total loss
        total_loss = loss_similar + loss_dissimilar
        
        return self._apply_reduction(total_loss)

class KLDivergenceLoss(AdvancedLossFunction):
    """Kullback-Leibler Divergence Loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        # Apply log softmax to predictions
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Apply softmax to targets
        target_probs = F.softmax(targets, dim=1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(log_probs, target_probs, reduction='none')
        
        return self._apply_reduction(kl_loss)

class HuberLoss(AdvancedLossFunction):
    """Huber Loss for robust regression."""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.delta = delta
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss."""
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Quadratic loss for small errors, linear loss for large errors
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        
        huber_loss = 0.5 * quadratic.pow(2) + self.delta * linear
        
        return self._apply_reduction(huber_loss)

# ============================================================================
# ADVANCED OPTIMIZATION ALGORITHMS
# ============================================================================

class AdvancedOptimizer(ABC):
    """Base class for advanced optimizers."""
    
    def __init__(self, params: List[torch.Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr
        self.step_count = 0
        
    @abstractmethod
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step."""
        pass
        
    def zero_grad(self):
        """Zero gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LionOptimizer(AdvancedOptimizer):
    """Lion optimizer implementation."""
    
    def __init__(self, params: List[torch.Tensor], lr: float = 0.0001, 
                 betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers
        self.momentum = [torch.zeros_like(p) for p in params]
        
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform Lion optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            momentum = self.momentum[i]
            
            # Update momentum
            momentum.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # Update parameter
            update = momentum.sign() * self.lr
            if self.weight_decay > 0:
                update.add_(param.data, alpha=self.weight_decay)
                
            param.data.sub_(update)
            
        self.step_count += 1
        return loss

class AdaBeliefOptimizer(AdvancedOptimizer):
    """AdaBelief optimizer implementation."""
    
    def __init__(self, params: List[torch.Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-16,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Initialize state
        self.state = {}
        for param in params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param),
                'exp_avg_sq': torch.zeros_like(param),
                'exp_avg_sq_hat': torch.zeros_like(param)
            }
            
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform AdaBelief optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for param in params:
            if param.grad is None:
                continue
                
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            step = state['step']
            
            # Update exponential moving averages
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).add_((grad - exp_avg).pow(2), alpha=1 - self.beta2)
            
            # Compute bias correction
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            
            # Update parameter
            step_size = self.lr / bias_correction1
            bias_correction2_sqrt = bias_correction2.sqrt()
            
            # AdaBelief update
            update = exp_avg / (exp_avg_sq.sqrt() / bias_correction2_sqrt + self.eps)
            
            if self.weight_decay > 0:
                update.add_(param.data, alpha=self.weight_decay)
                
            param.data.sub_(step_size * update)
            
        self.step_count += 1
        return loss

class RAdamOptimizer(AdvancedOptimizer):
    """RAdam optimizer implementation."""
    
    def __init__(self, params: List[torch.Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        self.state = {}
        for param in params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param),
                'exp_avg_sq': torch.zeros_like(param)
            }
            
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform RAdam optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for param in params:
            if param.grad is None:
                continue
                
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            step = state['step']
            
            # Update exponential moving averages
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).add_(grad.pow(2), alpha=1 - self.beta2)
            
            # Compute bias correction
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            
            # RAdam update
            if step > 4:
                # Rectified update
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                update = exp_avg_hat / (exp_avg_sq_hat.sqrt() + self.eps)
            else:
                # Standard update
                update = exp_avg / bias_correction1
                
            if self.weight_decay > 0:
                update.add_(param.data, alpha=self.weight_decay)
                
            param.data.sub_(self.lr * update)
            
        self.step_count += 1
        return loss

# ============================================================================
# ADVANCED LEARNING RATE SCHEDULERS
# ============================================================================

class AdvancedScheduler(ABC):
    """Base class for advanced learning rate schedulers."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        
    @abstractmethod
    def step(self, metrics: Optional[float] = None):
        """Update learning rate."""
        pass
        
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(AdvancedScheduler):
    """Cosine annealing with warm restarts."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, metrics: Optional[float] = None):
        """Update learning rate."""
        if self.T_cur >= self.T_0:
            self.T_0 *= self.T_mult
            self.T_cur = 0
            
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                         (1 + np.cos(np.pi * self.T_cur / self.T_0)) / 2
                         
        self.T_cur += 1

class OneCycleLR(AdvancedScheduler):
    """One Cycle Learning Rate scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, max_lr: float, epochs: int,
                 steps_per_epoch: int, pct_start: float = 0.3, anneal_strategy: str = 'cos'):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
        self.total_steps = epochs * steps_per_epoch
        self.step_size_up = int(self.total_steps * pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
        
    def step(self, metrics: Optional[float] = None):
        """Update learning rate."""
        for group in self.optimizer.param_groups:
            if self.step_count <= self.step_size_up:
                # Warm-up phase
                lr = self._annealing_cos(self.base_lr, self.max_lr, 
                                       self.step_count / self.step_size_up)
            else:
                # Annealing phase
                lr = self._annealing_cos(self.max_lr, self.base_lr / 100,
                                       (self.step_count - self.step_size_up) / self.step_size_down)
                                       
            group['lr'] = lr
            
        self.step_count += 1
        
    def _annealing_cos(self, start: float, end: float, pct: float) -> float:
        """Cosine annealing between start and end values."""
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

class ReduceLROnPlateau(AdvancedScheduler):
    """Reduce learning rate when a metric has stopped improving."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, verbose: bool = False, threshold: float = 1e-4,
                 threshold_mode: str = 'rel', cooldown: int = 0, min_lr: float = 0.0):
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        
    def step(self, metrics: Optional[float] = None):
        """Update learning rate based on metrics."""
        if metrics is None:
            return
            
        if self.best is None:
            self.best = metrics
        elif self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best - self.threshold
        else:
            return current > best + self.threshold
            
    def _reduce_lr(self):
        """Reduce learning rate."""
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            group['lr'] = new_lr
            
            if self.verbose:
                logger.info(f'Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')

# ============================================================================
# LOSS FUNCTION FACTORY
# ============================================================================

class LossFunctionFactory:
    """Factory for creating loss functions."""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """Create loss function by type."""
        loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'bce': nn.BCELoss,
            'bce_with_logits': nn.BCEWithLogitsLoss,
            'focal': FocalLoss,
            'dice': DiceLoss,
            'iou': IoULoss,
            'triplet': TripletLoss,
            'contrastive': ContrastiveLoss,
            'kl_div': KLDivergenceLoss,
            'huber': HuberLoss
        }
        
        if loss_type not in loss_functions:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        return loss_functions[loss_type](**kwargs)

# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================

class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create_optimizer(optimizer_type: str, params: List[torch.Tensor], **kwargs) -> torch.optim.Optimizer:
        """Create optimizer by type."""
        optimizers = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'lion': LionOptimizer,
            'adabelief': AdaBeliefOptimizer,
            'radam': RAdamOptimizer
        }
        
        if optimizer_type not in optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        return optimizers[optimizer_type](params, **kwargs)

# ============================================================================
# SCHEDULER FACTORY
# ============================================================================

class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer, **kwargs) -> AdvancedScheduler:
        """Create scheduler by type."""
        schedulers = {
            'step': lambda opt, **kw: lr_scheduler.StepLR(opt, **kw),
            'exponential': lambda opt, **kw: lr_scheduler.ExponentialLR(opt, **kw),
            'cosine': lambda opt, **kw: lr_scheduler.CosineAnnealingLR(opt, **kw),
            'cosine_warm_restarts': lambda opt, **kw: lr_scheduler.CosineAnnealingWarmRestarts(opt, **kw),
            'one_cycle': lambda opt, **kw: OneCycleLR(opt, **kw),
            'reduce_on_plateau': lambda opt, **kw: ReduceLROnPlateau(opt, **kw),
            'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts
        }
        
        if scheduler_type not in schedulers:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        return schedulers[scheduler_type](optimizer, **kwargs)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    """Main examples for loss functions and optimization."""
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create loss function
    loss_fn = LossFunctionFactory.create_loss('focal', alpha=1.0, gamma=2.0)
    
    # Create optimizer
    optimizer = OptimizerFactory.create_optimizer('adamw', model.parameters(), lr=0.001)
    
    # Create scheduler
    scheduler = SchedulerFactory.create_scheduler('cosine', optimizer, T_max=100)
    
    # Example data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    # Training loop
    for epoch in range(10):
        # Forward pass
        predictions = model(x)
        
        # Compute loss
        loss = loss_fn(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_lr()[0]:.6f}")
            
    print("Loss and optimization engine ready!")

if __name__ == "__main__":
    main()

