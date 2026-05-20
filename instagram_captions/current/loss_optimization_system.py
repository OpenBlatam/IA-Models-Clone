"""
Loss Functions and Optimization Algorithms System

This module provides comprehensive loss functions and optimization algorithms
for deep learning models. It includes:

1. Various loss functions (classification, regression, custom losses)
2. Optimization algorithms (SGD, Adam, RMSprop, etc.)
3. Learning rate schedulers and adaptive methods
4. Loss function analysis and debugging tools
5. Integration with custom model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings


class LossFunctions:
    """Comprehensive collection of loss functions for different tasks."""
    
    @staticmethod
    def cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                          weight: Optional[torch.Tensor] = None,
                          label_smoothing: float = 0.0) -> torch.Tensor:
        """Cross-entropy loss for classification tasks."""
        return F.cross_entropy(predictions, targets, weight=weight, 
                              label_smoothing=label_smoothing)
    
    @staticmethod
    def focal_loss(predictions: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0,
                   reduction: str = 'mean') -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    @staticmethod
    def dice_loss(predictions: torch.Tensor, targets: torch.Tensor,
                  smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss for segmentation tasks."""
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        return 1 - dice
    
    @staticmethod
    def huber_loss(predictions: torch.Tensor, targets: torch.Tensor,
                   delta: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
        """Huber loss for robust regression."""
        return F.huber_loss(predictions, targets, delta=delta, reduction=reduction)
    
    @staticmethod
    def smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor,
                       beta: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
        """Smooth L1 loss for regression tasks."""
        return F.smooth_l1_loss(predictions, targets, beta=beta, reduction=reduction)
    
    @staticmethod
    def kl_divergence_loss(predictions: torch.Tensor, targets: torch.Tensor,
                          reduction: str = 'mean') -> torch.Tensor:
        """KL divergence loss for distribution matching."""
        return F.kl_div(F.log_softmax(predictions, dim=1), 
                       F.softmax(targets, dim=1), reduction=reduction)
    
    @staticmethod
    def cosine_embedding_loss(predictions: torch.Tensor, targets: torch.Tensor,
                             margin: float = 0.0, reduction: str = 'mean') -> torch.Tensor:
        """Cosine embedding loss for similarity learning."""
        return F.cosine_embedding_loss(predictions, targets, 
                                      torch.ones(predictions.size(0)), 
                                      margin=margin, reduction=reduction)
    
    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                     margin: float = 1.0, p: int = 2) -> torch.Tensor:
        """Triplet loss for metric learning."""
        pos_dist = torch.norm(anchor - positive, p=p, dim=1)
        neg_dist = torch.norm(anchor - negative, p=p, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss.mean()
    
    @staticmethod
    def contrastive_loss(predictions: torch.Tensor, targets: torch.Tensor,
                         margin: float = 1.0) -> torch.Tensor:
        """Contrastive loss for similarity learning."""
        dist = torch.norm(predictions, p=2, dim=1, keepdim=True)
        dist = torch.cdist(predictions, predictions)
        
        # Create mask for positive pairs
        mask = targets.unsqueeze(0) == targets.unsqueeze(1)
        
        # Positive and negative distances
        pos_dist = dist[mask]
        neg_dist = dist[~mask]
        
        # Contrastive loss
        pos_loss = pos_dist.pow(2)
        neg_loss = torch.clamp(margin - neg_dist, min=0.0).pow(2)
        
        return (pos_loss.mean() + neg_loss.mean()) / 2
    
    @staticmethod
    def custom_loss(predictions: torch.Tensor, targets: torch.Tensor,
                   loss_type: str = 'mse', **kwargs) -> torch.Tensor:
        """Custom loss function with configurable type."""
        if loss_type == 'mse':
            return F.mse_loss(predictions, targets)
        elif loss_type == 'mae':
            return F.l1_loss(predictions, targets)
        elif loss_type == 'cross_entropy':
            return LossFunctions.cross_entropy_loss(predictions, targets, **kwargs)
        elif loss_type == 'focal':
            return LossFunctions.focal_loss(predictions, targets, **kwargs)
        elif loss_type == 'dice':
            return LossFunctions.dice_loss(predictions, targets, **kwargs)
        elif loss_type == 'huber':
            return LossFunctions.huber_loss(predictions, targets, **kwargs)
        elif loss_type == 'smooth_l1':
            return LossFunctions.smooth_l1_loss(predictions, targets, **kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class Optimizers:
    """Comprehensive collection of optimization algorithms."""
    
    @staticmethod
    def sgd_optimizer(model: nn.Module, lr: float = 0.01, momentum: float = 0.0,
                     weight_decay: float = 0.0, nesterov: bool = False) -> optim.SGD:
        """Stochastic Gradient Descent optimizer."""
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=nesterov)
    
    @staticmethod
    def adam_optimizer(model: nn.Module, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                      eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False) -> optim.Adam:
        """Adam optimizer."""
        return optim.Adam(model.parameters(), lr=lr, betas=betas,
                         eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    
    @staticmethod
    def adamw_optimizer(model: nn.Module, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                       eps: float = 1e-8, weight_decay: float = 0.01) -> optim.AdamW:
        """AdamW optimizer with decoupled weight decay."""
        return optim.AdamW(model.parameters(), lr=lr, betas=betas,
                          eps=eps, weight_decay=weight_decay)
    
    @staticmethod
    def rmsprop_optimizer(model: nn.Module, lr: float = 0.01, alpha: float = 0.99,
                         eps: float = 1e-8, weight_decay: float = 0.0,
                         momentum: float = 0.0, centered: bool = False) -> optim.RMSprop:
        """RMSprop optimizer."""
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha,
                            eps=eps, weight_decay=weight_decay,
                            momentum=momentum, centered=centered)
    
    @staticmethod
    def adagrad_optimizer(model: nn.Module, lr: float = 0.01, lr_decay: float = 0.0,
                         weight_decay: float = 0.0, eps: float = 1e-10) -> optim.Adagrad:
        """Adagrad optimizer."""
        return optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay,
                            weight_decay=weight_decay, eps=eps)
    
    @staticmethod
    def adadelta_optimizer(model: nn.Module, lr: float = 1.0, rho: float = 0.9,
                          eps: float = 1e-6, weight_decay: float = 0.0) -> optim.Adadelta:
        """Adadelta optimizer."""
        return optim.Adadelta(model.parameters(), lr=lr, rho=rho,
                             eps=eps, weight_decay=weight_decay)
    
    @staticmethod
    def lion_optimizer(model: nn.Module, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99),
                      weight_decay: float = 0.0) -> Any:
        """Lion optimizer (requires lion-pytorch package)."""
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        except ImportError:
            warnings.warn("lion-pytorch not available, falling back to Adam")
            return Optimizers.adam_optimizer(model, lr=lr, betas=betas, weight_decay=weight_decay)
    
    @staticmethod
    def custom_optimizer(model: nn.Module, optimizer_type: str = 'adam', **kwargs) -> optim.Optimizer:
        """Custom optimizer with configurable type."""
        if optimizer_type == 'sgd':
            return Optimizers.sgd_optimizer(model, **kwargs)
        elif optimizer_type == 'adam':
            return Optimizers.adam_optimizer(model, **kwargs)
        elif optimizer_type == 'adamw':
            return Optimizers.adamw_optimizer(model, **kwargs)
        elif optimizer_type == 'rmsprop':
            return Optimizers.rmsprop_optimizer(model, **kwargs)
        elif optimizer_type == 'adagrad':
            return Optimizers.adagrad_optimizer(model, **kwargs)
        elif optimizer_type == 'adadelta':
            return Optimizers.adadelta_optimizer(model, **kwargs)
        elif optimizer_type == 'lion':
            return Optimizers.lion_optimizer(model, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class LearningRateSchedulers:
    """Comprehensive collection of learning rate schedulers."""
    
    @staticmethod
    def step_scheduler(optimizer: optim.Optimizer, step_size: int, gamma: float = 0.1) -> optim.lr_scheduler.StepLR:
        """Step learning rate scheduler."""
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    @staticmethod
    def exponential_scheduler(optimizer: optim.Optimizer, gamma: float) -> optim.lr_scheduler.ExponentialLR:
        """Exponential learning rate scheduler."""
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    @staticmethod
    def cosine_scheduler(optimizer: optim.Optimizer, T_max: int, eta_min: float = 0.0) -> optim.lr_scheduler.CosineAnnealingLR:
        """Cosine annealing learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    @staticmethod
    def cosine_warm_restart_scheduler(optimizer: optim.Optimizer, T_0: int, T_mult: int = 1,
                                    eta_min: float = 0.0) -> optim.lr_scheduler.CosineAnnealingWarmRestarts:
        """Cosine annealing with warm restarts scheduler."""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    
    @staticmethod
    def reduce_lr_on_plateau_scheduler(optimizer: optim.Optimizer, mode: str = 'min', factor: float = 0.1,
                                     patience: int = 10, verbose: bool = False,
                                     threshold: float = 1e-4, threshold_mode: str = 'rel',
                                     cooldown: int = 0, min_lr: float = 0.0) -> optim.lr_scheduler.ReduceLROnPlateau:
        """Reduce learning rate on plateau scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                                   patience=patience, verbose=verbose,
                                                   threshold=threshold, threshold_mode=threshold_mode,
                                                   cooldown=cooldown, min_lr=min_lr)
    
    @staticmethod
    def one_cycle_scheduler(optimizer: optim.Optimizer, max_lr: float, epochs: int,
                           steps_per_epoch: int, pct_start: float = 0.3,
                           anneal_strategy: str = 'cos', cycle_momentum: bool = True,
                           base_momentum: float = 0.85, max_momentum: float = 0.95,
                           div_factor: float = 25.0, final_div_factor: float = 1e4) -> optim.lr_scheduler.OneCycleLR:
        """One cycle learning rate scheduler."""
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs,
                                            steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                                            anneal_strategy=anneal_strategy, cycle_momentum=cycle_momentum,
                                            base_momentum=base_momentum, max_momentum=max_momentum,
                                            div_factor=div_factor, final_div_factor=final_div_factor)
    
    @staticmethod
    def custom_scheduler(optimizer: optim.Optimizer, scheduler_type: str = 'step', **kwargs) -> optim.lr_scheduler._LRScheduler:
        """Custom scheduler with configurable type."""
        if scheduler_type == 'step':
            return LearningRateSchedulers.step_scheduler(optimizer, **kwargs)
        elif scheduler_type == 'exponential':
            return LearningRateSchedulers.exponential_scheduler(optimizer, **kwargs)
        elif scheduler_type == 'cosine':
            return LearningRateSchedulers.cosine_scheduler(optimizer, **kwargs)
        elif scheduler_type == 'cosine_warm_restart':
            return LearningRateSchedulers.cosine_warm_restart_scheduler(optimizer, **kwargs)
        elif scheduler_type == 'reduce_lr_on_plateau':
            return LearningRateSchedulers.reduce_lr_on_plateau_scheduler(optimizer, **kwargs)
        elif scheduler_type == 'one_cycle':
            return LearningRateSchedulers.one_cycle_scheduler(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class LossOptimizationAnalyzer:
    """Analyze and debug loss functions and optimization algorithms."""
    
    @staticmethod
    def analyze_loss_landscape(model: nn.Module, data: torch.Tensor, targets: torch.Tensor,
                             loss_fn: Callable, num_points: int = 100) -> Dict[str, Any]:
        """Analyze the loss landscape around the current model parameters."""
        model.eval()
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Get current loss
        with torch.no_grad():
            predictions = model(data)
            current_loss = loss_fn(predictions, targets).item()
        
        # Sample random directions
        losses = []
        directions = []
        
        for _ in range(num_points):
            # Generate random direction
            direction = {}
            for name, param in model.named_parameters():
                direction[name] = torch.randn_like(param) * 0.01
            
            # Apply direction and compute loss
            for name, param in model.named_parameters():
                param.data = original_params[name] + direction[name]
            
            with torch.no_grad():
                predictions = model(data)
                loss = loss_fn(predictions, targets).item()
                losses.append(loss)
                directions.append(direction)
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data = original_params[name]
        
        return {
            'current_loss': current_loss,
            'losses': losses,
            'directions': directions,
            'loss_std': np.std(losses),
            'loss_range': (min(losses), max(losses)),
            'loss_variance': np.var(losses)
        }
    
    @staticmethod
    def analyze_gradient_flow(model: nn.Module, loss_fn: Callable,
                            data: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient flow through the model."""
        model.train()
        model.zero_grad()
        
        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        gradient_stats = {}
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                gradient_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item()
                }
                
                total_norm += grad_norm ** 2
        
        total_norm = math.sqrt(total_norm)
        
        return {
            'gradient_stats': gradient_stats,
            'total_gradient_norm': total_norm,
            'loss_value': loss.item()
        }
    
    @staticmethod
    def check_optimization_convergence(loss_history: List[float], 
                                     patience: int = 10,
                                     tolerance: float = 1e-6) -> Dict[str, Any]:
        """Check if optimization has converged."""
        if len(loss_history) < patience:
            return {'converged': False, 'reason': 'Not enough iterations'}
        
        recent_losses = loss_history[-patience:]
        loss_variance = np.var(recent_losses)
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        
        converged = loss_variance < tolerance and loss_change < tolerance
        
        return {
            'converged': converged,
            'loss_variance': loss_variance,
            'loss_change': loss_change,
            'tolerance': tolerance,
            'recent_losses': recent_losses
        }


class CustomLossOptimizationSchemes:
    """Custom loss and optimization schemes for specific tasks."""
    
    @staticmethod
    def classification_scheme(model: nn.Module, num_classes: int,
                            class_weights: Optional[torch.Tensor] = None,
                            optimizer_type: str = 'adam',
                            scheduler_type: str = 'step') -> Tuple[Callable, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Complete scheme for classification tasks."""
        # Loss function
        if class_weights is not None:
            loss_fn = lambda pred, target: LossFunctions.cross_entropy_loss(pred, target, weight=class_weights)
        else:
            loss_fn = LossFunctions.cross_entropy_loss
        
        # Optimizer
        optimizer = Optimizers.custom_optimizer(model, optimizer_type, lr=0.001)
        
        # Scheduler
        scheduler = LearningRateSchedulers.custom_scheduler(optimizer, scheduler_type, step_size=30, gamma=0.1)
        
        return loss_fn, optimizer, scheduler
    
    @staticmethod
    def regression_scheme(model: nn.Module, loss_type: str = 'mse',
                         optimizer_type: str = 'adam',
                         scheduler_type: str = 'step') -> Tuple[Callable, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Complete scheme for regression tasks."""
        # Loss function
        loss_fn = lambda pred, target: LossFunctions.custom_loss(pred, target, loss_type=loss_type)
        
        # Optimizer
        optimizer = Optimizers.custom_optimizer(model, optimizer_type, lr=0.001)
        
        # Scheduler
        scheduler = LearningRateSchedulers.custom_scheduler(optimizer, scheduler_type, step_size=30, gamma=0.1)
        
        return loss_fn, optimizer, scheduler
    
    @staticmethod
    def segmentation_scheme(model: nn.Module, num_classes: int,
                           loss_weights: Dict[str, float] = None,
                           optimizer_type: str = 'adam',
                           scheduler_type: str = 'cosine') -> Tuple[Callable, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Complete scheme for segmentation tasks."""
        # Default loss weights
        if loss_weights is None:
            loss_weights = {'dice': 0.5, 'cross_entropy': 0.5}
        
        # Combined loss function
        def combined_loss(predictions, targets):
            total_loss = 0.0
            if 'dice' in loss_weights:
                total_loss += loss_weights['dice'] * LossFunctions.dice_loss(predictions, targets)
            if 'cross_entropy' in loss_weights:
                total_loss += loss_weights['cross_entropy'] * LossFunctions.cross_entropy_loss(predictions, targets)
            return total_loss
        
        # Optimizer
        optimizer = Optimizers.custom_optimizer(model, optimizer_type, lr=0.001)
        
        # Scheduler
        scheduler = LearningRateSchedulers.custom_scheduler(optimizer, scheduler_type, T_max=100)
        
        return combined_loss, optimizer, scheduler
    
    @staticmethod
    def metric_learning_scheme(model: nn.Module, margin: float = 1.0,
                              optimizer_type: str = 'adam',
                              scheduler_type: str = 'step') -> Tuple[Callable, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Complete scheme for metric learning tasks."""
        # Loss function (contrastive loss)
        loss_fn = lambda pred, target: LossFunctions.contrastive_loss(pred, target, margin=margin)
        
        # Optimizer
        optimizer = Optimizers.custom_optimizer(model, optimizer_type, lr=0.001)
        
        # Scheduler
        scheduler = LearningRateSchedulers.custom_scheduler(optimizer, scheduler_type, step_size=30, gamma=0.1)
        
        return loss_fn, optimizer, scheduler


def demonstrate_loss_optimization():
    """Demonstrate the loss functions and optimization system."""
    print("Loss Functions and Optimization System Demonstration")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    # Create sample data
    data = torch.randn(100, 10)
    targets = torch.randint(0, 5, (100,))
    
    print("1. Testing different loss functions...")
    
    # Test classification loss
    predictions = model(data)
    ce_loss = LossFunctions.cross_entropy_loss(predictions, targets)
    focal_loss = LossFunctions.focal_loss(predictions, targets)
    
    print(f"   Cross-entropy loss: {ce_loss.item():.4f}")
    print(f"   Focal loss: {focal_loss.item():.4f}")
    
    print("\n2. Testing different optimizers...")
    
    # Test optimizers
    sgd_opt = Optimizers.sgd_optimizer(model, lr=0.01)
    adam_opt = Optimizers.adam_optimizer(model, lr=0.001)
    
    print(f"   SGD optimizer created: {type(sgd_opt).__name__}")
    print(f"   Adam optimizer created: {type(adam_opt).__name__}")
    
    print("\n3. Testing learning rate schedulers...")
    
    # Test schedulers
    step_scheduler = LearningRateSchedulers.step_scheduler(adam_opt, step_size=10)
    cosine_scheduler = LearningRateSchedulers.cosine_scheduler(adam_opt, T_max=100)
    
    print(f"   Step scheduler created: {type(step_scheduler).__name__}")
    print(f"   Cosine scheduler created: {type(cosine_scheduler).__name__}")
    
    print("\n4. Testing task-specific schemes...")
    
    # Test classification scheme
    loss_fn, optimizer, scheduler = CustomLossOptimizationSchemes.classification_scheme(
        model, num_classes=5
    )
    
    print(f"   Classification scheme created:")
    print(f"     Loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else 'Custom'}")
    print(f"     Optimizer: {type(optimizer).__name__}")
    print(f"     Scheduler: {type(scheduler).__name__}")
    
    print("\nLoss functions and optimization demonstration completed!")


if __name__ == "__main__":
    demonstrate_loss_optimization()


