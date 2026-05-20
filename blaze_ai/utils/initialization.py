"""
Weight initialization, normalization, loss functions, and optimization utilities for deep learning models.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

class WeightInitializer:
    """Weight initialization strategies for neural networks."""
    
    @staticmethod
    def xavier_uniform(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Xavier/Glorot uniform initialization."""
        nn.init.xavier_uniform_(tensor, gain=gain)
    
    @staticmethod
    def xavier_normal(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Xavier/Glorot normal initialization."""
        nn.init.xavier_normal_(tensor, gain=gain)
    
    @staticmethod
    def kaiming_uniform(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> None:
        """Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def kaiming_normal(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> None:
        """Kaiming normal initialization."""
        nn.init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Orthogonal initialization."""
        nn.init.orthogonal_(tensor, gain=gain)
    
    @staticmethod
    def sparse(tensor: torch.Tensor, sparsity: float = 0.1, std: float = 0.01) -> None:
        """Sparse initialization."""
        nn.init.sparse_(tensor, sparsity=sparsity, std=std)
    
    @staticmethod
    def variance_scaling(tensor: torch.Tensor, scale: float = 1.0, mode: str = 'fan_in') -> None:
        """Variance scaling initialization."""
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        if mode == 'fan_in':
            n = fan_in
        elif mode == 'fan_out':
            n = fan_out
        else:
            n = (fan_in + fan_out) / 2
        
        std = math.sqrt(scale / n)
        with torch.no_grad():
            tensor.uniform_(-std, std)
    
    @staticmethod
    def delta_orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Delta orthogonal initialization for RNNs."""
        nn.init.delta_orthogonal_(tensor, gain=gain)

class NormalizationLayer(nn.Module):
    """Custom normalization layers."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_parameter('weight', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.weight * x_norm + self.bias

class LossFunctions:
    """Collection of loss functions for different tasks."""
    
    @staticmethod
    def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for classification tasks."""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss for segmentation tasks."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    @staticmethod
    def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Huber loss for regression tasks."""
        return F.huber_loss(pred, target, delta=delta)
    
    @staticmethod
    def cosine_embedding_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """Cosine embedding loss for similarity learning."""
        return F.cosine_embedding_loss(pred, target, torch.ones(pred.size(0)), margin=margin)
    
    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """Triplet loss for metric learning."""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss.mean()

class Optimizers:
    """Collection of optimization algorithms."""
    
    @staticmethod
    def adam(parameters: List[torch.nn.Parameter], lr: float = 1e-3, **kwargs) -> torch.optim.Adam:
        """Adam optimizer."""
        return torch.optim.Adam(parameters, lr=lr, **kwargs)
    
    @staticmethod
    def adamw(parameters: List[torch.nn.Parameter], lr: float = 1e-3, **kwargs) -> torch.optim.AdamW:
        """AdamW optimizer."""
        return torch.optim.AdamW(parameters, lr=lr, **kwargs)
    
    @staticmethod
    def sgd(parameters: List[torch.nn.Parameter], lr: float = 1e-2, **kwargs) -> torch.optim.SGD:
        """SGD optimizer."""
        return torch.optim.SGD(parameters, lr=lr, **kwargs)
    
    @staticmethod
    def rmsprop(parameters: List[torch.nn.Parameter], lr: float = 1e-2, **kwargs) -> torch.optim.RMSprop:
        """RMSprop optimizer."""
        return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
    
    @staticmethod
    def adagrad(parameters: List[torch.nn.Parameter], lr: float = 1e-2, **kwargs) -> torch.optim.Adagrad:
        """Adagrad optimizer."""
        return torch.optim.Adagrad(parameters, lr=lr, **kwargs)
    
    @staticmethod
    def adadelta(parameters: List[torch.nn.Parameter], lr: float = 1.0, **kwargs) -> torch.optim.Adadelta:
        """Adadelta optimizer."""
        return torch.optim.Adadelta(parameters, lr=lr, **kwargs)

class GradientClipper:
    """Gradient clipping utilities with NaN/Inf handling."""
    
    @staticmethod
    def clip_grad_norm_(parameters: Union[torch.Tensor, List[torch.Tensor]], max_norm: float, norm_type: float = 2.0) -> float:
        """Clip gradients by norm with NaN/Inf handling."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Handle NaN/Inf gradients
        for p in parameters:
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    logger.warning(f"NaN/Inf detected in gradients for parameter {p.shape}")
                    p.grad.data = torch.zeros_like(p.grad.data)
        
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm
    
    @staticmethod
    def clip_grad_value_(parameters: Union[torch.Tensor, List[torch.Tensor]], clip_value: float) -> None:
        """Clip gradients by value with NaN/Inf handling."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        for p in parameters:
            if p.grad is not None:
                # Handle NaN/Inf gradients
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    logger.warning(f"NaN/Inf detected in gradients for parameter {p.shape}")
                    p.grad.data = torch.zeros_like(p.grad.data)
                else:
                    p.grad.data.clamp_(min=-clip_value, max=clip_value)

class ModelInitializer:
    """Model initialization utilities."""
    
    def __init__(self, init_method: str = 'xavier_uniform', **kwargs):
        self.init_method = init_method
        self.kwargs = kwargs
    
    def initialize_model(self, model: nn.Module) -> None:
        """Initialize model weights using specified method."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU)):
                if hasattr(module, 'weight'):
                    self._apply_initialization(module.weight, f"{name}.weight")
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _apply_initialization(self, tensor: torch.Tensor, name: str) -> None:
        """Apply initialization to a tensor."""
        try:
            if self.init_method == 'xavier_uniform':
                WeightInitializer.xavier_uniform(tensor, **self.kwargs)
            elif self.init_method == 'xavier_normal':
                WeightInitializer.xavier_normal(tensor, **self.kwargs)
            elif self.init_method == 'kaiming_uniform':
                WeightInitializer.kaiming_uniform(tensor, **self.kwargs)
            elif self.init_method == 'kaiming_normal':
                WeightInitializer.kaiming_normal(tensor, **self.kwargs)
            elif self.init_method == 'orthogonal':
                WeightInitializer.orthogonal(tensor, **self.kwargs)
            elif self.init_method == 'sparse':
                WeightInitializer.sparse(tensor, **self.kwargs)
            elif self.init_method == 'variance_scaling':
                WeightInitializer.variance_scaling(tensor, **self.kwargs)
            else:
                logger.warning(f"Unknown initialization method: {self.init_method}")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")

class TrainingUtilities:
    """Training utility functions."""
    
    @staticmethod
    def check_gradients(model: nn.Module) -> Dict[str, Any]:
        """Check for NaN/Inf gradients in model."""
        grad_stats = {
            'has_nan': False,
            'has_inf': False,
            'total_params': 0,
            'params_with_grad': 0,
            'nan_params': [],
            'inf_params': []
        }
        
        for name, param in model.named_parameters():
            grad_stats['total_params'] += 1
            
            if param.grad is not None:
                grad_stats['params_with_grad'] += 1
                
                if torch.isnan(param.grad).any():
                    grad_stats['has_nan'] = True
                    grad_stats['nan_params'].append(name)
                
                if torch.isinf(param.grad).any():
                    grad_stats['has_inf'] = True
                    grad_stats['inf_params'].append(name)
        
        return grad_stats
    
    @staticmethod
    def zero_gradients(model: nn.Module) -> None:
        """Zero gradients with NaN/Inf handling."""
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad.data = torch.zeros_like(param.grad.data)
                else:
                    param.grad.zero_()
    
    @staticmethod
    def compute_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
        """Compute gradient norm with NaN/Inf handling."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                # Handle NaN/Inf gradients
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    continue
                
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

# Factory functions for easy access
def get_weight_initializer(method: str = 'xavier_uniform', **kwargs) -> ModelInitializer:
    """Get a weight initializer instance."""
    return ModelInitializer(method, **kwargs)

def get_loss_function(loss_type: str) -> Callable:
    """Get a loss function by name."""
    loss_functions = {
        'focal': LossFunctions.focal_loss,
        'dice': LossFunctions.dice_loss,
        'huber': LossFunctions.huber_loss,
        'cosine_embedding': LossFunctions.cosine_embedding_loss,
        'triplet': LossFunctions.triplet_loss,
        'cross_entropy': F.cross_entropy,
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'smooth_l1': F.smooth_l1_loss,
        'bce': F.binary_cross_entropy,
        'bce_with_logits': F.binary_cross_entropy_with_logits,
        'kl_div': F.kl_div,
        'ctc': F.ctc_loss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_type}")
    
    return loss_functions[loss_type]

def get_optimizer(optimizer_type: str, parameters: List[torch.nn.Parameter], **kwargs) -> torch.optim.Optimizer:
    """Get an optimizer instance."""
    optimizer_functions = {
        'adam': Optimizers.adam,
        'adamw': Optimizers.adamw,
        'sgd': Optimizers.sgd,
        'rmsprop': Optimizers.rmsprop,
        'adagrad': Optimizers.adagrad,
        'adadelta': Optimizers.adadelta
    }
    
    if optimizer_type not in optimizer_functions:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer_functions[optimizer_type](parameters, **kwargs)

__all__ = [
    "WeightInitializer",
    "NormalizationLayer", 
    "LossFunctions",
    "Optimizers",
    "GradientClipper",
    "ModelInitializer",
    "TrainingUtilities",
    "get_weight_initializer",
    "get_loss_function",
    "get_optimizer"
]
