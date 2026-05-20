from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Loss Functions
Comprehensive implementation of loss functions with proper PyTorch practices.
"""



class LossType(Enum):
    """Types of loss functions."""
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"
    FOCAL = "focal"
    DICE = "dice"
    COMBINED = "combined"
    CUSTOM = "custom"


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    loss_type: LossType = LossType.CROSS_ENTROPY
    reduction: str = "mean"
    label_smoothing: float = 0.0
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    huber_delta: float = 1.0
    smooth_l1_beta: float = 1.0
    dice_epsilon: float = 1e-6
    class_weights: Optional[torch.Tensor] = None
    ignore_index: int = -100


class AdvancedCrossEntropyLoss(nn.Module):
    """Advanced cross-entropy loss with label smoothing and class weights."""
    
    def __init__(self, num_classes: int, label_smoothing: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None, 
                 ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        
        # Validate parameters
        if label_smoothing < 0 or label_smoothing > 1:
            raise ValueError("label_smoothing must be between 0 and 1")
        
        if class_weights is not None and class_weights.size(0) != num_classes:
            raise ValueError("class_weights size must match num_classes")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with advanced cross-entropy loss."""
        if self.label_smoothing > 0:
            return self._label_smoothing_loss(predictions, targets)
        else:
            return F.cross_entropy(predictions, targets, 
                                 weight=self.class_weights,
                                 ignore_index=self.ignore_index)
    
    def _label_smoothing_loss(self, predictions: torch.Tensor, 
                             targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with label smoothing."""
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / self.num_classes
        
        # Compute log probabilities
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            log_probs = log_probs * self.class_weights.unsqueeze(0)
        
        # Compute loss
        loss = -(targets_one_hot * log_probs).sum(dim=1)
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            loss = loss[mask]
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with focal loss."""
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - target_probs) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(target_probs + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, epsilon: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with dice loss."""
        # Apply sigmoid for binary classification
        if predictions.dim() == targets.dim():
            predictions = torch.sigmoid(predictions)
        else:
            predictions = F.softmax(predictions, dim=1)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Compute dice coefficient
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        
        # Return dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    
    def __init__(self, loss_components: List[Tuple[nn.Module, float]]):
        super().__init__()
        self.loss_components = nn.ModuleList([loss for loss, _ in loss_components])
        self.weights = [weight for _, weight in loss_components]
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with combined loss."""
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.loss_components, self.weights):
            loss = loss_fn(predictions, targets)
            total_loss += weight * loss
        
        return total_loss


class CustomLossFunction(nn.Module):
    """Custom loss function with advanced features."""
    
    def __init__(self, loss_type: str = "custom", alpha: float = 0.5, 
                 beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with custom loss function."""
        if self.loss_type == "combined_regression":
            return self._combined_regression_loss(predictions, targets)
        elif self.loss_type == "robust_classification":
            return self._robust_classification_loss(predictions, targets)
        elif self.loss_type == "attention_loss":
            return self._attention_loss(predictions, targets)
        else:
            return self._default_custom_loss(predictions, targets)
    
    def _combined_regression_loss(self, predictions: torch.Tensor, 
                                 targets: torch.Tensor) -> torch.Tensor:
        """Combined regression loss with multiple components."""
        # MSE component
        mse_loss = F.mse_loss(predictions, targets)
        
        # MAE component
        mae_loss = F.l1_loss(predictions, targets)
        
        # Huber component
        huber_loss = F.smooth_l1_loss(predictions, targets)
        
        # Combined loss
        total_loss = (self.alpha * mse_loss + 
                     self.beta * mae_loss + 
                     self.gamma * huber_loss)
        
        return total_loss
    
    def _robust_classification_loss(self, predictions: torch.Tensor, 
                                   targets: torch.Tensor) -> torch.Tensor:
        """Robust classification loss with noise handling."""
        # Standard cross-entropy
        ce_loss = F.cross_entropy(predictions, targets)
        
        # Confidence penalty
        probs = F.softmax(predictions, dim=1)
        confidence_penalty = torch.mean(torch.max(probs, dim=1)[0])
        
        # Entropy regularization
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
        
        # Combined loss
        total_loss = ce_loss + self.alpha * confidence_penalty - self.beta * entropy
        
        return total_loss
    
    def _attention_loss(self, predictions: torch.Tensor, 
                       targets: torch.Tensor) -> torch.Tensor:
        """Attention-based loss for sequence tasks."""
        # Standard loss
        base_loss = F.cross_entropy(predictions, targets)
        
        # Attention regularization
        attention_weights = F.softmax(predictions, dim=-1)
        attention_entropy = -torch.mean(torch.sum(attention_weights * 
                                                 torch.log(attention_weights + 1e-8), dim=-1))
        
        # Sparsity penalty
        sparsity_penalty = torch.mean(torch.sum(attention_weights ** 2, dim=-1))
        
        # Combined loss
        total_loss = base_loss + self.alpha * attention_entropy + self.beta * sparsity_penalty
        
        return total_loss
    
    def _default_custom_loss(self, predictions: torch.Tensor, 
                            targets: torch.Tensor) -> torch.Tensor:
        """Default custom loss implementation."""
        # MSE with regularization
        mse_loss = F.mse_loss(predictions, targets)
        
        # L1 regularization on predictions
        l1_reg = torch.mean(torch.abs(predictions))
        
        # L2 regularization on predictions
        l2_reg = torch.mean(predictions ** 2)
        
        # Combined loss
        total_loss = mse_loss + self.alpha * l1_reg + self.beta * l2_reg
        
        return total_loss


class LossFunctionFactory:
    """Factory for creating different loss functions."""
    
    @staticmethod
    def create_loss_function(config: LossConfig) -> nn.Module:
        """Create loss function based on configuration."""
        if config.loss_type == LossType.CROSS_ENTROPY:
            return AdvancedCrossEntropyLoss(
                num_classes=10,  # Default, should be configurable
                label_smoothing=config.label_smoothing,
                class_weights=config.class_weights,
                ignore_index=config.ignore_index
            )
        elif config.loss_type == LossType.BINARY_CROSS_ENTROPY:
            return nn.BCEWithLogitsLoss(reduction=config.reduction)
        elif config.loss_type == LossType.MSE:
            return nn.MSELoss(reduction=config.reduction)
        elif config.loss_type == LossType.MAE:
            return nn.L1Loss(reduction=config.reduction)
        elif config.loss_type == LossType.HUBER:
            return nn.HuberLoss(delta=config.huber_delta, reduction=config.reduction)
        elif config.loss_type == LossType.SMOOTH_L1:
            return nn.SmoothL1Loss(beta=config.smooth_l1_beta, reduction=config.reduction)
        elif config.loss_type == LossType.FOCAL:
            return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        elif config.loss_type == LossType.DICE:
            return DiceLoss(epsilon=config.dice_epsilon)
        elif config.loss_type == LossType.COMBINED:
            # Create combined loss with multiple components
            loss_components = [
                (nn.MSELoss(), 0.6),
                (nn.L1Loss(), 0.3),
                (nn.HuberLoss(), 0.1)
            ]
            return CombinedLoss(loss_components)
        elif config.loss_type == LossType.CUSTOM:
            return CustomLossFunction()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")


class LossAnalyzer:
    """Analyze loss function behavior and statistics."""
    
    def __init__(self) -> Any:
        self.loss_history = []
        self.gradient_history = []
    
    def analyze_loss(self, loss_function: nn.Module, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor) -> Dict[str, Any]:
        """Analyze loss function behavior."""
        # Compute loss
        loss = loss_function(predictions, targets)
        
        # Compute gradients
        loss.backward()
        
        # Analyze gradients
        gradient_stats = self._analyze_gradients(predictions)
        
        # Store history
        self.loss_history.append(loss.item())
        self.gradient_history.append(gradient_stats)
        
        return {
            'loss_value': loss.item(),
            'gradient_stats': gradient_stats,
            'predictions_stats': self._analyze_predictions(predictions),
            'targets_stats': self._analyze_targets(targets)
        }
    
    def _analyze_gradients(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Analyze gradient statistics."""
        if tensor.grad is None:
            return {}
        
        grad = tensor.grad
        return {
            'grad_norm': grad.norm().item(),
            'grad_mean': grad.mean().item(),
            'grad_std': grad.std().item(),
            'grad_max': grad.max().item(),
            'grad_min': grad.min().item()
        }
    
    def _analyze_predictions(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Analyze prediction statistics."""
        return {
            'mean': predictions.mean().item(),
            'std': predictions.std().item(),
            'max': predictions.max().item(),
            'min': predictions.min().item(),
            'norm_l2': predictions.norm(p=2).item()
        }
    
    def _analyze_targets(self, targets: torch.Tensor) -> Dict[str, float]:
        """Analyze target statistics."""
        return {
            'mean': targets.mean().item(),
            'std': targets.std().item(),
            'max': targets.max().item(),
            'min': targets.min().item(),
            'norm_l2': targets.norm(p=2).item()
        }


def demonstrate_loss_functions():
    """Demonstrate different loss functions."""
    print("Loss Functions Demonstration")
    print("=" * 40)
    
    # Create sample data
    batch_size = 4
    num_classes = 3
    predictions = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    loss_configs = [
        LossConfig(loss_type=LossType.CROSS_ENTROPY, label_smoothing=0.1),
        LossConfig(loss_type=LossType.FOCAL, focal_alpha=1.0, focal_gamma=2.0),
        LossConfig(loss_type=LossType.COMBINED),
        LossConfig(loss_type=LossType.CUSTOM)
    ]
    
    analyzer = LossAnalyzer()
    results = {}
    
    for config in loss_configs:
        print(f"\nTesting {config.loss_type.value}:")
        
        try:
            # Create loss function
            loss_function = LossFunctionFactory.create_loss_function(config)
            
            # Analyze loss
            analysis = analyzer.analyze_loss(loss_function, predictions, targets)
            
            print(f"  Loss value: {analysis['loss_value']:.6f}")
            print(f"  Gradient norm: {analysis['gradient_stats'].get('grad_norm', 0):.6f}")
            
            results[config.loss_type.value] = analysis
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config.loss_type.value] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Demonstrate loss functions
    results = demonstrate_loss_functions()
    print("\nLoss functions demonstration completed!") 