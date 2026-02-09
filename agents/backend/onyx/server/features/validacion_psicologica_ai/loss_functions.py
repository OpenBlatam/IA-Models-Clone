"""
Custom Loss Functions
=====================
Specialized loss functions for psychological analysis
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger()


class PersonalityTraitLoss(nn.Module):
    """
    Loss function for personality trait prediction
    Combines multiple traits with weighted importance
    """
    
    def __init__(
        self,
        trait_weights: Optional[Dict[str, float]] = None,
        reduction: str = "mean"
    ):
        """
        Initialize loss
        
        Args:
            trait_weights: Weights for each trait (None = equal weights)
            reduction: Reduction method (mean, sum, none)
        """
        super().__init__()
        self.trait_weights = trait_weights or {
            "openness": 1.0,
            "conscientiousness": 1.0,
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "neuroticism": 1.0
        }
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Dictionary of trait predictions
            targets: Target trait values [batch_size, num_traits]
            
        Returns:
            Loss value
        """
        total_loss = 0.0
        trait_names = list(self.trait_weights.keys())
        
        for i, trait in enumerate(trait_names):
            if trait in predictions:
                pred = predictions[trait].squeeze()
                target = targets[:, i]
                
                trait_loss = self.mse_loss(pred, target)
                weighted_loss = trait_loss * self.trait_weights[trait]
                total_loss += weighted_loss
        
        return total_loss / len(trait_names)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor for each class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target classes [batch_size]
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1
    ):
        """
        Initialize label smoothing loss
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target classes [batch_size]
            
        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-task learning
    """
    
    def __init__(
        self,
        loss_functions: Dict[str, nn.Module],
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize combined loss
        
        Args:
            loss_functions: Dictionary of loss functions
            loss_weights: Weights for each loss (None = equal weights)
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        
        if loss_weights is None:
            self.loss_weights = {name: 1.0 for name in loss_functions.keys()}
        else:
            self.loss_weights = loss_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        for name, loss_fn in self.loss_functions.items():
            if name in predictions and name in targets:
                loss = loss_fn(predictions[name], targets[name])
                weighted_loss = loss * self.loss_weights.get(name, 1.0)
                total_loss += weighted_loss
        
        return total_loss


# Factory function for creating loss functions
def create_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss (mse, cross_entropy, focal, label_smoothing, personality)
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function
    """
    if loss_type == "mse":
        return nn.MSELoss(**kwargs)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "personality":
        return PersonalityTraitLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")




