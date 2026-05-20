"""
Advanced Loss Functions - Funciones de pérdida avanzadas
=========================================================
Loss functions personalizadas para diferentes tareas
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss para manejar clases desbalanceadas"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class DiceLoss(nn.Module):
    """Dice Loss para segmentación"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets_one_hot.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice


class ContrastiveLoss(nn.Module):
    """Contrastive Loss para aprendizaje de representaciones"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        
        loss = torch.mean(
            torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        )
        return loss


class TripletLoss(nn.Module):
    """Triplet Loss"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class CombinedLoss(nn.Module):
    """Loss combinado de múltiples funciones"""
    
    def __init__(
        self,
        losses: List[Callable],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(*args, **kwargs)
            total_loss += weight * loss_value
        return total_loss




