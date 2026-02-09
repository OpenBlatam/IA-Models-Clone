"""
Funciones de pérdida personalizadas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss para clases desbalanceadas"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
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
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class ContrastiveLoss(nn.Module):
    """Contrastive Loss para embeddings"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: [batch_size, embedding_dim]
            embeddings2: [batch_size, embedding_dim]
            labels: [batch_size] (1 para similar, 0 para diferente)
        """
        distance = F.pairwise_distance(embeddings1, embeddings2)
        
        positive_loss = labels * torch.pow(distance, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class CombinedLoss(nn.Module):
    """Combina múltiples losses"""
    
    def __init__(self, losses: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            if name in predictions and name in targets:
                loss = loss_fn(predictions[name], targets[name])
                weight = self.weights.get(name, 1.0)
                total_loss += weight * loss
        
        return total_loss




