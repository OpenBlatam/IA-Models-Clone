"""
Loss Function Manager - Gestor de funciones de pérdida
=======================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LossType(Enum):
    """Tipos de funciones de pérdida"""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    MAE = "mae"
    BCE = "bce"
    BCE_WITH_LOGITS = "bce_with_logits"
    FOCAL = "focal"
    DICE = "dice"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"
    COSINE_SIMILARITY = "cosine_similarity"
    TRIPLET = "triplet"
    CONTRASTIVE = "contrastive"


@dataclass
class LossConfig:
    """Configuración de función de pérdida"""
    loss_type: LossType
    weight: float = 1.0
    reduction: str = "mean"  # "mean", "sum", "none"
    ignore_index: int = -100
    label_smoothing: float = 0.0
    # Focal loss params
    alpha: float = 0.25
    gamma: float = 2.0
    # Triplet loss params
    margin: float = 1.0


class FocalLoss(nn.Module):
    """Focal Loss para clases desbalanceadas"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss para segmentación"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice


class TripletLoss(nn.Module):
    """Triplet Loss para embeddings"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class LossFunctionManager:
    """Gestor de funciones de pérdida"""
    
    def __init__(self):
        self.loss_functions: Dict[str, nn.Module] = {}
    
    def get_loss_function(self, config: LossConfig) -> nn.Module:
        """Obtiene una función de pérdida"""
        if config.loss_type == LossType.CROSS_ENTROPY:
            return nn.CrossEntropyLoss(
                weight=None,
                reduction=config.reduction,
                ignore_index=config.ignore_index,
                label_smoothing=config.label_smoothing
            )
        
        elif config.loss_type == LossType.MSE:
            return nn.MSELoss(reduction=config.reduction)
        
        elif config.loss_type == LossType.MAE:
            return nn.L1Loss(reduction=config.reduction)
        
        elif config.loss_type == LossType.BCE:
            return nn.BCELoss(reduction=config.reduction)
        
        elif config.loss_type == LossType.BCE_WITH_LOGITS:
            return nn.BCEWithLogitsLoss(reduction=config.reduction)
        
        elif config.loss_type == LossType.FOCAL:
            return FocalLoss(
                alpha=config.alpha,
                gamma=config.gamma,
                reduction=config.reduction
            )
        
        elif config.loss_type == LossType.DICE:
            return DiceLoss()
        
        elif config.loss_type == LossType.HUBER:
            return nn.HuberLoss(reduction=config.reduction)
        
        elif config.loss_type == LossType.SMOOTH_L1:
            return nn.SmoothL1Loss(reduction=config.reduction)
        
        elif config.loss_type == LossType.COSINE_SIMILARITY:
            return nn.CosineEmbeddingLoss()
        
        elif config.loss_type == LossType.TRIPLET:
            return TripletLoss(margin=config.margin)
        
        else:
            raise ValueError(f"Tipo de pérdida {config.loss_type} no soportado")
    
    def create_combined_loss(
        self,
        loss_configs: List[LossConfig]
    ) -> Callable:
        """Crea una función de pérdida combinada"""
        loss_functions = [self.get_loss_function(config) for config in loss_configs]
        weights = [config.weight for config in loss_configs]
        
        def combined_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
            total_loss = 0.0
            
            for loss_fn, weight, config in zip(loss_functions, weights, loss_configs):
                if config.loss_type == LossType.TRIPLET:
                    # Triplet loss requiere inputs especiales
                    loss = loss_fn(
                        outputs.get("anchor"),
                        outputs.get("positive"),
                        outputs.get("negative")
                    )
                else:
                    output = outputs.get("logits") or outputs.get("prediction")
                    target = targets.get("labels") or targets.get("target")
                    loss = loss_fn(output, target)
                
                total_loss += weight * loss
            
            return total_loss
        
        return combined_loss




