"""
Loss Functions Service - Funciones de pérdida
=============================================

Sistema para crear y gestionar funciones de pérdida.
Sigue mejores prácticas de PyTorch.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuración de función de pérdida"""
    loss_type: str = "cross_entropy"  # cross_entropy, mse, mae, bce, focal, etc.
    reduction: str = "mean"  # mean, sum, none
    weight: Optional[torch.Tensor] = None  # Class weights
    ignore_index: int = -100  # For CrossEntropyLoss
    label_smoothing: float = 0.0  # For CrossEntropyLoss
    # Focal loss
    alpha: float = 1.0  # For Focal Loss
    gamma: float = 2.0  # For Focal Loss
    # Custom parameters
    custom_params: Dict[str, Any] = None


class LossFunctionsService:
    """Servicio para funciones de pérdida"""
    
    @staticmethod
    def create_loss(config: LossConfig) -> nn.Module:
        """
        Crear función de pérdida según configuración.
        
        Args:
            config: Configuración de pérdida
        
        Returns:
            Función de pérdida configurada
        """
        loss_type = config.loss_type.lower()
        
        if loss_type == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss(
                weight=config.weight,
                reduction=config.reduction,
                ignore_index=config.ignore_index,
                label_smoothing=config.label_smoothing,
            )
        
        elif loss_type == "mse" or loss_type == "mean_squared_error":
            loss_fn = nn.MSELoss(reduction=config.reduction)
        
        elif loss_type == "mae" or loss_type == "mean_absolute_error" or loss_type == "l1":
            loss_fn = nn.L1Loss(reduction=config.reduction)
        
        elif loss_type == "bce" or loss_type == "binary_cross_entropy":
            loss_fn = nn.BCELoss(
                weight=config.weight,
                reduction=config.reduction,
            )
        
        elif loss_type == "bce_with_logits":
            loss_fn = nn.BCEWithLogitsLoss(
                weight=config.weight,
                reduction=config.reduction,
            )
        
        elif loss_type == "focal":
            # Focal Loss implementation
            loss_fn = FocalLoss(
                alpha=config.alpha,
                gamma=config.gamma,
                reduction=config.reduction,
            )
        
        elif loss_type == "smooth_l1":
            loss_fn = nn.SmoothL1Loss(reduction=config.reduction)
        
        elif loss_type == "huber":
            beta = config.custom_params.get("beta", 1.0) if config.custom_params else 1.0
            loss_fn = nn.HuberLoss(reduction=config.reduction, delta=beta)
        
        elif loss_type == "kl_div":
            loss_fn = nn.KLDivLoss(reduction=config.reduction)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        logger.info(f"Created {loss_type} loss function")
        return loss_fn
    
    @staticmethod
    def compute_class_weights(
        labels: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Calcular pesos de clases para balancear dataset.
        
        Args:
            labels: Tensor de etiquetas
            num_classes: Número de clases (None = auto-detect)
        
        Returns:
            Tensor de pesos
        """
        if num_classes is None:
            num_classes = int(labels.max().item() + 1)
        
        # Count occurrences of each class
        class_counts = torch.bincount(labels.long(), minlength=num_classes).float()
        
        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)
        
        # Calculate weights: total_samples / (num_classes * class_count)
        total_samples = labels.size(0)
        weights = total_samples / (num_classes * class_counts)
        
        # Normalize
        weights = weights / weights.sum() * num_classes
        
        logger.info(f"Computed class weights: {weights.tolist()}")
        return weights


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar desbalance de clases.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Logits (N, C)
            targets: Targets (N,)
        
        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss.
    
    Mejora la generalización suavizando las etiquetas.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean"
    ):
        """
        Args:
            num_classes: Número de clases
            smoothing: Factor de suavizado (0.0 = no smoothing)
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Logits (N, C)
            targets: Targets (N,)
        
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute KL divergence
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss




