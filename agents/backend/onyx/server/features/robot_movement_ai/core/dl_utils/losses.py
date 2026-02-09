"""
Loss Functions - Modular Loss Implementations
=============================================

Funciones de pérdida modulares para diferentes tareas.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TrajectoryLoss(nn.Module):
    """
    Pérdida para trayectorias que combina múltiples componentes.
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 0.5,
        smoothness_weight: float = 0.3,
        position_loss_type: str = 'mse'
    ):
        """
        Inicializar pérdida de trayectoria.
        
        Args:
            position_weight: Peso para pérdida de posición
            velocity_weight: Peso para pérdida de velocidad
            smoothness_weight: Peso para suavidad
            position_loss_type: Tipo de pérdida ('mse', 'mae', 'huber')
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight
        
        if position_loss_type == 'mse':
            self.position_loss = nn.MSELoss()
        elif position_loss_type == 'mae':
            self.position_loss = nn.L1Loss()
        elif position_loss_type == 'huber':
            self.position_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {position_loss_type}")
    
    def forward(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular pérdida.
        
        Args:
            pred_trajectory: Trayectoria predicha [batch, seq_len, dim]
            target_trajectory: Trayectoria objetivo [batch, seq_len, dim]
            
        Returns:
            Pérdida total
        """
        # Pérdida de posición
        position_loss = self.position_loss(pred_trajectory, target_trajectory)
        
        # Pérdida de velocidad (derivada)
        if self.velocity_weight > 0:
            pred_velocity = pred_trajectory[:, 1:] - pred_trajectory[:, :-1]
            target_velocity = target_trajectory[:, 1:] - target_trajectory[:, :-1]
            velocity_loss = F.mse_loss(pred_velocity, target_velocity)
        else:
            velocity_loss = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Pérdida de suavidad (segunda derivada)
        if self.smoothness_weight > 0:
            pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
            target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
            smoothness_loss = F.mse_loss(pred_acceleration, target_acceleration)
        else:
            smoothness_loss = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Pérdida total
        total_loss = (
            self.position_weight * position_loss +
            self.velocity_weight * velocity_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss para problemas de clasificación desbalanceados.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Inicializar Focal Loss.
        
        Args:
            alpha: Factor de balanceo
            gamma: Factor de enfoque
            reduction: Tipo de reducción
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcular Focal Loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss para aprendizaje de representaciones.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Inicializar Contrastive Loss.
        
        Args:
            margin: Margen para pares negativos
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular Contrastive Loss.
        
        Args:
            anchor: Embeddings de ancla [batch, dim]
            positive: Embeddings positivos [batch, dim]
            negative: Embeddings negativos [batch, dim]
            
        Returns:
            Pérdida
        """
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        
        loss = torch.mean(
            torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        )
        
        return loss


def get_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Obtener función de pérdida por nombre.
    
    Args:
        loss_type: Tipo de pérdida
        **kwargs: Argumentos adicionales
        
    Returns:
        Función de pérdida
    """
    loss_functions = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': nn.HuberLoss,
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCELoss,
        'bce_with_logits': nn.BCEWithLogitsLoss,
        'trajectory': TrajectoryLoss,
        'focal': FocalLoss,
        'contrastive': ContrastiveLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_functions[loss_type](**kwargs)













