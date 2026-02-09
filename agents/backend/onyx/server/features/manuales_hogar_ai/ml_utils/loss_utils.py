"""
Loss Utils - Funciones de Pérdida Avanzadas
===========================================

Funciones de pérdida personalizadas para diferentes tareas.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar desbalance de clases.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Inicializar Focal Loss.
        
        Args:
            alpha: Factor de balanceo
            gamma: Factor de focusing
            reduction: Reducción ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular Focal Loss.
        
        Args:
            inputs: Predicciones (logits)
            targets: Targets (clases)
            
        Returns:
            Loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Inicializar Dice Loss.
        
        Args:
            smooth: Factor de suavizado
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular Dice Loss.
        
        Args:
            inputs: Predicciones (probabilidades)
            targets: Targets (binarios)
            
        Returns:
            Loss
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class IoULoss(nn.Module):
    """
    IoU Loss (Intersection over Union) para segmentación.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Inicializar IoU Loss.
        
        Args:
            smooth: Factor de suavizado
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular IoU Loss.
        
        Args:
            inputs: Predicciones (probabilidades)
            targets: Targets (binarios)
            
        Returns:
            Loss
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss para regularización.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Inicializar Label Smoothing Loss.
        
        Args:
            num_classes: Número de clases
            smoothing: Factor de suavizado
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular Label Smoothing Loss.
        
        Args:
            inputs: Predicciones (logits)
            targets: Targets (clases)
            
        Returns:
            Loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class TripletLoss(nn.Module):
    """
    Triplet Loss para aprendizaje de embeddings.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Inicializar Triplet Loss.
        
        Args:
            margin: Margen para triplets
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
        Calcular Triplet Loss.
        
        Args:
            anchor: Embeddings de anclas
            positive: Embeddings positivos
            negative: Embeddings negativos
            
        Returns:
            Loss
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss para aprendizaje de embeddings.
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
        x1: torch.Tensor,
        x2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular Contrastive Loss.
        
        Args:
            x1: Primer embedding
            x2: Segundo embedding
            label: 1 si son similares, 0 si son diferentes
            
        Returns:
            Loss
        """
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss_positive = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_negative = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(loss_positive + loss_negative)
        return loss


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1) para regresión robusta.
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Inicializar Huber Loss.
        
        Args:
            delta: Umbral de transición
        """
        super().__init__()
        self.delta = delta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular Huber Loss.
        
        Args:
            inputs: Predicciones
            targets: Targets
            
        Returns:
            Loss
        """
        error = inputs - targets
        is_small_error = torch.abs(error) < self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * torch.abs(error) - 0.5 * self.delta ** 2
        return torch.where(is_small_error, squared_loss, linear_loss).mean()


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss para knowledge distillation.
    """
    
    def __init__(self, temperature: float = 4.0):
        """
        Inicializar KL Divergence Loss.
        
        Args:
            temperature: Temperatura para softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular KL Divergence Loss.
        
        Args:
            student_logits: Logits del estudiante
            teacher_logits: Logits del profesor
            
        Returns:
            Loss
        """
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)


class CombinedLoss(nn.Module):
    """
    Combinación de múltiples losses.
    """
    
    def __init__(
        self,
        losses: list,
        weights: Optional[list] = None
    ):
        """
        Inicializar Combined Loss.
        
        Args:
            losses: Lista de funciones de pérdida
            weights: Pesos para cada loss (opcional)
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)
        
        if len(self.weights) != len(self.losses):
            raise ValueError("Number of weights must match number of losses")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Calcular combined loss.
        
        Args:
            *args: Argumentos para losses
            **kwargs: Keyword arguments para losses
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(*args, **kwargs)
            total_loss += weight * loss_value
        return total_loss




