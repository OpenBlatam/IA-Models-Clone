"""
Knowledge Distillation

Utilities for knowledge distillation.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """Distill knowledge from teacher to student model."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        """
        Initialize knowledge distiller.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels (optional)
            
        Returns:
            Distillation loss
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss (if labels provided)
        if labels is not None:
            student_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        else:
            total_loss = distillation_loss
        
        return total_loss


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    temperature: float = 3.0,
    alpha: float = 0.7
) -> nn.Module:
    """
    Distill knowledge from teacher to student.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataloader: Training data loader
        optimizer: Optimizer
        num_epochs: Number of epochs
        temperature: Temperature for distillation
        alpha: Weight for distillation loss
        
    Returns:
        Trained student model
    """
    distiller = KnowledgeDistiller(teacher_model, student_model, temperature, alpha)
    
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch['input'] if isinstance(batch, dict) else batch[0]
            labels = batch.get('label') if isinstance(batch, dict) else (batch[1] if len(batch) > 1 else None)
            
            # Teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # Student predictions
            student_logits = student_model(inputs)
            
            # Distillation loss
            loss = distiller.distill_loss(student_logits, teacher_logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return student_model


def create_distillation_loss(
    temperature: float = 3.0,
    alpha: float = 0.7
) -> callable:
    """
    Create distillation loss function.
    
    Args:
        temperature: Temperature for softmax
        alpha: Weight for distillation loss
        
    Returns:
        Loss function
    """
    def loss_fn(student_logits, teacher_logits, labels=None):
        distiller = KnowledgeDistiller(
            None, None, temperature, alpha
        )
        return distiller.distill_loss(student_logits, teacher_logits, labels)
    
    return loss_fn



