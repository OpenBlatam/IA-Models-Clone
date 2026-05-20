"""
Knowledge Distillation
======================

Implementation of knowledge distillation for model compression.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KnowledgeDistillation(nn.Module):
    """
    Knowledge Distillation loss.
    
    Distills knowledge from teacher model to student model.
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            temperature: Temperature for softmax
            alpha: Weight for distillation loss vs hard target loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Hard targets
            
        Returns:
            Combined loss
        """
        # Soft targets (distillation loss)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets (standard loss)
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


class DistillationTrainer:
    """
    Trainer for knowledge distillation.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7,
        device: Optional[torch.device] = None
    ):
        """
        Initialize distillation trainer.
        
        Args:
            student_model: Student model (smaller)
            teacher_model: Teacher model (larger, pre-trained)
            temperature: Distillation temperature
            alpha: Weight for distillation loss
            device: Device to run on
        """
        self.student = student_model
        self.teacher = teacher_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.student = self.student.to(self.device)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()  # Teacher is frozen
        
        self.distillation_loss = KnowledgeDistillation(temperature, alpha)
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            inputs: Input data
            targets: Ground truth labels
            optimizer: Optimizer
            
        Returns:
            Dictionary with losses
        """
        self.student.train()
        
        # Forward pass
        student_logits = self.student(inputs)
        
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'distillation_loss': loss.item(),
            'student_accuracy': (student_logits.argmax(dim=1) == targets).float().mean().item()
        }



