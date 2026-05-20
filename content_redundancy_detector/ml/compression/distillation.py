"""
Knowledge Distillation
Knowledge distillation for model compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration"""
    temperature: float = 3.0
    alpha: float = 0.5  # Weight for soft target loss
    hard_target_weight: float = 0.5  # Weight for hard target loss


class KnowledgeDistiller:
    """
    Knowledge distillation implementation
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
    ):
        """
        Initialize distiller
        
        Args:
            teacher_model: Teacher model (large, pre-trained)
            student_model: Student model (small, to be trained)
            config: Distillation configuration
        """
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: True labels
            
        Returns:
            Distillation loss
        """
        # Soft targets (teacher predictions)
        soft_targets = F.softmax(teacher_logits / self.config.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.config.temperature, dim=1)
        
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (
            self.config.temperature ** 2
        )
        
        # Hard targets (true labels)
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = (
            self.config.alpha * soft_loss +
            self.config.hard_target_weight * hard_loss
        )
        
        return total_loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        student_optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform one distillation training step
        
        Args:
            inputs: Input batch
            targets: Target labels
            student_optimizer: Student model optimizer
            
        Returns:
            Dictionary with loss values
        """
        self.teacher.eval()
        self.student.train()
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Student forward
        student_logits = self.student(inputs)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # Backward
        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()
        
        return {
            'distillation_loss': float(loss.item()),
            'student_accuracy': self._compute_accuracy(student_logits, targets),
        }
    
    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute accuracy"""
        preds = torch.argmax(logits, dim=1)
        return float((preds == targets).float().mean().item())



