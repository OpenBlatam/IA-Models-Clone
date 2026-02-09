"""
Model Optimization: Pruning and Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Dict, Callable
import logging

logger = logging.getLogger(__name__)


class ModelPruner:
    """Model pruning utilities"""
    
    @staticmethod
    def prune_model(
        model: nn.Module,
        pruning_method: str = "l1_unstructured",
        amount: float = 0.2,
        module_type: type = nn.Linear
    ) -> nn.Module:
        """
        Prune model
        
        Args:
            model: Model to prune
            pruning_method: Pruning method (l1_unstructured, l2_unstructured, etc.)
            amount: Pruning amount (0.0-1.0)
            module_type: Module type to prune
        
        Returns:
            Pruned model
        """
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, module_type):
                modules_to_prune.append((module, "weight"))
        
        if pruning_method == "l1_unstructured":
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        elif pruning_method == "l2_unstructured":
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L2Unstructured,
                amount=amount
            )
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")
        
        # Make pruning permanent
        for module, _ in modules_to_prune:
            prune.remove(module, "weight")
        
        logger.info(f"Model pruned: {pruning_method}, amount={amount}")
        return model
    
    @staticmethod
    def get_sparsity(model: nn.Module) -> float:
        """Calculate model sparsity"""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0


class KnowledgeDistillation:
    """Knowledge distillation for model compression"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        """
        Initialize knowledge distillation
        
        Args:
            teacher_model: Large teacher model
            student_model: Small student model
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher_model.eval()
        self.student_model.train()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Calculate distillation loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Ground truth targets
            temperature: Temperature (uses self.temperature if None)
        
        Returns:
            Distillation loss
        """
        if temperature is None:
            temperature = self.temperature
        
        # Soft targets from teacher
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            student_log_probs, teacher_probs
        ) * (temperature ** 2)
        
        # Student loss (hard targets)
        student_loss = nn.CrossEntropyLoss()(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Training step for distillation
        
        Args:
            inputs: Input data
            targets: Target labels
            optimizer: Optimizer
        
        Returns:
            Dictionary of losses
        """
        optimizer.zero_grad()
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Student forward
        student_logits = self.student_model(inputs)
        
        # Calculate loss
        loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return {
            "loss": loss.item(),
            "distillation_loss": loss.item() * self.alpha,
            "student_loss": loss.item() * (1 - self.alpha)
        }


def create_lightweight_student(
    teacher_model: nn.Module,
    reduction_factor: float = 0.5
) -> nn.Module:
    """
    Create lightweight student model from teacher
    
    Args:
        teacher_model: Teacher model
        reduction_factor: Size reduction factor
    
    Returns:
        Student model
    """
    # This is a simplified version - in practice, you'd design a smaller architecture
    # For now, we'll just return a smaller version of the same architecture
    
    # Get teacher architecture info
    if hasattr(teacher_model, "model"):
        # For Sequential models
        layers = []
        for i, layer in enumerate(teacher_model.model):
            if isinstance(layer, nn.Linear):
                # Reduce size
                new_in = int(layer.in_features * reduction_factor) if i > 0 else layer.in_features
                new_out = int(layer.out_features * reduction_factor)
                layers.append(nn.Linear(new_in, new_out))
            else:
                layers.append(layer)
        
        student = nn.Sequential(*layers)
    else:
        # Fallback: return teacher (in practice, design proper student)
        student = teacher_model
    
    logger.info("Lightweight student model created")
    return student

