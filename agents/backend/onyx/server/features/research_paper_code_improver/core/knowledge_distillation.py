"""
Knowledge Distillation - Distilación de conocimiento
=====================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuración de distilación"""
    temperature: float = 3.0
    alpha: float = 0.5  # Weight para soft targets vs hard targets
    loss_type: str = "kl_div"  # "kl_div" or "mse"


class KnowledgeDistiller:
    """Distilador de conocimiento"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """Calcula pérdida de distilación"""
        temp = temperature or self.config.temperature
        alpha_val = alpha or self.config.alpha
        
        # Soft targets (distribución del teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        
        # Hard targets (ground truth)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combinar
        total_loss = alpha_val * soft_loss + (1 - alpha_val) * hard_loss
        
        return total_loss
    
    def distill(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader,
        optimizer,
        num_epochs: int = 10,
        device: str = "cuda"
    ):
        """Entrena estudiante usando distilación"""
        student_model.train()
        teacher_model.eval()
        device = torch.device(device)
        
        student_model = student_model.to(device)
        teacher_model = teacher_model.to(device)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Mover batch a device
                inputs = batch.get("input_ids") or batch.get("inputs")
                labels = batch.get("labels")
                
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                
                # Forward pass - Teacher (no gradients)
                with torch.no_grad():
                    if isinstance(batch, dict):
                        teacher_outputs = teacher_model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                                          for k, v in batch.items()})
                    else:
                        teacher_outputs = teacher_model(inputs)
                    
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
                
                # Forward pass - Student
                if isinstance(batch, dict):
                    student_outputs = student_model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                                       for k, v in batch.items()})
                else:
                    student_outputs = student_model(inputs)
                
                student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
                
                # Distillation loss
                loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels
                )
                
                # Backward
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch}: Distillation loss = {avg_loss:.4f}")
        
        logger.info("Distilación completada")




