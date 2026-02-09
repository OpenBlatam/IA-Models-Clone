"""
Knowledge Distillation para transferir conocimiento de modelo grande a pequeño
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KnowledgeDistillationLoss(nn.Module):
    """Loss function para knowledge distillation"""
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula loss de knowledge distillation
        
        Args:
            student_logits: Logits del estudiante
            teacher_logits: Logits del profesor
            labels: Labels verdaderos (opcional)
        """
        # Softmax con temperatura
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Distillation loss
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Si hay labels, agregar loss de clasificación
        if labels is not None:
            classification_loss = self.ce_loss(student_logits, labels)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * classification_loss
        else:
            total_loss = distillation_loss
        
        return total_loss


class KnowledgeDistiller:
    """Distiller de conocimiento"""
    
    def __init__(self):
        pass
    
    def distill(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        temperature: float = 3.0,
        alpha: float = 0.7,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Distilla conocimiento de teacher a student
        
        Args:
            teacher_model: Modelo profesor (grande)
            student_model: Modelo estudiante (pequeño)
            train_loader: DataLoader de entrenamiento
            num_epochs: Número de épocas
            temperature: Temperatura para softmax
            alpha: Balance entre distillation y classification loss
            device: Device
            
        Returns:
            Resultados del entrenamiento
        """
        teacher_model.eval()
        student_model.train()
        
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
        criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
        
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Teacher (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
                
                # Student
                student_outputs = student_model(inputs)
                student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
                
                # Loss
                loss = criterion(student_logits, teacher_logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return {
            "success": True,
            "losses": losses,
            "final_loss": losses[-1] if losses else None
        }




