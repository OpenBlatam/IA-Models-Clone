"""
Distillation Utils - Utilidades de Knowledge Distillation
==========================================================

Utilidades para knowledge distillation de modelos.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Loss para knowledge distillation.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        reduction: str = 'mean'
    ):
        """
        Inicializar distillation loss.
        
        Args:
            temperature: Temperatura para softmax
            alpha: Peso entre soft targets y hard targets
            reduction: Reducción ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular distillation loss.
        
        Args:
            student_logits: Logits del estudiante
            teacher_logits: Logits del profesor
            targets: Hard targets
            
        Returns:
            Loss
        """
        # Soft targets (distillation loss)
        student_log_softmax = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_softmax = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_log_softmax, teacher_softmax)
        soft_loss = soft_loss.sum(dim=1) * (self.temperature ** 2)
        
        # Hard targets (classification loss)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DistillationTrainer:
    """
    Trainer para knowledge distillation.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = "cuda"
    ):
        """
        Inicializar distillation trainer.
        
        Args:
            student_model: Modelo estudiante
            teacher_model: Modelo profesor
            temperature: Temperatura
            alpha: Peso de soft targets
            device: Dispositivo
        """
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()
        
        self.distillation_loss = DistillationLoss(temperature, alpha)
        self.device = device
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Realizar paso de entrenamiento.
        
        Args:
            inputs: Inputs
            targets: Targets
            optimizer: Optimizador
            
        Returns:
            Diccionario con métricas
        """
        self.student_model.train()
        optimizer.zero_grad()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass del estudiante
        student_logits = self.student_model(inputs)
        
        # Forward pass del profesor (sin gradientes)
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Calcular loss
        loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calcular accuracy
        preds = student_logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Entrenar modelo con distillation.
        
        Args:
            train_loader: DataLoader de entrenamiento
            optimizer: Optimizador
            epochs: Número de épocas
            val_loader: DataLoader de validación (opcional)
            
        Returns:
            Historial de entrenamiento
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Entrenamiento
            self.student_model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                metrics = self.train_step(inputs, targets, optimizer)
                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(avg_acc)
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
            
            # Validación
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        return history
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluar modelo.
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Métricas de evaluación
        """
        self.student_model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                student_logits = self.student_model(inputs)
                
                with torch.no_grad():
                    teacher_logits = self.teacher_model(inputs)
                
                loss = self.distillation_loss(student_logits, teacher_logits, targets)
                preds = student_logits.argmax(dim=1)
                accuracy = (preds == targets).float().mean()
                
                total_loss += loss.item()
                total_acc += accuracy.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches
        }


class FeatureDistillation:
    """
    Feature-based distillation.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        feature_layers: Dict[str, str],
        temperature: float = 4.0
    ):
        """
        Inicializar feature distillation.
        
        Args:
            student_model: Modelo estudiante
            teacher_model: Modelo profesor
            feature_layers: Mapeo de capas (student_layer: teacher_layer)
            temperature: Temperatura
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.feature_layers = feature_layers
        self.temperature = temperature
        
        self.student_features = {}
        self.teacher_features = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Registrar hooks para capturar features."""
        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook
        
        for student_layer, teacher_layer in self.feature_layers.items():
            student_module = dict(self.student_model.named_modules())[student_layer]
            teacher_module = dict(self.teacher_model.named_modules())[teacher_layer]
            
            student_module.register_forward_hook(get_student_hook(student_layer))
            teacher_module.register_forward_hook(get_teacher_hook(teacher_layer))
    
    def compute_loss(self) -> torch.Tensor:
        """
        Calcular feature distillation loss.
        
        Returns:
            Loss
        """
        total_loss = 0.0
        
        for student_layer, teacher_layer in self.feature_layers.items():
            student_feat = self.student_features[student_layer]
            teacher_feat = self.teacher_features[teacher_layer]
            
            # Alinear dimensiones si es necesario
            if student_feat.shape != teacher_feat.shape:
                # Interpolación o proyección
                if len(student_feat.shape) == 4:  # Conv features
                    student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[2:])
                else:
                    # Proyección lineal
                    if student_feat.shape[1] != teacher_feat.shape[1]:
                        projection = nn.Linear(student_feat.shape[1], teacher_feat.shape[1])
                        student_feat = projection(student_feat)
            
            # MSE loss entre features
            loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += loss
        
        return total_loss / len(self.feature_layers)




