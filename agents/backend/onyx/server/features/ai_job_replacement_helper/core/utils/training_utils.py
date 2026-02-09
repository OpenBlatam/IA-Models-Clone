"""
Training Utilities - Utilidades para entrenamiento
====================================================

Funciones de utilidad para entrenamiento de modelos.
Sigue mejores prácticas de PyTorch.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None


class EarlyStopping:
    """Early stopping para detener entrenamiento cuando no hay mejora"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Args:
            patience: Número de epochs sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' para minimizar, 'max' para maximizar
            restore_best_weights: Si True, restaurar mejores pesos al parar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Verificar si se debe hacer early stopping.
        
        Args:
            score: Score actual (loss o métrica)
            model: Modelo para guardar pesos si es necesario
        
        Returns:
            True si se debe parar, False en caso contrario
        """
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
        
        return self.early_stop


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> optim.Optimizer:
    """
    Crear optimizador con configuración estándar.
    
    Args:
        model: Modelo PyTorch
        optimizer_type: Tipo de optimizador ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Argumentos adicionales para el optimizador
    
    Returns:
        Optimizador configurado
    """
    params = model.parameters()
    
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        optimizer = optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Crear learning rate scheduler.
    
    Args:
        optimizer: Optimizador
        scheduler_type: Tipo de scheduler ('cosine', 'step', 'plateau', 'exponential')
        **kwargs: Argumentos adicionales
    
    Returns:
        Scheduler configurado
    """
    if scheduler_type.lower() == "cosine":
        T_max = kwargs.pop("T_max", 10)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, **kwargs)
    elif scheduler_type.lower() == "step":
        step_size = kwargs.pop("step_size", 10)
        gamma = kwargs.pop("gamma", 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)
    elif scheduler_type.lower() == "plateau":
        mode = kwargs.pop("mode", "min")
        factor = kwargs.pop("factor", 0.1)
        patience = kwargs.pop("patience", 10)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, **kwargs
        )
    elif scheduler_type.lower() == "exponential":
        gamma = kwargs.pop("gamma", 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 10
) -> Tuple[float, Optional[float], Dict[str, Any]]:
    """
    Entrenar un epoch completo.
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con datos de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo (CPU/GPU)
        use_mixed_precision: Usar mixed precision training
        gradient_accumulation_steps: Pasos de acumulación de gradientes
        max_grad_norm: Norma máxima para gradient clipping
        scaler: GradScaler para mixed precision
    
    Returns:
        Tupla con (loss promedio, accuracy promedio)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0].to(device), batch[1].to(device)
        else:
            inputs = batch.to(device)
            targets = None
        
        # Forward pass
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = outputs.mean()
        else:
            outputs = model(inputs)
            if targets is not None:
                loss = criterion(outputs, targets)
            else:
                loss = outputs.mean()
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if max_grad_norm is not None and max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Calculate metrics
        total_loss += loss.item() * gradient_accumulation_steps
        
        if targets is not None and outputs.dim() > 1:
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else None
    
    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = True
) -> Tuple[float, Optional[float]]:
    """
    Validar un epoch completo.
    
    Args:
        model: Modelo a validar
        dataloader: DataLoader con datos de validación
        criterion: Función de pérdida
        device: Dispositivo (CPU/GPU)
        use_mixed_precision: Usar mixed precision
    
    Returns:
        Tupla con (loss promedio, accuracy promedio)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = None
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if targets is not None:
                        loss = criterion(outputs, targets)
                    else:
                        loss = outputs.mean()
            else:
                outputs = model(inputs)
                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = outputs.mean()
            
            total_loss += loss.item()
            
            if targets is not None and outputs.dim() > 1:
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else None
    
    return avg_loss, accuracy

