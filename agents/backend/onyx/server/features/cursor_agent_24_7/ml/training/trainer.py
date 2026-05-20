"""
Trainer - Entrenador de modelos
================================

Clase para entrenar modelos de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

try:
    from ..models.base import BaseModel
except ImportError:
    # Fallback para cuando se usa directamente
    BaseModel = torch.nn.Module

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./checkpoints"


class TrainingCallback(ABC):
    """Callback abstracto para entrenamiento"""
    
    @abstractmethod
    def on_train_begin(self, trainer: 'Trainer'):
        """Llamado al inicio del entrenamiento"""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: 'Trainer'):
        """Llamado al final del entrenamiento"""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, trainer: 'Trainer', epoch: int):
        """Llamado al inicio de cada época"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]):
        """Llamado al final de cada época"""
        pass
    
    @abstractmethod
    def on_step_end(self, trainer: 'Trainer', step: int, loss: float):
        """Llamado al final de cada paso"""
        pass


class Trainer:
    """Entrenador de modelos"""
    
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        
        # Optimizador
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
        
        # Estado
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Entrenar modelo"""
        # Callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)
                
                # Entrenar época
                train_metrics = self._train_epoch()
                
                # Validar
                val_metrics = {}
                if self.val_loader:
                    val_metrics = self._validate()
                
                # Callbacks
                metrics = {**train_metrics, **val_metrics}
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, metrics)
                
                # Guardar checkpoint
                if (epoch + 1) % (self.config.save_steps // len(self.train_loader)) == 0:
                    self._save_checkpoint(epoch)
        
        finally:
            # Callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Entrenar una época"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass con mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Callbacks
            if self.global_step % self.config.logging_steps == 0:
                for callback in self.callbacks:
                    callback.on_step_end(self, self.global_step, loss.item())
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def _validate(self) -> Dict[str, float]:
        """Validar modelo"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Guardar mejor modelo
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(self.current_epoch, is_best=True)
        
        return {"val_loss": avg_loss}
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Guardar checkpoint"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-epoch-{epoch}.pt"
        )
        
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            self.model.save(best_path)
        
        self.model.save(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

