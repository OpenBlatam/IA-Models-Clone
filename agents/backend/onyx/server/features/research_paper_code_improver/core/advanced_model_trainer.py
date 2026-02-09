"""
Advanced Model Trainer - Entrenador avanzado de modelos con PyTorch
====================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import os
import json

from .base_classes import BaseTrainer, BaseConfig
from .common_utils import get_device, move_to_device, timing_decorator
from .constants import (
    DEFAULT_DEVICE, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS, DEFAULT_WEIGHT_DECAY, DEFAULT_MAX_GRAD_NORM,
    DEFAULT_SEED
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig(BaseConfig):
    """Configuración de entrenamiento"""
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gradient_clip_norm: float = DEFAULT_MAX_GRAD_NORM
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    use_mixed_precision: bool = True
    device: Optional[str] = None  # Se resuelve con get_device()
    seed: int = DEFAULT_SEED


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp.isoformat()
        }


class AdvancedModelTrainer(BaseTrainer):
    """Entrenador avanzado de modelos con PyTorch"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config = config
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler = None
        self.training_history: List[TrainingMetrics] = []
        self.best_model_state: Optional[Dict] = None
        self.best_loss = float('inf')
        
        # Configurar device usando utilidades compartidas
        self.device = get_device(config.device)
        
        # Configurar mixed precision
        if config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Set seed
        self._set_seed(config.seed)
    
    def _set_seed(self, seed: int):
        """Establece seed para reproducibilidad"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    def setup_model(self, model: nn.Module):
        """Configura el modelo"""
        self.model = model.to(self.device)
        self.log_event("model_setup", {"device": str(self.device)})
        logger.info(f"Modelo configurado en {self.device}")
    
    def setup_optimizer(self, optimizer_class=optim.AdamW, **kwargs):
        """Configura el optimizador"""
        if self.model is None:
            raise ValueError("Modelo no configurado")
        
        default_params = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay
        }
        default_params.update(kwargs)
        
        self.optimizer = optimizer_class(self.model.parameters(), **default_params)
        logger.info(f"Optimizador configurado: {optimizer_class.__name__}")
    
    def setup_scheduler(self, scheduler_class=optim.lr_scheduler.CosineAnnealingLR, **kwargs):
        """Configura el scheduler"""
        if self.optimizer is None:
            raise ValueError("Optimizador no configurado")
        
        default_params = {"T_max": self.config.num_epochs}
        default_params.update(kwargs)
        
        self.scheduler = scheduler_class(self.optimizer, **default_params)
        logger.info(f"Scheduler configurado: {scheduler_class.__name__}")
    
    @timing_decorator
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable
    ) -> float:
        """Ejecuta un paso de entrenamiento"""
        if self.model is None or self.optimizer is None:
            raise ValueError("Modelo u optimizador no configurado")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Mover batch a device usando utilidades compartidas
        batch = move_to_device(batch, self.device)
        
        # Forward pass con mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                loss = loss_fn(outputs, batch)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
            loss = loss_fn(outputs, batch)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None
    ):
        """Entrena el modelo"""
        if self.model is None:
            raise ValueError("Modelo no configurado")
        
        if loss_fn is None:
            raise ValueError("Función de pérdida no especificada")
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Mover batch a device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                loss = self.train_step(batch, loss_fn)
                epoch_loss += loss
                num_batches += 1
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        step=global_step,
                        loss=loss,
                        learning_rate=current_lr
                    )
                    self.training_history.append(metrics)
                    logger.info(
                        f"Epoch {epoch}, Step {global_step}, Loss: {loss:.4f}, LR: {current_lr:.2e}"
                    )
                
                # Evaluation
                if eval_loader and global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_loader, eval_fn)
                    logger.info(f"Evaluation at step {global_step}: {eval_metrics}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{global_step}")
            
            # Epoch end
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint("best_model")
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        eval_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Evalúa el modelo"""
        if self.model is None:
            raise ValueError("Modelo no configurado")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        if eval_fn:
                            loss = eval_fn(outputs, batch)
                        else:
                            loss = outputs.loss if hasattr(outputs, 'loss') else 0
                else:
                    outputs = self.model(**batch)
                    if eval_fn:
                        loss = eval_fn(outputs, batch)
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else 0
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {"eval_loss": avg_loss}
    
    def save_checkpoint(self, checkpoint_name: str, save_dir: str = "./checkpoints"):
        """Guarda un checkpoint"""
        if self.model is None:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"{checkpoint_name}.pt")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config.__dict__,
            "best_loss": self.best_loss,
            "training_history": [m.to_dict() for m in self.training_history]
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint guardado: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Carga un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.model:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        self.training_history = [
            TrainingMetrics(**m) for m in checkpoint.get("training_history", [])
        ]
        
        logger.info(f"Checkpoint cargado: {checkpoint_path}")

