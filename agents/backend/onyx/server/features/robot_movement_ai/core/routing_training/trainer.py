"""
Route Trainer
=============

Entrenador para modelos de enrutamiento.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from tqdm import tqdm

from ..routing_models.base_model import BaseRouteModel
from ..routing_data.dataset import DataLoader
from .callbacks import Callback
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: Optional[str] = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10


class RouteTrainer:
    """
    Entrenador para modelos de enrutamiento.
    """
    
    def __init__(
        self,
        model: BaseRouteModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Inicializar entrenador.
        
        Args:
            model: Modelo a entrenar
            config: Configuración de entrenamiento
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            callbacks: Lista de callbacks (opcional)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        
        # Optimizador
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Scaler para mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
        
        # Criterio de pérdida
        self.criterion = nn.MSELoss()
        
        # Métricas
        self.metrics_calculator = MetricsCalculator()
        
        # Historial
        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Crear optimizador."""
        optimizers = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD
        }
        
        optimizer_class = optimizers.get(self.config.optimizer.lower(), optim.AdamW)
        
        if self.config.optimizer.lower() == "sgd":
            return optimizer_class(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            return optimizer_class(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self) -> Optional[Any]:
        """Crear scheduler."""
        if not self.config.scheduler:
            return None
        
        if self.config.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        return None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Entrenar una época.
        
        Returns:
            Métricas de la época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_features, batch_targets, batch_metadata in progress_bar:
            batch_features = batch_features.to(self.model.device)
            batch_targets = batch_targets.to(self.model.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Actualizar progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validar modelo.
        
        Returns:
            Métricas de validación
        """
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets, batch_metadata in self.val_loader:
                batch_features = batch_features.to(self.model.device)
                batch_targets = batch_targets.to(self.model.device)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(batch_features)
                        loss = self.criterion(outputs, batch_targets)
                else:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_outputs.append(outputs.cpu())
                all_targets.append(batch_targets.cpu())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calcular métricas adicionales
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate(all_outputs, all_targets)
        metrics["val_loss"] = avg_loss
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Entrenar modelo.
        
        Returns:
            Historial de entrenamiento
        """
        logger.info(f"Iniciando entrenamiento por {self.config.epochs} épocas")
        
        # Callbacks: on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Callbacks: on_epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # Entrenar
            train_metrics = self.train_epoch()
            
            # Validar
            val_metrics = self.validate()
            
            # Actualizar scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()
            
            # Combinar métricas
            epoch_metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics
            }
            
            self.history.append(epoch_metrics)
            
            # Actualizar mejor pérdida
            val_loss = val_metrics.get("val_loss", float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.config.save_best_model:
                    self._save_checkpoint(is_best=True)
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Época {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f}" +
                    (f" - Val Loss: {val_loss:.4f}" if val_loss != float('inf') else "")
                )
            
            # Callbacks: on_epoch_end
            for callback in self.callbacks:
                should_stop = callback.on_epoch_end(self, epoch, epoch_metrics)
                if should_stop:
                    logger.info(f"Early stopping en época {epoch + 1}")
                    break
        
        # Callbacks: on_train_end
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        logger.info("Entrenamiento completado")
        
        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss
        }
    
    def _save_checkpoint(self, is_best: bool = False):
        """Guardar checkpoint."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        self.model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata={
                "epoch": self.current_epoch + 1,
                "val_loss": self.best_val_loss,
                "history": self.history
            }
        )


