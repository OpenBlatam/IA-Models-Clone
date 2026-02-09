"""
Training Utils - Utilidades de Entrenamiento
=============================================

Utilidades para entrenamiento de modelos PyTorch con mejores prácticas.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints"
    save_every: int = 1
    eval_every: int = 1
    log_every: int = 100
    warmup_steps: int = 0
    use_amp: bool = True


class EarlyStopping:
    """
    Early stopping para prevenir overfitting.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Inicializar early stopping.
        
        Args:
            patience: Número de epochs sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: "min" o "max" (minimizar o maximizar métrica)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Verificar si debe parar.
        
        Args:
            score: Score actual
            
        Returns:
            True si debe parar
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Verificar si score actual es mejor"""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class LearningRateScheduler:
    """
    Scheduler de learning rate con múltiples estrategias.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        strategy: str = "cosine",
        **kwargs
    ):
        """
        Inicializar scheduler.
        
        Args:
            optimizer: Optimizador
            strategy: Estrategia (cosine, linear, step, plateau)
            **kwargs: Argumentos adicionales según estrategia
        """
        self.optimizer = optimizer
        self.strategy = strategy
        
        if strategy == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 10),
                eta_min=kwargs.get("eta_min", 0)
            )
        elif strategy == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=kwargs.get("start_factor", 1.0),
                end_factor=kwargs.get("end_factor", 0.0),
                total_iters=kwargs.get("total_iters", 10)
            )
        elif strategy == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 5),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif strategy == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 3)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def step(self, metric: Optional[float] = None) -> None:
        """
        Actualizar learning rate.
        
        Args:
            metric: Métrica para plateau scheduler
        """
        if self.strategy == "plateau" and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Obtener learning rate actual"""
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    Trainer para modelos PyTorch con mejores prácticas.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[LearningRateScheduler] = None
    ):
        """
        Inicializar trainer.
        
        Args:
            model: Modelo PyTorch
            config: Configuración de entrenamiento
            optimizer: Optimizador (se crea si no se proporciona)
            criterion: Función de pérdida (se crea si no se proporciona)
            scheduler: Scheduler de LR (opcional)
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizador por defecto
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Criterion por defecto
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        self.scheduler = scheduler
        
        # Mixed precision
        self.scaler = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Early stopping
        self.early_stopping = None
        
        # Historial
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Entrenar un epoch.
        
        Args:
            train_loader: DataLoader de entrenamiento
            
        Returns:
            Diccionario con métricas
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Mover batch a device
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None
            
            # Forward pass con mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_amp and self.device.type == "cuda"):
                if targets is not None:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs.get('loss', torch.tensor(0.0))
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Métricas
            total_loss += loss.item()
            total_samples += inputs.size(0)
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        avg_loss = total_loss / len(train_loader)
        return {"loss": avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validar modelo.
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Diccionario con métricas
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                with torch.cuda.amp.autocast(enabled=self.config.use_amp and self.device.type == "cuda"):
                    if targets is not None:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        # Accuracy
                        if outputs.dim() > 1:
                            _, predicted = torch.max(outputs.data, 1)
                            total += targets.size(0)
                            correct += (predicted == targets).sum().item()
                    else:
                        outputs = self.model(inputs)
                        loss = outputs if isinstance(outputs, torch.Tensor) else outputs.get('loss', torch.tensor(0.0))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        early_stopping: Optional[EarlyStopping] = None
    ) -> Dict[str, List[float]]:
        """
        Entrenar modelo completo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            early_stopping: Early stopping (opcional)
            
        Returns:
            Historial de entrenamiento
        """
        self.early_stopping = early_stopping
        
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Entrenar
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            
            # Validar
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                if "accuracy" in val_metrics:
                    self.history["val_acc"].append(val_metrics["accuracy"])
                
                logger.info(
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.2f}%"
                )
                
                # Early stopping
                if early_stopping:
                    if early_stopping(val_metrics["loss"]):
                        logger.info("Early stopping triggered")
                        break
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step(val_metrics["loss"])
            else:
                if self.scheduler:
                    self.scheduler.step()
            
            # Guardar checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        return self.history
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        Guardar checkpoint.
        
        Args:
            epoch: Número de epoch
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "history": self.history
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = Path(self.config.save_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Cargar checkpoint.
        
        Args:
            path: Ruta del checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded: {path}")




