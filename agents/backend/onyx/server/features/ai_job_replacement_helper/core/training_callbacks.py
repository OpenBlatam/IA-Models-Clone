"""
Training Callbacks - Sistema de callbacks para entrenamiento
============================================================

Sistema de callbacks para personalizar el comportamiento del entrenamiento.
Sigue mejores prácticas de callbacks en deep learning.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Clase base para callbacks"""
    
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamado al inicio del entrenamiento"""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamado al final del entrenamiento"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamado al inicio de cada época"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamado al final de cada época"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamado al inicio de cada batch"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamado al final de cada batch"""
        pass


class EarlyStoppingCallback(Callback):
    """Callback para early stopping"""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Args:
            monitor: Métrica a monitorear
            patience: Paciencia (epochs sin mejora)
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' o 'max'
            restore_best_weights: Restaurar mejores pesos
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.wait = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Inicializar al inicio del entrenamiento"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('inf') if self.mode == "min" else float('-inf')
        self.counter = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verificar early stopping al final de época.
        
        Returns:
            True si se debe detener el entrenamiento
        """
        if logs is None:
            return False
        
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Early stopping monitor '{self.monitor}' not found in logs")
            return False
        
        if self.mode == "min":
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current
            self.wait = 0
            if self.restore_best_weights and logs.get("model") is not None:
                model = logs["model"]
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"Early stopping triggered at epoch {epoch}")
            if self.restore_best_weights and self.best_weights and logs.get("model") is not None:
                logs["model"].load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
            return True
        
        return False
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Finalizar al final del entrenamiento"""
        if self.stopped_epoch > 0:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")


class ModelCheckpointCallback(Callback):
    """Callback para guardar checkpoints"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = "min",
        period: int = 1
    ):
        """
        Args:
            filepath: Ruta donde guardar (puede incluir {epoch}, {monitor})
            monitor: Métrica a monitorear
            save_best_only: Solo guardar si es mejor
            save_weights_only: Solo guardar pesos del modelo
            mode: 'min' o 'max'
            period: Guardar cada N épocas
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.epochs_since_last_save = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Inicializar"""
        self.epochs_since_last_save = 0
        self.best_score = float('inf') if self.mode == "min" else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Guardar checkpoint si es necesario"""
        if logs is None:
            return
        
        self.epochs_since_last_save += 1
        
        # Check if should save
        if self.epochs_since_last_save < self.period:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Monitor '{self.monitor}' not found in logs")
            return
        
        # Check if best
        if self.save_best_only:
            if self.mode == "min":
                is_best = current < (self.best_score - 1e-6)
            else:
                is_best = current > (self.best_score + 1e-6)
            
            if not is_best:
                return
            
            self.best_score = current
        
        # Format filepath
        filepath = self.filepath.format(epoch=epoch, monitor=current)
        
        # Save checkpoint
        try:
            model = logs.get("model")
            if model is None:
                logger.warning("Model not found in logs, skipping checkpoint")
                return
            
            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": logs.get("optimizer").state_dict() if logs.get("optimizer") else None,
                    "scheduler_state_dict": logs.get("scheduler").state_dict() if logs.get("scheduler") else None,
                    "loss": logs.get("loss"),
                    "metric": current,
                }
                torch.save(checkpoint, filepath)
            
            logger.info(f"Checkpoint saved: {filepath}")
            self.epochs_since_last_save = 0
        
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}", exc_info=True)


class LearningRateSchedulerCallback(Callback):
    """Callback para learning rate scheduling"""
    
    def __init__(self, scheduler: Any):
        """
        Args:
            scheduler: Learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Actualizar learning rate"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if logs and self.scheduler.mode == "min":
                metric = logs.get("val_loss", logs.get("loss"))
            else:
                metric = logs.get("val_acc", logs.get("acc")) if logs else None
            
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        if logs:
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else None
            if current_lr:
                logs["learning_rate"] = current_lr


class TensorBoardCallback(Callback):
    """Callback para logging a TensorBoard"""
    
    def __init__(self, log_dir: str = "./logs/tensorboard"):
        """
        Args:
            log_dir: Directorio de logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.available = True
        except ImportError:
            logger.warning("TensorBoard not available")
            self.available = False
            self.writer = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics to TensorBoard"""
        if not self.available or logs is None:
            return
        
        try:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)
        except Exception as e:
            logger.error(f"Error logging to TensorBoard: {e}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Cerrar writer"""
        if self.writer:
            self.writer.close()


class CallbackList:
    """Lista de callbacks"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Args:
            callbacks: Lista de callbacks
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Agregar callback"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamar on_train_begin en todos los callbacks"""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamar on_train_end en todos los callbacks"""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamar on_epoch_begin en todos los callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Llamar on_epoch_end en todos los callbacks.
        
        Returns:
            True si algún callback indica que se debe detener
        """
        should_stop = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs)
            if result is True:
                should_stop = True
        return should_stop
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamar on_batch_begin en todos los callbacks"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Llamar on_batch_end en todos los callbacks"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)




