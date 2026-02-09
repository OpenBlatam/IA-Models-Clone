"""
Training Callbacks - Callbacks de Entrenamiento
================================================

Sistema de callbacks para entrenamiento.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Callback base"""
    
    @abstractmethod
    def on_train_begin(self, trainer: Any):
        """Al inicio del entrenamiento"""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: Any):
        """Al final del entrenamiento"""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, trainer: Any):
        """Al inicio de una época"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Al final de una época"""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        """Al final de un batch"""
        pass


class EarlyStoppingCallback(Callback):
    """Callback de early stopping"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        """
        Inicializar early stopping
        
        Args:
            patience: Paciencia (épocas sin mejora)
            min_delta: Delta mínimo para considerar mejora
            monitor: Métrica a monitorear
            mode: Modo (min o max)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.stopped = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Verificar early stopping"""
        if self.monitor not in metrics:
            return
        
        current_score = metrics[self.monitor]
        
        if self.mode == "min":
            is_better = current_score < (self.best_score - self.min_delta)
        else:
            is_better = current_score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = current_score
            self.counter = 0
            logger.info(f"Mejora en {self.monitor}: {current_score:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                trainer.should_stop = True
                logger.info(f"Early stopping activado después de {epoch} épocas")
    
    def on_train_begin(self, trainer: Any):
        self.counter = 0
        self.stopped = False
    
    def on_train_end(self, trainer: Any):
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: Any):
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        pass


class ModelCheckpointCallback(Callback):
    """Callback para guardar checkpoints"""
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        """
        Inicializar checkpoint callback
        
        Args:
            save_dir: Directorio para guardar
            save_best: Guardar mejor modelo
            save_last: Guardar último modelo
            monitor: Métrica a monitorear
            mode: Modo (min o max)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Guardar checkpoint"""
        if self.save_last:
            checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metrics": metrics
            }, checkpoint_path)
            logger.info(f"Checkpoint guardado: {checkpoint_path}")
        
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            is_better = (
                current_score < (self.best_score - 1e-6) if self.mode == "min"
                else current_score > (self.best_score + 1e-6)
            )
            
            if is_better:
                self.best_score = current_score
                best_path = f"{self.save_dir}/best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "metrics": metrics
                }, best_path)
                logger.info(f"Mejor modelo guardado: {best_path}")
    
    def on_train_begin(self, trainer: Any):
        pass
    
    def on_train_end(self, trainer: Any):
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: Any):
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        pass


class LearningRateSchedulerCallback(Callback):
    """Callback para learning rate scheduling"""
    
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        """
        Inicializar scheduler callback
        
        Args:
            scheduler: Scheduler de learning rate
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Actualizar learning rate"""
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        logger.info(f"Learning rate actualizado: {current_lr:.6f}")
    
    def on_train_begin(self, trainer: Any):
        pass
    
    def on_train_end(self, trainer: Any):
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: Any):
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        pass


class CallbackManager:
    """Gestor de callbacks"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Inicializar gestor
        
        Args:
            callbacks: Lista de callbacks
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback):
        """Agregar callback"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any):
        """Ejecutar al inicio del entrenamiento"""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any):
        """Ejecutar al final del entrenamiento"""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, epoch: int, trainer: Any):
        """Ejecutar al inicio de época"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Ejecutar al final de época"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        """Ejecutar al final de batch"""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, trainer)




