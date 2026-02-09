"""
Callbacks Module
=================

Sistema de callbacks profesional para pipelines de entrenamiento.
Permite extensibilidad y monitoreo avanzado.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Clase base abstracta para callbacks."""
    
    @abstractmethod
    def on_train_begin(self, logs: Dict[str, Any]):
        """Llamado al inicio del entrenamiento."""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Dict[str, Any]):
        """Llamado al final del entrenamiento."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Llamado al inicio de cada época."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Llamado al final de cada época."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        """Llamado al inicio de cada batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        """Llamado al final de cada batch."""
        pass


class LearningRateSchedulerCallback(Callback):
    """Callback para logging de learning rate."""
    
    def __init__(self, log_interval: int = 10):
        """
        Inicializar callback.
        
        Args:
            log_interval: Intervalo de logging
        """
        self.log_interval = log_interval
        self.lr_history: List[float] = []
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Inicializar historial."""
        self.lr_history = []
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Registrar learning rate."""
        if 'learning_rate' in logs:
            self.lr_history.append(logs['learning_rate'])
            if epoch % self.log_interval == 0:
                logger.info(f"Epoch {epoch}: LR = {logs['learning_rate']:.6f}")
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Finalizar callback."""
        logger.info(f"Learning rate history: {len(self.lr_history)} entries")
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        pass


class ModelCheckpointCallback(Callback):
    """Callback para guardar checkpoints."""
    
    def __init__(
        self,
        save_path: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        """
        Inicializar callback.
        
        Args:
            save_path: Ruta para guardar checkpoints
            monitor: Métrica a monitorear
            mode: "min" o "max"
            save_best_only: Solo guardar el mejor modelo
            save_freq: Frecuencia de guardado (cada N épocas)
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.model = None
    
    def set_model(self, model: nn.Module):
        """Establecer modelo a guardar."""
        self.model = model
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Guardar checkpoint si es necesario."""
        if self.model is None:
            return
        
        if self.monitor not in logs:
            logger.warning(f"Metric {self.monitor} not found in logs")
            return
        
        current_score = logs[self.monitor]
        should_save = False
        
        if self.save_best_only:
            if self.mode == "min":
                if current_score < self.best_score:
                    self.best_score = current_score
                    should_save = True
            else:
                if current_score > self.best_score:
                    self.best_score = current_score
                    should_save = True
        else:
            if epoch % self.save_freq == 0:
                should_save = True
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'metrics': logs
            }
            torch.save(checkpoint, f"{self.save_path}/checkpoint_epoch_{epoch}.pt")
            logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def on_train_begin(self, logs: Dict[str, Any]):
        pass
    
    def on_train_end(self, logs: Dict[str, Any]):
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        pass


class ProgressBarCallback(Callback):
    """Callback para mostrar barra de progreso."""
    
    def __init__(self):
        """Inicializar callback."""
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.TQDM_AVAILABLE = True
        except ImportError:
            self.TQDM_AVAILABLE = False
            logger.warning("tqdm not available. Progress bar disabled.")
        
        self.pbar = None
        self.current_epoch = 0
        self.total_epochs = 0
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Inicializar barra de progreso."""
        self.total_epochs = logs.get('num_epochs', 100)
        if self.TQDM_AVAILABLE:
            self.pbar = self.tqdm(total=self.total_epochs, desc="Training")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Actualizar barra de progreso."""
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix(logs)
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Cerrar barra de progreso."""
        if self.pbar:
            self.pbar.close()
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        pass


class TimingCallback(Callback):
    """Callback para medir tiempos de entrenamiento."""
    
    def __init__(self):
        """Inicializar callback."""
        self.start_time = None
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Iniciar cronómetro."""
        self.start_time = time.time()
        self.epoch_times = []
        self.batch_times = []
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Iniciar cronómetro de época."""
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Registrar tiempo de época."""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        logs['epoch_time'] = epoch_time
        logger.info(f"Epoch {epoch} took {epoch_time:.2f} seconds")
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        """Iniciar cronómetro de batch."""
        self.batch_start = time.time()
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        """Registrar tiempo de batch."""
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        logs['batch_time'] = batch_time
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Finalizar y reportar tiempos."""
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        
        logger.info(f"Total training time: {total_time:.2f} seconds")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        logger.info(f"Average batch time: {avg_batch_time:.4f} seconds")
        
        logs['total_time'] = total_time
        logs['avg_epoch_time'] = avg_epoch_time
        logs['avg_batch_time'] = avg_batch_time


class CallbackList:
    """Lista de callbacks para ejecutar en orden."""
    
    def __init__(self, callbacks: List[Callback]):
        """
        Inicializar lista de callbacks.
        
        Args:
            callbacks: Lista de callbacks
        """
        self.callbacks = callbacks
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Ejecutar on_train_begin en todos los callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Ejecutar on_train_end en todos los callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Ejecutar on_epoch_begin en todos los callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Ejecutar on_epoch_end en todos los callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        """Ejecutar on_batch_begin en todos los callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        """Ejecutar on_batch_end en todos los callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

