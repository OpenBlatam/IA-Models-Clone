"""
Training Callbacks System - Sistema de callbacks para entrenamiento
=====================================================================
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CallbackEvent(Enum):
    """Eventos de callback"""
    ON_TRAIN_START = "on_train_start"
    ON_TRAIN_END = "on_train_end"
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_VALIDATION_START = "on_validation_start"
    ON_VALIDATION_END = "on_validation_end"
    ON_BACKWARD_END = "on_backward_end"
    ON_OPTIMIZER_STEP = "on_optimizer_step"


@dataclass
class CallbackState:
    """Estado del callback"""
    epoch: int = 0
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    loss: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class TrainingCallback:
    """Callback base para entrenamiento"""
    
    def __init__(self):
        self.enabled = True
    
    def on_train_start(self, state: CallbackState):
        """Llamado al inicio del entrenamiento"""
        pass
    
    def on_train_end(self, state: CallbackState):
        """Llamado al final del entrenamiento"""
        pass
    
    def on_epoch_start(self, state: CallbackState):
        """Llamado al inicio de cada epoch"""
        pass
    
    def on_epoch_end(self, state: CallbackState):
        """Llamado al final de cada epoch"""
        pass
    
    def on_batch_start(self, state: CallbackState):
        """Llamado al inicio de cada batch"""
        pass
    
    def on_batch_end(self, state: CallbackState):
        """Llamado al final de cada batch"""
        pass
    
    def on_validation_start(self, state: CallbackState):
        """Llamado al inicio de validación"""
        pass
    
    def on_validation_end(self, state: CallbackState):
        """Llamado al final de validación"""
        pass
    
    def on_backward_end(self, state: CallbackState):
        """Llamado después de backward"""
        pass
    
    def on_optimizer_step(self, state: CallbackState):
        """Llamado después de optimizer step"""
        pass


class CallbackManager:
    """Gestor de callbacks"""
    
    def __init__(self):
        self.callbacks: List[TrainingCallback] = []
        self.state = CallbackState()
    
    def add_callback(self, callback: TrainingCallback):
        """Agrega un callback"""
        self.callbacks.append(callback)
        logger.info(f"Callback {type(callback).__name__} agregado")
    
    def remove_callback(self, callback: TrainingCallback):
        """Remueve un callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def trigger(self, event: CallbackEvent, **kwargs):
        """Dispara un evento"""
        self.state.__dict__.update(kwargs)
        
        for callback in self.callbacks:
            if not callback.enabled:
                continue
            
            try:
                if event == CallbackEvent.ON_TRAIN_START:
                    callback.on_train_start(self.state)
                elif event == CallbackEvent.ON_TRAIN_END:
                    callback.on_train_end(self.state)
                elif event == CallbackEvent.ON_EPOCH_START:
                    callback.on_epoch_start(self.state)
                elif event == CallbackEvent.ON_EPOCH_END:
                    callback.on_epoch_end(self.state)
                elif event == CallbackEvent.ON_BATCH_START:
                    callback.on_batch_start(self.state)
                elif event == CallbackEvent.ON_BATCH_END:
                    callback.on_batch_end(self.state)
                elif event == CallbackEvent.ON_VALIDATION_START:
                    callback.on_validation_start(self.state)
                elif event == CallbackEvent.ON_VALIDATION_END:
                    callback.on_validation_end(self.state)
                elif event == CallbackEvent.ON_BACKWARD_END:
                    callback.on_backward_end(self.state)
                elif event == CallbackEvent.ON_OPTIMIZER_STEP:
                    callback.on_optimizer_step(self.state)
            except Exception as e:
                logger.error(f"Error en callback {type(callback).__name__}: {e}")


class LoggingCallback(TrainingCallback):
    """Callback para logging"""
    
    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
    
    def on_epoch_end(self, state: CallbackState):
        """Log al final de epoch"""
        logger.info(
            f"Epoch {state.epoch} - Loss: {state.loss:.4f}, "
            f"Metrics: {state.metrics}"
        )
    
    def on_batch_end(self, state: CallbackState):
        """Log periódico durante entrenamiento"""
        if state.step % self.log_interval == 0:
            logger.info(
                f"Step {state.step} - Loss: {state.loss:.4f}"
            )


class CheckpointCallback(TrainingCallback):
    """Callback para checkpointing"""
    
    def __init__(self, checkpointer, save_interval: int = 1):
        super().__init__()
        self.checkpointer = checkpointer
        self.save_interval = save_interval
    
    def on_epoch_end(self, state: CallbackState):
        """Guarda checkpoint al final de epoch"""
        if state.epoch % self.save_interval == 0:
            self.checkpointer.save_checkpoint(
                epoch=state.epoch,
                step=state.step,
                metrics=state.metrics
            )


class EarlyStoppingCallback(TrainingCallback):
    """Callback para early stopping"""
    
    def __init__(self, early_stopper):
        super().__init__()
        self.early_stopper = early_stopper
        self.should_stop = False
    
    def on_epoch_end(self, state: CallbackState):
        """Verifica early stopping"""
        if self.early_stopper(state.epoch, state.metrics, None):
            self.should_stop = True
            logger.info("Early stopping activado")




